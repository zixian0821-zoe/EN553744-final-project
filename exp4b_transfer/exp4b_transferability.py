\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
   
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_PIPELINE_DIR = _Path(__file__).resolve().parent.parent / "pipeline"
if str(_PIPELINE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_PIPELINE_DIR))

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from torch_geometric.nn import ChebConv

from recommendation_config import RecommendationConfig
from recommendation_evaluate import build_eval_split_cache, evaluate_test_split
from recommendation_graphs import build_fixed_fused_operator
from recommendation_loss_sampling import (
    build_negative_sampling_distribution,
    build_train_positive_cache,
    build_target_train_item_popularity,
    bpr_loss,
    sample_bpr_triplets,
)
from recommendation_models import ItemRepresentationModule, score
from train_recommendation import (
    Stage1Artifacts,
    TrainConfig,
    load_stage1_artifacts,
    resolve_device,
    scipy_to_torch_sparse,
    set_seed,
)

GRAPH_NAMES = ["source", "target", "fused"]
METRIC_KEYS = ["NDCG@10", "NDCG@20", "Recall@10", "Recall@20", "HitRate@20"]
PRIMARY_METRIC = "NDCG@20"
CHEB_K = 3

class ChebNetRecommender(nn.Module):
                                                                            

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        in_dim: int,
        n_items: int,
        genre_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        K: int = 3,
    ) -> None:
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.conv1 = ChebConv(in_dim, hidden_dim, K=K, normalization="sym")
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K, normalization="sym")
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.item_module = ItemRepresentationModule(n_items, genre_dim, hidden_dim)

    def get_user_embeddings(self, x_user: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x_user, self.edge_index, self.edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, self.edge_index, self.edge_weight)
        h = F.relu(h)
        return self.fc(h)

    def get_item_embeddings(self, item_genre_features: torch.Tensor) -> torch.Tensor:
        return self.item_module(item_genre_features)

    def get_alpha(self) -> float | None:
        return None

def build_operators_scipy(stage1: Stage1Artifacts) -> dict[str, sparse.csr_matrix]:
    fused_op = build_fixed_fused_operator(
        stage1.source_operator, stage1.target_operator, alpha=0.5
    )
    return {
        "source": stage1.source_operator,
        "target": stage1.target_operator,
        "fused": fused_op,
    }

def operator_to_edge_repr(
    operator: sparse.csr_matrix, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
                                                                               
    coo = operator.tocoo()
    edge_index = torch.tensor(
        np.vstack((coo.row, coo.col)), dtype=torch.long, device=device
    )
    edge_weight = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return edge_index, edge_weight

def build_edge_reprs(
    operators_scipy: dict[str, sparse.csr_matrix], device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        name: operator_to_edge_repr(op, device)
        for name, op in operators_scipy.items()
    }

def train_chebnet_on_graph(
    graph_name: str,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    stage1: Stage1Artifacts,
    config: RecommendationConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> dict:
                                                      
    set_seed(train_config.seed)

    x_user = torch.tensor(stage1.user_features, dtype=torch.float32, device=device)
    item_genre_features = scipy_to_torch_sparse(stage1.item_genre_features, device=device)

    n_items = stage1.item_genre_features.shape[0]
    genre_dim = stage1.item_genre_features.shape[1]
    in_dim = stage1.user_features.shape[1]

    model = ChebNetRecommender(
        edge_index=edge_index,
        edge_weight=edge_weight,
        in_dim=in_dim,
        n_items=n_items,
        genre_dim=genre_dim,
        hidden_dim=train_config.hidden_dim,
        dropout=train_config.dropout,
        K=CHEB_K,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    train_popularity = build_target_train_item_popularity(stage1.train_matrix)
    neg_dist = build_negative_sampling_distribution(
        train_item_popularity=train_popularity,
        candidate_item_universe=stage1.candidate_item_universe,
        power=config.negative_sampling_power,
    )
    pos_cache = build_train_positive_cache(stage1.train_matrix)
    rng = np.random.default_rng(train_config.seed)
    candidate_ids_t = torch.tensor(
        stage1.candidate_item_universe, dtype=torch.long, device=device
    )

    val_exclude = (stage1.train_matrix + stage1.test_matrix).sign().tocsr()
    val_eval_cache = build_eval_split_cache(
        split_name="val",
        relevant_matrix=stage1.val_matrix,
        exclude_matrix=val_exclude,
        metric_user_mask=stage1.user_masks["val_metric_user_mask"],
        candidate_item_universe=stage1.candidate_item_universe,
        device=device,
    )

    best_state = None
    best_val_ndcg = float("-inf")
    best_epoch = 0
    no_improve = 0
    t0 = time.perf_counter()

    for epoch in range(1, train_config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        users_np, pos_np, neg_np = sample_bpr_triplets(
            positive_cache=pos_cache, distribution=neg_dist, rng=rng,
        )
        user_ids = torch.tensor(users_np, dtype=torch.long, device=device)
        pos_ids = torch.tensor(pos_np, dtype=torch.long, device=device)
        neg_ids = torch.tensor(neg_np, dtype=torch.long, device=device)

        u_emb = model.get_user_embeddings(x_user)
        i_emb = model.get_item_embeddings(item_genre_features)
        loss = bpr_loss(
            score(u_emb[user_ids], i_emb[pos_ids]),
            score(u_emb[user_ids], i_emb[neg_ids]),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=train_config.grad_clip_norm
        )
        optimizer.step()

        model.eval()
        with torch.no_grad():
            u_emb_v = model.get_user_embeddings(x_user)
            i_emb_v = model.get_item_embeddings(item_genre_features)
            cand_emb = i_emb_v.index_select(0, candidate_ids_t)
            val_result = evaluate_test_split(
                user_embeddings=u_emb_v,
                candidate_item_embeddings=cand_emb,
                eval_cache=val_eval_cache,
                batch_size_users=train_config.eval_batch_size_users,
            )
        val_ndcg = val_result.metrics["NDCG@20"]

        if epoch <= 3 or epoch % 20 == 0:
            print(
                f"  [{graph_name}] epoch {epoch:>3d}  "
                f"bpr={loss.item():.4f}  val_ndcg@20={val_ndcg:.6f}"
            )

        if val_ndcg > best_val_ndcg + 1e-12:
            best_val_ndcg = val_ndcg
            best_epoch = epoch
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= train_config.patience:
                break

    elapsed = time.perf_counter() - t0
    print(
        f"  [{graph_name}] done — best epoch {best_epoch}, "
        f"val NDCG@20={best_val_ndcg:.6f}, {elapsed:.1f}s"
    )

    return {
        "graph_name": graph_name,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_ndcg": best_val_ndcg,
        "train_seconds": elapsed,
    }

@torch.no_grad()
def evaluate_with_graph(
    state_dict: dict,
    eval_edge_index: torch.Tensor,
    eval_edge_weight: torch.Tensor,
    stage1: Stage1Artifacts,
    train_config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
                                                                                 
    in_dim = stage1.user_features.shape[1]
    n_items = stage1.item_genre_features.shape[0]
    genre_dim = stage1.item_genre_features.shape[1]

    model = ChebNetRecommender(
        edge_index=eval_edge_index,
        edge_weight=eval_edge_weight,
        in_dim=in_dim,
        n_items=n_items,
        genre_dim=genre_dim,
        hidden_dim=train_config.hidden_dim,
        dropout=train_config.dropout,
        K=CHEB_K,
    ).to(device)

    learned_params = {
        k: v for k, v in state_dict.items()
        if k not in ("edge_index", "edge_weight")
    }
    model.load_state_dict(learned_params, strict=False)
    model.edge_index = eval_edge_index
    model.edge_weight = eval_edge_weight
    model.eval()

    x_user = torch.tensor(stage1.user_features, dtype=torch.float32, device=device)
    item_genre_features = scipy_to_torch_sparse(
        stage1.item_genre_features, device=device
    )
    candidate_ids_t = torch.tensor(
        stage1.candidate_item_universe, dtype=torch.long, device=device
    )

    test_exclude = (stage1.train_matrix + stage1.val_matrix).sign().tocsr()
    test_eval_cache = build_eval_split_cache(
        split_name="test",
        relevant_matrix=stage1.test_matrix,
        exclude_matrix=test_exclude,
        metric_user_mask=stage1.user_masks["test_metric_user_mask"],
        candidate_item_universe=stage1.candidate_item_universe,
        device=device,
    )

    u_emb = model.get_user_embeddings(x_user)
    i_emb = model.get_item_embeddings(item_genre_features)
    cand_emb = i_emb.index_select(0, candidate_ids_t)
    result = evaluate_test_split(
        user_embeddings=u_emb,
        candidate_item_embeddings=cand_emb,
        eval_cache=test_eval_cache,
        batch_size_users=train_config.eval_batch_size_users,
    )
    return result.metrics

def operator_frobenius_distance(
    ops: dict[str, sparse.csr_matrix],
) -> dict[tuple[str, str], float]:
                                                             
    distances = {}
    names = list(ops.keys())
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i <= j:
                diff = ops[a] - ops[b]
                dist = sparse.linalg.norm(diff, ord="fro")
                distances[(a, b)] = float(dist)
                distances[(b, a)] = float(dist)
    return distances

def plot_transferability_heatmap(
    matrix: np.ndarray,
    metric_name: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"Eval: {g}" for g in GRAPH_NAMES], fontsize=11)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Train: {g}" for g in GRAPH_NAMES], fontsize=11)
    ax.set_title(
        f"Transferability Matrix ({metric_name}) — ChebNet K={CHEB_K}",
        fontsize=12, fontweight="bold",
    )

    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            color = "white" if val > (matrix.max() + matrix.min()) / 2 else "black"
            ax.text(
                j, i, f"{val:.6f}", ha="center", va="center",
                fontsize=10, color=color,
            )

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

def plot_distance_vs_degradation(
    distances: dict[tuple[str, str], float],
    transfer_matrix: np.ndarray,
    save_path: Path,
) -> None:
                                                                             
    xs, ys, labels = [], [], []
    for i, train_g in enumerate(GRAPH_NAMES):
        diag_val = transfer_matrix[i, i]
        for j, eval_g in enumerate(GRAPH_NAMES):
            if i == j:
                continue
            dist = distances[(train_g, eval_g)]
            degradation = (diag_val - transfer_matrix[i, j]) / diag_val * 100.0
            xs.append(dist)
            ys.append(degradation)
            labels.append(f"{train_g}\u2192{eval_g}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=80, c="steelblue", edgecolors="navy", zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(
            lab, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9,
        )

    if len(xs) >= 2:
        coeffs = np.polyfit(xs, ys, 1)
        x_line = np.linspace(min(xs) * 0.9, max(xs) * 1.1, 50)
        ax.plot(
            x_line, np.polyval(coeffs, x_line), "--", color="salmon",
            linewidth=1.5, label=f"linear fit (slope={coeffs[0]:.2f})",
        )
        ax.legend(fontsize=9)

    ax.set_xlabel("Operator Frobenius Distance \u2016S_i \u2212 S_j\u2016_F", fontsize=11)
    ax.set_ylabel("NDCG@20 Degradation (%)", fontsize=11)
    ax.set_title(
        f"Operator Distance vs Performance Drop \u2014 ChebNet K={CHEB_K}",
        fontsize=12, fontweight="bold",
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

def plot_operator_distance_bars(
    distances: dict[tuple[str, str], float],
    save_path: Path,
) -> None:
    pairs = [("source", "target"), ("source", "fused"), ("target", "fused")]
    pair_labels = ["source\u2013target", "source\u2013fused", "target\u2013fused"]
    vals = [distances[p] for p in pairs]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        pair_labels, vals,
        color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="black", width=0.5,
    )
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{v:.2f}", ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Frobenius Distance", fontsize=11)
    ax.set_title("Pairwise Operator Distance", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp4B: Transferability (ChebNet)")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick sanity run (20 epochs, patience 5)",
    )
    args = parser.parse_args()

    config = RecommendationConfig()
    train_config = TrainConfig(
        max_epochs=20 if args.smoke else 200,
        patience=5 if args.smoke else 20,
    )
    device = resolve_device()
    print(f"Device: {device}")
    print(f"Model: ChebNet (K={CHEB_K})")

    print("\n[1/5] Loading Stage 1 artifacts ...")
    stage1 = load_stage1_artifacts(config)
    operators_scipy = build_operators_scipy(stage1)
    edge_reprs = build_edge_reprs(operators_scipy, device)
    print(
        f"  Users: {stage1.user_features.shape[0]}, "
        f"Items: {stage1.item_genre_features.shape[0]}"
    )
    for gname, (ei, ew) in edge_reprs.items():
        print(f"  {gname} graph: {ei.shape[1]} edges")

    print(f"\n[2/5] Training 3 ChebNet (K={CHEB_K}) models (one per graph) ...")
    trained_models: dict[str, dict] = {}
    for gname in GRAPH_NAMES:
        print(f"\n--- Training ChebNet on {gname} graph ---")
        ei, ew = edge_reprs[gname]
        result = train_chebnet_on_graph(
            graph_name=gname,
            edge_index=ei,
            edge_weight=ew,
            stage1=stage1,
            config=config,
            train_config=train_config,
            device=device,
        )
        trained_models[gname] = result

    print("\n[3/5] Evaluating 3\u00d73 transferability matrix ...")
    all_metrics: dict[tuple[str, str], dict[str, float]] = {}
    for train_g in GRAPH_NAMES:
        for eval_g in GRAPH_NAMES:
            print(f"  Evaluating: train={train_g}, eval={eval_g}")
            ei_eval, ew_eval = edge_reprs[eval_g]
            metrics = evaluate_with_graph(
                state_dict=trained_models[train_g]["best_state"],
                eval_edge_index=ei_eval,
                eval_edge_weight=ew_eval,
                stage1=stage1,
                train_config=train_config,
                device=device,
            )
            all_metrics[(train_g, eval_g)] = metrics
            print(
                f"    NDCG@20={metrics['NDCG@20']:.6f}  "
                f"Recall@20={metrics['Recall@20']:.6f}  "
                f"HitRate@20={metrics['HitRate@20']:.6f}"
            )

    metric_matrices: dict[str, np.ndarray] = {}
    for mk in METRIC_KEYS:
        mat = np.zeros((3, 3), dtype=np.float64)
        for i, tg in enumerate(GRAPH_NAMES):
            for j, eg in enumerate(GRAPH_NAMES):
                mat[i, j] = all_metrics[(tg, eg)][mk]
        metric_matrices[mk] = mat

    print(f"\n  Transferability Matrix ({PRIMARY_METRIC}) — ChebNet K={CHEB_K}:")
    _header = "Train \\ Eval"
    print(f"  {_header:>15s}", end="")
    for g in GRAPH_NAMES:
        print(f"  {g:>10s}", end="")
    print()
    for i, tg in enumerate(GRAPH_NAMES):
        print(f"  {tg:>15s}", end="")
        for j in range(3):
            val = metric_matrices[PRIMARY_METRIC][i, j]
            marker = " *" if i == j else "  "
            print(f"  {val:>10.6f}{marker}", end="")
        print()

    print("\n[4/5] Computing operator Frobenius distances ...")
    frob_distances = operator_frobenius_distance(operators_scipy)
    print("  Pairwise distances:")
    for (a, b), d in sorted(frob_distances.items()):
        if a <= b:
            print(f"    \u2016S_{a} - S_{b}\u2016_F = {d:.4f}")

    print("\n[5/5] Saving results and plots ...")
    out_dir = config.results_dir / "exp4b_transferability"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_results = {
        "model": f"ChebNet (K={CHEB_K})",
        "graph_names": GRAPH_NAMES,
        "metric_keys": METRIC_KEYS,
        "primary_metric": PRIMARY_METRIC,
        "transferability_matrices": {
            mk: metric_matrices[mk].tolist() for mk in METRIC_KEYS
        },
        "operator_frobenius_distances": {
            f"{a}-{b}": d for (a, b), d in frob_distances.items() if a <= b
        },
        "training_info": {
            gname: {
                "best_epoch": info["best_epoch"],
                "best_val_ndcg": info["best_val_ndcg"],
                "train_seconds": info["train_seconds"],
            }
            for gname, info in trained_models.items()
        },
    }
    json_path = out_dir / "transferability_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved: {json_path}")

    plot_transferability_heatmap(
        metric_matrices[PRIMARY_METRIC],
        PRIMARY_METRIC,
        out_dir / "transferability_heatmap_ndcg20.png",
    )
    for mk in ["Recall@20", "HitRate@20"]:
        safe_name = mk.replace("@", "at")
        plot_transferability_heatmap(
            metric_matrices[mk], mk,
            out_dir / f"transferability_heatmap_{safe_name}.png",
        )
    plot_distance_vs_degradation(
        frob_distances,
        metric_matrices[PRIMARY_METRIC],
        out_dir / "distance_vs_degradation.png",
    )
    plot_operator_distance_bars(
        frob_distances, out_dir / "operator_distance_bars.png",
    )

    print("\n\u2713 Experiment 4B (ChebNet) complete!")

if __name__ == "__main__":
    main()
