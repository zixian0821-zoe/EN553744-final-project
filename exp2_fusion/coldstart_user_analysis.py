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
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from torch_geometric.nn import ChebConv

from recommendation_config import RecommendationConfig
from recommendation_evaluate import (
    EvalSplitCache,
    build_eval_split_cache,
    _rank_topk_gpu,
    _compute_hits_gpu,
)
from recommendation_graphs import build_fixed_fused_operator
from recommendation_loss_sampling import (
    build_negative_sampling_distribution,
    build_train_positive_cache,
    build_target_train_item_popularity,
    bpr_loss,
    sample_bpr_triplets,
)
from recommendation_models import (
    GraphRecommender,
    ItemRepresentationModule,
    NoGraphRecommender,
    score,
)
from train_recommendation import (
    Stage1Artifacts,
    TrainConfig,
    load_stage1_artifacts,
    resolve_device,
    scipy_to_torch_sparse,
    set_seed,
)

CHEB_K = 3

class ChebNetRecommender(nn.Module):
    def __init__(self, edge_index, edge_weight, in_dim, n_items, genre_dim,
                 hidden_dim=64, dropout=0.3, K=3):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.conv1 = ChebConv(in_dim, hidden_dim, K=K, normalization="sym")
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K, normalization="sym")
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.item_module = ItemRepresentationModule(n_items, genre_dim, hidden_dim)

    def get_user_embeddings(self, x_user):
        h = F.relu(self.conv1(x_user, self.edge_index, self.edge_weight))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, self.edge_index, self.edge_weight))
        return self.fc(h)

    def get_item_embeddings(self, item_genre_features):
        return self.item_module(item_genre_features)

    def get_alpha(self):
        return None

EPS = 1e-12

@torch.no_grad()
def compute_per_user_metrics(
    user_embeddings: torch.Tensor,
    candidate_item_embeddings: torch.Tensor,
    eval_cache: EvalSplitCache,
    batch_size_users: int = 1024,
) -> dict[str, np.ndarray]:
\
\
\
\
       
    max_k = 20
    topk_tensor = _rank_topk_gpu(
        user_embeddings=user_embeddings,
        candidate_item_embeddings=candidate_item_embeddings,
        eval_cache=eval_cache,
        topk=max_k,
        batch_size_users=batch_size_users,
    )
    device = topk_tensor.device
    hits_20 = _compute_hits_gpu(topk_tensor, eval_cache.relevant_padded)

    rel_f = eval_cache.relevant_counts.float().clamp(min=1.0)

    recall_20 = (hits_20.sum(1).float() / rel_f).cpu().numpy()

    discounts = 1.0 / torch.log2(
        torch.arange(2, max_k + 2, device=device, dtype=torch.float32)
    )
    dcg_20 = (hits_20.float() * discounts[:max_k]).sum(1)
    cum_disc = torch.cumsum(discounts, dim=0)
    twenty_t = torch.tensor(max_k, device=device)
    ideal_20 = torch.minimum(eval_cache.relevant_counts, twenty_t)
    idcg_20 = cum_disc[(ideal_20.clamp(min=1) - 1).long()]
    ndcg_20 = (dcg_20 / idcg_20.clamp(min=EPS)).cpu().numpy()

    hitrate_20 = hits_20.any(dim=1).float().cpu().numpy()

    return {
        "NDCG@20": ndcg_20,
        "Recall@20": recall_20,
        "HitRate@20": hitrate_20,
        "user_indices": eval_cache.user_indices,
    }

def build_degree_groups(target_adj: sparse.csr_matrix) -> dict:
\
\
       
    degrees = np.array(target_adj.sum(axis=1)).flatten()
    degrees_unweighted = np.diff(target_adj.indptr)

    groups = {
        "isolated (deg=0)": degrees_unweighted == 0,
        "sparse (deg 1-5)": (degrees_unweighted >= 1) & (degrees_unweighted <= 5),
        "moderate (deg 6-10)": (degrees_unweighted >= 6) & (degrees_unweighted <= 10),
        "connected (deg>10)": degrees_unweighted > 10,
    }
    return groups, degrees_unweighted

def train_model_and_get_embeddings(
    model: nn.Module,
    model_name: str,
    stage1: Stage1Artifacts,
    config: RecommendationConfig,
    train_config: TrainConfig,
    device: torch.device,
    x_user: torch.Tensor,
    item_genre_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
                                                                           
    set_seed(train_config.seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    train_pop = build_target_train_item_popularity(stage1.train_matrix)
    neg_dist = build_negative_sampling_distribution(
        train_item_popularity=train_pop,
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
        split_name="val", relevant_matrix=stage1.val_matrix,
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
        loss = bpr_loss(score(u_emb[user_ids], i_emb[pos_ids]),
                        score(u_emb[user_ids], i_emb[neg_ids]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            u_emb_v = model.get_user_embeddings(x_user)
            i_emb_v = model.get_item_embeddings(item_genre_features)
            cand_emb = i_emb_v.index_select(0, candidate_ids_t)
            from recommendation_evaluate import evaluate_test_split
            val_result = evaluate_test_split(
                user_embeddings=u_emb_v, candidate_item_embeddings=cand_emb,
                eval_cache=val_eval_cache, batch_size_users=1024,
            )
        val_ndcg = val_result.metrics["NDCG@20"]

        if epoch <= 3 or epoch % 20 == 0:
            print(f"  [{model_name}] epoch {epoch:>3d}  bpr={loss.item():.4f}  "
                  f"val_ndcg@20={val_ndcg:.6f}")

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
    print(f"  [{model_name}] done — best epoch {best_epoch}, "
          f"val NDCG@20={best_val_ndcg:.6f}, {elapsed:.1f}s")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        u_emb = model.get_user_embeddings(x_user)
        i_emb = model.get_item_embeddings(item_genre_features)
    return u_emb, i_emb

def operator_to_edge_repr(op, device):
    coo = op.tocoo()
    ei = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
    ew = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return ei, ew

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args, _ = parser.parse_known_args()

    config = RecommendationConfig()
    tc = TrainConfig(
        max_epochs=5 if args.smoke else 200,
        patience=3 if args.smoke else 20,
    )
    device = resolve_device()
    print(f"Device: {device}")

    print("\n[1/5] Loading Stage 1 artifacts ...")
    stage1 = load_stage1_artifacts(config)
    n_users = stage1.user_features.shape[0]
    n_items = stage1.item_genre_features.shape[0]
    genre_dim = stage1.item_genre_features.shape[1]
    in_dim = stage1.user_features.shape[1]
    print(f"  Users: {n_users}, Items: {n_items}, Features: {in_dim}d")

    x_user = torch.tensor(stage1.user_features, dtype=torch.float32, device=device)
    item_genre_features = scipy_to_torch_sparse(stage1.item_genre_features, device=device)

    print("\n[2/5] Building user stratification by target graph degree ...")
    target_adj = sparse.load_npz(config.artifact_path("target_graph_adjacency"))
    source_adj = sparse.load_npz(config.artifact_path("source_graph_adjacency"))
    groups, target_degrees = build_degree_groups(target_adj)
    source_degrees = np.diff(source_adj.indptr)

    print(f"\n  User stratification (all {n_users} users):")
    for gname, gmask in groups.items():
        n = gmask.sum()
        src_deg = source_degrees[gmask]
        print(f"    {gname:25s}: n={n:>6d} ({n/n_users:>5.1%})  "
              f"source_deg: mean={src_deg.mean():.1f}, median={np.median(src_deg):.0f}")

    test_exclude = (stage1.train_matrix + stage1.val_matrix).sign().tocsr()
    test_eval_cache = build_eval_split_cache(
        split_name="test",
        relevant_matrix=stage1.test_matrix,
        exclude_matrix=test_exclude,
        metric_user_mask=stage1.user_masks["test_metric_user_mask"],
        candidate_item_universe=stage1.candidate_item_universe,
        device=device,
    )
    candidate_ids_t = torch.tensor(
        stage1.candidate_item_universe, dtype=torch.long, device=device
    )

    print("\n[3/5] Training 5 models ...")

    source_op = scipy_to_torch_sparse(stage1.source_operator, device=device)
    target_op = scipy_to_torch_sparse(stage1.target_operator, device=device)
    fused_op_scipy = build_fixed_fused_operator(
        stage1.source_operator, stage1.target_operator, alpha=0.5
    )
    fused_op = scipy_to_torch_sparse(fused_op_scipy, device=device)
    fused_ei, fused_ew = operator_to_edge_repr(fused_op_scipy, device)

    models_to_train = [
        ("No-graph MLP", NoGraphRecommender(
            in_dim=in_dim, n_items=n_items, genre_dim=genre_dim,
            hidden_dim=tc.hidden_dim, dropout=tc.dropout,
        ).to(device)),
        ("Source-only GCN", GraphRecommender(
            operator=source_op, in_dim=in_dim, n_items=n_items,
            genre_dim=genre_dim, hidden_dim=tc.hidden_dim,
            dropout=tc.dropout, fixed_alpha=1.0,
        ).to(device)),
        ("Target-only GCN", GraphRecommender(
            operator=target_op, in_dim=in_dim, n_items=n_items,
            genre_dim=genre_dim, hidden_dim=tc.hidden_dim,
            dropout=tc.dropout, fixed_alpha=0.0,
        ).to(device)),
        ("Fused GCN (a=0.5)", GraphRecommender(
            operator=fused_op, in_dim=in_dim, n_items=n_items,
            genre_dim=genre_dim, hidden_dim=tc.hidden_dim,
            dropout=tc.dropout, fixed_alpha=0.5,
        ).to(device)),
        ("ChebNet (K=3) fused", ChebNetRecommender(
            edge_index=fused_ei, edge_weight=fused_ew,
            in_dim=in_dim, n_items=n_items, genre_dim=genre_dim,
            hidden_dim=tc.hidden_dim, dropout=tc.dropout, K=CHEB_K,
        ).to(device)),
    ]

    print("\n[4/5] Training and evaluating per-user metrics ...\n")
    all_results = {}

    for model_name, model in models_to_train:
        print(f"\n--- {model_name} ---")
        u_emb, i_emb = train_model_and_get_embeddings(
            model=model, model_name=model_name,
            stage1=stage1, config=config, train_config=tc,
            device=device, x_user=x_user, item_genre_features=item_genre_features,
        )
        cand_emb = i_emb.index_select(0, candidate_ids_t)
        per_user = compute_per_user_metrics(
            user_embeddings=u_emb,
            candidate_item_embeddings=cand_emb,
            eval_cache=test_eval_cache,
        )
        all_results[model_name] = per_user

    print("\n\n[5/5] Stratified results ...\n")

    eval_user_ids = test_eval_cache.user_indices
    eval_group_masks = {}
    for gname, gmask in groups.items():
        eval_group_masks[gname] = gmask[eval_user_ids]

    group_names = list(groups.keys()) + ["ALL"]
    eval_group_masks["ALL"] = np.ones(len(eval_user_ids), dtype=bool)

    metric = "NDCG@20"
    print(f"\n{'='*90}")
    print(f"  Cold-Start Analysis: {metric} by Target Graph Degree Group")
    print(f"{'='*90}")

    header = f"{'Model':>25s}"
    for gname in group_names:
        n = eval_group_masks[gname].sum()
        header += f"  {gname:>15s}(n={n})"
    print(header)
    print("-" * len(header))

    results_table = []
    for model_name in all_results:
        per_user = all_results[model_name]
        row = {"model": model_name}
        line = f"{model_name:>25s}"
        for gname in group_names:
            mask = eval_group_masks[gname]
            if mask.sum() == 0:
                val = 0.0
            else:
                val = float(per_user[metric][mask].mean())
            row[gname] = val
            line += f"  {val:>20.6f}"
        results_table.append(row)
        print(line)

    print(f"\n{'='*90}")
    print(f"  Lift: Fused GCN vs Target-only GCN ({metric})")
    print(f"{'='*90}")
    fused_results = all_results["Fused GCN (a=0.5)"]
    target_results = all_results["Target-only GCN"]
    for gname in group_names:
        mask = eval_group_masks[gname]
        if mask.sum() == 0:
            continue
        fused_val = fused_results[metric][mask].mean()
        target_val = target_results[metric][mask].mean()
        if target_val > 1e-10:
            lift = (fused_val - target_val) / target_val * 100
        else:
            lift = float("inf")
        print(f"  {gname:>25s}: fused={fused_val:.6f}  target={target_val:.6f}  "
              f"lift={lift:>+.1f}%")

    for extra_metric in ["Recall@20", "HitRate@20"]:
        print(f"\n{'='*90}")
        print(f"  Cold-Start Analysis: {extra_metric} by Target Graph Degree Group")
        print(f"{'='*90}")
        for model_name in all_results:
            per_user = all_results[model_name]
            line = f"{model_name:>25s}"
            for gname in group_names:
                mask = eval_group_masks[gname]
                val = per_user[extra_metric][mask].mean() if mask.sum() > 0 else 0.0
                line += f"  {val:>20.6f}"
            print(line)

    print(f"\n{'='*90}")
    print(f"  Cross-Domain Information Availability (eval users only)")
    print(f"{'='*90}")
    for gname in group_names:
        mask = eval_group_masks[gname]
        if mask.sum() == 0:
            continue
        global_ids = eval_user_ids[mask]
        src = source_degrees[global_ids]
        tgt = target_degrees[global_ids]
        print(f"  {gname:>25s}: n={mask.sum():>5d}  "
              f"src_deg: {src.mean():.1f}\u00B1{src.std():.1f} (med={np.median(src):.0f})  "
              f"tgt_deg: {tgt.mean():.1f}\u00B1{tgt.std():.1f} (med={np.median(tgt):.0f})")

    out_dir = config.results_dir / "coldstart_user_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = {
        "experiment": "coldstart_user_analysis",
        "metric": metric,
        "user_groups": {
            gname: {
                "n_eval_users": int(eval_group_masks[gname].sum()),
                "source_degree_mean": float(source_degrees[eval_user_ids[eval_group_masks[gname]]].mean())
                    if eval_group_masks[gname].sum() > 0 else 0,
                "target_degree_mean": float(target_degrees[eval_user_ids[eval_group_masks[gname]]].mean())
                    if eval_group_masks[gname].sum() > 0 else 0,
            }
            for gname in group_names
        },
        "results": results_table,
    }
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_path = out_dir / "coldstart_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, cls=_NumpyEncoder)
    print(f"\n  Saved: {json_path}")

    df = pd.DataFrame(results_table)
    csv_path = out_dir / "coldstart_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    model_names = list(all_results.keys())
    x_groups = [g for g in group_names if g != "ALL"]
    x = np.arange(len(x_groups))
    width = 0.15
    colors = ["#888888", "#E8913A", "#4A90D9", "#2E7D32", "#9C27B0"]

    for i, mname in enumerate(model_names):
        vals = []
        for gname in x_groups:
            mask = eval_group_masks[gname]
            vals.append(all_results[mname]["NDCG@20"][mask].mean() if mask.sum() > 0 else 0)
        ax.bar(x + i * width, vals, width, label=mname, color=colors[i], alpha=0.85)

    ax.set_xlabel("Target Graph Degree Group", fontsize=12)
    ax.set_ylabel("NDCG@20", fontsize=12)
    ax.set_title("Cold-Start Analysis: NDCG@20 by User Group", fontsize=13)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(x_groups, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    lifts = []
    for gname in x_groups:
        mask = eval_group_masks[gname]
        if mask.sum() == 0:
            lifts.append(0)
            continue
        fv = fused_results["NDCG@20"][mask].mean()
        tv = target_results["NDCG@20"][mask].mean()
        lifts.append((fv - tv) / tv * 100 if tv > 1e-10 else 0)

    bar_colors = ["#2E7D32" if l > 0 else "#D32F2F" for l in lifts]
    ax.bar(x_groups, lifts, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Target Graph Degree Group", fontsize=12)
    ax.set_ylabel("Lift (%)", fontsize=12)
    ax.set_title("Fused GCN vs Target-only GCN: NDCG@20 Lift", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    for i, (g, l) in enumerate(zip(x_groups, lifts)):
        ax.text(i, l + (1 if l >= 0 else -2), f"{l:+.1f}%", ha="center", fontsize=10,
                fontweight="bold")

    plt.tight_layout()
    fig_path = out_dir / "coldstart_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.show()
    plt.close()

    print(f"\n\u2713 Cold-Start User Analysis complete!")

if __name__ == "__main__":
    main()
else:
    main()
