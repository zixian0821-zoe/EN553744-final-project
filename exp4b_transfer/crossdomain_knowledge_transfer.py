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
CHEB_K = 3
EPS = 1e-12

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

@torch.no_grad()
def compute_per_user_ndcg20(
    user_embeddings: torch.Tensor,
    candidate_item_embeddings: torch.Tensor,
    eval_cache: EvalSplitCache,
    batch_size: int = 1024,
) -> np.ndarray:
                                                                           
    max_k = 20
    topk = _rank_topk_gpu(user_embeddings, candidate_item_embeddings,
                          eval_cache, topk=max_k, batch_size_users=batch_size)
    device = topk.device
    hits = _compute_hits_gpu(topk, eval_cache.relevant_padded)
    discounts = 1.0 / torch.log2(
        torch.arange(2, max_k + 2, device=device, dtype=torch.float32))
    dcg = (hits.float() * discounts).sum(1)
    cum_disc = torch.cumsum(discounts, dim=0)
    ideal = torch.minimum(eval_cache.relevant_counts,
                          torch.tensor(max_k, device=device))
    idcg = cum_disc[(ideal.clamp(min=1) - 1).long()]
    ndcg = (dcg / idcg.clamp(min=EPS)).cpu().numpy()
    return ndcg

def operator_to_edge_repr(op, device):
    coo = op.tocoo()
    ei = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
    ew = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return ei, ew

def train_chebnet(name, ei, ew, stage1, config, tc, device, x_user_override, igf):
\
\
\
\
       
    set_seed(tc.seed)
    x_user = x_user_override
    model = ChebNetRecommender(
        edge_index=ei, edge_weight=ew,
        in_dim=x_user.shape[1], n_items=stage1.item_genre_features.shape[0],
        genre_dim=stage1.item_genre_features.shape[1],
        hidden_dim=tc.hidden_dim, dropout=tc.dropout, K=CHEB_K,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc.learning_rate,
                                 weight_decay=tc.weight_decay)
    pop = build_target_train_item_popularity(stage1.train_matrix)
    neg_dist = build_negative_sampling_distribution(
        train_item_popularity=pop,
        candidate_item_universe=stage1.candidate_item_universe,
        power=config.negative_sampling_power)
    pos_cache = build_train_positive_cache(stage1.train_matrix)
    rng = np.random.default_rng(tc.seed)
    cand_t = torch.tensor(stage1.candidate_item_universe, dtype=torch.long, device=device)

    val_exclude = (stage1.train_matrix + stage1.test_matrix).sign().tocsr()
    val_cache = build_eval_split_cache(
        "val", stage1.val_matrix, val_exclude,
        stage1.user_masks["val_metric_user_mask"],
        stage1.candidate_item_universe, device)

    best_state, best_ndcg, best_ep, no_imp = None, -1.0, 0, 0
    t0 = time.perf_counter()

    for ep in range(1, tc.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        u_np, p_np, n_np = sample_bpr_triplets(pos_cache, neg_dist, rng)
        uid = torch.tensor(u_np, dtype=torch.long, device=device)
        pid = torch.tensor(p_np, dtype=torch.long, device=device)
        nid = torch.tensor(n_np, dtype=torch.long, device=device)
        ue = model.get_user_embeddings(x_user)
        ie = model.get_item_embeddings(igf)
        loss = bpr_loss(score(ue[uid], ie[pid]), score(ue[uid], ie[nid]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            ue_v = model.get_user_embeddings(x_user)
            ie_v = model.get_item_embeddings(igf)
            from recommendation_evaluate import evaluate_test_split
            vr = evaluate_test_split(ue_v, ie_v.index_select(0, cand_t),
                                     val_cache, 1024)
        vn = vr.metrics["NDCG@20"]
        if ep <= 3 or ep % 20 == 0:
            print(f"  [{name}] ep {ep:>3d}  bpr={loss.item():.4f}  val={vn:.6f}")
        if vn > best_ndcg + 1e-12:
            best_ndcg, best_ep, no_imp = vn, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
            if no_imp >= tc.patience:
                break

    el = time.perf_counter() - t0
    print(f"  [{name}] done — best ep {best_ep}, val={best_ndcg:.6f}, {el:.1f}s")
    return best_state

@torch.no_grad()
def evaluate_with_graph(state_dict, eval_ei, eval_ew, stage1, tc, device,
                        x_user, igf, test_cache, cand_t):
                                                                    
    model = ChebNetRecommender(
        edge_index=eval_ei, edge_weight=eval_ew,
        in_dim=x_user.shape[1], n_items=stage1.item_genre_features.shape[0],
        genre_dim=stage1.item_genre_features.shape[1],
        hidden_dim=tc.hidden_dim, dropout=tc.dropout, K=CHEB_K,
    ).to(device)
    learned = {k: v for k, v in state_dict.items()
               if k not in ("edge_index", "edge_weight")}
    model.load_state_dict(learned, strict=False)
    model.edge_index = eval_ei
    model.edge_weight = eval_ew
    model.eval()

    ue = model.get_user_embeddings(x_user)
    ie = model.get_item_embeddings(igf)
    cand_emb = ie.index_select(0, cand_t)

    from recommendation_evaluate import evaluate_test_split
    result = evaluate_test_split(ue, cand_emb, test_cache, 1024)
    agg = result.metrics

    per_user = compute_per_user_ndcg20(ue, cand_emb, test_cache)

    return agg, per_user

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args, _ = parser.parse_known_args()

    config = RecommendationConfig()
    tc = TrainConfig(max_epochs=5 if args.smoke else 200,
                     patience=3 if args.smoke else 20)
    device = resolve_device()
    print(f"Device: {device}")
    print(f"Model: ChebNet (K={CHEB_K})")

    print("\n[1/6] Loading artifacts ...")
    stage1 = load_stage1_artifacts(config)
    n_users = stage1.user_features.shape[0]
    print(f"  Users: {n_users}, Items: {stage1.item_genre_features.shape[0]}")

    x_user = torch.tensor(stage1.user_features, dtype=torch.float32, device=device)
    igf = scipy_to_torch_sparse(stage1.item_genre_features, device=device)
    cand_t = torch.tensor(stage1.candidate_item_universe, dtype=torch.long, device=device)

    fused_op = build_fixed_fused_operator(
        stage1.source_operator, stage1.target_operator, alpha=0.5)
    ops_scipy = {"source": stage1.source_operator,
                 "target": stage1.target_operator,
                 "fused": fused_op}
    edge_reprs = {n: operator_to_edge_repr(o, device) for n, o in ops_scipy.items()}

    test_exclude = (stage1.train_matrix + stage1.val_matrix).sign().tocsr()
    test_cache = build_eval_split_cache(
        "test", stage1.test_matrix, test_exclude,
        stage1.user_masks["test_metric_user_mask"],
        stage1.candidate_item_universe, device)

    print("\n[2/6] Building user groups ...")
    target_adj = sparse.load_npz(config.artifact_path("target_graph_adjacency"))
    tgt_deg = np.diff(target_adj.indptr)
    groups = {
        "isolated (deg=0)": tgt_deg == 0,
        "connected (deg>0)": tgt_deg > 0,
    }
    eval_uids = test_cache.user_indices
    eval_groups = {g: m[eval_uids] for g, m in groups.items()}
    eval_groups["ALL"] = np.ones(len(eval_uids), dtype=bool)
    for g, m in eval_groups.items():
        print(f"  {g:>25s}: {m.sum()} eval users")

    print("\n[3/6] Training 3 ChebNet models (one per graph) ...")
    trained = {}
    for gn in GRAPH_NAMES:
        print(f"\n--- Training ChebNet on {gn} graph ---")
        ei, ew = edge_reprs[gn]
        state = train_chebnet(gn, ei, ew, stage1, config, tc, device, x_user, igf)
        trained[gn] = state

    print("\n[4/6] Evaluating 3x3 transferability matrix ...")
    agg_matrix = {}
    per_user_matrix = {}

    for train_g in GRAPH_NAMES:
        for eval_g in GRAPH_NAMES:
            ei, ew = edge_reprs[eval_g]
            agg, pu = evaluate_with_graph(
                trained[train_g], ei, ew, stage1, tc, device,
                x_user, igf, test_cache, cand_t)
            agg_matrix[(train_g, eval_g)] = agg
            per_user_matrix[(train_g, eval_g)] = pu
            print(f"  train={train_g:>6s}, eval={eval_g:>6s}  "
                  f"NDCG@20={agg['NDCG@20']:.6f}  Recall@20={agg['Recall@20']:.6f}")

    print(f"\n  Transferability Matrix (NDCG@20):")
    print(f"  {'Train/Eval':>12s}  {'source':>10s}  {'target':>10s}  {'fused':>10s}")
    for tg in GRAPH_NAMES:
        vals = [f"{agg_matrix[(tg, eg)]['NDCG@20']:.6f}" for eg in GRAPH_NAMES]
        marks = []
        for i, eg in enumerate(GRAPH_NAMES):
            if tg == eg:
                marks.append(f"{vals[i]} *")
            else:
                marks.append(f"{vals[i]}  ")
        print(f"  {tg:>12s}  {'  '.join(marks)}")

    print(f"\n  Stratified Transferability (NDCG@20):")
    for group_name, group_mask in eval_groups.items():
        if group_name == "ALL":
            continue
        print(f"\n  --- {group_name} (n={group_mask.sum()}) ---")
        print(f"  {'Train/Eval':>12s}  {'source':>10s}  {'target':>10s}  {'fused':>10s}")
        for tg in GRAPH_NAMES:
            vals = []
            for eg in GRAPH_NAMES:
                pu = per_user_matrix[(tg, eg)]
                v = pu[group_mask].mean() if group_mask.sum() > 0 else 0
                vals.append(f"{v:.6f}")
            print(f"  {tg:>12s}  {'    '.join(vals)}")

    print("\n[5/6] Feature ablation: random features on source graph ...")
    set_seed(tc.seed)
    x_random = torch.randn_like(x_user)
    x_random = x_random / (x_random.norm(dim=1, keepdim=True) + 1e-8)
    x_random = x_random * x_user.norm(dim=1, keepdim=True).mean()

    ei_src, ew_src = edge_reprs["source"]
    state_random = train_chebnet("random_feat_source", ei_src, ew_src,
                                  stage1, config, tc, device, x_random, igf)

    ei_tgt, ew_tgt = edge_reprs["target"]
    agg_rand_tgt, pu_rand_tgt = evaluate_with_graph(
        state_random, ei_tgt, ew_tgt, stage1, tc, device,
        x_random, igf, test_cache, cand_t)
    agg_rand_src, pu_rand_src = evaluate_with_graph(
        state_random, ei_src, ew_src, stage1, tc, device,
        x_random, igf, test_cache, cand_t)

    print(f"\n  Feature Ablation Results (NDCG@20):")
    print(f"    Source-trained (real features) → eval target:  "
          f"{agg_matrix[('source', 'target')]['NDCG@20']:.6f}")
    print(f"    Source-trained (RANDOM features) → eval target: "
          f"{agg_rand_tgt['NDCG@20']:.6f}")
    print(f"    Source-trained (RANDOM features) → eval source: "
          f"{agg_rand_src['NDCG@20']:.6f}")

    real_val = agg_matrix[("source", "target")]["NDCG@20"]
    rand_val = agg_rand_tgt["NDCG@20"]
    if rand_val > 1e-10:
        ratio = real_val / rand_val
        print(f"    Real/Random ratio: {ratio:.2f}x")
    print(f"    → {'Music genre features carry significant cross-domain signal' if real_val > rand_val * 1.1 else 'Transfer relies primarily on graph topology'}")

    print("\n[6/6] Operator distances ...")
    distances = {}
    for i, a in enumerate(GRAPH_NAMES):
        for j, b in enumerate(GRAPH_NAMES):
            if i <= j:
                diff = ops_scipy[a] - ops_scipy[b]
                d = sparse.linalg.norm(diff, ord="fro")
                distances[f"{a}-{b}"] = float(d)
                print(f"  ||S_{a} - S_{b}||_F = {d:.4f}")

    out_dir = config.results_dir / "crossdomain_knowledge_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "crossdomain_knowledge_transfer",
        "model": f"ChebNet (K={CHEB_K})",
        "aggregate_matrix": {
            f"train={tg},eval={eg}": agg_matrix[(tg, eg)]
            for tg in GRAPH_NAMES for eg in GRAPH_NAMES
        },
        "stratified_matrix": {
            group_name: {
                f"train={tg},eval={eg}": float(per_user_matrix[(tg, eg)][gmask].mean())
                    if gmask.sum() > 0 else 0.0
                for tg in GRAPH_NAMES for eg in GRAPH_NAMES
            }
            for group_name, gmask in eval_groups.items()
        },
        "feature_ablation": {
            "real_features_source_to_target": agg_matrix[("source", "target")]["NDCG@20"],
            "random_features_source_to_target": agg_rand_tgt["NDCG@20"],
            "random_features_source_to_source": agg_rand_src["NDCG@20"],
        },
        "operator_distances": distances,
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

    with open(out_dir / "transfer_reframed_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"\n  Saved: {out_dir / 'transfer_reframed_results.json'}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    mat = np.array([[agg_matrix[(tg, eg)]["NDCG@20"]
                     for eg in GRAPH_NAMES] for tg in GRAPH_NAMES])
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(GRAPH_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(GRAPH_NAMES)
    ax.set_xlabel("Evaluation Graph"); ax.set_ylabel("Training Graph")
    ax.set_title("Transferability Matrix (NDCG@20)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i,j]:.4f}", ha="center", va="center",
                    fontsize=10, fontweight="bold" if i == j else "normal")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    bar_data = {}
    for gname, gmask in eval_groups.items():
        if gname == "ALL":
            continue
        bar_data[gname] = {}
        for tg in GRAPH_NAMES:
            pu = per_user_matrix[(tg, "target")]
            bar_data[gname][tg] = pu[gmask].mean() if gmask.sum() > 0 else 0

    x = np.arange(len([g for g in eval_groups if g != "ALL"]))
    gnames = [g for g in eval_groups if g != "ALL"]
    width = 0.25
    colors = ["#E8913A", "#4A90D9", "#2E7D32"]
    for i, tg in enumerate(GRAPH_NAMES):
        vals = [bar_data[g][tg] for g in gnames]
        ax.bar(x + i * width, vals, width, label=f"Trained on {tg}",
               color=colors[i], alpha=0.85)
    ax.set_xlabel("User Group")
    ax.set_ylabel("NDCG@20")
    ax.set_title("Cross-Domain Transfer by User Type\n(Evaluated on TARGET graph)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(gnames, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    labels = ["Real features\n(source→target)",
              "Random features\n(source→target)",
              "Random features\n(source→source)"]
    values = [agg_matrix[("source", "target")]["NDCG@20"],
              agg_rand_tgt["NDCG@20"],
              agg_rand_src["NDCG@20"]]
    colors_abl = ["#2E7D32", "#D32F2F", "#FF9800"]
    ax.bar(labels, values, color=colors_abl, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("NDCG@20")
    ax.set_title("Feature Ablation:\nDo Music Features Matter for Transfer?")
    for i, v in enumerate(values):
        ax.text(i, v + 0.00005, f"{v:.5f}", ha="center", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "transfer_reframed_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.show()
    plt.close()

    src_to_tgt = agg_matrix[("source", "target")]["NDCG@20"]
    tgt_to_tgt = agg_matrix[("target", "target")]["NDCG@20"]
    transfer_lift = (src_to_tgt - tgt_to_tgt) / tgt_to_tgt * 100

    print(f"\n{'='*70}")
    print(f"  KEY FINDINGS — Cross-Domain Knowledge Transfer")
    print(f"{'='*70}")
    print(f"  1. Source→Target transfer NDCG@20: {src_to_tgt:.6f}")
    print(f"     Target→Target baseline NDCG@20: {tgt_to_tgt:.6f}")
    print(f"     Transfer lift: {transfer_lift:+.1f}%")
    print(f"  2. Feature ablation: real features {'outperform' if real_val > rand_val * 1.1 else 'comparable to'} random")
    print(f"     → Music genre features {'are critical' if real_val > rand_val * 1.1 else 'are not the main driver'} for cross-domain transfer")
    print(f"  3. Operator distance source↔target: {distances.get('source-target', 0):.2f}")
    print(f"{'='*70}")

    print(f"\n✓ Cross-Domain Knowledge Transfer analysis complete!")

if __name__ == "__main__":
    main()
else:
    main()
