# EN.553.744 — Cross-Domain Recommendation as a Graph Signal Processing Problem

**Course:** EN.553.744 Data Science for Large-Scale Graphs (Spring 2026)
**Team:** Yunwei Chai · Yang Song · Zixian Zhou
**Dataset:** Amazon Reviews 2023 — CDs/Vinyl (source) → Books (target)
**Reference:** Liu et al., *Graph Signal Processing for Cross-Domain Recommendation* (arXiv:2407.12374)

We test whether the geometry of the music domain can substitute for missing
structure in the book domain when the user catalogs are disjoint. Book
preferences are a graph signal, the user-similarity network is the support,
and fusion is a linear combination of two graph shift operators
**S_fused = α·S_src + (1−α)·S_tgt**.

---

## Repository layout & ownership

| Folder | Owner | Push status | Contents |
|---|---|---|
| `data_preprocessing/` | Zixian Zhou | Included in this push | Download + materialize Amazon Reviews 2023 |
| `pipeline/` | Zixian Zhou | Included in this push | Shared core: features, graphs, BPR loss, training loop, Exp 2 models (MLP / GCN / fused-α / learned-α / per-user-α) |
| `exp1_signal_smoothness/` | Zixian Zhou | Included in this push | Dirichlet energy + permutation test |
| `exp2_fusion/` | Zixian Zhou | Included in this push | α schemes (fixed / learned / per-user) + cold-start audits |
| `exp4b_transfer/` | Zixian Zhou | Included in this push | 3×3 transferability matrix + feature ablation + spectral analysis |
| `exp3_architecture/` | Exp 3 contributor | Not included in this push | ChebNet K=3 / GraphSAGE / GAT comparison |
| `exp4a_stability/` | Exp 4a contributor | Not included in this push | Within-graph perturbation analysis |

`pipeline/` is added to `sys.path` by every experiment runner via a small
bootstrap header (`_PIPELINE_DIR = ../pipeline`). No package install needed.

---

## Headline results (NDCG@20, test split)

| Experiment | Result |
|---|---|
| Exp 1 — Smoothness | z = −10.13, p < 0.005 (0/200 perms below observed) |
| Exp 2 — Fusion | fixed α = 0.5: 0.001904 (+15.7 % vs MLP); learned-α → 0.478 |
| Exp 4b — Transfer | src→tgt = 0.002292 (93 % of tgt self-ceiling, only −6.9 %); topology ≈ 84 %, features ≈ 19 %; W₁(S_src, S_tgt) = 0.023 |

---

## Reproducing each experiment

All runners assume Stage-1 artifacts (graphs, features, splits) exist under
`pipeline/results/`. Build them first via `data_preprocessing/` and the
Stage-1 scripts in `pipeline/`. The exact paths are configured in
`pipeline/recommendation_config.py`.

### Stage 0 — Data preprocessing
```
data_preprocessing/amazon_data_preprocess.ipynb
python data_preprocessing/materialize_amazon_interactions.py
```

### Stage 1 — Build graphs + features
```
python pipeline/stage1_preprocess_graphs.py
python pipeline/build_recommendation_stage1.py
```

### Exp 1 — Signal smoothness
```
python exp1_signal_smoothness/total_variation_colab.py
python exp1_signal_smoothness/rerun_on_exp2_graph.py
```

### Exp 2 — Fusion (α schemes)
```
python pipeline/run_recommendation_experiment.py
python exp2_fusion/find_per_user_alpha.py
python exp2_fusion/rebuild_p80_comparison.py
```

### Exp 4b — Cross-graph transfer + spectral
```
python exp4b_transfer/exp4b_transferability.py
python exp4b_transfer/crossdomain_knowledge_transfer.py
python exp4b_transfer/exp4b_spectral_analysis.py
```

---

## Dependencies

Tested on Python 3.11+, CUDA 12.x. Core stack: PyTorch 2.x, torch-geometric,
scipy, numpy, pandas, matplotlib.

```
pip install -r requirements.txt
```

`torch-geometric` requires PyTorch wheels; on Colab use
`!pip install torch_geometric` after the matching torch is loaded.
