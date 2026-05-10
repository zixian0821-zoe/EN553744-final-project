# EN.553.744 — Cross-Domain Recommendation as a Graph Signal Processing Problem

**Course:** EN.553.744 Data Science for Large-Scale Graphs (Spring 2026, JHU AMS)
**Team:** Zixian Zhou · Yunwei Chai · Yang Song
**Dataset:** Amazon Reviews 2023 — CDs/Vinyl (source) → Books (target), disjoint user catalogs
**Reference:** Liu et al., *Graph Signal Processing for Cross-Domain Recommendation* (arXiv:2407.12374)

We test whether the geometry of the music domain can substitute for missing
structure in the book domain when the user catalogs are disjoint. Book
preferences are treated as a graph signal, the user-similarity network as the
support, and fusion as a linear combination of two graph shift operators:

> **S_fused = α · S_src + (1 − α) · S_tgt**

The repo contains the full pipeline (data → graphs → features → BPR training
→ evaluation) plus five experiments that probe smoothness, fusion, GNN
architecture, stability under perturbation, and cross-graph transfer.

---

## Repository layout

| Folder | Owner | Contents |
|---|---|---|
| `data_preprocessing/` | Zixian Zhou | Download + materialize Amazon Reviews 2023 (CDs/Vinyl, Books) |
| `pipeline/` | Zixian Zhou | Shared core: features, graphs, BPR loss, training loop, Exp 2 models (MLP / GCN / fused-α / learned-α / per-user-α). Used directly by Exp 1 / Exp 2 / Exp 4b |
| `exp1_signal_smoothness/` | Zixian Zhou | Dirichlet energy + permutation test on the book signal |
| `exp2_fusion/` | Zixian Zhou | α schemes (fixed / learned / per-user) + cold-start audits |
| `exp3_architecture/` | Yang Song | ChebNet K=3 / GraphSAGE / GCN / GAT / MLP comparison on the fused graph. Contains an **extended fork** of `pipeline/recommendation_models.py`, `pipeline/train_recommendation.py`, `pipeline/run_recommendation_experiment.py` that adds the new architectures (not merged back into `pipeline/`) |
| `exp4a_stability/` | Yunwei Chai | Within-graph perturbation analysis (edge dropout / edge-weight noise / feature noise). Imports the **exp3-forked** modules — see `exp4a_stability/README.md` |
| `exp4b_transfer/` | Zixian Zhou | 3×3 transferability matrix + feature ablation + spectral (Wasserstein) analysis |

Most experiment runners add `pipeline/` to `sys.path` via a small bootstrap
header (`_PIPELINE_DIR = ../pipeline`); no package install needed.
**Exception:** `exp4a_stability/stability.py` is a Colab export and hardcodes
a Drive path (`/content/drive/MyDrive/.../Experiment2`) where the two
exp3-forked modules and a copy of `recommendation_config.py` were placed
at run time. See `exp4a_stability/README.md` for the exact module-source
mapping and the steps to re-execute it outside Colab.

---

## Headline results (NDCG@20, test split)

| Experiment | Result |
|---|---|
| Exp 1 — Smoothness | z = −10.13, p < 0.005 (0/200 random permutations below the observed Dirichlet energy) |
| Exp 2 — Fusion | fixed α = 0.5 → 0.001904 (+15.7 % vs MLP); learned-α converges to **α* ≈ 0.478** |
| Exp 3 — Architecture | ChebNet K=3 = 0.002302 (+20.9 % vs GCN); GraphSAGE = 0.001932; GCN = 0.001904; MLP = 0.001645; GAT = 0.001630 |
| Exp 4a — Stability | All three top architectures degrade gracefully; structural edge dropout is the most damaging perturbation; ChebNet is most accurate but drops ~7.6 % under edge dropout, GraphSAGE is most robust, and GCN is least stable |
| Exp 4b — Transfer | src → tgt = 0.002292 (93 % of the tgt self-ceiling, only −6.9 %); topology ≈ 84 %, features ≈ 19 %; W₁(S_src, S_tgt) = 0.023 |

The fusion gain in Exp 2 is consistent with the smoothness gap measured in
Exp 1 and the topology-vs-features decomposition in Exp 4b.

---

## Reproducing each experiment

All runners assume Stage-1 artifacts (graphs, features, splits) exist under
`pipeline/results/`. Build them first via `data_preprocessing/` and the
Stage-1 scripts in `pipeline/`. Paths are configured in
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
python exp2_fusion/run_supplement_strict_coldstart.py
```

### Exp 3 — Architecture comparison
```
python exp3_architecture/run_recommendation_experiment.py
python exp3_architecture/train_recommendation.py
python "exp3_architecture/learning curve.py"
python exp3_architecture/raking.py
```

### Exp 4a — Stability
```
# stability.py is a Colab export: it mounts Drive and imports the
# exp3-forked train_recommendation.py / recommendation_models.py plus
# pipeline's recommendation_config.py from a Drive path.
# See exp4a_stability/README.md for the exact module-source mapping
# and Colab-vs-CLI instructions.
python exp4a_stability/stability.py
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
scipy, numpy, pandas, scikit-learn, matplotlib, tqdm.

```
pip install -r requirements.txt
```

`torch-geometric` needs PyTorch wheels first; on Colab use
`!pip install torch_geometric` after the matching torch is loaded.

---

## Data and artifacts

Raw Amazon Reviews 2023 dumps, intermediate `.parquet` / `.npz` graphs,
trained checkpoints, and per-run result CSVs are excluded from version
control (see `.gitignore`). Only the small JSON summaries
(`ranking_summary.json`, `metrics.json`, `spectral_analysis_results.json`)
are kept under `results/` so reviewers can see headline numbers without
rebuilding the pipeline.
