# Exp 4a — Within-graph stability

Perturbs the fused (α = 0.5) graph and re-trains the top three architectures
from Exp 3 to measure graceful degradation under realistic noise.

## Files in this folder

| File | Purpose |
|---|---|
| `stability.py` | Working runner — Colab export. Mounts Drive, applies the three perturbation families, re-trains ChebNet K=3 / GraphSAGE / GCN under three seeds, writes summary CSVs |

## Perturbations swept

On the fused α = 0.5 graph:

- **Edge dropout** — keep ratio ∈ {0.95, 0.90, 0.80, 0.70} (drop ∈ {5, 10, 20, 30 %})
- **Edge-weight Gaussian noise** — σ ∈ {0.05, 0.10, 0.20}
- **Node-feature Gaussian noise** — σ ∈ {0.05, 0.10, 0.20}

Three architectures × three seeds (42, 43, 44).

## Running

`stability.py` is a Colab export. Either:

1. **Colab (intended path)** — open the file in Colab, mount Drive, and run end-to-end. The script hardcodes its `PROJECT_ROOT` to `/content/drive/MyDrive/large_scale_graph_final_project-yang-744/Experiment2`, where the exp3-forked `recommendation_models.py`, `train_recommendation.py`, and `recommendation_config.py` were placed at run time. The Stage-1 artifacts must also be reachable from `recommendation_config.RecommendationConfig()`.
2. **CLI** — strip the `google.colab` import, the `drive.mount(...)` call, and the `!pip install` lines; assemble the three required modules (see the Dependency table below) into a single directory; point `PROJECT_ROOT` at it; then:
   ```
   python exp4a_stability/stability.py
   ```
   Output lands in `<PROJECT_ROOT>/results/stability/{stability_results.csv, stability_summary.csv}` plus per-perturbation degradation curves.

## Dependency

`stability.py` imports three modules from `PROJECT_ROOT`:

| Module | Source on GitHub | Notes |
|---|---|---|
| `recommendation_config` | `pipeline/recommendation_config.py` | Not forked by Exp 3 — pipeline copy is fine |
| `train_recommendation` | `exp3_architecture/train_recommendation.py` | **Exp 3 fork** — adds ChebNet / GraphSAGE support; pipeline version does not |
| `recommendation_models` | `exp3_architecture/recommendation_models.py` | **Exp 3 fork** — defines ChebNet / GraphSAGE / fixed-fused GCN; pipeline version does not |

At Colab run-time all three were placed together inside the Drive
folder `Experiment2/` (which is what `PROJECT_ROOT` points to). To run
this script outside Colab, assemble the same three files into a single
directory and point `PROJECT_ROOT` at it.

## Headline behaviour

All three architectures degrade gracefully under all perturbations. The
notebook-level conclusion is that structural perturbation has the largest
impact on recommendation quality, with relative destructiveness:

`structural edge dropout > edge-weight noise ≈ node-feature noise`.

The accuracy–stability trade-off:

- **ChebNet K=3** — most accurate baseline, moderately sensitive to structural perturbation (~7.6 % NDCG@20 drop under edge dropout), and relatively stable under feature noise.
- **GraphSAGE** — most robust, with minimal degradation across perturbations and only ~2.1 % drop under edge dropout.
- **GCN** — least stable overall, with the largest drops in the notebook summary (~9.0 % under edge dropout and ~7.1 % under edge-weight noise).

Across perturbation families, edge dropout is the dominant failure mode; edge
weight noise and node-feature noise are milder and roughly comparable.
