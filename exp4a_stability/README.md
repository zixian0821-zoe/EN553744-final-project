# Exp 4a — Within-graph stability

Perturbs the fused (α = 0.5) graph and re-trains the top three architectures
from Exp 3 to measure graceful degradation under realistic noise.

## Files in this folder

| File | Purpose |
|---|---|
| `stability (2).py` | Working runner — Colab export (391 lines). Mounts Drive, applies the three perturbation families, re-trains ChebNet K=3 / GraphSAGE / GCN under three seeds, writes summary CSVs |
| `stability.py` | Empty placeholder for a CLI-cleaned version (Colab cells stripped). Not yet populated |

## Perturbations swept

On the fused α = 0.5 graph:

- **Edge dropout** — keep ratio ∈ {0.95, 0.90, 0.80, 0.70} (drop ∈ {5, 10, 20, 30 %})
- **Edge-weight Gaussian noise** — σ ∈ {0.05, 0.10, 0.20}
- **Node-feature Gaussian noise** — σ ∈ {0.05, 0.10, 0.20}

Three architectures × three seeds (42, 43, 44).

## Running

The current copy is a Colab-style script. Either:

1. **Run in Colab** — upload `stability (2).py` as a notebook cell and run end-to-end. Requires the same `pipeline/` artifacts mounted under Drive.
2. **CLI** — first strip the `google.colab` import, the Drive mount, and the `!pip install` lines, then:
   ```
   python "exp4a_stability/stability (2).py"
   ```
   Output lands in `pipeline/results/stability/{stability_results.csv, stability_summary.csv}` plus per-perturbation degradation curves.

## Dependency

Imports `pipeline/recommendation_models.py` and
`pipeline/train_recommendation.py` and expects them to expose ChebNet /
GraphSAGE — i.e. it depends on Exp 3's extensions to those two files. Run
Exp 3 first (or copy `exp3_architecture/recommendation_models.py` and
`exp3_architecture/train_recommendation.py` into `pipeline/`).

## Headline behaviour

All three architectures degrade gracefully under all perturbations. The
accuracy–stability trade-off:

- **ChebNet K=3** — most accurate, but most feature-noise sensitive (~21 % NDCG@20 drop at σ = 0.20).
- **GraphSAGE** — essentially flat under all perturbations; lowest accuracy ceiling, highest robustness.
- **GCN** — in between on both axes.

Across perturbation families: edge-weight noise is the mildest, node-feature
noise the most damaging, edge dropout in between.
