# Exp 4b — Cross-graph transfer + spectral alignment

Stability *across* graphs (Exp 4a covers stability *within* a graph).
We train a ChebNet K=3 on each of the three graphs (source / target /
fused) and evaluate it on every other graph — a 3×3 transferability
matrix — then ablate features vs topology and measure the spectral
distance between the source and target operators.

## Files in this folder

| File | Purpose |
|---|---|
| `04b_transfer.ipynb` | Notebook view of the full experiment (used as a deliverable) |
| `exp4b_transferability.py` | Trains and evaluates the 3×3 (train graph × eval graph) ChebNet matrix |
| `crossdomain_knowledge_transfer.py` | Feature ablation — replaces real music features with random noise and re-trains the source→target transfer to isolate the topology contribution |
| `exp4b_spectral_analysis.py` | Operator-level comparison of S_src and S_tgt: Frobenius distance, normalized-Laplacian eigenvalue distributions, and Wasserstein-1 between the two spectra |

## Running

```
python exp4b_transfer/exp4b_transferability.py
python exp4b_transfer/crossdomain_knowledge_transfer.py
python exp4b_transfer/exp4b_spectral_analysis.py
```

Reads Stage-1 artifacts from `pipeline/results/`; writes per-script
metrics to `pipeline/results/exp4b/`.

## Results (NDCG@20, test split)

| Direction | NDCG@20 | vs self-baseline |
|---|---|---|
| target → target (self-ceiling) | 0.002461 | — |
| **source → target** (cross-domain transfer) | **0.002292** | **−6.9 %** (= 93 % of the self-ceiling) |
| target → source (reverse) | — | **+9.6 %** above the source self-baseline |

Feature ablation:

| Feature source on transfer leg | NDCG@20 | Share of real-feature performance |
|---|---|---|
| real music features | 0.002292 | 100 % |
| random features | 0.001919 | **84 %** (so topology carries ≈84%, features ≈19% lift) |

Spectral comparison of the two operators:

| Quantity | S_src vs S_tgt |
|---|---|
| Frobenius distance | **40.77** |
| Normalized-Laplacian Wasserstein-1 distance | **0.023** |

Takeaway: as matrices the two operators are far apart, but their
*spectra* nearly coincide. A polynomial filter only sees the operator
through its eigenvalues, which is the structural reason a single
ChebNet remains accurate when transplanted across domains.
