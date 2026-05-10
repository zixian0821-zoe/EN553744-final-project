# Exp 2 — Fusion (α schemes)

Holds the architecture fixed (GCN) and varies the fusion operator
**S_fused = α·S_src + (1−α)·S_tgt** to ask: how much lift comes from
combining the two domain graphs, and is the optimal α universal or
per-user?

Three α schemes are compared against the no-graph MLP baseline:
**fixed α = 0.5**, **learned scalar α**, **per-user α**.

## Files in this folder

| File | Purpose |
|---|---|
| `02_fusion.ipynb` | Notebook view of the full experiment (used as a deliverable) |
| `find_per_user_alpha.py` | Per-user α optimizer + diagnostics on the resulting α distribution |
| `rebuild_p80_comparison.py` | Rebuilds the cold-vs-warm comparison on the bottom 80% interaction-count users (the cold-start audit) |
| `run_supplement_strict_coldstart.py` | Strict cold-start variant — drops users with any test-side leakage; uses `pipeline/recommendation_data_strict_coldstart.py` |
| `coldstart_user_analysis.py` | User-level breakdown of how each α scheme behaves across activity buckets |

## Running

The headline numbers are produced by the Exp 2 sweep in `pipeline/`:

```
python pipeline/run_recommendation_experiment.py
```

Then the supplementary scripts in this folder:

```
python exp2_fusion/find_per_user_alpha.py
python exp2_fusion/rebuild_p80_comparison.py
python exp2_fusion/run_supplement_strict_coldstart.py
```

## Results (NDCG@20, test split)

| Scheme | α | NDCG@20 | vs no-graph MLP |
|---|---|---|---|
| **Fixed fused** | 0.500 | **0.001904** | **+15.7 %** |
| Learned-α | 0.478 (converged) | 0.001853 | +12.6 % |
| Per-user-α | 0.422 (mean) | 0.001832 | +11.3 % |

Takeaways: (i) α = 0.5 already captures essentially all the fusion lift —
the two domain graphs contribute symmetrically; (ii) the learned global α
converges near 0.5 (0.478), corroborating (i); (iii) per-user-α slightly
underperforms the global schemes — extra capacity overfits with no
matching held-out signal to support it.
