# Exp 1 — Signal smoothness on the music graph

Tests whether the book preference signal is smooth on the music
user-similarity graph. We compute the Dirichlet energy of the book
signal on the source (music) graph and compare it against the same
energy under random label permutations of the signal.

If books and music live on truly unrelated geometries, the observed
energy should be indistinguishable from the permutation null.

## Files in this folder

| File | Purpose |
|---|---|
| `01_smoothness.ipynb` | Notebook view of the full experiment (used as a deliverable) |
| `total_variation_colab.py` | Main runner — builds the music k-NN graph, computes the Dirichlet energy of the book signal, runs 200 random permutations |
| `total_variation_colab.ipynb` | Colab-export of the runner |
| `rerun_on_exp2_graph.py` | Re-runs the same test on the exact graph used in Exp 2 (sanity check that the smoothness gap is not an artifact of the standalone graph build) |
| `total_variation_test.py` | Light integration test on a toy graph |
| `genre_profile_colab.py` | Builds the 30-dim music-genre user features used as the basis for the k-NN graph |

## Running

```
python exp1_signal_smoothness/total_variation_colab.py
python exp1_signal_smoothness/rerun_on_exp2_graph.py
```

## Result

| Metric | Value |
|---|---|
| Dirichlet energy z-score (vs 200 permutations) | **−10.13** |
| p-value | **< 0.005** |
| Permutations strictly below observed energy | **0 / 200** |

The book signal is far smoother on the music graph than chance allows,
so cross-domain signal exists in the topology — which is what justifies
the fusion in Exp 2.
