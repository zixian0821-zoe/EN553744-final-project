# Pipeline — shared Stage-1 core + Exp 2 runner

Everything the experiments share lives here: feature extraction, k-NN
graph construction, BPR loss + sampling, the training loop, the Exp 2
recommender models (MLP / GCN / fused-α / learned-α / per-user-α), and
evaluation.

Used directly by **Exp 1**, **Exp 2**, and **Exp 4b**. Exp 3 maintains a
fork of three of these modules (see `exp3_architecture/README.md`); Exp
4a imports the exp3 fork.

## Files in this folder

| File | Purpose |
|---|---|
| `recommendation_config.py` | Single source of truth for all paths, hyperparameters, and split ratios. Edit this if your data lives elsewhere |
| `recommendation_features.py` | Builds the 30-dim L1-normalized music-genre user features |
| `recommendation_graphs.py` | Cosine k-NN (k = 10) + symmetric normalization S = D⁻¹ᐟ²(A + I)D⁻¹ᐟ² for source / target / fused graphs |
| `recommendation_data.py` | Per-user 60/20/20 temporal split, BPR triplet sampler, leakage check |
| `recommendation_data_strict_coldstart.py` | Variant split for the strict cold-start audit used in Exp 2 |
| `recommendation_loss_sampling.py` | Negative-sampling utilities for BPR |
| `recommendation_models.py` | MLP / GCN / fused-α / learned-α / per-user-α recommenders (Exp 2 architectures) |
| `recommendation_evaluate.py` | NDCG@K / Recall@K on the held-out test split |
| `train_recommendation.py` | Training loop + `instantiate_model()` dispatcher |
| `stage1_preprocess_graphs.py` | One-shot Stage-1 build of the two single-domain graphs |
| `build_recommendation_stage1.py` | Bundles Stage-1 artifacts (graphs + features + splits) under `pipeline/results/` |
| `run_recommendation_experiment.py` | Top-level Exp 2 runner — sweeps the five Exp 2 model kinds on the fused graph |

## Running Stage 1 (build the artifacts everything else depends on)

```
python pipeline/stage1_preprocess_graphs.py
python pipeline/build_recommendation_stage1.py
```

This produces `pipeline/results/` containing the two graphs, the user
features, and the 60/20/20 split. All other experiments expect these
files to already exist.

## Running Exp 2 (fusion sweep)

```
python pipeline/run_recommendation_experiment.py
```

Writes per-model metrics under `pipeline/results/`. See `exp2_fusion/`
for the supplementary α-sweep and cold-start audits.
