# Stage 0 — Data preprocessing

Downloads the Amazon Reviews 2023 dumps (CDs/Vinyl as source, Books as
target) and materializes flat `interactions` tables that the rest of the
pipeline consumes.

## Files in this folder

| File | Purpose |
|---|---|
| `amazon_data_preprocess.ipynb` | One-shot notebook: pulls the two raw category dumps from the McAuley-Lab HuggingFace mirror and writes per-category parquet under `raw/` |
| `materialize_amazon_interactions.py` | Flattens raw parquet into `source_interactions.csv` and `target_interactions.csv` (user, item, rating, timestamp), restricted to the 15,824 users that appear on both sides |

## Running

```
# 1. download raw dumps (notebook)
data_preprocessing/amazon_data_preprocess.ipynb

# 2. materialize the two interaction tables
python data_preprocessing/materialize_amazon_interactions.py
```

## Outputs

| File | Rows |
|---|---|
| `data/source_interactions.csv` (CDs/Vinyl) | 555,963 |
| `data/target_interactions.csv` (Books) | 347,212 |

These two CSVs are the only inputs Stage 1 (`pipeline/`) needs.
