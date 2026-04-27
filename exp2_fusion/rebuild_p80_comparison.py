\
\
\
\
\
\
\
\
\
   
import sys as _sys
from pathlib import Path as _Path
_PIPELINE_DIR = _Path(__file__).resolve().parent.parent / "pipeline"
if str(_PIPELINE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_PIPELINE_DIR))

import json
from pathlib import Path
import pandas as pd

ROOT = Path("/content/drive/MyDrive/Experiment2/results/recommendation_learned_alpha")
assert ROOT.exists(), f"Not found: {ROOT}"

ROWS = [
    ("No-graph",                "no_graph",             "none",   "none",     None, "p20"),
    ("Source-only GCN",         "source_only",          "source", "fixed",    1.0,  "p20"),
    ("Target-only GCN",         "target_only",          "target", "fixed",    0.0,  "p20"),
    ("Fixed-α GCN (α=0.1)",     "fixed_alpha_0.1_p80",  "fused",  "fixed",    0.1,  "p80"),
    ("Fixed-α GCN (α=0.3)",     "fixed_alpha_0.3_p80",  "fused",  "fixed",    0.3,  "p80"),
    ("Fixed-α GCN (α=0.5)",     "fixed_alpha_0.5_p80",  "fused",  "fixed",    0.5,  "p80"),
    ("Fixed-α GCN (α=0.7)",     "fixed_alpha_0.7_p80",  "fused",  "fixed",    0.7,  "p80"),
    ("Fixed-α GCN (α=0.9)",     "fixed_alpha_0.9_p80",  "fused",  "fixed",    0.9,  "p80"),
    ("Learned-α GCN",           "learned_alpha_fused_p80",  "fused", "learned", None, "p80"),
    ("Per-user-α GCN",          "per_user_alpha_fused_p80", "fused", "per_user", None, "p80"),
]

def first_present(d: dict, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def load_metrics(subdir: str) -> dict:
    p = ROOT / subdir / "metrics.json"
    if not p.exists():
        print(f"  MISSING: {p}")
        return {}
    with open(p) as f:
        return json.load(f)

rows = []
print("Reading metrics.json from each subdir:\n")
for name, sub, graph, amode, aval, pat in ROWS:
    m = load_metrics(sub)
    if not m:
        continue
    if not rows:
        print(f"  schema sample ({sub}): keys = {sorted(m.keys())}\n")
    row = {
        "model": name,
        "graph": graph,
        "alpha_mode": amode,
        "alpha_value": first_present(m, ["alpha_value", "alpha", "learned_alpha"], aval),
        "patience_tag": pat,
        "Recall@10":  first_present(m, ["test_recall@10",  "Recall@10",  "recall_10"]),
        "Recall@20":  first_present(m, ["test_recall@20",  "Recall@20",  "recall_20"]),
        "NDCG@10":    first_present(m, ["test_ndcg@10",    "NDCG@10",    "ndcg_10"]),
        "NDCG@20":    first_present(m, ["test_ndcg@20",    "NDCG@20",    "ndcg_20"]),
        "HitRate@20": first_present(m, ["test_hitrate@20", "HitRate@20", "hitrate_20"]),
        "val_NDCG@20": first_present(m, ["val_ndcg@20", "val_NDCG@20"]),
        "Best_Epoch":  first_present(m, ["best_epoch",  "Best_Epoch"]),
        "Epochs_Ran":  first_present(m, ["epochs_ran",  "Epochs_Ran"]),
        "Train_Time_Total_Seconds": first_present(
            m, ["train_time_total_seconds", "Train_Time_Total_Seconds", "train_time"]
        ),
    }
    rows.append(row)

df = pd.DataFrame(rows)

df = df.sort_values("NDCG@20", ascending=False).reset_index(drop=True)

out = ROOT / "model_comparison_p80_rebuilt.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")

pd.set_option("display.float_format", lambda x: f"{x:.6f}")
print("\n=== Rebuilt comparison table (sorted by test NDCG@20) ===\n")
print(df[["model","patience_tag","alpha_value","NDCG@20","Recall@20","NDCG@10","Recall@10",
         "HitRate@20","val_NDCG@20","Best_Epoch","Epochs_Ran"]].to_string(index=False))

old = ROOT / "model_comparison.csv"
if old.exists():
    print(f"\nOld file still present at: {old}  (rebuilt file is separate — not overwriting)")
