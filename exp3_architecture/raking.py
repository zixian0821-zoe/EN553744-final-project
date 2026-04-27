from pyparsing import results
import json
import pandas as pd
from pathlib import Path

base = Path("results/recommendation_learned_alpha")

# 只保留 Exp 3 相关模型 + 对照 baseline
keep_runs = {
    "no_graph",
    "fixed_fused_alpha_0.5",
    "chebnet_fused",
    "graphsage_fused",
    "gat_fused",
}

rows = []
for run_dir in sorted(base.iterdir()):
    if not run_dir.is_dir():
        continue
    run = run_dir.name
    if run not in keep_runs:
        continue

    mf = run_dir / "metrics.json"
    if not mf.exists():
        continue

    with open(mf, "r", encoding="utf-8") as f:
        m = json.load(f)

    rows.append({
        "run_name": run,
        "model": m.get("model", run),
        "graph": m.get("graph"),
        "alpha_value": m.get("alpha_value"),
        "Recall@20": m.get("test_Recall@20"),
        "NDCG@20": m.get("test_NDCG@20"),
        "HitRate@20": m.get("test_HitRate@20"),
        "Best_Epoch": m.get("best_epoch"),
        "Train_s": round(m.get("train_time_total_seconds", 0), 1),
    })

df = pd.DataFrame(rows).sort_values("NDCG@20", ascending=False).reset_index(drop=True)
df.index += 1
df.index.name = "Rank"

print(df.to_string())
df.to_csv(base / "exp3_architecture_ranking.csv", index=True)