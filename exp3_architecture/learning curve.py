from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base = Path("results/recommendation_learned_alpha")

runs = {
    "GCN (fused α=0.5)": "fixed_fused_alpha_0.5",
    "ChebNet": "chebnet_fused",
    "GraphSAGE": "graphsage_fused",
    "GAT": "gat_fused",
}

def find_col(cols, include_keywords):
    cols = list(cols)
    for c in cols:
        lc = c.lower()
        if all(k in lc for k in include_keywords):
            return c
    return None

histories = {}
for label, run in runs.items():
    fp = base / run / "training_history.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        histories[label] = df
    else:
        print(f"[WARN] missing: {fp}")

# ----------------------------
# Figure 1: Validation NDCG@20
# ----------------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))

for label, df in histories.items():
    epoch_col = find_col(df.columns, ["epoch"]) or "epoch"
    ndcg_col = find_col(df.columns, ["val", "ndcg", "20"])
    if ndcg_col is None:
        print(f"[WARN] {label}: no val_ndcg@20 column found")
        continue
    ax1.plot(df[epoch_col], df[ndcg_col], label=label)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Validation NDCG@20")
ax1.set_title("Exp 3 Learning Curve: Validation NDCG@20")
ax1.legend()
fig1.tight_layout()

fig1_path = base / "exp3_learning_curve_val_ndcg20.png"
fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
print(f"Saved: {fig1_path}")

# 不依赖窗口保留，直接关闭
plt.close(fig1)

# ----------------------------
# Figure 2: Training Loss
# ----------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))

for label, df in histories.items():
    epoch_col = find_col(df.columns, ["epoch"]) or "epoch"
    loss_col = (
        find_col(df.columns, ["bpr"]) or
        find_col(df.columns, ["train", "loss"]) or
        find_col(df.columns, ["loss"])
    )
    if loss_col is None:
        print(f"[WARN] {label}: no training loss column found")
        continue
    ax2.plot(df[epoch_col], df[loss_col], label=label)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Training Loss")
ax2.set_title("Exp 3 Learning Curve: Training Loss")
ax2.legend()
fig2.tight_layout()

fig2_path = base / "exp3_learning_curve_train_loss.png"
fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
print(f"Saved: {fig2_path}")

plt.close(fig2)

print("\nDone.")