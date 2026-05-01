# Exp 3 — Architecture comparison

Holds the graph fixed (fused, α = 0.5) and varies the GNN architecture to ask:
*given the same support, how much of the lift comes from the operator class?*

Architectures compared: **MLP** (no-graph baseline), **GCN** (Exp 2 baseline),
**ChebNet (K = 3)**, **GraphSAGE-mean**, **GAT**.

## Files in this folder

| File | Purpose |
|---|---|
| `recommendation_models.py` | Drop-in replacement extending `pipeline/recommendation_models.py` with `ChebNetRecommender`, `GraphSAGERecommender`, `GATRecommender` (PyG: `ChebConv`, `SAGEConv`, `GATConv`) |
| `train_recommendation.py` | Extended `instantiate_model()` that dispatches `model_kind ∈ {chebnet, graphsage, gat}` in addition to the Exp 2 kinds |
| `run_recommendation_experiment.py` | Top-level runner — sweeps the five architectures on the fused graph and writes the comparison table |
| `learning curve.py` | Per-architecture learning curves (NDCG@20 vs epoch) |
| `raking.py` | Final ranking / summary table generation |

## Running

```
python exp3_architecture/run_recommendation_experiment.py
```

Reads Stage-1 artifacts from `pipeline/results/` (built via
`pipeline/stage1_preprocess_graphs.py` + `pipeline/build_recommendation_stage1.py`).
Writes per-architecture metrics under `pipeline/results/exp3/`.

## Results (NDCG@20, test split)

| Architecture | NDCG@20 | vs GCN |
|---|---|---|
| **ChebNet K=3** | **0.002302** | **+20.9 %** |
| GraphSAGE-mean | 0.001932 | +1.5 % |
| GCN | 0.001904 | baseline |
| MLP | 0.001645 | −13.6 % |
| GAT | 0.001630 | −14.4 % |

Takeaways: (i) the polynomial filter (ChebNet) extracts more information from
the same fused graph than the single-hop GCN; (ii) GAT's learned attention
re-weighting underperforms here — the fused similarity weights are already
informative and attention adds variance; (iii) the no-graph MLP confirms that
the lift is from topology, not from encoder capacity.
