from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from recommendation_config import RecommendationConfig
from train_recommendation import ModelSpec, TrainConfig, load_stage1_artifacts, train_one_model


FIXED_ALPHA_RUN_NAMES = [f"fixed_alpha_{alpha:.1f}" for alpha in RecommendationConfig().fixed_alpha_grid]


def save_json(path: Path, payload: dict[str, object], indent: int = 2) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True)


def no_graph_spec() -> ModelSpec:
    return ModelSpec("no_graph", "No-graph", "no_graph", "none", "none", None)


def source_only_spec() -> ModelSpec:
    return ModelSpec("source_only", "Source-only GCN", "source_only", "source", "fixed", 1.0)


def target_only_spec() -> ModelSpec:
    return ModelSpec("target_only", "Target-only GCN", "target_only", "target", "fixed", 0.0)

def chebnet_spec() -> ModelSpec:
    return ModelSpec(
        "chebnet_fused",
        "ChebNet (K=3)",
        "chebnet",
        "fused",
        "fixed",
        0.5,
    )


def graphsage_spec() -> ModelSpec:
    return ModelSpec(
        "graphsage_fused",
        "GraphSAGE-mean",
        "graphsage",
        "fused",
        "fixed",
        0.5,
    )


def gat_spec() -> ModelSpec:
    return ModelSpec(
        "gat_fused",
        "GAT",
        "gat",
        "fused",
        "fixed",
        0.5,
    )
def fixed_alpha_spec(alpha: float) -> ModelSpec:
    return ModelSpec(
        run_name=f"fixed_alpha_{alpha:.1f}",
        display_name=f"Fixed-alpha GCN (alpha={alpha:.1f})",
        model_kind="fixed_fused",
        graph="fused",
        alpha_mode="fixed",
        alpha_value=float(alpha),
    )

def chebnet_spec() -> ModelSpec:
    return ModelSpec(
        "chebnet_fused",
        "ChebNet (K=3)",
        "chebnet",
        "fused",
        "fixed",
        0.5,
    )


def graphsage_spec() -> ModelSpec:
    return ModelSpec(
        "graphsage_fused",
        "GraphSAGE-mean",
        "graphsage",
        "fused",
        "fixed",
        0.5,
    )


def gat_spec() -> ModelSpec:
    return ModelSpec(
        "gat_fused",
        "GAT",
        "gat",
        "fused",
        "fixed",
        0.5,
    )
def learned_alpha_spec() -> ModelSpec:
    return ModelSpec("learned_alpha_fused", "Learned-alpha GCN", "learned_alpha", "fused", "learned", None)


def per_user_alpha_spec() -> ModelSpec:
    return ModelSpec("per_user_alpha_fused", "Per-user-alpha GCN", "per_user_alpha", "fused", "per_user", None)


def save_model_outputs(
    model_dir: Path,
    results: dict[str, object],
    history: list[dict[str, object]],
    ranking_summary: dict[str, object],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    save_json(model_dir / "metrics.json", results)
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    save_json(model_dir / "ranking_summary.json", ranking_summary)


def build_comparison_tables(rows: list[dict[str, object]], results_dir: Path, partial: bool) -> None:
    if not rows:
        return

    rows_df = pd.DataFrame(rows)
    rows_df = rows_df.sort_values("run_name").reset_index(drop=True)

    fixed_alpha_rows = rows_df[rows_df["run_name"].str.startswith("fixed_alpha_")].copy()

    comparison_rows = []
    row_map = {row["run_name"]: row for row in rows_df.to_dict(orient="records")}

    if "no_graph" in row_map:
        comparison_rows.append(_comparison_row(row_map["no_graph"], model_label="No-graph"))
    if "source_only" in row_map:
        comparison_rows.append(_comparison_row(row_map["source_only"], model_label="Source-only GCN"))
    if "target_only" in row_map:
        comparison_rows.append(_comparison_row(row_map["target_only"], model_label="Target-only GCN"))

    fixed_fused_source = None
    if "fixed_fused_alpha_0.5" in row_map:
        fixed_fused_source = row_map["fixed_fused_alpha_0.5"]
    elif "fixed_alpha_0.5" in row_map:
        fixed_fused_source = row_map["fixed_alpha_0.5"]
    if fixed_fused_source is not None:
        comparison_rows.append(_comparison_row(fixed_fused_source, model_label="Fixed fused GCN"))

    if not fixed_alpha_rows.empty:
        best_fixed_alpha_row = fixed_alpha_rows.sort_values(["val_NDCG@20", "test_NDCG@20"], ascending=False).iloc[0].to_dict()
        comparison_rows.append(_comparison_row(best_fixed_alpha_row, model_label="Best fixed-alpha GCN"))

    if "learned_alpha_fused" in row_map:
        comparison_rows.append(_comparison_row(row_map["learned_alpha_fused"], model_label="Learned-alpha GCN"))
    if "per_user_alpha_fused" in row_map:
        comparison_rows.append(_comparison_row(row_map["per_user_alpha_fused"], model_label="Per-user-alpha GCN"))

    if comparison_rows:
        model_comparison = pd.DataFrame(comparison_rows)
        model_csv_name = "partial_model_comparison.csv" if partial else "model_comparison.csv"
        model_json_name = "partial_model_comparison.json" if partial else "model_comparison.json"
        model_comparison.to_csv(results_dir / model_csv_name, index=False)
        model_comparison.to_json(results_dir / model_json_name, orient="records", indent=2)

    alpha_rows = []
    if not fixed_alpha_rows.empty:
        for row in fixed_alpha_rows.sort_values("alpha_numeric").to_dict(orient="records"):
            alpha_rows.append(
                {
                    "model": row["display_name"],
                    "alpha_mode": row["alpha_mode"],
                    "alpha_value": row["alpha_numeric"],
                    "val_ndcg@20": row["val_NDCG@20"],
                    "test_ndcg@20": row["test_NDCG@20"],
                    "test_recall@20": row["test_Recall@20"],
                }
            )
    if "learned_alpha_fused" in row_map:
        learned_row = row_map["learned_alpha_fused"]
        alpha_rows.append(
            {
                "model": learned_row["display_name"],
                "alpha_mode": learned_row["alpha_mode"],
                "alpha_value": learned_row["alpha_numeric"],
                "val_ndcg@20": learned_row["val_NDCG@20"],
                "test_ndcg@20": learned_row["test_NDCG@20"],
                "test_recall@20": learned_row["test_Recall@20"],
            }
        )
    if alpha_rows:
        alpha_name = "partial_alpha_comparison.csv" if partial else "alpha_comparison.csv"
        pd.DataFrame(alpha_rows).to_csv(results_dir / alpha_name, index=False)


def completed_rows_from_disk(results_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        run_name = metrics_path.parent.name
        if run_name.startswith("smoke_"):
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            results = json.load(handle)
        display_name = str(results["model"])
        if run_name == "fixed_fused_alpha_0.5":
            display_name = "Fixed fused GCN"
        rows.append(
            {
                "run_name": run_name,
                "display_name": display_name,
                "graph": results["graph"],
                "alpha_mode": results["alpha_mode"],
                "alpha_numeric": float(results["alpha_value"]) if results["alpha_value"] is not None else float("nan"),
                "val_Recall@10": results["val_Recall@10"],
                "val_Recall@20": results["val_Recall@20"],
                "val_NDCG@10": results["val_NDCG@10"],
                "val_NDCG@20": results["val_NDCG@20"],
                "val_HitRate@20": results["val_HitRate@20"],
                "test_Recall@10": results["test_Recall@10"],
                "test_Recall@20": results["test_Recall@20"],
                "test_NDCG@10": results["test_NDCG@10"],
                "test_NDCG@20": results["test_NDCG@20"],
                "test_HitRate@20": results["test_HitRate@20"],
                "Best_Epoch": results["best_epoch"],
                "Epochs_Ran": results["epochs_ran"],
                "Train_Time_Total_Seconds": results["train_time_total_seconds"],
            }
        )
    return rows


def refresh_rolling_tables(results_dir: Path) -> None:
    rows = completed_rows_from_disk(results_dir)
    build_comparison_tables(rows, results_dir, partial=True)
    if final_table_ready(rows):
        build_comparison_tables(rows, results_dir, partial=False)


def final_table_ready(rows: list[dict[str, object]]) -> bool:
    run_names = {row["run_name"] for row in rows}
    required = {"no_graph", "source_only", "target_only", "learned_alpha_fused", "fixed_alpha_0.5"}
    return required.issubset(run_names) and all(run_name in run_names for run_name in FIXED_ALPHA_RUN_NAMES)


def ensure_fixed_fused_alias(results_dir: Path) -> None:
    ensure_fixed_fused_alias_named(
        results_dir=results_dir,
        source_run_name="fixed_alpha_0.5",
        alias_run_name="fixed_fused_alpha_0.5",
    )


def ensure_fixed_fused_alias_named(results_dir: Path, source_run_name: str, alias_run_name: str) -> None:
    source_dir = results_dir / source_run_name
    if not source_dir.exists():
        return
    alias_dir = results_dir / alias_run_name
    alias_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["metrics.json", "training_history.csv", "ranking_summary.json"]:
        shutil.copy2(source_dir / filename, alias_dir / filename)


def selected_specs(run_tokens: list[str], config: RecommendationConfig) -> list[tuple[ModelSpec, bool]]:
    resolved: list[tuple[ModelSpec, bool]] = []
    for token in run_tokens:
        if token == "all":
            resolved.extend(
                [
                    (no_graph_spec(), False),
                    (source_only_spec(), False),
                    (target_only_spec(), False),
                    (fixed_alpha_spec(0.1), False),
                    (fixed_alpha_spec(0.3), False),
                    (fixed_alpha_spec(0.5), True),
                    (fixed_alpha_spec(0.7), False),
                    (fixed_alpha_spec(0.9), False),
                    (learned_alpha_spec(), False),
                    (per_user_alpha_spec(), False),
                    (chebnet_spec(), False),
                    (graphsage_spec(), False),
                    (gat_spec(), False),
                ]
            )
            continue
        if token == "no_graph":
            resolved.append((no_graph_spec(), False))
            continue
        if token == "source_only":
            resolved.append((source_only_spec(), False))
            continue
        if token == "target_only":
            resolved.append((target_only_spec(), False))
            continue
        if token == "fixed_fused":
            resolved.append((fixed_alpha_spec(0.5), True))
            continue
        if token == "fixed_alpha_sweep":
            for alpha in config.fixed_alpha_grid:
                resolved.append((fixed_alpha_spec(float(alpha)), float(alpha) == 0.5))
            continue
        if token == "learned_alpha":
            resolved.append((learned_alpha_spec(), False))
            continue
        if token == "per_user_alpha":
            resolved.append((per_user_alpha_spec(), False))
            continue
        if token.startswith("fixed_alpha_"):
            try:
                alpha = float(token.split("_")[-1])
            except ValueError as exc:
                raise ValueError(f"Invalid fixed-alpha token: {token}") from exc
            resolved.append((fixed_alpha_spec(alpha), alpha == 0.5))
            continue

        if token == "chebnet":
            resolved.append((chebnet_spec(), False))
            continue

        if token == "graphsage":
            resolved.append((graphsage_spec(), False))
            continue

        if token == "gat":
            resolved.append((gat_spec(), False))
            continue
        raise ValueError(f"Unsupported run token: {token}")
    return dedupe_specs(resolved)


def dedupe_specs(specs: list[tuple[ModelSpec, bool]]) -> list[tuple[ModelSpec, bool]]:
    deduped: dict[str, tuple[ModelSpec, bool]] = {}
    for spec, make_fixed_alias in specs:
        previous = deduped.get(spec.run_name)
        if previous is None:
            deduped[spec.run_name] = (spec, make_fixed_alias)
        else:
            deduped[spec.run_name] = (spec, previous[1] or make_fixed_alias)
    return list(deduped.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        help="Run selection: all, no_graph, source_only, target_only, fixed_fused, fixed_alpha_sweep, fixed_alpha_0.1, ..., fixed_alpha_0.9, learned_alpha, per_user_alpha",
    )
    parser.add_argument("--max-epochs", type=int, default=None, help="Optional override for smoke/debug runs")
    parser.add_argument("--patience", type=int, default=None, help="Optional override for smoke/debug runs")
    parser.add_argument(
        "--smoke-eval-users",
        type=int,
        default=None,
        help="Optional cap on validation/test ranking users for smoke runs",
    )
    parser.add_argument(
        "--smoke-skip-ranking",
        action="store_true",
        help="Skip expensive ranking evaluation and still write smoke outputs",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Prefix output directories with smoke_ so debug runs never overwrite real baselines",
    )
    parser.add_argument(
        "--enriched-features",
        action="store_true",
        help="Enrich user features with book_train_top30 (music_top30 + book_train_top30 = 60 dims)",
    )
    parser.add_argument(
        "--highrated-sampling",
        action="store_true",
        help="Only use rating≥4 target-train interactions as BPR positives",
    )
    parser.add_argument(
        "--highrated-threshold",
        type=float,
        default=4.0,
        help="Rating threshold for --highrated-sampling (default: 4.0)",
    )
    args = parser.parse_args()

    config = RecommendationConfig()
    train_config = TrainConfig(
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        max_epochs=args.max_epochs if args.max_epochs is not None else config.max_epochs,
        patience=args.patience if args.patience is not None else config.patience,
        seed=config.seed,
        smoke_eval_user_limit=args.smoke_eval_users,
        smoke_skip_ranking_eval=bool(args.smoke_skip_ranking),
        use_enriched_features=bool(args.enriched_features),
        highrated_sampling=bool(args.highrated_sampling),
        highrated_threshold=float(args.highrated_threshold),
    )
    stage1 = load_stage1_artifacts(config)
    run_tokens = args.runs if args.runs else ["all"]
    specs_to_run = selected_specs(run_tokens, config)

    # Build run-name prefix from active variant flags
    variant_prefix = _build_variant_prefix(args)

    for spec, make_fixed_alias in specs_to_run:
        resolved_spec = spec
        if args.smoke:
            resolved_spec = apply_smoke_prefix(resolved_spec)
        if variant_prefix:
            resolved_spec = apply_variant_prefix(resolved_spec, variant_prefix)
        results, history, ranking_summary = train_one_model(resolved_spec, stage1, train_config, config)
        model_dir = config.results_dir / resolved_spec.run_name
        save_model_outputs(model_dir, results, history, ranking_summary)
        if make_fixed_alias and not variant_prefix:
            if args.smoke:
                ensure_fixed_fused_alias_named(
                    results_dir=config.results_dir,
                    source_run_name=resolved_spec.run_name,
                    alias_run_name="smoke_fixed_fused_alpha_0.5",
                )
            else:
                ensure_fixed_fused_alias(config.results_dir)
        if not args.smoke:
            refresh_rolling_tables(config.results_dir)


def _comparison_row(row: dict[str, object], model_label: str) -> dict[str, object]:
    alpha_value = row["alpha_numeric"]
    alpha_serialized = None if pd.isna(alpha_value) else float(alpha_value)
    return {
        "model": model_label,
        "graph": row["graph"],
        "alpha_mode": row["alpha_mode"],
        "alpha_value": alpha_serialized,
        "Recall@10": row["test_Recall@10"],
        "Recall@20": row["test_Recall@20"],
        "NDCG@10": row["test_NDCG@10"],
        "NDCG@20": row["test_NDCG@20"],
        "HitRate@20": row["test_HitRate@20"],
        "Best_Epoch": row["Best_Epoch"],
        "Epochs_Ran": row["Epochs_Ran"],
        "Train_Time_Total_Seconds": row["Train_Time_Total_Seconds"],
    }


def apply_smoke_prefix(spec: ModelSpec) -> ModelSpec:
    return ModelSpec(
        run_name=f"smoke_{spec.run_name}",
        display_name=f"{spec.display_name} [smoke]",
        model_kind=spec.model_kind,
        graph=spec.graph,
        alpha_mode=spec.alpha_mode,
        alpha_value=spec.alpha_value,
    )


def _build_variant_prefix(args) -> str:
    """Compose a short directory prefix from active feature-variant flags."""
    parts = []
    if getattr(args, "enriched_features", False):
        parts.append("ef")
    if getattr(args, "highrated_sampling", False):
        parts.append("hr")
    return "_".join(parts)  # e.g. "", "ef", "hr", "ef_hr"


def apply_variant_prefix(spec: ModelSpec, prefix: str) -> ModelSpec:
    """Prefix run_name and display_name with the active variant tag."""
    tag = prefix.upper().replace("_", "+")  # e.g. "EF", "HR", "EF+HR"
    return ModelSpec(
        run_name=f"{prefix}_{spec.run_name}",
        display_name=f"{spec.display_name} [{tag}]",
        model_kind=spec.model_kind,
        graph=spec.graph,
        alpha_mode=spec.alpha_mode,
        alpha_value=spec.alpha_value,
    )


if __name__ == "__main__":
    main()
