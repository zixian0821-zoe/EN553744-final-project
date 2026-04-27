from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_PIPELINE_DIR = _Path(__file__).resolve().parent.parent / "pipeline"
if str(_PIPELINE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_PIPELINE_DIR))

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from build_recommendation_stage1 import save_json, save_stage1_artifacts
from recommendation_config import RecommendationConfig
from recommendation_data import (
    build_candidate_item_universe,
    build_eval_user_masks,
    build_positive_sets_by_user,
    data_summary_dict,
    load_recommendation_inputs,
    split_summary_dict,
    verify_user_alignment,
)
from recommendation_data_strict_coldstart import build_target_user_holdout_splits
from recommendation_features import TargetItemFeatureArtifacts, UserFeatureArtifacts, build_target_item_genre_multi_hot
from recommendation_graphs import GraphArtifacts, build_source_operator, build_target_operator, graph_stats_dict
from train_recommendation import ModelSpec, TrainConfig, load_stage1_artifacts, train_one_model

SUPPLEMENT_DIRNAME = "supplement_strict_coldstart"
COLAB_MAIN_RESULTS_FALLBACKS = (
    Path("/content/drive/MyDrive/results 2/recommendation_learned_alpha"),
    Path("/content/drive/MyDrive/Experiment2/results/recommendation_learned_alpha"),
)

def supplement_config(
    base_config: RecommendationConfig,
    supplement_results_dir: Path | None = None,
) -> RecommendationConfig:
    resolved_dir = supplement_results_dir if supplement_results_dir is not None else base_config.root / "results" / SUPPLEMENT_DIRNAME
    return replace(base_config, results_dir=resolved_dir)

def reusable_main_artifacts_ready(config: RecommendationConfig) -> bool:
    required = [
        config.artifact_path("user_features"),
        config.artifact_path("target_item_genre_features"),
        config.artifact_path("target_item_index"),
        config.artifact_path("source_graph_adjacency"),
        config.artifact_path("source_operator"),
    ]
    return all(path.exists() for path in required)

def resolve_main_artifact_config(
    base_config: RecommendationConfig,
    main_results_dir: Path | None = None,
) -> RecommendationConfig:
    candidate_dirs: list[Path] = []
    if main_results_dir is not None:
        candidate_dirs.append(main_results_dir)
    candidate_dirs.append(base_config.results_dir)
    candidate_dirs.extend(COLAB_MAIN_RESULTS_FALLBACKS)

    seen: set[Path] = set()
    for candidate in candidate_dirs:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidate_config = replace(base_config, results_dir=resolved)
        if reusable_main_artifacts_ready(candidate_config):
            if resolved != base_config.results_dir:
                print(f"[supplement] using main stage-1 artifacts from: {resolved}")
            return candidate_config

    if main_results_dir is not None:
        print(f"[supplement] main artifact override not found yet, will attempt rebuild fallbacks from: {main_results_dir}")
        return replace(base_config, results_dir=main_results_dir.expanduser().resolve())
    return base_config

def _require_exists(path: Path, message: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(message)
    return path

def load_main_user_features(base_config: RecommendationConfig) -> UserFeatureArtifacts:
    artifact = base_config.artifact_path("user_features")
    _require_exists(
        artifact,
        "Main user feature artifact is missing. Rebuild the main Experiment2 stage-1 artifacts first.",
    )
    matrix = np.load(artifact)["matrix"].astype(np.float32)
    summary = {
        "n_users": int(matrix.shape[0]),
        "feature_dim": int(matrix.shape[1]),
        "loaded_from_main_artifact": str(artifact),
        "row_sum_min": float(matrix.sum(axis=1).min()),
        "row_sum_max": float(matrix.sum(axis=1).max()),
    }
    checks = {
        "loaded_from_main_artifact": True,
        "user_feature_shape_matches_protocol": bool(matrix.ndim == 2 and matrix.shape[1] == base_config.top_music_genres),
    }
    return UserFeatureArtifacts(matrix=matrix, summary=summary, checks=checks)

def load_or_build_target_item_features(base_config: RecommendationConfig) -> TargetItemFeatureArtifacts:
    matrix_path = base_config.artifact_path("target_item_genre_features")
    item_index_path = base_config.artifact_path("target_item_index")
    if matrix_path.exists() and item_index_path.exists():
        matrix = sparse.load_npz(matrix_path).tocsr().astype(np.float32)
        item_index = pd.read_csv(item_index_path)
        summary = {
            "target_item_count": int(matrix.shape[0]),
            "genre_feature_dim": int(matrix.shape[1]),
            "loaded_from_main_artifact": str(matrix_path),
            "nonzero_feature_entries": int(matrix.nnz),
        }
        checks = {
            "loaded_from_main_artifact": True,
            "target_item_feature_rows_match_items": matrix.shape[0] == len(item_index),
            "target_item_feature_dim_matches_protocol": matrix.shape[1] == base_config.book_genre_dim,
        }
        return TargetItemFeatureArtifacts(matrix=matrix, item_index=item_index, summary=summary, checks=checks)
    return build_target_item_genre_multi_hot(base_config)

def load_or_build_source_graph(
    base_config: RecommendationConfig,
    user_feature_artifacts: UserFeatureArtifacts,
) -> GraphArtifacts:
    adj_path = base_config.artifact_path("source_graph_adjacency")
    op_path = base_config.artifact_path("source_operator")
    if adj_path.exists() and op_path.exists():
        adjacency = sparse.load_npz(adj_path).tocsr().astype(np.float32)
        operator = sparse.load_npz(op_path).tocsr().astype(np.float32)
        stats = graph_stats_dict(adjacency)
        checks = {
            "loaded_from_main_artifact": True,
            "source_graph_uses_music_user_features_only": True,
            "source_graph_is_symmetric": stats["is_symmetric"],
            "source_graph_has_no_self_loops_in_raw_adjacency": bool(adjacency.diagonal().sum() == 0.0),
        }
        return GraphArtifacts(adjacency=adjacency, operator=operator, stats=stats, checks=checks)
    return build_source_operator(user_feature_artifacts.matrix, base_config)

def no_graph_spec() -> ModelSpec:
    return ModelSpec("no_graph", "No-graph MLP", "no_graph", "none", "none", None)

def source_only_spec() -> ModelSpec:
    return ModelSpec("source_only", "Source-only GCN", "source_only", "source", "fixed", 1.0)

def target_only_spec() -> ModelSpec:
    return ModelSpec("target_only", "Target-only GCN", "target_only", "target", "fixed", 0.0)

def fused_spec(alpha: float = 0.5) -> ModelSpec:
    return ModelSpec(
        run_name=f"fused_alpha_{alpha:.1f}",
        display_name=f"Fused GCN (alpha={alpha:.1f})",
        model_kind="fixed_fused",
        graph="fused",
        alpha_mode="fixed",
        alpha_value=float(alpha),
    )

def apply_smoke_prefix(spec: ModelSpec) -> ModelSpec:
    return ModelSpec(
        run_name=f"smoke_{spec.run_name}",
        display_name=f"{spec.display_name} [smoke]",
        model_kind=spec.model_kind,
        graph=spec.graph,
        alpha_mode=spec.alpha_mode,
        alpha_value=spec.alpha_value,
    )

def selected_specs(run_tokens: list[str]) -> list[ModelSpec]:
    resolved: list[ModelSpec] = []
    for token in run_tokens:
        if token == "all":
            resolved.extend([no_graph_spec(), source_only_spec(), target_only_spec(), fused_spec(0.5)])
            continue
        if token == "no_graph":
            resolved.append(no_graph_spec())
            continue
        if token == "source_only":
            resolved.append(source_only_spec())
            continue
        if token == "target_only":
            resolved.append(target_only_spec())
            continue
        if token in {"fused", "fixed_fused", "fused_alpha_0.5"}:
            resolved.append(fused_spec(0.5))
            continue
        raise ValueError(f"Unsupported supplement run token: {token}")

    deduped: dict[str, ModelSpec] = {}
    for spec in resolved:
        deduped[spec.run_name] = spec
    return list(deduped.values())

def save_model_outputs(
    model_dir: Path,
    results: dict[str, object],
    history: list[dict[str, object]],
    ranking_summary: dict[str, object],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    save_json(model_dir / "metrics.json", results, indent=2)
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    save_json(model_dir / "ranking_summary.json", ranking_summary, indent=2)

def print_run_banner(spec: ModelSpec, model_dir: Path) -> None:
    print("\n" + "=" * 72)
    print(f"[supplement] running: {spec.display_name}")
    print(f"[supplement] run_name: {spec.run_name}")
    print(f"[supplement] output_dir: {model_dir}")
    print("=" * 72)

def print_run_summary(results: dict[str, object]) -> None:
    print(
        "[supplement] finished: "
        f"{results['model']} | "
        f"best_epoch={results['best_epoch']} | "
        f"test_NDCG@20={results['test_NDCG@20']:.6f} | "
        f"test_Recall@20={results['test_Recall@20']:.6f} | "
        f"time={results['train_time_total_seconds']:.1f}s"
    )

def build_comparison_tables(results_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        run_name = metrics_path.parent.name
        if run_name.startswith("smoke_"):
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "run_name": run_name,
                "model": metrics["model"],
                "graph": metrics["graph"],
                "alpha_mode": metrics["alpha_mode"],
                "alpha_value": metrics["alpha_value"],
                "Recall@10": metrics["test_Recall@10"],
                "Recall@20": metrics["test_Recall@20"],
                "NDCG@10": metrics["test_NDCG@10"],
                "NDCG@20": metrics["test_NDCG@20"],
                "HitRate@20": metrics["test_HitRate@20"],
                "Best_Epoch": metrics["best_epoch"],
                "Epochs_Ran": metrics["epochs_ran"],
                "Train_Time_Total_Seconds": metrics["train_time_total_seconds"],
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("NDCG@20", ascending=False).reset_index(drop=True)
    no_graph_ndcg = df.loc[df["run_name"] == "no_graph", "NDCG@20"]
    target_ndcg = df.loc[df["run_name"] == "target_only", "NDCG@20"]
    if not no_graph_ndcg.empty:
        baseline = float(no_graph_ndcg.iloc[0])
        df["NDCG20_vs_no_graph_pct"] = ((df["NDCG@20"] / baseline) - 1.0) * 100.0
    if not target_ndcg.empty:
        baseline = float(target_ndcg.iloc[0])
        df["NDCG20_vs_target_only_pct"] = ((df["NDCG@20"] / baseline) - 1.0) * 100.0

    df.to_csv(results_dir / "supplement_model_comparison.csv", index=False)
    df.to_json(results_dir / "supplement_model_comparison.json", orient="records", indent=2)

def build_stage1(main_config: RecommendationConfig, cold_config: RecommendationConfig) -> None:
    cold_config.results_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_recommendation_inputs(main_config)
    alignment_checks = verify_user_alignment(inputs)
    split_artifacts = build_target_user_holdout_splits(inputs.target_interactions, main_config)
    user_feature_artifacts = load_main_user_features(main_config)
    item_feature_artifacts = load_or_build_target_item_features(main_config)

    book_genre_index = pd.read_csv(main_config.book_genre_index_path)
    source_graph_artifacts = load_or_build_source_graph(main_config, user_feature_artifacts)
    target_profile_artifacts, target_graph_artifacts = build_target_operator(split_artifacts, book_genre_index, main_config)

    fused_support = source_graph_artifacts.adjacency.maximum(target_graph_artifacts.adjacency).tocsr().astype(np.float32)

    data_summary = data_summary_dict(inputs, alignment_checks)
    split_summary = split_summary_dict(split_artifacts)
    item_feature_summary = item_feature_artifacts.summary
    graph_stats_source = {
        **source_graph_artifacts.stats,
        "graph_definition": "weighted symmetric user k-NN on music top-30 L1-normalized user profiles",
        "normalization": "S = D^(-1/2) (A + I) D^(-1/2)",
    }
    graph_stats_target = {
        **target_graph_artifacts.stats,
        **target_profile_artifacts.summary,
        "graph_definition": "weighted symmetric user k-NN on strict-cold-start train-user book genre profiles",
        "normalization": "S = D^(-1/2) (A + I) D^(-1/2)",
    }
    protocol_summary = {
        "experiment_name": "supplement_strict_coldstart_user_holdout",
        "mainline_unchanged": True,
        "purpose": "Stress-test cross-domain fusion under a stricter user-holdout cold-start protocol.",
        "split_protocol": "Users with >=5 target interactions are split 60/20/20 by user into train/val/test. Low-activity users remain train-only. Validation/test users contribute zero target-train interactions.",
        "main_results_dir": str(main_config.results_dir),
        "supplement_results_dir": str(cold_config.results_dir),
        "graph_stats_fused_support": {
            **graph_stats_dict(fused_support),
            "definition": "support union of source_graph_adjacency and target_graph_adjacency",
        },
    }
    validation_checks = {
        "alignment_checks": alignment_checks,
        "split_checks": split_artifacts.checks,
        "user_feature_checks": user_feature_artifacts.checks,
        "item_feature_checks": item_feature_artifacts.checks,
        "source_graph_checks": source_graph_artifacts.checks,
        "target_graph_checks": {
            **target_profile_artifacts.checks,
            **target_graph_artifacts.checks,
        },
    }

    save_json(cold_config.artifact_path("config"), cold_config.to_serializable_dict(), indent=2)
    save_json(cold_config.artifact_path("data_summary"), data_summary, indent=2)
    save_json(cold_config.artifact_path("split_summary"), split_summary, indent=2)
    save_json(cold_config.artifact_path("item_feature_summary"), item_feature_summary, indent=2)
    save_json(cold_config.artifact_path("graph_stats_source"), graph_stats_source, indent=2)
    save_json(cold_config.artifact_path("graph_stats_target"), graph_stats_target, indent=2)
    save_json(cold_config.artifact_path("validation_checks"), validation_checks, indent=2)
    save_json(cold_config.artifact_path("target_graph_leakage_checks"), target_graph_artifacts.checks, indent=2)
    save_json(cold_config.results_dir / "supplement_protocol_summary.json", protocol_summary, indent=2)
    save_json(cold_config.results_dir / "graph_stats_fused_support.json", protocol_summary["graph_stats_fused_support"], indent=2)

    save_stage1_artifacts(
        config=cold_config,
        user_features=user_feature_artifacts.matrix,
        item_feature_matrix=item_feature_artifacts.matrix,
        target_item_index=item_feature_artifacts.item_index,
        split_interactions=split_artifacts.split_interactions,
        user_masks=build_eval_user_masks(split_artifacts),
        positive_sets=build_positive_sets_by_user(split_artifacts),
        candidate_item_universe=build_candidate_item_universe(split_artifacts),
        source_adjacency=source_graph_artifacts.adjacency,
        target_adjacency=target_graph_artifacts.adjacency,
        source_operator=source_graph_artifacts.operator,
        target_operator=target_graph_artifacts.operator,
        target_train_user_profiles=target_profile_artifacts.matrix,
    )

def stage1_ready(config: RecommendationConfig) -> bool:
    required = [
        config.artifact_path("user_features"),
        config.artifact_path("target_item_genre_features"),
        config.artifact_path("train_user_item"),
        config.artifact_path("val_user_item"),
        config.artifact_path("test_user_item"),
        config.artifact_path("source_operator"),
        config.artifact_path("target_operator"),
    ]
    return all(path.exists() for path in required)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", dest="runs", help="Run selection: all, no_graph, source_only, target_only, fused")
    parser.add_argument(
        "--main-results-dir",
        type=Path,
        default=None,
        help="Optional location of the main Experiment2 stage-1 artifacts (defaults to Experiment2/results/recommendation_learned_alpha).",
    )
    parser.add_argument(
        "--supplement-results-dir",
        type=Path,
        default=None,
        help="Optional output directory for strict-cold-start artifacts and model runs.",
    )
    parser.add_argument("--rebuild-stage1", action="store_true", help="Force regeneration of strict cold-start stage-1 artifacts")
    parser.add_argument("--stage1-only", action="store_true", help="Build stage-1 artifacts only, then exit")
    parser.add_argument("--max-epochs", type=int, default=None, help="Optional override for debug or supplement runs")
    parser.add_argument("--patience", type=int, default=None, help="Optional override for debug or supplement runs")
    parser.add_argument("--smoke", action="store_true", help="Prefix run directories with smoke_ so quick tests never overwrite real outputs")
    parser.add_argument("--smoke-eval-users", type=int, default=None, help="Optional cap on ranking users for smoke runs")
    parser.add_argument("--smoke-skip-ranking", action="store_true", help="Skip expensive ranking eval for smoke runs")
    args = parser.parse_args()

    base_config = RecommendationConfig()
    main_config = resolve_main_artifact_config(base_config, main_results_dir=args.main_results_dir)
    cold_config = supplement_config(base_config, supplement_results_dir=args.supplement_results_dir)

    print(f"[supplement] data root: {base_config.root}")
    print(f"[supplement] main stage-1 dir: {main_config.results_dir}")
    print(f"[supplement] supplement dir: {cold_config.results_dir}")

    if args.rebuild_stage1 or not stage1_ready(cold_config):
        build_stage1(main_config, cold_config)

    if args.stage1_only:
        print(f"Built strict cold-start stage-1 artifacts in {cold_config.results_dir}")
        return

    train_config = TrainConfig(
        hidden_dim=main_config.hidden_dim,
        dropout=main_config.dropout,
        max_epochs=args.max_epochs if args.max_epochs is not None else main_config.max_epochs,
        patience=args.patience if args.patience is not None else main_config.patience,
        seed=main_config.seed,
        smoke_eval_user_limit=args.smoke_eval_users,
        smoke_skip_ranking_eval=bool(args.smoke_skip_ranking),
    )
    stage1 = load_stage1_artifacts(cold_config)
    specs_to_run = selected_specs(args.runs if args.runs else ["all"])

    for spec in specs_to_run:
        resolved_spec = apply_smoke_prefix(spec) if args.smoke else spec
        model_dir = cold_config.results_dir / resolved_spec.run_name
        print_run_banner(resolved_spec, model_dir)
        results, history, ranking_summary = train_one_model(resolved_spec, stage1, train_config, cold_config)
        save_model_outputs(model_dir, results, history, ranking_summary)
        print_run_summary(results)

    if not args.smoke:
        build_comparison_tables(cold_config.results_dir)
        print(f"\n[supplement] comparison table written to: {cold_config.results_dir / 'supplement_model_comparison.csv'}")

if __name__ == "__main__":
    main()
