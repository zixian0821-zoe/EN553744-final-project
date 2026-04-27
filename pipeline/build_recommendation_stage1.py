from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from recommendation_config import RecommendationConfig
from recommendation_data import (
    build_candidate_item_universe,
    build_eval_user_masks,
    build_positive_sets_by_user,
    build_target_interaction_splits,
    data_summary_dict,
    load_recommendation_inputs,
    split_summary_dict,
    verify_user_alignment,
)
from recommendation_features import build_target_item_genre_multi_hot, build_user_music_features_top30_l1
from recommendation_graphs import build_source_operator, build_target_operator


def save_json(path: Path, payload: dict[str, object], indent: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True)


def save_stage1_artifacts(
    config: RecommendationConfig,
    user_features: np.ndarray,
    item_feature_matrix: sparse.csr_matrix,
    target_item_index: pd.DataFrame,
    split_interactions: pd.DataFrame,
    user_masks: dict[str, np.ndarray],
    positive_sets: dict[str, sparse.csr_matrix],
    candidate_item_universe: np.ndarray,
    source_adjacency: sparse.csr_matrix,
    target_adjacency: sparse.csr_matrix,
    source_operator: sparse.csr_matrix,
    target_operator: sparse.csr_matrix,
    target_train_user_profiles: sparse.csr_matrix,
) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(config.artifact_path("user_features"), matrix=user_features.astype(np.float32, copy=False))
    sparse.save_npz(config.artifact_path("target_item_genre_features"), item_feature_matrix.tocsr().astype(np.float32))
    target_item_index.to_csv(config.artifact_path("target_item_index"), index=False)
    split_interactions.to_csv(config.artifact_path("target_split_assignments"), index=False)
    np.savez_compressed(
        config.artifact_path("user_masks"),
        **{name: values.astype(bool, copy=False) for name, values in user_masks.items()},
    )
    sparse.save_npz(config.artifact_path("train_user_item"), positive_sets["train"].tocsr().astype(np.float32))
    sparse.save_npz(config.artifact_path("val_user_item"), positive_sets["val"].tocsr().astype(np.float32))
    sparse.save_npz(config.artifact_path("test_user_item"), positive_sets["test"].tocsr().astype(np.float32))
    np.save(config.artifact_path("candidate_items"), candidate_item_universe.astype(np.int32, copy=False))
    sparse.save_npz(config.artifact_path("source_graph_adjacency"), source_adjacency.tocsr().astype(np.float32))
    sparse.save_npz(config.artifact_path("target_graph_adjacency"), target_adjacency.tocsr().astype(np.float32))
    sparse.save_npz(config.artifact_path("source_operator"), source_operator.tocsr().astype(np.float32))
    sparse.save_npz(config.artifact_path("target_operator"), target_operator.tocsr().astype(np.float32))
    sparse.save_npz(config.artifact_path("target_train_user_profiles"), target_train_user_profiles.tocsr().astype(np.float32))


def main() -> None:
    config = RecommendationConfig()
    config.results_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_recommendation_inputs(config)
    alignment_checks = verify_user_alignment(inputs)
    split_artifacts = build_target_interaction_splits(inputs.target_interactions, config)
    user_feature_artifacts = build_user_music_features_top30_l1(config)
    item_feature_artifacts = build_target_item_genre_multi_hot(config)

    book_genre_index = pd.read_csv(config.book_genre_index_path)
    source_graph_artifacts = build_source_operator(user_feature_artifacts.matrix, config)
    target_profile_artifacts, target_graph_artifacts = build_target_operator(split_artifacts, book_genre_index, config)

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
        "graph_definition": "weighted symmetric user k-NN on target-train-only book genre user profiles",
        "normalization": "S = D^(-1/2) (A + I) D^(-1/2)",
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

    save_json(config.artifact_path("config"), config.to_serializable_dict(), indent=config.json_indent)
    save_json(config.artifact_path("data_summary"), data_summary, indent=config.json_indent)
    save_json(config.artifact_path("split_summary"), split_summary, indent=config.json_indent)
    save_json(config.artifact_path("item_feature_summary"), item_feature_summary, indent=config.json_indent)
    save_json(config.artifact_path("graph_stats_source"), graph_stats_source, indent=config.json_indent)
    save_json(config.artifact_path("graph_stats_target"), graph_stats_target, indent=config.json_indent)
    save_json(config.artifact_path("validation_checks"), validation_checks, indent=config.json_indent)
    save_json(
        config.artifact_path("target_graph_leakage_checks"),
        target_graph_artifacts.checks,
        indent=config.json_indent,
    )

    save_stage1_artifacts(
        config=config,
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


if __name__ == "__main__":
    main()
