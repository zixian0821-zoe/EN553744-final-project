from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from recommendation_config import RecommendationConfig


@dataclass(frozen=True)
class UserFeatureArtifacts:
    matrix: np.ndarray
    summary: dict[str, object]
    checks: dict[str, object]


@dataclass(frozen=True)
class TargetItemFeatureArtifacts:
    matrix: sparse.csr_matrix
    item_index: pd.DataFrame
    summary: dict[str, object]
    checks: dict[str, object]


def build_user_music_features_top30_l1(config: RecommendationConfig) -> UserFeatureArtifacts:
    music_matrix = sparse.load_npz(config.music_user_genre_matrix_path).tocsr().astype(np.float32)
    selected_music = pd.read_csv(config.selected_music_genres_path)
    selected_indices = selected_music["genre_idx"].to_numpy(dtype=np.int32)

    if len(selected_indices) != config.top_music_genres:
        raise ValueError("Selected music genre file does not contain the expected top-30 genres")

    selected_matrix = music_matrix[:, selected_indices].tocsr()
    dense = selected_matrix.toarray().astype(np.float32, copy=False)
    row_sums = dense.sum(axis=1, keepdims=True)
    nonzero_mask = row_sums.squeeze(1) > 0.0
    dense[nonzero_mask] = dense[nonzero_mask] / row_sums[nonzero_mask]
    dense[~nonzero_mask] = 0.0

    summary = {
        "n_users": int(dense.shape[0]),
        "feature_dim": int(dense.shape[1]),
        "zero_rows_after_top30": int((~nonzero_mask).sum()),
        "row_sum_min": float(dense.sum(axis=1).min()),
        "row_sum_max": float(dense.sum(axis=1).max()),
        "selected_music_genres_path": str(config.selected_music_genres_path),
    }
    checks = {
        "user_feature_shape_matches_protocol": dense.shape == (music_matrix.shape[0], config.top_music_genres),
        "row_sums_are_zero_or_one": bool(np.all(np.isclose(dense.sum(axis=1), 0.0) | np.isclose(dense.sum(axis=1), 1.0))),
        "selected_genre_count_matches_protocol": len(selected_indices) == config.top_music_genres,
    }
    return UserFeatureArtifacts(matrix=dense, summary=summary, checks=checks)


def parse_genre_label_strings(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if text == "":
        return []
    return [label.strip() for label in text.split("|") if label.strip()]


def build_target_item_genre_multi_hot(config: RecommendationConfig) -> TargetItemFeatureArtifacts:
    target_interactions = pd.read_csv(config.target_interactions_path)
    book_genre_index = pd.read_csv(config.book_genre_index_path)
    genre_to_idx = {
        label: int(idx) for label, idx in zip(book_genre_index["genre_label"], book_genre_index["genre_idx"], strict=True)
    }

    item_level = (
        target_interactions[
            ["item_idx", "item_id", "genre_labels", "genre_label_count", "primary_genre_label", "has_genre_labels"]
        ]
        .drop_duplicates()
        .sort_values("item_idx")
        .reset_index(drop=True)
    )

    if len(item_level) != target_interactions["item_idx"].nunique():
        raise ValueError("Target item metadata could not be reduced to one row per target item")

    item_idx_expected = np.arange(len(item_level), dtype=np.int32)
    if not np.array_equal(item_level["item_idx"].to_numpy(dtype=np.int32), item_idx_expected):
        raise ValueError("Target item_idx values are not contiguous from zero")

    grouped = target_interactions.groupby(["item_idx", "item_id"], sort=True, dropna=False)
    inconsistent_genre_labels = int((grouped["genre_labels"].nunique(dropna=False) > 1).sum())
    inconsistent_primary = int((grouped["primary_genre_label"].nunique(dropna=False) > 1).sum())
    inconsistent_has = int((grouped["has_genre_labels"].nunique(dropna=False) > 1).sum())
    if inconsistent_genre_labels or inconsistent_primary or inconsistent_has:
        raise ValueError("Target item genre metadata is inconsistent across interaction rows")

    rows: list[int] = []
    cols: list[int] = []
    unknown_labels: set[str] = set()
    label_counts: list[int] = []

    for row in item_level.itertuples(index=False):
        labels = parse_genre_label_strings(row.genre_labels)
        deduped_labels = list(dict.fromkeys(labels))
        label_counts.append(len(deduped_labels))
        for label in deduped_labels:
            genre_idx = genre_to_idx.get(label)
            if genre_idx is None:
                unknown_labels.add(label)
                continue
            rows.append(int(row.item_idx))
            cols.append(genre_idx)

    if unknown_labels:
        raise ValueError(f"Found target item genre labels missing from book_genre_index: {sorted(unknown_labels)[:10]}")

    data = np.ones(len(rows), dtype=np.float32)
    matrix = sparse.coo_matrix(
        (data, (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(len(item_level), len(book_genre_index)),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1.0

    summary = {
        "target_item_count": int(len(item_level)),
        "genre_feature_dim": int(len(book_genre_index)),
        "items_with_genre_labels": int(item_level["has_genre_labels"].sum()),
        "items_without_genre_labels": int((~item_level["has_genre_labels"]).sum()),
        "avg_labels_per_item": float(np.mean(label_counts)) if label_counts else 0.0,
        "max_labels_per_item": int(max(label_counts)) if label_counts else 0,
        "nonzero_feature_entries": int(matrix.nnz),
    }
    checks = {
        "target_item_feature_rows_match_items": matrix.shape[0] == len(item_level),
        "target_item_feature_dim_matches_protocol": matrix.shape[1] == config.book_genre_dim,
        "target_item_genre_metadata_consistent": True,
        "unknown_item_genre_labels_count": 0,
    }

    item_index = item_level[
        ["item_idx", "item_id", "has_genre_labels", "genre_label_count", "primary_genre_label"]
    ].copy()
    return TargetItemFeatureArtifacts(matrix=matrix, item_index=item_index, summary=summary, checks=checks)
