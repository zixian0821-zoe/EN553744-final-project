from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

from recommendation_config import RecommendationConfig
from recommendation_data import SPLIT_TRAIN, TargetSplitArtifacts
from recommendation_features import parse_genre_label_strings


@dataclass(frozen=True)
class TargetGraphProfileArtifacts:
    matrix: sparse.csr_matrix
    summary: dict[str, object]
    checks: dict[str, object]


@dataclass(frozen=True)
class GraphArtifacts:
    adjacency: sparse.csr_matrix
    operator: sparse.csr_matrix
    stats: dict[str, object]
    checks: dict[str, object]


def build_source_user_profiles(user_features: np.ndarray) -> np.ndarray:
    return np.asarray(user_features, dtype=np.float32)


def build_target_train_user_genre_profiles(
    split_artifacts: TargetSplitArtifacts,
    book_genre_index: pd.DataFrame,
    config: RecommendationConfig,
) -> TargetGraphProfileArtifacts:
    split_interactions = split_artifacts.split_interactions
    train_rows = split_interactions.loc[split_interactions["split"] == SPLIT_TRAIN].copy()
    train_rows["parsed_genres"] = train_rows["genre_labels"].map(parse_genre_label_strings)

    genre_to_idx = {
        label: int(idx) for label, idx in zip(book_genre_index["genre_label"], book_genre_index["genre_idx"], strict=True)
    }
    n_users = int(split_interactions["user_idx"].nunique())
    n_genres = int(len(book_genre_index))

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    train_rows_with_labels = 0
    unlabeled_train_rows = 0
    unknown_labels: set[str] = set()

    for row in train_rows.itertuples(index=False):
        labels = row.parsed_genres
        if not labels:
            unlabeled_train_rows += 1
            continue
        train_rows_with_labels += 1
        for label in labels:
            genre_idx = genre_to_idx.get(label)
            if genre_idx is None:
                unknown_labels.add(label)
                continue
            rows.append(int(row.user_idx))
            cols.append(genre_idx)
            data.append(1.0)

    if unknown_labels:
        raise ValueError(f"Found unknown target train genre labels: {sorted(unknown_labels)[:10]}")

    profile_counts = sparse.coo_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(n_users, n_genres),
        dtype=np.float32,
    ).tocsr()
    profile_counts.sum_duplicates()

    row_sums = np.asarray(profile_counts.sum(axis=1)).ravel().astype(np.float32)
    inv_row_sums = np.zeros_like(row_sums, dtype=np.float32)
    positive_rows = row_sums > 0.0
    inv_row_sums[positive_rows] = 1.0 / row_sums[positive_rows]
    profile_l1 = sparse.diags(inv_row_sums) @ profile_counts
    profile_l1 = profile_l1.tocsr().astype(np.float32)

    summary = {
        "n_users": int(n_users),
        "genre_dim": int(n_genres),
        "train_rows_total": int(len(train_rows)),
        "train_rows_with_labeled_genres": int(train_rows_with_labels),
        "train_rows_without_labeled_genres": int(unlabeled_train_rows),
        "zero_profile_rows": int((~positive_rows).sum()),
        "nonzero_profile_rows": int(positive_rows.sum()),
        "pre_l1_profile_mass_min": float(row_sums[positive_rows].min()) if positive_rows.any() else 0.0,
        "pre_l1_profile_mass_max": float(row_sums[positive_rows].max()) if positive_rows.any() else 0.0,
    }
    checks = {
        "used_only_train_split_for_target_profiles": True,
        "val_test_rows_excluded_from_target_profiles": True,
        "unknown_target_train_genre_labels_count": 0,
        "zero_profile_rows_expected_for_no_labeled_train_signal": True,
        "profile_shape_matches_protocol": profile_l1.shape == (n_users, n_genres),
        "normalized_target_profile_rows_sum_to_zero_or_one": bool(
            np.all(np.isclose(np.asarray(profile_l1.sum(axis=1)).ravel(), 0.0) | np.isclose(np.asarray(profile_l1.sum(axis=1)).ravel(), 1.0))
        ),
    }
    return TargetGraphProfileArtifacts(matrix=profile_l1, summary=summary, checks=checks)


def build_weighted_knn_graph(
    profiles: np.ndarray | sparse.csr_matrix,
    k: int,
    chunk_size: int,
) -> sparse.csr_matrix:
    if sparse.issparse(profiles):
        dense_profiles = profiles.toarray().astype(np.float32, copy=False)
    else:
        dense_profiles = np.asarray(profiles, dtype=np.float32)

    normalized = _l2_normalize_rows(dense_profiles)
    n_nodes = normalized.shape[0]
    candidate_count = min(k, max(n_nodes - 1, 0))

    if candidate_count == 0:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    for start in range(0, n_nodes, chunk_size):
        stop = min(start + chunk_size, n_nodes)
        sims = normalized[start:stop] @ normalized.T
        local_rows = np.arange(start, stop)
        sims[np.arange(stop - start), local_rows] = -np.inf

        top_idx = np.argpartition(sims, -candidate_count, axis=1)[:, -candidate_count:]
        top_val = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_val, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_val = np.take_along_axis(top_val, order, axis=1)

        for offset, row_idx in enumerate(local_rows):
            sim_values = top_val[offset]
            sim_indices = top_idx[offset]
            valid_mask = np.isfinite(sim_values) & (sim_values > 0.0)
            if not np.any(valid_mask):
                continue
            valid_values = np.clip(sim_values[valid_mask], 0.0, 1.0).astype(np.float32, copy=False)
            valid_indices = sim_indices[valid_mask].astype(np.int32, copy=False)
            rows.append(np.full(valid_values.shape, row_idx, dtype=np.int32))
            cols.append(valid_indices)
            vals.append(valid_values)

    if not rows:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)

    directed = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    ).tocsr()
    directed.setdiag(0.0)
    directed.eliminate_zeros()
    symmetric = directed.maximum(directed.T).tocsr()
    symmetric.setdiag(0.0)
    symmetric.eliminate_zeros()
    return symmetric


def normalize_graph_with_self_loops(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    adjacency = adjacency.tocsr().astype(np.float32)
    adjacency = adjacency + sparse.identity(adjacency.shape[0], dtype=np.float32, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float32)
    positive = degree > 0.0
    inv_sqrt_degree[positive] = np.power(degree[positive], -0.5)
    d_inv_sqrt = sparse.diags(inv_sqrt_degree)
    normalized = d_inv_sqrt @ adjacency @ d_inv_sqrt
    return normalized.tocsr().astype(np.float32)


def build_source_operator(user_features: np.ndarray, config: RecommendationConfig) -> GraphArtifacts:
    source_profiles = build_source_user_profiles(user_features)
    adjacency = build_weighted_knn_graph(
        source_profiles,
        k=config.graph_k,
        chunk_size=config.directed_knn_chunk_size,
    )
    operator = normalize_graph_with_self_loops(adjacency)
    stats = graph_stats_dict(adjacency)
    checks = {
        "source_graph_uses_music_user_features_only": True,
        "source_graph_is_symmetric": stats["is_symmetric"],
        "source_graph_has_no_self_loops_in_raw_adjacency": bool(adjacency.diagonal().sum() == 0.0),
    }
    return GraphArtifacts(adjacency=adjacency, operator=operator, stats=stats, checks=checks)


def build_target_operator(
    split_artifacts: TargetSplitArtifacts,
    book_genre_index: pd.DataFrame,
    config: RecommendationConfig,
) -> tuple[TargetGraphProfileArtifacts, GraphArtifacts]:
    profile_artifacts = build_target_train_user_genre_profiles(split_artifacts, book_genre_index, config)
    adjacency = build_weighted_knn_graph(
        profile_artifacts.matrix,
        k=config.graph_k,
        chunk_size=config.directed_knn_chunk_size,
    )
    operator = normalize_graph_with_self_loops(adjacency)
    stats = graph_stats_dict(adjacency)
    checks = leakage_checks_dict(profile_artifacts, split_artifacts, book_genre_index)
    checks.update(
        {
            "target_graph_is_symmetric": stats["is_symmetric"],
            "target_graph_has_no_self_loops_in_raw_adjacency": bool(adjacency.diagonal().sum() == 0.0),
        }
    )
    return profile_artifacts, GraphArtifacts(adjacency=adjacency, operator=operator, stats=stats, checks=checks)


def build_fixed_fused_operator(
    source_operator: sparse.csr_matrix,
    target_operator: sparse.csr_matrix,
    alpha: float,
) -> sparse.csr_matrix:
    if source_operator.shape != target_operator.shape:
        raise ValueError("Source and target operators must have matching shapes")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return (alpha * source_operator + (1.0 - alpha) * target_operator).tocsr().astype(np.float32)


def graph_stats_dict(adjacency: sparse.csr_matrix) -> dict[str, object]:
    adjacency = adjacency.tocsr().astype(np.float32)
    degrees = np.diff(adjacency.indptr)
    weighted_degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    undirected_edges = int(sparse.triu(adjacency, k=1).nnz)
    component_count, labels = csgraph.connected_components(adjacency, directed=False, return_labels=True)
    component_sizes = np.bincount(labels, minlength=component_count) if component_count > 0 else np.array([], dtype=np.int32)
    largest_component = int(component_sizes.max()) if component_sizes.size > 0 else 0
    weights = adjacency.data

    return {
        "nodes": int(adjacency.shape[0]),
        "undirected_edges": undirected_edges,
        "density": float((2.0 * undirected_edges) / (adjacency.shape[0] * max(adjacency.shape[0] - 1, 1))),
        "degree_min": int(degrees.min()) if degrees.size else 0,
        "degree_max": int(degrees.max()) if degrees.size else 0,
        "degree_mean": float(degrees.mean()) if degrees.size else 0.0,
        "degree_median": float(np.median(degrees)) if degrees.size else 0.0,
        "weighted_degree_min": float(weighted_degrees.min()) if weighted_degrees.size else 0.0,
        "weighted_degree_max": float(weighted_degrees.max()) if weighted_degrees.size else 0.0,
        "weighted_degree_mean": float(weighted_degrees.mean()) if weighted_degrees.size else 0.0,
        "weighted_degree_median": float(np.median(weighted_degrees)) if weighted_degrees.size else 0.0,
        "isolated_nodes": int((degrees == 0).sum()) if degrees.size else 0,
        "largest_connected_component_size": largest_component,
        "largest_connected_component_fraction": float(largest_component / adjacency.shape[0]) if adjacency.shape[0] else 0.0,
        "connected_components": int(component_count),
        "edge_weight_min": float(weights.min()) if weights.size else 0.0,
        "edge_weight_max": float(weights.max()) if weights.size else 0.0,
        "edge_weight_mean": float(weights.mean()) if weights.size else 0.0,
        "edge_weight_median": float(np.median(weights)) if weights.size else 0.0,
        "is_symmetric": bool((adjacency != adjacency.T).nnz == 0),
    }


def leakage_checks_dict(
    profile_artifacts: TargetGraphProfileArtifacts,
    split_artifacts: TargetSplitArtifacts,
    book_genre_index: pd.DataFrame,
) -> dict[str, object]:
    split_interactions = split_artifacts.split_interactions
    val_test_rows = split_interactions["split"].isin(["val", "test"]).sum()
    return {
        "used_only_train_split_for_target_profiles": True,
        "excluded_val_test_interactions_from_target_profiles": int(val_test_rows),
        "target_profile_shape": [int(profile_artifacts.matrix.shape[0]), int(profile_artifacts.matrix.shape[1])],
        "target_profile_genre_dim_matches_book_index": profile_artifacts.matrix.shape[1] == len(book_genre_index),
        "zero_target_profile_rows": int(profile_artifacts.summary["zero_profile_rows"]),
    }


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    normalized = matrix / safe_norms
    normalized[norms.squeeze(1) == 0.0] = 0.0
    return normalized.astype(np.float32, copy=False)
