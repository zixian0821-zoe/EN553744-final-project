from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from recommendation_config import RecommendationConfig


SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"


@dataclass(frozen=True)
class RecommendationInputs:
    source_interactions: pd.DataFrame
    target_interactions: pd.DataFrame
    aligned_users: pd.DataFrame
    music_user_index: pd.DataFrame
    book_user_index: pd.DataFrame


@dataclass(frozen=True)
class TargetSplitArtifacts:
    split_interactions: pd.DataFrame
    train_matrix: sparse.csr_matrix
    val_matrix: sparse.csr_matrix
    test_matrix: sparse.csr_matrix
    candidate_item_universe: np.ndarray
    user_masks: dict[str, np.ndarray]
    summary: dict[str, object]
    checks: dict[str, object]
    train_item_popularity: np.ndarray


def load_recommendation_inputs(config: RecommendationConfig) -> RecommendationInputs:
    return RecommendationInputs(
        source_interactions=pd.read_csv(config.source_interactions_path),
        target_interactions=pd.read_csv(config.target_interactions_path),
        aligned_users=pd.read_csv(config.aligned_users_path),
        music_user_index=pd.read_csv(config.music_user_index_path),
        book_user_index=pd.read_csv(config.book_user_index_path),
    )


def verify_user_alignment(inputs: RecommendationInputs) -> dict[str, object]:
    aligned = inputs.aligned_users.sort_values("user_idx").reset_index(drop=True)
    music_user_index = inputs.music_user_index.sort_values("user_idx").reset_index(drop=True)
    book_user_index = inputs.book_user_index.sort_values("user_idx").reset_index(drop=True)

    if not aligned["user_idx"].equals(music_user_index["user_idx"]):
        raise ValueError("aligned_users and music_user_index have mismatched user_idx values")
    if not aligned["user_idx"].equals(book_user_index["user_idx"]):
        raise ValueError("aligned_users and book_user_index have mismatched user_idx values")
    if not aligned["user_id"].equals(music_user_index["user_id"]):
        raise ValueError("aligned_users and music_user_index have mismatched user_id values")
    if not aligned["user_id"].equals(book_user_index["user_id"]):
        raise ValueError("aligned_users and book_user_index have mismatched user_id values")

    checks: dict[str, object] = {
        "aligned_user_rows": int(len(aligned)),
        "music_user_index_rows": int(len(music_user_index)),
        "book_user_index_rows": int(len(book_user_index)),
        "aligned_matches_music_index": True,
        "aligned_matches_book_index": True,
    }

    aligned_user_ids = set(aligned["user_id"])
    aligned_user_idx = set(aligned["user_idx"])

    for domain_name, interactions in (
        ("source", inputs.source_interactions),
        ("target", inputs.target_interactions),
    ):
        distinct_users = interactions[["user_idx", "user_id"]].drop_duplicates().sort_values("user_idx").reset_index(drop=True)
        if not distinct_users["user_idx"].equals(aligned["user_idx"]):
            raise ValueError(f"{domain_name} interactions do not match aligned user_idx ordering")
        if not distinct_users["user_id"].equals(aligned["user_id"]):
            raise ValueError(f"{domain_name} interactions do not match aligned user_id ordering")
        if set(interactions["user_id"]) != aligned_user_ids:
            raise ValueError(f"{domain_name} interactions do not cover the aligned user_id set exactly")
        if set(interactions["user_idx"]) != aligned_user_idx:
            raise ValueError(f"{domain_name} interactions do not cover the aligned user_idx set exactly")

        duplicate_user_item = int(interactions.duplicated(["user_idx", "item_idx"]).sum())
        if duplicate_user_item != 0:
            raise ValueError(f"{domain_name} interactions contain duplicate user-item pairs")

        item_pairs = interactions[["item_id", "item_idx"]].drop_duplicates()
        item_id_counts = item_pairs["item_id"].value_counts()
        item_idx_counts = item_pairs["item_idx"].value_counts()
        if int(item_id_counts.max()) != 1 or int(item_idx_counts.max()) != 1:
            raise ValueError(f"{domain_name} item_id and item_idx are not one-to-one")

        checks[f"{domain_name}_users_match_aligned"] = True
        checks[f"{domain_name}_duplicate_user_item_rows"] = duplicate_user_item
        checks[f"{domain_name}_item_id_idx_one_to_one"] = True
        checks[f"{domain_name}_item_idx_contiguous_from_zero"] = bool(
            interactions["item_idx"].nunique() == int(interactions["item_idx"].max()) + 1
            and int(interactions["item_idx"].min()) == 0
        )

    return checks


def build_target_interaction_splits(
    target_interactions: pd.DataFrame,
    config: RecommendationConfig,
) -> TargetSplitArtifacts:
    required_columns = {
        "user_idx",
        "user_id",
        "item_idx",
        "item_id",
        "timestamp",
        "has_genre_labels",
    }
    missing = required_columns.difference(target_interactions.columns)
    if missing:
        raise ValueError(f"Target interactions missing required columns: {sorted(missing)}")

    split_frames: list[pd.DataFrame] = []
    n_users = int(target_interactions["user_idx"].nunique())
    n_items = int(target_interactions["item_idx"].max()) + 1

    eval_user_mask = np.zeros(n_users, dtype=bool)
    interaction_count_by_user = np.zeros(n_users, dtype=np.int32)

    sorted_target = target_interactions.sort_values(["user_idx", "timestamp", "item_idx"], kind="mergesort")

    for user_idx, user_df in sorted_target.groupby("user_idx", sort=True):
        user_frame = user_df.copy()
        interaction_count = len(user_frame)
        interaction_count_by_user[user_idx] = interaction_count

        if interaction_count < config.min_eval_target_interactions:
            split_labels = np.full(interaction_count, SPLIT_TRAIN, dtype=object)
        else:
            eval_user_mask[user_idx] = True
            n_test = max(1, int(np.floor(config.test_fraction * interaction_count)))
            n_val = max(1, int(np.floor(config.val_fraction * interaction_count)))
            n_train = interaction_count - n_val - n_test
            if n_train <= 0:
                raise ValueError(f"User {user_idx} has non-positive train interactions after split")
            split_labels = np.empty(interaction_count, dtype=object)
            split_labels[:n_train] = SPLIT_TRAIN
            split_labels[n_train : n_train + n_val] = SPLIT_VAL
            split_labels[n_train + n_val :] = SPLIT_TEST

        user_frame["split"] = split_labels
        user_frame["is_eval_user"] = bool(eval_user_mask[user_idx])
        split_frames.append(user_frame)

    split_interactions = pd.concat(split_frames, ignore_index=True)
    split_interactions = split_interactions.sort_values(["user_idx", "timestamp", "item_idx"], kind="mergesort").reset_index(drop=True)

    split_counts = split_interactions["split"].value_counts().to_dict()
    if int(sum(split_counts.values())) != len(split_interactions):
        raise ValueError("Split counts do not sum to the total number of target interactions")

    train_matrix = _user_item_matrix(split_interactions, SPLIT_TRAIN, n_users=n_users, n_items=n_items)
    val_matrix = _user_item_matrix(split_interactions, SPLIT_VAL, n_users=n_users, n_items=n_items)
    test_matrix = _user_item_matrix(split_interactions, SPLIT_TEST, n_users=n_users, n_items=n_items)

    train_item_popularity = np.asarray(train_matrix.sum(axis=0)).ravel().astype(np.int32)
    candidate_item_universe = np.flatnonzero(train_item_popularity > 0).astype(np.int32)
    candidate_mask = np.zeros(n_items, dtype=bool)
    candidate_mask[candidate_item_universe] = True

    val_warm_counts = np.asarray(val_matrix[:, candidate_item_universe].sum(axis=1)).ravel().astype(np.int32)
    test_warm_counts = np.asarray(test_matrix[:, candidate_item_universe].sum(axis=1)).ravel().astype(np.int32)
    val_metric_user_mask = val_warm_counts > 0
    test_metric_user_mask = test_warm_counts > 0

    checks = {
        "split_rows_match_target_rows": int(len(split_interactions)) == int(len(target_interactions)),
        "train_val_test_partition_complete": True,
        "split_labels_valid": sorted(split_interactions["split"].unique().tolist()) == [SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL],
        "eval_users_have_min_target_interactions": bool(np.all(interaction_count_by_user[eval_user_mask] >= config.min_eval_target_interactions)),
        "train_only_users_below_threshold": bool(np.all(~eval_user_mask[interaction_count_by_user < config.min_eval_target_interactions])),
        "candidate_items_nonempty": candidate_item_universe.size > 0,
        "train_matrix_user_item_duplicates_removed": True,
    }

    summary: dict[str, object] = {
        "target_interactions_total": int(len(split_interactions)),
        "target_users_total": int(n_users),
        "target_items_total": int(n_items),
        "eval_user_threshold": int(config.min_eval_target_interactions),
        "eval_user_count": int(eval_user_mask.sum()),
        "train_only_user_count": int((~eval_user_mask).sum()),
        "train_interactions": int(split_counts.get(SPLIT_TRAIN, 0)),
        "val_interactions": int(split_counts.get(SPLIT_VAL, 0)),
        "test_interactions": int(split_counts.get(SPLIT_TEST, 0)),
        "candidate_item_universe_size": int(candidate_item_universe.size),
        "val_warm_interactions_total": int(val_warm_counts.sum()),
        "test_warm_interactions_total": int(test_warm_counts.sum()),
        "val_metric_user_count": int(val_metric_user_mask.sum()),
        "test_metric_user_count": int(test_metric_user_mask.sum()),
        "users_with_at_least_5_target_interactions": int(eval_user_mask.sum()),
        "users_with_zero_warm_val_items": int((eval_user_mask & ~val_metric_user_mask).sum()),
        "users_with_zero_warm_test_items": int((eval_user_mask & ~test_metric_user_mask).sum()),
    }

    user_masks = {
        "eval_user_mask": eval_user_mask,
        "train_only_user_mask": ~eval_user_mask,
        "val_metric_user_mask": val_metric_user_mask,
        "test_metric_user_mask": test_metric_user_mask,
        "candidate_item_mask": candidate_mask,
    }

    return TargetSplitArtifacts(
        split_interactions=split_interactions,
        train_matrix=train_matrix,
        val_matrix=val_matrix,
        test_matrix=test_matrix,
        candidate_item_universe=candidate_item_universe,
        user_masks=user_masks,
        summary=summary,
        checks=checks,
        train_item_popularity=train_item_popularity,
    )


def build_eval_user_masks(split_artifacts: TargetSplitArtifacts) -> dict[str, np.ndarray]:
    return dict(split_artifacts.user_masks)


def build_positive_sets_by_user(split_artifacts: TargetSplitArtifacts) -> dict[str, sparse.csr_matrix]:
    return {
        SPLIT_TRAIN: split_artifacts.train_matrix,
        SPLIT_VAL: split_artifacts.val_matrix,
        SPLIT_TEST: split_artifacts.test_matrix,
    }


def build_candidate_item_universe(split_artifacts: TargetSplitArtifacts) -> np.ndarray:
    return split_artifacts.candidate_item_universe.copy()


def split_summary_dict(split_artifacts: TargetSplitArtifacts) -> dict[str, object]:
    return dict(split_artifacts.summary)


def data_summary_dict(inputs: RecommendationInputs, alignment_checks: dict[str, object]) -> dict[str, object]:
    source = inputs.source_interactions
    target = inputs.target_interactions
    payload: dict[str, object] = {
        "alignment_checks": alignment_checks,
        "source_rows": int(len(source)),
        "source_users": int(source["user_idx"].nunique()),
        "source_items": int(source["item_idx"].nunique()),
        "target_rows": int(len(target)),
        "target_users": int(target["user_idx"].nunique()),
        "target_items": int(target["item_idx"].nunique()),
        "source_rating_values": sorted(source["rating"].unique().tolist()),
        "target_rating_values": sorted(target["rating"].unique().tolist()),
        "source_timestamp_min": int(source["timestamp"].min()),
        "source_timestamp_max": int(source["timestamp"].max()),
        "target_timestamp_min": int(target["timestamp"].min()),
        "target_timestamp_max": int(target["timestamp"].max()),
        "source_has_genre_labels_rate": float(source["has_genre_labels"].mean()),
        "target_has_genre_labels_rate": float(target["has_genre_labels"].mean()),
    }
    return payload


def _user_item_matrix(
    interactions: pd.DataFrame,
    split_name: str,
    n_users: int,
    n_items: int,
) -> sparse.csr_matrix:
    split_frame = interactions.loc[interactions["split"] == split_name, ["user_idx", "item_idx"]]
    if split_frame.empty:
        return sparse.csr_matrix((n_users, n_items), dtype=np.float32)
    data = np.ones(len(split_frame), dtype=np.float32)
    matrix = sparse.coo_matrix(
        (data, (split_frame["user_idx"].to_numpy(), split_frame["item_idx"].to_numpy())),
        shape=(n_users, n_items),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix
