from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from recommendation_config import RecommendationConfig
from recommendation_data import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, TargetSplitArtifacts


def build_target_user_holdout_splits(
    target_interactions: pd.DataFrame,
    config: RecommendationConfig,
    *,
    seed: int | None = None,
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

    n_users = int(target_interactions["user_idx"].nunique())
    n_items = int(target_interactions["item_idx"].max()) + 1
    sorted_target = target_interactions.sort_values(["user_idx", "timestamp", "item_idx"], kind="mergesort")

    interaction_count_by_user = (
        sorted_target.groupby("user_idx", sort=True)
        .size()
        .reindex(np.arange(n_users, dtype=np.int32), fill_value=0)
        .to_numpy(dtype=np.int32)
    )
    eligible_eval_users = np.flatnonzero(interaction_count_by_user >= config.min_eval_target_interactions).astype(np.int32)
    low_activity_users = np.flatnonzero(interaction_count_by_user < config.min_eval_target_interactions).astype(np.int32)

    if eligible_eval_users.size == 0:
        raise ValueError("No users satisfy the minimum interaction threshold for strict cold-start evaluation")

    split_seed = config.seed if seed is None else seed
    rng = np.random.default_rng(split_seed)
    shuffled = rng.permutation(eligible_eval_users)

    n_train_eval = int(np.floor(config.train_fraction * shuffled.size))
    n_val = int(np.floor(config.val_fraction * shuffled.size))
    n_test = int(shuffled.size - n_train_eval - n_val)
    if min(n_train_eval, n_val, n_test) <= 0:
        raise ValueError(
            "Strict cold-start split produced an empty train/val/test user partition; "
            "increase the number of eligible evaluation users or adjust fractions."
        )

    train_eval_users = shuffled[:n_train_eval]
    val_users = shuffled[n_train_eval : n_train_eval + n_val]
    test_users = shuffled[n_train_eval + n_val :]

    train_user_mask = np.zeros(n_users, dtype=bool)
    val_user_mask = np.zeros(n_users, dtype=bool)
    test_user_mask = np.zeros(n_users, dtype=bool)

    train_user_mask[low_activity_users] = True
    train_user_mask[train_eval_users] = True
    val_user_mask[val_users] = True
    test_user_mask[test_users] = True

    if not np.all(train_user_mask | val_user_mask | test_user_mask):
        raise ValueError("Strict cold-start user partition does not cover all users")
    if np.any(train_user_mask & val_user_mask) or np.any(train_user_mask & test_user_mask) or np.any(val_user_mask & test_user_mask):
        raise ValueError("Strict cold-start user partition overlaps")

    eval_user_mask = val_user_mask | test_user_mask

    split_frames: list[pd.DataFrame] = []
    for user_idx, user_df in sorted_target.groupby("user_idx", sort=True):
        user_frame = user_df.copy()
        if train_user_mask[user_idx]:
            split_label = SPLIT_TRAIN
        elif val_user_mask[user_idx]:
            split_label = SPLIT_VAL
        elif test_user_mask[user_idx]:
            split_label = SPLIT_TEST
        else:
            raise ValueError(f"User {user_idx} was not assigned to any strict cold-start split")

        user_frame["split"] = split_label
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
    val_metric_user_mask = val_user_mask & (val_warm_counts > 0)
    test_metric_user_mask = test_user_mask & (test_warm_counts > 0)

    checks = {
        "split_protocol": "strict_user_holdout_coldstart",
        "split_rows_match_target_rows": int(len(split_interactions)) == int(len(target_interactions)),
        "train_val_test_partition_complete": True,
        "split_labels_valid": sorted(split_interactions["split"].unique().tolist()) == [SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL],
        "eligible_eval_users_meet_threshold": bool(np.all(interaction_count_by_user[eligible_eval_users] >= config.min_eval_target_interactions)),
        "low_activity_users_assigned_to_train": bool(np.all(train_user_mask[interaction_count_by_user < config.min_eval_target_interactions])),
        "val_users_have_no_train_rows": bool(np.all(np.asarray(train_matrix[val_user_mask].sum(axis=1)).ravel() == 0)),
        "test_users_have_no_train_rows": bool(np.all(np.asarray(train_matrix[test_user_mask].sum(axis=1)).ravel() == 0)),
        "candidate_items_nonempty": candidate_item_universe.size > 0,
        "train_matrix_user_item_duplicates_removed": True,
    }

    summary: dict[str, object] = {
        "split_protocol": "strict_user_holdout_coldstart",
        "target_interactions_total": int(len(split_interactions)),
        "target_users_total": int(n_users),
        "target_items_total": int(n_items),
        "eval_user_threshold": int(config.min_eval_target_interactions),
        "eligible_eval_user_pool_count": int(eligible_eval_users.size),
        "train_user_count": int(train_user_mask.sum()),
        "train_eval_user_count": int(train_eval_users.size),
        "val_user_count": int(val_user_mask.sum()),
        "test_user_count": int(test_user_mask.sum()),
        "low_activity_train_only_user_count": int(low_activity_users.size),
        "eval_user_count": int(eval_user_mask.sum()),
        "train_only_user_count": int(train_user_mask.sum()),
        "train_interactions": int(split_counts.get(SPLIT_TRAIN, 0)),
        "val_interactions": int(split_counts.get(SPLIT_VAL, 0)),
        "test_interactions": int(split_counts.get(SPLIT_TEST, 0)),
        "candidate_item_universe_size": int(candidate_item_universe.size),
        "val_warm_interactions_total": int(val_warm_counts[val_user_mask].sum()),
        "test_warm_interactions_total": int(test_warm_counts[test_user_mask].sum()),
        "val_metric_user_count": int(val_metric_user_mask.sum()),
        "test_metric_user_count": int(test_metric_user_mask.sum()),
        "users_with_at_least_5_target_interactions": int(eligible_eval_users.size),
        "users_with_zero_warm_val_items": int((val_user_mask & ~val_metric_user_mask).sum()),
        "users_with_zero_warm_test_items": int((test_user_mask & ~test_metric_user_mask).sum()),
    }

    user_masks = {
        "eval_user_mask": eval_user_mask,
        "train_only_user_mask": train_user_mask,
        "train_user_mask": train_user_mask,
        "val_user_mask": val_user_mask,
        "test_user_mask": test_user_mask,
        "strict_coldstart_eval_user_mask": eval_user_mask,
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
