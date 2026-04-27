from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = ["user_id", "parent_asin", "rating", "timestamp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize source_interactions.csv and target_interactions.csv "
            "from local Amazon Reviews parquet files using the preprocessing "
            "logic from Amazon_data_preprocess.ipynb."
        )
    )
    parser.add_argument(
        "--source-parquet",
        required=True,
        help="Local parquet file for CDs/Vinyl interactions",
    )
    parser.add_argument(
        "--target-parquet",
        nargs="+",
        required=True,
        help="One or more local parquet files for Books interactions",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/zhouzixian/Documents/New project/data",
        help="Directory where source_interactions.csv and target_interactions.csv will be written",
    )
    parser.add_argument(
        "--user-threshold-both",
        type=int,
        default=10,
        help="Keep users with at least this many interactions in both domains",
    )
    parser.add_argument(
        "--min-target-item-freq",
        type=int,
        default=2,
        help="One-pass minimum target item frequency before split-capable user filtering",
    )
    parser.add_argument(
        "--min-target-inter-per-user-for-split",
        type=int,
        default=3,
        help="Minimum target interactions required to support the notebook's chronological split",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Target-domain chronological train ratio used to derive the train-seen item set",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Target-domain chronological validation ratio used to derive the train-seen item set",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, path_label: str) -> None:
    missing = [column for column in REQUIRED_COLS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path_label} is missing required columns: {missing}")


def load_source_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=REQUIRED_COLS)
    require_columns(frame, str(path))
    return frame.copy()


def load_target_frame(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, columns=REQUIRED_COLS)
        require_columns(frame, str(path))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def basic_clean(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame[REQUIRED_COLS].dropna(subset=["user_id", "parent_asin", "timestamp"]).copy()
    cleaned["timestamp"] = pd.to_numeric(cleaned["timestamp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp"]).copy()
    return cleaned


def deduplicate_latest(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values(["user_id", "parent_asin", "timestamp"], kind="mergesort")
        .drop_duplicates(subset=["user_id", "parent_asin"], keep="last")
        .reset_index(drop=True)
    )


def keep_overlap_users(source_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap_users = set(source_df["user_id"].unique()) & set(target_df["user_id"].unique())
    source_df = source_df[source_df["user_id"].isin(overlap_users)].copy()
    target_df = target_df[target_df["user_id"].isin(overlap_users)].copy()
    return source_df, target_df


def apply_user_thresholds(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    user_threshold_both: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_user_counts = source_df["user_id"].value_counts()
    target_user_counts = target_df["user_id"].value_counts()
    qualified_users = (
        set(source_user_counts[source_user_counts >= user_threshold_both].index)
        & set(target_user_counts[target_user_counts >= user_threshold_both].index)
    )
    source_df = source_df[source_df["user_id"].isin(qualified_users)].copy()
    target_df = target_df[target_df["user_id"].isin(qualified_users)].copy()
    return source_df, target_df


def apply_target_item_filter(
    target_df: pd.DataFrame,
    min_target_item_freq: int,
) -> pd.DataFrame:
    target_item_counts = target_df["parent_asin"].value_counts()
    keep_target_items = set(target_item_counts[target_item_counts >= min_target_item_freq].index)
    return target_df[target_df["parent_asin"].isin(keep_target_items)].copy()


def keep_split_capable_users(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    min_target_inter_per_user_for_split: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    target_user_counts = target_df["user_id"].value_counts()
    split_capable_users = set(
        target_user_counts[target_user_counts >= min_target_inter_per_user_for_split].index
    )
    final_users = sorted(set(source_df["user_id"].unique()) & split_capable_users)
    source_df = source_df[source_df["user_id"].isin(final_users)].copy()
    target_df = target_df[target_df["user_id"].isin(final_users)].copy()
    return source_df, target_df, final_users


def compute_train_target_items(
    target_df: pd.DataFrame,
    final_users: list[str],
    train_ratio: float,
    val_ratio: float,
) -> set[str]:
    user2id = {user_id: idx for idx, user_id in enumerate(final_users)}
    target_df = target_df.copy()
    target_df["user_idx"] = target_df["user_id"].map(user2id)
    target_df = target_df.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

    train_parts: list[pd.DataFrame] = []
    for _, group in target_df.groupby("user_idx", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n_rows = len(group)
        if n_rows < 3:
            train_parts.append(group.copy())
            continue

        n_train = int(np.floor(n_rows * train_ratio))
        n_val = int(np.floor(n_rows * val_ratio))
        n_test = n_rows - n_train - n_val

        if n_val == 0:
            n_val = 1
            n_train -= 1
        if n_test == 0:
            n_test = 1
            n_train -= 1
        if n_train <= 0:
            n_train = max(1, n_rows - 2)
            n_val = 1
            n_test = n_rows - n_train - n_val

        train_parts.append(group.iloc[:n_train].copy())

    train_df = pd.concat(train_parts, ignore_index=True)
    return set(train_df["parent_asin"].unique())


def attach_indices_and_trim_target(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    final_users: list[str],
    train_target_items: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user2id = {user_id: idx for idx, user_id in enumerate(final_users)}
    source_items = sorted(source_df["parent_asin"].unique())
    target_items = sorted(train_target_items)
    source_item2id = {item_id: idx for idx, item_id in enumerate(source_items)}
    target_item2id = {item_id: idx for idx, item_id in enumerate(target_items)}

    source_df = source_df.copy()
    target_df = target_df[target_df["parent_asin"].isin(train_target_items)].copy()

    source_df["user_idx"] = source_df["user_id"].map(user2id)
    target_df["user_idx"] = target_df["user_id"].map(user2id)
    source_df["item_idx"] = source_df["parent_asin"].map(source_item2id)
    target_df["item_idx"] = target_df["parent_asin"].map(target_item2id)

    source_out = source_df[["user_id", "parent_asin", "rating", "timestamp", "user_idx", "item_idx"]].copy()
    target_out = target_df[["user_id", "parent_asin", "rating", "timestamp", "user_idx", "item_idx"]].copy()
    return source_out, target_out


def materialize_interactions(
    source_parquet: Path,
    target_parquets: list[Path],
    output_dir: Path,
    user_threshold_both: int,
    min_target_item_freq: int,
    min_target_inter_per_user_for_split: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_df = load_source_frame(source_parquet)
    target_df = load_target_frame(target_parquets)

    source_df = basic_clean(source_df)
    target_df = basic_clean(target_df)
    source_df = deduplicate_latest(source_df)
    target_df = deduplicate_latest(target_df)
    source_df, target_df = keep_overlap_users(source_df, target_df)
    source_df, target_df = apply_user_thresholds(source_df, target_df, user_threshold_both)

    if source_df.empty or target_df.empty:
        raise ValueError("Data became empty after the overlap or user-threshold filtering steps.")

    target_df = apply_target_item_filter(target_df, min_target_item_freq)
    source_df, target_df, final_users = keep_split_capable_users(
        source_df,
        target_df,
        min_target_inter_per_user_for_split,
    )

    if source_df.empty or target_df.empty or not final_users:
        raise ValueError(
            "Data became empty after the target-item or split-capable-user filtering steps. "
            "Try lowering --min-target-item-freq to 1."
        )

    train_target_items = compute_train_target_items(target_df, final_users, train_ratio, val_ratio)
    source_out, target_out = attach_indices_and_trim_target(
        source_df,
        target_df,
        final_users,
        train_target_items,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_out.to_csv(output_dir / "source_interactions.csv", index=False)
    target_out.to_csv(output_dir / "target_interactions.csv", index=False)
    return source_out, target_out


def main() -> None:
    args = parse_args()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    target_parquets = [Path(path).expanduser().resolve() for path in args.target_parquet]
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not source_parquet.exists():
        raise FileNotFoundError(f"Source parquet not found: {source_parquet}")
    missing_targets = [str(path) for path in target_parquets if not path.exists()]
    if missing_targets:
        raise FileNotFoundError(f"Target parquet files not found: {missing_targets}")

    source_out, target_out = materialize_interactions(
        source_parquet=source_parquet,
        target_parquets=target_parquets,
        output_dir=output_dir,
        user_threshold_both=args.user_threshold_both,
        min_target_item_freq=args.min_target_item_freq,
        min_target_inter_per_user_for_split=args.min_target_inter_per_user_for_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print(f"Wrote {len(source_out)} rows to {output_dir / 'source_interactions.csv'}")
    print(f"Wrote {len(target_out)} rows to {output_dir / 'target_interactions.csv'}")


if __name__ == "__main__":
    main()
