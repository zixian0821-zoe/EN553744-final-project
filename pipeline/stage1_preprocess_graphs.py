from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

from graph_builder import build_fused_graph, build_source_graph, build_target_graph


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "experiment2_stage1"

TOP_MUSIC_GENRES = 30
TOP_BOOK_GENRES = 20
K_NEIGHBORS = 10
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
SEED = 42


def safe_min(values: np.ndarray | list[float] | list[int], default: float = 0.0) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return float(default)
    return float(arr.min())


def safe_max(values: np.ndarray | list[float] | list[int], default: float = 0.0) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return float(default)
    return float(arr.max())


def load_inputs() -> tuple[sparse.csr_matrix, sparse.csr_matrix, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    music_matrix = sparse.load_npz(DATA_DIR / "music_user_genre_matrix.npz").tocsr().astype(np.float32)
    book_matrix = sparse.load_npz(DATA_DIR / "book_user_genre_matrix.npz").tocsr().astype(np.float32)
    aligned_users = pd.read_csv(DATA_DIR / "aligned_users.csv")
    music_user_index = pd.read_csv(DATA_DIR / "music_user_index.csv")
    book_user_index = pd.read_csv(DATA_DIR / "book_user_index.csv")
    music_genre_index = pd.read_csv(DATA_DIR / "music_genre_index.csv")
    book_genre_index = pd.read_csv(DATA_DIR / "book_genre_index.csv")
    return (
        music_matrix,
        book_matrix,
        aligned_users,
        music_user_index,
        book_user_index,
        music_genre_index,
        book_genre_index,
    )


def verify_alignment(
    music_matrix: sparse.csr_matrix,
    book_matrix: sparse.csr_matrix,
    aligned_users: pd.DataFrame,
    music_user_index: pd.DataFrame,
    book_user_index: pd.DataFrame,
) -> dict[str, bool | int]:
    if music_matrix.shape[0] != len(aligned_users):
        raise ValueError("Music matrix row count does not match aligned_users.csv")
    if book_matrix.shape[0] != len(aligned_users):
        raise ValueError("Book matrix row count does not match aligned_users.csv")
    if len(music_user_index) != len(aligned_users):
        raise ValueError("music_user_index.csv row count does not match aligned_users.csv")
    if len(book_user_index) != len(aligned_users):
        raise ValueError("book_user_index.csv row count does not match aligned_users.csv")

    same_user_idx_music = aligned_users["user_idx"].equals(music_user_index["user_idx"])
    same_user_idx_book = aligned_users["user_idx"].equals(book_user_index["user_idx"])
    same_user_id_music = aligned_users["user_id"].equals(music_user_index["user_id"])
    same_user_id_book = aligned_users["user_id"].equals(book_user_index["user_id"])

    if not all([same_user_idx_music, same_user_idx_book, same_user_id_music, same_user_id_book]):
        raise ValueError("User alignment mismatch across aligned_users and index CSVs")

    return {
        "music_rows_match": True,
        "book_rows_match": True,
        "same_user_idx_music": same_user_idx_music,
        "same_user_idx_book": same_user_idx_book,
        "same_user_id_music": same_user_id_music,
        "same_user_id_book": same_user_id_book,
    }


def get_valid_user_mask(aligned_users: pd.DataFrame) -> np.ndarray:
    if "has_book_profile" not in aligned_users.columns:
        raise ValueError("aligned_users.csv must contain has_book_profile")
    valid_mask = aligned_users["has_book_profile"].astype(bool).to_numpy()
    if valid_mask.sum() == 0:
        raise ValueError("No valid supervised users found with has_book_profile == True")
    return valid_mask


def _column_totals(matrix: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)


def select_top_genres(matrix: sparse.csr_matrix, top_k: int) -> np.ndarray:
    totals = _column_totals(matrix)
    ranked = np.lexsort((np.arange(totals.size), -totals))
    return ranked[:top_k]


def l1_normalize_rows(matrix: sparse.csr_matrix) -> np.ndarray:
    dense = matrix.toarray().astype(np.float32, copy=False)
    row_sums = dense.sum(axis=1, keepdims=True)
    nonzero_mask = row_sums.squeeze(1) > 0.0
    dense[nonzero_mask] = dense[nonzero_mask] / row_sums[nonzero_mask]
    dense[~nonzero_mask] = 0.0
    return dense


def build_split(n_users: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(n_users)
    train_end = int(np.floor(TRAIN_RATIO * n_users))
    val_end = train_end + int(np.floor(VAL_RATIO * n_users))

    train_mask = np.zeros(n_users, dtype=bool)
    val_mask = np.zeros(n_users, dtype=bool)
    test_mask = np.zeros(n_users, dtype=bool)

    train_mask[permutation[:train_end]] = True
    val_mask[permutation[train_end:val_end]] = True
    test_mask[permutation[val_end:]] = True

    if not np.all(train_mask | val_mask | test_mask):
        raise ValueError("Split masks do not cover all valid users")
    if np.any(train_mask & val_mask) or np.any(train_mask & test_mask) or np.any(val_mask & test_mask):
        raise ValueError("Split masks overlap")
    return train_mask, val_mask, test_mask, permutation


def save_feature_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savez_compressed(path, matrix=matrix.astype(np.float32, copy=False))


def adjacency_to_edge_frame(adjacency: sparse.csr_matrix) -> pd.DataFrame:
    coo = sparse.triu(adjacency, k=1, format="coo")
    return pd.DataFrame(
        {
            "src": coo.row.astype(np.int32),
            "dst": coo.col.astype(np.int32),
            "weight": coo.data.astype(np.float32),
        }
    )


def graph_stats(adjacency: sparse.csr_matrix) -> dict[str, float | int]:
    adjacency = adjacency.tocsr()
    degrees = np.diff(adjacency.indptr)
    weighted_degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    undirected_edges = int(sparse.triu(adjacency, k=1).nnz)

    component_count, labels = csgraph.connected_components(adjacency, directed=False, return_labels=True)
    component_sizes = np.bincount(labels, minlength=component_count) if component_count > 0 else np.array([], dtype=int)
    largest_component = int(component_sizes.max()) if component_sizes.size > 0 else 0
    weights = adjacency.data

    return {
        "nodes": int(adjacency.shape[0]),
        "undirected_edges": undirected_edges,
        "density": float((2.0 * undirected_edges) / (adjacency.shape[0] * max(adjacency.shape[0] - 1, 1))),
        "degree_min": int(degrees.min()) if degrees.size else 0,
        "degree_max": int(degrees.max()) if degrees.size else 0,
        "degree_mean": float(degrees.mean()),
        "degree_median": float(np.median(degrees)),
        "weighted_degree_min": safe_min(weighted_degrees),
        "weighted_degree_max": safe_max(weighted_degrees),
        "weighted_degree_mean": float(weighted_degrees.mean()),
        "weighted_degree_median": float(np.median(weighted_degrees)),
        "isolated_nodes": int((degrees == 0).sum()),
        "largest_connected_component_size": largest_component,
        "largest_connected_component_fraction": float(largest_component / adjacency.shape[0]),
        "connected_components": int(component_count),
        "edge_weight_min": safe_min(weights),
        "edge_weight_max": safe_max(weights),
        "edge_weight_mean": float(weights.mean()) if weights.size else 0.0,
        "edge_weight_median": float(np.median(weights)) if weights.size else 0.0,
        "is_symmetric": bool((adjacency != adjacency.T).nnz == 0),
    }


def genre_selection_frame(
    selected_indices: np.ndarray,
    genre_index: pd.DataFrame,
    totals: np.ndarray,
) -> pd.DataFrame:
    selection = genre_index.iloc[selected_indices].copy()
    selection["selected_rank"] = np.arange(1, len(selected_indices) + 1)
    selection["matrix_column_total"] = totals[selected_indices]
    return selection[["selected_rank", "genre_idx", "genre_label", "interaction_count", "matrix_column_total"]]


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (
        music_matrix,
        book_matrix,
        aligned_users,
        music_user_index,
        book_user_index,
        music_genre_index,
        book_genre_index,
    ) = load_inputs()

    alignment_checks = verify_alignment(
        music_matrix=music_matrix,
        book_matrix=book_matrix,
        aligned_users=aligned_users,
        music_user_index=music_user_index,
        book_user_index=book_user_index,
    )

    valid_mask_full = get_valid_user_mask(aligned_users)
    valid_indices = np.flatnonzero(valid_mask_full)
    valid_users = aligned_users.loc[valid_indices].reset_index(drop=True).copy()
    valid_users["valid_user_position"] = np.arange(len(valid_users), dtype=np.int32)

    if "has_music_profile" in valid_users.columns:
        valid_users["music_profile_consistent"] = valid_users["has_music_profile"].astype(bool)

    music_valid = music_matrix[valid_indices]
    book_valid = book_matrix[valid_indices]

    music_totals = _column_totals(music_valid)
    book_totals = _column_totals(book_valid)
    top_music_indices = select_top_genres(music_valid, TOP_MUSIC_GENRES)
    top_book_indices = select_top_genres(book_valid, TOP_BOOK_GENRES)

    x_music_sparse = music_valid[:, top_music_indices].tocsr()
    y_book_sparse = book_valid[:, top_book_indices].tocsr()
    x_music = l1_normalize_rows(x_music_sparse)
    y_book = l1_normalize_rows(y_book_sparse)

    train_mask, val_mask, test_mask, permutation = build_split(len(valid_users))

    source_graph, source_meta = build_source_graph(x_music, k=K_NEIGHBORS)
    target_graph, target_graph_profiles, target_meta = build_target_graph(
        y_book,
        k=K_NEIGHBORS,
        train_mask=train_mask,
    )
    fused_graph = build_fused_graph(source_graph, target_graph, alpha=0.5)

    np.savez_compressed(
        RESULTS_DIR / "split_masks.npz",
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        valid_indices=valid_indices,
        permutation=permutation,
    )
    save_feature_matrix(RESULTS_DIR / "x_music_top30_l1.npz", x_music)
    save_feature_matrix(RESULTS_DIR / "y_book_top20_l1.npz", y_book)
    save_feature_matrix(RESULTS_DIR / "y_book_target_graph_profiles.npz", target_graph_profiles)
    sparse.save_npz(RESULTS_DIR / "source_graph_adjacency.npz", source_graph)
    sparse.save_npz(RESULTS_DIR / "target_graph_adjacency.npz", target_graph)
    sparse.save_npz(RESULTS_DIR / "fused_graph_adjacency.npz", fused_graph)

    adjacency_to_edge_frame(source_graph).to_csv(RESULTS_DIR / "source_graph_edges.csv", index=False)
    adjacency_to_edge_frame(target_graph).to_csv(RESULTS_DIR / "target_graph_edges.csv", index=False)
    adjacency_to_edge_frame(fused_graph).to_csv(RESULTS_DIR / "fused_graph_edges.csv", index=False)

    valid_users["split"] = np.where(train_mask, "train", np.where(val_mask, "val", "test"))
    valid_users.to_csv(RESULTS_DIR / "valid_users.csv", index=False)

    genre_selection_frame(top_music_indices, music_genre_index, music_totals).to_csv(
        RESULTS_DIR / "selected_music_genres_top30.csv",
        index=False,
    )
    genre_selection_frame(top_book_indices, book_genre_index, book_totals).to_csv(
        RESULTS_DIR / "selected_book_genres_top20.csv",
        index=False,
    )

    preprocessing_diagnostics = {
        "seed": SEED,
        "top_music_genres": TOP_MUSIC_GENRES,
        "top_book_genres": TOP_BOOK_GENRES,
        "k_neighbors": K_NEIGHBORS,
        "alignment_checks": alignment_checks,
        "input_shapes": {
            "music_matrix": list(music_matrix.shape),
            "book_matrix": list(book_matrix.shape),
        },
        "valid_user_definition": "has_book_profile == True",
        "valid_user_count": int(len(valid_users)),
        "valid_user_original_index_min": int(valid_indices.min()) if valid_indices.size else 0,
        "valid_user_original_index_max": int(valid_indices.max()) if valid_indices.size else 0,
        "music_consistency_check_enabled": bool("has_music_profile" in aligned_users.columns),
        "music_consistency_all_valid_true": bool(
            valid_users["music_profile_consistent"].all()
        ) if "music_profile_consistent" in valid_users.columns else None,
        "final_feature_shape": list(x_music.shape),
        "final_target_shape": list(y_book.shape),
        "x_music_zero_rows_after_top30": int((x_music.sum(axis=1) == 0.0).sum()),
        "y_book_zero_rows_after_top20": int((y_book.sum(axis=1) == 0.0).sum()),
        "x_music_row_sum_min": safe_min(x_music.sum(axis=1)),
        "x_music_row_sum_max": safe_max(x_music.sum(axis=1)),
        "y_book_row_sum_min": safe_min(y_book.sum(axis=1)),
        "y_book_row_sum_max": safe_max(y_book.sum(axis=1)),
        "train_count": int(train_mask.sum()),
        "val_count": int(val_mask.sum()),
        "test_count": int(test_mask.sum()),
        "target_graph_zeroed_rows": int((~train_mask).sum()),
        "target_graph_train_rows_preserved": int(train_mask.sum()),
        "target_graph_val_test_rows_all_zero": bool(np.all(target_graph_profiles[~train_mask] == 0.0)),
        "target_graph_train_row_sum_min": safe_min(target_graph_profiles[train_mask].sum(axis=1)),
        "target_graph_train_row_sum_max": safe_max(target_graph_profiles[train_mask].sum(axis=1)),
    }
    save_json(RESULTS_DIR / "preprocessing_diagnostics.json", preprocessing_diagnostics)

    target_edges = adjacency_to_edge_frame(target_graph)
    target_edge_train_only = (
        target_edges.empty
        or (
            train_mask[target_edges["src"].to_numpy(dtype=np.int64)]
            & train_mask[target_edges["dst"].to_numpy(dtype=np.int64)]
        ).all()
    )

    graph_diagnostics = {
        "source_graph": {
            **source_meta,
            **graph_stats(source_graph),
        },
        "target_graph": {
            **target_meta,
            **graph_stats(target_graph),
            "all_edges_train_train_only": bool(target_edge_train_only),
        },
        "fused_graph": graph_stats(fused_graph),
    }
    save_json(RESULTS_DIR / "graph_statistics.json", graph_diagnostics)


if __name__ == "__main__":
    main()
