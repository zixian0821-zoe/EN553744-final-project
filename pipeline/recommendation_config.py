from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "recommendation_learned_alpha"


@dataclass(frozen=True)
class RecommendationConfig:
    root: Path = ROOT
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR

    source_interactions_path: Path = ROOT / "source_interactions_labeled.csv"
    target_interactions_path: Path = ROOT / "target_interactions_labeled.csv"
    aligned_users_path: Path = DATA_DIR / "aligned_users.csv"
    music_user_index_path: Path = DATA_DIR / "music_user_index.csv"
    book_user_index_path: Path = DATA_DIR / "book_user_index.csv"
    music_user_genre_matrix_path: Path = DATA_DIR / "music_user_genre_matrix.npz"
    book_genre_index_path: Path = DATA_DIR / "book_genre_index.csv"
    selected_music_genres_path: Path = ROOT / "results" / "experiment2_stage1" / "selected_music_genres_top30.csv"

    top_music_genres: int = 30
    book_genre_dim: int = 571
    graph_k: int = 10

    min_eval_target_interactions: int = 5
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    test_fraction: float = 0.2

    hidden_dim: int = 64
    dropout: float = 0.3
    max_epochs: int = 200
    patience: int = 20
    seed: int = 42

    negative_sampling_power: float = 0.75
    eval_ks: tuple[int, ...] = (10, 20)
    fixed_alpha_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

    json_indent: int = 2
    directed_knn_chunk_size: int = 1024

    artifact_names: dict[str, str] = field(
        default_factory=lambda: {
            "config": "config.json",
            "data_summary": "data_summary.json",
            "split_summary": "split_summary.json",
            "item_feature_summary": "item_feature_summary.json",
            "graph_stats_source": "graph_stats_source.json",
            "graph_stats_target": "graph_stats_target.json",
            "validation_checks": "validation_checks.json",
            "target_graph_leakage_checks": "target_graph_leakage_checks.json",
            "user_features": "x_user_music_top30_l1.npz",
            "target_item_genre_features": "target_item_genre_multi_hot.npz",
            "target_item_index": "target_item_index.csv",
            "target_split_assignments": "target_interaction_splits.csv",
            "user_masks": "user_masks.npz",
            "train_user_item": "target_train_user_item_csr.npz",
            "val_user_item": "target_val_user_item_csr.npz",
            "test_user_item": "target_test_user_item_csr.npz",
            "candidate_items": "candidate_item_universe.npy",
            "source_graph_adjacency": "source_graph_adjacency.npz",
            "target_graph_adjacency": "target_graph_adjacency.npz",
            "source_operator": "source_operator_norm.npz",
            "target_operator": "target_operator_norm.npz",
            "target_train_user_profiles": "target_train_user_genre_profiles_l1.npz",
        }
    )

    def artifact_path(self, key: str) -> Path:
        return self.results_dir / self.artifact_names[key]

    def to_serializable_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
            elif isinstance(value, dict):
                payload[key] = {
                    nested_key: (str(nested_value) if isinstance(nested_value, Path) else nested_value)
                    for nested_key, nested_value in value.items()
                }
        return payload
