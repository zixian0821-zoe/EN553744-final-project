from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from recommendation_config import RecommendationConfig
from recommendation_evaluate import build_eval_split_cache, evaluate_test_split, evaluate_validation_split
from recommendation_loss_sampling import (
    build_negative_sampling_distribution,
    build_train_positive_cache,
    build_target_train_item_popularity,
    bpr_loss,
    sample_bpr_triplets,
)
from recommendation_models import GraphRecommender, LearnedAlphaGraphRecommender, NoGraphRecommender, PerUserAlphaGraphRecommender, score

@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 64
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    grad_clip_norm: float = 5.0
    seed: int = 42
    eval_batch_size_users: int = 1024
    smoke_eval_user_limit: int | None = None
    smoke_skip_ranking_eval: bool = False
    use_enriched_features: bool = False
    top_book_genres: int = 30
    highrated_sampling: bool = False
    highrated_threshold: float = 4.0

@dataclass(frozen=True)
class Stage1Artifacts:
    user_features: np.ndarray
    item_genre_features: sparse.csr_matrix
    train_matrix: sparse.csr_matrix
    val_matrix: sparse.csr_matrix
    test_matrix: sparse.csr_matrix
    candidate_item_universe: np.ndarray
    user_masks: dict[str, np.ndarray]
    source_operator: sparse.csr_matrix
    target_operator: sparse.csr_matrix

@dataclass(frozen=True)
class ModelSpec:
    run_name: str
    display_name: str
    model_kind: str
    graph: str
    alpha_mode: str
    alpha_value: float | None

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_stage1_artifacts(config: RecommendationConfig) -> Stage1Artifacts:
    root = config.results_dir
    user_features = np.load(root / config.artifact_names["user_features"])["matrix"].astype(np.float32)
    item_genre_features = sparse.load_npz(root / config.artifact_names["target_item_genre_features"]).tocsr().astype(np.float32)
    train_matrix = sparse.load_npz(root / config.artifact_names["train_user_item"]).tocsr().astype(np.float32)
    val_matrix = sparse.load_npz(root / config.artifact_names["val_user_item"]).tocsr().astype(np.float32)
    test_matrix = sparse.load_npz(root / config.artifact_names["test_user_item"]).tocsr().astype(np.float32)
    candidate_item_universe = np.load(root / config.artifact_names["candidate_items"]).astype(np.int32)
    user_masks_npz = np.load(root / config.artifact_names["user_masks"])
    user_masks = {name: user_masks_npz[name].astype(bool) for name in user_masks_npz.files}
    source_operator = sparse.load_npz(root / config.artifact_names["source_operator"]).tocsr().astype(np.float32)
    target_operator = sparse.load_npz(root / config.artifact_names["target_operator"]).tocsr().astype(np.float32)
    return Stage1Artifacts(
        user_features=user_features,
        item_genre_features=item_genre_features,
        train_matrix=train_matrix,
        val_matrix=val_matrix,
        test_matrix=test_matrix,
        candidate_item_universe=candidate_item_universe,
        user_masks=user_masks,
        source_operator=source_operator,
        target_operator=target_operator,
    )

def build_enriched_user_features(
    music_features: np.ndarray,
    config: RecommendationConfig,
    top_book_genres: int = 30,
) -> np.ndarray:
\
\
\
\
\
\
       
    d = np.load(config.artifact_path("target_train_user_profiles"))
    book_sparse = sparse.csr_matrix(
        (d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"])
    )
    book_top_k = book_sparse[:, :top_book_genres].toarray().astype(np.float32)
    enriched = np.concatenate([music_features, book_top_k], axis=1)
    print(
        f"  [features] enriched: music_top30({music_features.shape[1]}) "
        f"+ book_train_top{top_book_genres}({book_top_k.shape[1]}) "
        f"= {enriched.shape[1]} dims  "
        f"(all-zero rows: {(enriched.sum(1)==0).sum()})"
    )
    return enriched

def build_highrated_positive_cache(
    config: RecommendationConfig,
    stage1: Stage1Artifacts,
    rating_threshold: float = 4.0,
) -> object:
\
\
\
\
       
    splits = pd.read_csv(config.artifact_path("target_split_assignments"))
    train_hr = splits[
        (splits["split"] == "train") & (splits["rating"] >= rating_threshold)
    ]
    n_users, n_items = stage1.train_matrix.shape
    rows = train_hr["user_idx"].values
    cols = train_hr["item_idx"].values
    data = np.ones(len(rows), dtype=np.float32)
    hr_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()
    n_retained = len(train_hr)
    n_total = int((splits["split"] == "train").sum())
    print(
        f"  [sampling] high-rated BPR positives (rating≥{rating_threshold}): "
        f"{n_retained:,} / {n_total:,} = {n_retained/n_total:.1%}"
    )
    return build_train_positive_cache(hr_matrix)

def train_one_model(
    model_spec: ModelSpec,
    stage1: Stage1Artifacts,
    train_config: TrainConfig,
    config: RecommendationConfig,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    set_seed(train_config.seed)
    device = resolve_device()

    if train_config.use_enriched_features:
        user_features_np = build_enriched_user_features(
            stage1.user_features, config, top_book_genres=train_config.top_book_genres
        )
    else:
        user_features_np = stage1.user_features
    x_user = torch.tensor(user_features_np, dtype=torch.float32, device=device)
    item_genre_features = scipy_to_torch_sparse(stage1.item_genre_features, device=device)
    source_operator = scipy_to_torch_sparse(stage1.source_operator, device=device)
    target_operator = scipy_to_torch_sparse(stage1.target_operator, device=device)

    n_items = stage1.item_genre_features.shape[0]
    genre_dim = stage1.item_genre_features.shape[1]
    in_dim = user_features_np.shape[1]

    model = instantiate_model(
        model_spec=model_spec,
        x_user_dim=in_dim,
        n_items=n_items,
        genre_dim=genre_dim,
        source_operator=source_operator,
        target_operator=target_operator,
        train_config=train_config,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    train_popularity = build_target_train_item_popularity(stage1.train_matrix)
    negative_distribution = build_negative_sampling_distribution(
        train_item_popularity=train_popularity,
        candidate_item_universe=stage1.candidate_item_universe,
        power=config.negative_sampling_power,
    )

    if train_config.highrated_sampling:
        positive_cache = build_highrated_positive_cache(
            config, stage1, rating_threshold=train_config.highrated_threshold
        )
    else:
        positive_cache = build_train_positive_cache(stage1.train_matrix)
    train_rng = np.random.default_rng(train_config.seed)
    candidate_item_ids_t = torch.tensor(stage1.candidate_item_universe, dtype=torch.long, device=device)

    val_exclude = (stage1.train_matrix + stage1.test_matrix).sign().tocsr()
    test_exclude = (stage1.train_matrix + stage1.val_matrix).sign().tocsr()
    val_metric_mask = build_eval_mask(
        stage1.user_masks["val_metric_user_mask"],
        limit=train_config.smoke_eval_user_limit,
        seed=train_config.seed,
    )
    test_metric_mask = build_eval_mask(
        stage1.user_masks["test_metric_user_mask"],
        limit=train_config.smoke_eval_user_limit,
        seed=train_config.seed + 1,
    )
    val_metric_user_count = int(np.asarray(val_metric_mask, dtype=bool).sum())
    test_metric_user_count = int(np.asarray(test_metric_mask, dtype=bool).sum())
    val_eval_cache = None
    test_eval_cache = None
    if not train_config.smoke_skip_ranking_eval:
        val_eval_cache = build_eval_split_cache(
            split_name="val",
            relevant_matrix=stage1.val_matrix,
            exclude_matrix=val_exclude,
            metric_user_mask=val_metric_mask,
            candidate_item_universe=stage1.candidate_item_universe,
            device=device,
        )
        test_eval_cache = build_eval_split_cache(
            split_name="test",
            relevant_matrix=stage1.test_matrix,
            exclude_matrix=test_exclude,
            metric_user_mask=test_metric_mask,
            candidate_item_universe=stage1.candidate_item_universe,
            device=device,
        )

    history: list[dict[str, object]] = []
    best_state = None
    best_val_ndcg20 = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    total_start = time.perf_counter()

    for epoch in range(1, train_config.max_epochs + 1):
        epoch_start = time.perf_counter()

        t_sample = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        users_np, pos_np, neg_np = sample_bpr_triplets(
            positive_cache=positive_cache,
            distribution=negative_distribution,
            rng=train_rng,
        )
        sample_seconds = time.perf_counter() - t_sample

        t_train = time.perf_counter()
        user_ids = torch.tensor(users_np, dtype=torch.long, device=device)
        pos_item_ids = torch.tensor(pos_np, dtype=torch.long, device=device)
        neg_item_ids = torch.tensor(neg_np, dtype=torch.long, device=device)

        user_embeddings = model.get_user_embeddings(x_user)
        item_embeddings = model.get_item_embeddings(item_genre_features)
        pos_scores = score(user_embeddings[user_ids], item_embeddings[pos_item_ids])
        neg_scores = score(user_embeddings[user_ids], item_embeddings[neg_item_ids])
        train_bpr = bpr_loss(pos_scores, neg_scores)
        train_bpr.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.grad_clip_norm)
        optimizer.step()
        train_seconds = time.perf_counter() - t_train

        t_eval = time.perf_counter()
        model.eval()
        with torch.no_grad():
            user_embeddings_eval = model.get_user_embeddings(x_user)
            item_embeddings_eval = model.get_item_embeddings(item_genre_features)
            candidate_item_embeddings = item_embeddings_eval.index_select(0, candidate_item_ids_t)
            val_metrics = maybe_evaluate_split(
                split_name="val",
                skip_ranking=train_config.smoke_skip_ranking_eval,
                user_embeddings=user_embeddings_eval,
                candidate_item_embeddings=candidate_item_embeddings,
                eval_cache=val_eval_cache,
                candidate_item_universe_size=int(stage1.candidate_item_universe.size),
                eligible_user_count=val_metric_user_count,
                batch_size_users=train_config.eval_batch_size_users,
            )
        eval_seconds = time.perf_counter() - t_eval

        epoch_seconds = time.perf_counter() - epoch_start
        alpha_value = model.get_alpha()
        history.append(
            {
                "epoch": epoch,
                "train_bpr": float(train_bpr.item()),
                "val_recall@20": float(val_metrics.metrics["Recall@20"]),
                "val_ndcg@20": float(val_metrics.metrics["NDCG@20"]),
                "alpha": alpha_value if alpha_value is not None else "",
                "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                "epoch_seconds": float(epoch_seconds),
                "sample_seconds": float(sample_seconds),
                "train_seconds": float(train_seconds),
                "eval_seconds": float(eval_seconds),
            }
        )

        if epoch <= 3 or epoch % 10 == 0:
            print(
                f"  [epoch {epoch:>3d}] bpr={train_bpr.item():.4f}  "
                f"val_ndcg@20={val_metrics.metrics['NDCG@20']:.4f}  "
                f"time={epoch_seconds:.1f}s "
                f"(sample={sample_seconds:.2f}s train={train_seconds:.2f}s eval={eval_seconds:.2f}s)"
                + (f"  alpha={alpha_value:.4f}" if alpha_value is not None else ""),
                flush=True,
            )

        if val_metrics.metrics["NDCG@20"] > best_val_ndcg20 + 1e-12:
            best_val_ndcg20 = float(val_metrics.metrics["NDCG@20"])
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_config.patience:
                break

    if best_state is None:
        raise RuntimeError(f"Training failed to produce a best validation checkpoint for {model_spec.run_name}")

    total_seconds = time.perf_counter() - total_start
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        user_embeddings = model.get_user_embeddings(x_user)
        item_embeddings = model.get_item_embeddings(item_genre_features)
        candidate_item_embeddings = item_embeddings.index_select(0, candidate_item_ids_t)
        val_metrics = maybe_evaluate_split(
            split_name="val",
            skip_ranking=train_config.smoke_skip_ranking_eval,
            user_embeddings=user_embeddings,
            candidate_item_embeddings=candidate_item_embeddings,
            eval_cache=val_eval_cache,
            candidate_item_universe_size=int(stage1.candidate_item_universe.size),
            eligible_user_count=val_metric_user_count,
            batch_size_users=train_config.eval_batch_size_users,
        )
        test_metrics = maybe_evaluate_split(
            split_name="test",
            skip_ranking=train_config.smoke_skip_ranking_eval,
            user_embeddings=user_embeddings,
            candidate_item_embeddings=candidate_item_embeddings,
            eval_cache=test_eval_cache,
            candidate_item_universe_size=int(stage1.candidate_item_universe.size),
            eligible_user_count=test_metric_user_count,
            batch_size_users=train_config.eval_batch_size_users,
        )

    alpha_stats = {}
    if isinstance(model, PerUserAlphaGraphRecommender):
        alpha_stats = model.get_alpha_stats(x_user)
        print(f"  [per-user alpha] mean={alpha_stats['alpha_mean']:.4f} "
              f"std={alpha_stats['alpha_std']:.4f} "
              f"min={alpha_stats['alpha_min']:.4f} max={alpha_stats['alpha_max']:.4f}")

    results: dict[str, object] = {
        "model": model_spec.display_name,
        "graph": model_spec.graph,
        "alpha_mode": model_spec.alpha_mode,
        "alpha_value": alpha_stats.get("alpha_mean", model.get_alpha()),
        "device": str(device),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "train_time_total_seconds": float(total_seconds),
        "use_enriched_features": bool(train_config.use_enriched_features),
        "highrated_sampling": bool(train_config.highrated_sampling),
        "user_feature_dim": int(user_features_np.shape[1]),
        **alpha_stats,
        **{f"val_{key}": float(value) for key, value in val_metrics.metrics.items()},
        **{f"test_{key}": float(value) for key, value in test_metrics.metrics.items()},
    }
    ranking_summary = {
        "val": val_metrics.summary,
        "test": test_metrics.summary,
    }
    return results, history, ranking_summary

def instantiate_model(
    model_spec: ModelSpec,
    x_user_dim: int,
    n_items: int,
    genre_dim: int,
    source_operator: torch.Tensor,
    target_operator: torch.Tensor,
    train_config: TrainConfig,
) -> torch.nn.Module:
    if model_spec.model_kind == "no_graph":
        return NoGraphRecommender(
            in_dim=x_user_dim,
            n_items=n_items,
            genre_dim=genre_dim,
            hidden_dim=train_config.hidden_dim,
            dropout=train_config.dropout,
        )
    if model_spec.model_kind == "source_only":
        return GraphRecommender(
            operator=source_operator,
            in_dim=x_user_dim,
            n_items=n_items,
            genre_dim=genre_dim,
            hidden_dim=train_config.hidden_dim,
            dropout=train_config.dropout,
            fixed_alpha=1.0,
        )
    if model_spec.model_kind == "target_only":
        return GraphRecommender(
            operator=target_operator,
            in_dim=x_user_dim,
            n_items=n_items,
            genre_dim=genre_dim,
            hidden_dim=train_config.hidden_dim,
            dropout=train_config.dropout,
            fixed_alpha=0.0,
        )
    if model_spec.model_kind == "fixed_fused":
        fused_operator = (model_spec.alpha_value * source_operator) + ((1.0 - model_spec.alpha_value) * target_operator)
        return GraphRecommender(
            operator=fused_operator.coalesce(),
            in_dim=x_user_dim,
            n_items=n_items,
            genre_dim=genre_dim,
            hidden_dim=train_config.hidden_dim,
            dropout=train_config.dropout,
            fixed_alpha=model_spec.alpha_value,
        )
    if model_spec.model_kind == "learned_alpha":
        return LearnedAlphaGraphRecommender(
            source_operator=source_operator,
            target_operator=target_operator,
            in_dim=x_user_dim,
            n_items=n_items,
            genre_dim=genre_dim,
            hidden_dim=train_config.hidden_dim,
            dropout=train_config.dropout,
        )
    if model_spec.model_kind == "per_user_alpha":
        return PerUserAlphaGraphRecommender(
            source_operator=source_operator,
            target_operator=target_operator,
            in_dim=x_user_dim,
            n_items=n_items,
            genre_dim=genre_dim,
            hidden_dim=train_config.hidden_dim,
            dropout=train_config.dropout,
        )
    raise ValueError(f"Unsupported model kind: {model_spec.model_kind}")

def scipy_to_torch_sparse(matrix: sparse.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device).coalesce()

def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_eval_mask(metric_user_mask: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    metric_user_mask = np.asarray(metric_user_mask, dtype=bool)
    if limit is None:
        return metric_user_mask
    eligible = np.flatnonzero(metric_user_mask)
    if eligible.size <= limit:
        return metric_user_mask
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(eligible, size=limit, replace=False))
    subset_mask = np.zeros_like(metric_user_mask, dtype=bool)
    subset_mask[selected] = True
    return subset_mask

def maybe_evaluate_split(
    split_name: str,
    skip_ranking: bool,
    user_embeddings: torch.Tensor,
    candidate_item_embeddings: torch.Tensor,
    eval_cache,
    candidate_item_universe_size: int,
    eligible_user_count: int,
    batch_size_users: int,
):
    if skip_ranking:
        return _skipped_ranking_metrics(split_name, eligible_user_count, candidate_item_universe_size)
    if split_name == "val":
        return evaluate_validation_split(
            user_embeddings=user_embeddings,
            candidate_item_embeddings=candidate_item_embeddings,
            eval_cache=eval_cache,
            batch_size_users=batch_size_users,
        )
    if split_name == "test":
        return evaluate_test_split(
            user_embeddings=user_embeddings,
            candidate_item_embeddings=candidate_item_embeddings,
            eval_cache=eval_cache,
            batch_size_users=batch_size_users,
        )
    raise ValueError(f"Unsupported split name: {split_name}")

def _skipped_ranking_metrics(split_name: str, eligible_users: int, candidate_item_universe_size: int):
    class _SkippedMetrics:
        def __init__(self) -> None:
            self.metrics = {
                "Recall@10": 0.0,
                "Recall@20": 0.0,
                "NDCG@10": 0.0,
                "NDCG@20": 0.0,
                "HitRate@20": 0.0,
            }
            self.summary = {
                "split": split_name,
                "eligible_user_count": int(eligible_users),
                "relevant_warm_items_total": 0,
                "candidate_item_universe_size": int(candidate_item_universe_size),
                "ranking_skipped": True,
            }

    return _SkippedMetrics()
