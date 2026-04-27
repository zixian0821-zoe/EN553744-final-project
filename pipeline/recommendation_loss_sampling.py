from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse

@dataclass(frozen=True)
class NegativeSamplingDistribution:
    candidate_items: np.ndarray
    probabilities: np.ndarray

@dataclass(frozen=True)
class TrainPositiveCache:
    user_ids: np.ndarray
    pos_item_ids: np.ndarray
    user_indptr: np.ndarray
    positive_csr: sparse.csr_matrix

def build_target_train_item_popularity(train_matrix: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(train_matrix.sum(axis=0)).ravel().astype(np.int64)

def build_train_positive_cache(train_matrix: sparse.csr_matrix) -> TrainPositiveCache:
    train_matrix = train_matrix.tocsr()
    user_counts = np.diff(train_matrix.indptr).astype(np.int32, copy=False)
    user_ids = np.repeat(np.arange(train_matrix.shape[0], dtype=np.int32), user_counts)
    pos_item_ids = train_matrix.indices.astype(np.int32, copy=True)
    user_indptr = train_matrix.indptr.astype(np.int64, copy=True)
    positive_csr = train_matrix.astype(bool).tocsr()
    return TrainPositiveCache(
        user_ids=user_ids,
        pos_item_ids=pos_item_ids,
        user_indptr=user_indptr,
        positive_csr=positive_csr,
    )

def build_negative_sampling_distribution(
    train_item_popularity: np.ndarray,
    candidate_item_universe: np.ndarray,
    power: float,
) -> NegativeSamplingDistribution:
    candidate_items = candidate_item_universe.astype(np.int32, copy=False)
    weights = np.power(train_item_popularity[candidate_items].astype(np.float64), power)
    if np.any(weights <= 0.0):
        raise ValueError("Negative sampling received non-positive popularity weights in the candidate universe")
    probabilities = weights / weights.sum()
    return NegativeSamplingDistribution(candidate_items=candidate_items, probabilities=probabilities)

def sample_bpr_triplets(
    positive_cache: TrainPositiveCache,
    distribution: NegativeSamplingDistribution,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
\
\
\
\
\
\
       
    candidate_items = distribution.candidate_items
    probabilities = distribution.probabilities
    n_edges = positive_cache.pos_item_ids.size

    if n_edges == 0:
        raise ValueError("No BPR triplets could be sampled because the training matrix is empty")

    neg_items = rng.choice(
        candidate_items,
        size=n_edges,
        replace=True,
        p=probabilities,
    ).astype(np.int32, copy=False)

    user_ids = positive_cache.user_ids
    positive_csr = positive_cache.positive_csr
    _MAX_RETRIES = 3

    for _ in range(_MAX_RETRIES):
        collision_vals = positive_csr[user_ids, neg_items]
        collision_mask = np.asarray(collision_vals).ravel().astype(bool)
        n_collisions = int(collision_mask.sum())
        if n_collisions == 0:
            break
        neg_items[collision_mask] = rng.choice(
            candidate_items,
            size=n_collisions,
            replace=True,
            p=probabilities,
        ).astype(np.int32, copy=False)

    return (
        positive_cache.user_ids,
        positive_cache.pos_item_ids,
        neg_items,
    )

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

def regularization_loss(model: torch.nn.Module) -> torch.Tensor:
    penalties = [parameter.pow(2).sum() for parameter in model.parameters() if parameter.requires_grad]
    if not penalties:
        return torch.tensor(0.0)
    return torch.stack(penalties).sum()
