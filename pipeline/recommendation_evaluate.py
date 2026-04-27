from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse

EPS = 1e-12

@dataclass(frozen=True)
class RankingMetrics:
    metrics: dict[str, float]
    summary: dict[str, int | float | str]

@dataclass(frozen=True)
class EvalSplitCache:
    split_name: str
    user_indices: np.ndarray
    user_indices_t: torch.Tensor
    relevant_padded: torch.Tensor
    relevant_counts: torch.Tensor
    exclude_positions_tensors: list[torch.Tensor]
    candidate_item_universe_size: int
    relevant_warm_items_total: int

def build_eval_split_cache(
    split_name: str,
    relevant_matrix: sparse.csr_matrix,
    exclude_matrix: sparse.csr_matrix,
    metric_user_mask: np.ndarray,
    candidate_item_universe: np.ndarray,
    device: torch.device,
) -> EvalSplitCache:
    metric_user_mask = np.asarray(metric_user_mask, dtype=bool)
    candidate_positions = np.full(relevant_matrix.shape[1], -1, dtype=np.int32)
    candidate_positions[candidate_item_universe] = np.arange(candidate_item_universe.size, dtype=np.int32)

    cached_user_indices: list[int] = []
    cached_relevant_list: list[np.ndarray] = []
    cached_exclude_list: list[torch.Tensor] = []
    relevant_warm_items_total = 0

    for user_idx in np.flatnonzero(metric_user_mask).astype(np.int32):
        rel_start = relevant_matrix.indptr[user_idx]
        rel_stop = relevant_matrix.indptr[user_idx + 1]
        relevant_global = relevant_matrix.indices[rel_start:rel_stop]
        relevant_pos = candidate_positions[relevant_global]
        relevant_pos = relevant_pos[relevant_pos >= 0].astype(np.int32, copy=False)
        if relevant_pos.size == 0:
            continue

        exc_start = exclude_matrix.indptr[user_idx]
        exc_stop = exclude_matrix.indptr[user_idx + 1]
        excluded_global = exclude_matrix.indices[exc_start:exc_stop]
        excluded_pos = candidate_positions[excluded_global]
        excluded_pos = excluded_pos[excluded_pos >= 0].astype(np.int64, copy=False)

        cached_user_indices.append(int(user_idx))
        cached_relevant_list.append(relevant_pos)
        cached_exclude_list.append(torch.tensor(excluded_pos, dtype=torch.long, device=device))
        relevant_warm_items_total += int(relevant_pos.size)

    user_indices = np.asarray(cached_user_indices, dtype=np.int32)
    user_indices_t = torch.tensor(user_indices, dtype=torch.long, device=device)

    n_eval = len(cached_relevant_list)
    if n_eval > 0:
        max_rel = max(rp.size for rp in cached_relevant_list)
        rel_padded_np = np.full((n_eval, max_rel), -1, dtype=np.int64)
        rel_counts_np = np.zeros(n_eval, dtype=np.int64)
        for i, rp in enumerate(cached_relevant_list):
            rel_padded_np[i, : rp.size] = rp
            rel_counts_np[i] = rp.size
        relevant_padded = torch.tensor(rel_padded_np, dtype=torch.long, device=device)
        relevant_counts = torch.tensor(rel_counts_np, dtype=torch.long, device=device)
    else:
        relevant_padded = torch.empty((0, 1), dtype=torch.long, device=device)
        relevant_counts = torch.empty(0, dtype=torch.long, device=device)

    return EvalSplitCache(
        split_name=split_name,
        user_indices=user_indices,
        user_indices_t=user_indices_t,
        relevant_padded=relevant_padded,
        relevant_counts=relevant_counts,
        exclude_positions_tensors=cached_exclude_list,
        candidate_item_universe_size=int(candidate_item_universe.size),
        relevant_warm_items_total=int(relevant_warm_items_total),
    )

def _rank_topk_gpu(
    user_embeddings: torch.Tensor,
    candidate_item_embeddings: torch.Tensor,
    eval_cache: EvalSplitCache,
    topk: int,
    batch_size_users: int,
) -> torch.Tensor:
                                                                                
    device = user_embeddings.device
    n_eval = eval_cache.user_indices.size
    topk_all = torch.empty((n_eval, topk), dtype=torch.long, device=device)
    cand_t = candidate_item_embeddings.T.contiguous()

    for start in range(0, n_eval, batch_size_users):
        stop = min(start + batch_size_users, n_eval)
        batch_users = eval_cache.user_indices_t[start:stop]
        scores = user_embeddings.index_select(0, batch_users) @ cand_t

        for row_off, excl in enumerate(eval_cache.exclude_positions_tensors[start:stop]):
            if excl.numel() > 0:
                scores[row_off, excl] = -torch.inf

        topk_all[start:stop] = torch.topk(scores, k=topk, dim=1).indices

    return topk_all

def _compute_hits_gpu(
    topk_tensor: torch.Tensor,
    rel_padded: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
                                                                         
    N, k = topk_tensor.shape
    device = topk_tensor.device
    hits = torch.zeros(N, k, dtype=torch.bool, device=device)

    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        hits[s:e] = (
            topk_tensor[s:e].unsqueeze(2) == rel_padded[s:e].unsqueeze(1)
        ).any(dim=2)

    return hits

def _gpu_metrics(
    topk_tensor: torch.Tensor,
    eval_cache: EvalSplitCache,
) -> dict[str, float]:
                                                                  
    device = topk_tensor.device
    max_k = topk_tensor.shape[1]

    hits_20 = _compute_hits_gpu(topk_tensor, eval_cache.relevant_padded)
    hits_10 = hits_20[:, :10]

    rel_f = eval_cache.relevant_counts.float().clamp(min=1.0)
    n = float(topk_tensor.shape[0])

    recall_10 = hits_10.sum(1).float() / rel_f
    recall_20 = hits_20.sum(1).float() / rel_f

    discounts = 1.0 / torch.log2(
        torch.arange(2, max_k + 2, device=device, dtype=torch.float32)
    )
    dcg_10 = (hits_10.float() * discounts[:10]).sum(1)
    dcg_20 = (hits_20.float() * discounts[:max_k]).sum(1)

    cum_disc = torch.cumsum(discounts, dim=0)
    ten_t = torch.tensor(10, device=device)
    twenty_t = torch.tensor(max_k, device=device)
    ideal_10 = torch.minimum(eval_cache.relevant_counts, ten_t)
    ideal_20 = torch.minimum(eval_cache.relevant_counts, twenty_t)
    idcg_10 = cum_disc[(ideal_10.clamp(min=1) - 1).long()]
    idcg_20 = cum_disc[(ideal_20.clamp(min=1) - 1).long()]
    ndcg_10 = dcg_10 / idcg_10.clamp(min=EPS)
    ndcg_20 = dcg_20 / idcg_20.clamp(min=EPS)

    hitrate_20 = hits_20.any(dim=1).float()

    return {
        "Recall@10": float(recall_10.sum().item() / n),
        "Recall@20": float(recall_20.sum().item() / n),
        "NDCG@10": float(ndcg_10.sum().item() / n),
        "NDCG@20": float(ndcg_20.sum().item() / n),
        "HitRate@20": float(hitrate_20.sum().item() / n),
    }

def evaluate_validation_split(
    user_embeddings: torch.Tensor,
    candidate_item_embeddings: torch.Tensor,
    eval_cache: EvalSplitCache,
    batch_size_users: int = 128,
) -> RankingMetrics:
    return _evaluate_split(
        user_embeddings=user_embeddings,
        candidate_item_embeddings=candidate_item_embeddings,
        eval_cache=eval_cache,
        batch_size_users=batch_size_users,
    )

def evaluate_test_split(
    user_embeddings: torch.Tensor,
    candidate_item_embeddings: torch.Tensor,
    eval_cache: EvalSplitCache,
    batch_size_users: int = 128,
) -> RankingMetrics:
    return _evaluate_split(
        user_embeddings=user_embeddings,
        candidate_item_embeddings=candidate_item_embeddings,
        eval_cache=eval_cache,
        batch_size_users=batch_size_users,
    )

def ranking_summary_dict(
    split_name: str,
    eligible_users: int,
    relevant_items_total: int,
    candidate_item_universe_size: int,
) -> dict[str, int | str]:
    return {
        "split": split_name,
        "eligible_user_count": int(eligible_users),
        "relevant_warm_items_total": int(relevant_items_total),
        "candidate_item_universe_size": int(candidate_item_universe_size),
    }

def _evaluate_split(
    user_embeddings: torch.Tensor,
    candidate_item_embeddings: torch.Tensor,
    eval_cache: EvalSplitCache,
    batch_size_users: int,
) -> RankingMetrics:
    if eval_cache.user_indices.size == 0:
        raise ValueError(f"No eligible users found for {eval_cache.split_name} evaluation")

    max_k = 20
    topk_tensor = _rank_topk_gpu(
        user_embeddings=user_embeddings,
        candidate_item_embeddings=candidate_item_embeddings,
        eval_cache=eval_cache,
        topk=max_k,
        batch_size_users=batch_size_users,
    )

    metrics = _gpu_metrics(topk_tensor, eval_cache)
    summary = ranking_summary_dict(
        split_name=eval_cache.split_name,
        eligible_users=int(topk_tensor.shape[0]),
        relevant_items_total=eval_cache.relevant_warm_items_total,
        candidate_item_universe_size=eval_cache.candidate_item_universe_size,
    )
    return RankingMetrics(metrics=metrics, summary=summary)
