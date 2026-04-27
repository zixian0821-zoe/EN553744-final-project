from __future__ import annotations

import math

import torch
from torch import nn

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import ChebConv, SAGEConv, GATConv



class ItemRepresentationModule(nn.Module):
    def __init__(self, n_items: int, genre_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, hidden_dim)
        self.genre_projection = nn.Parameter(torch.empty(genre_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.genre_projection)

    def forward(self, item_genre_features: torch.Tensor) -> torch.Tensor:
        projected = torch.sparse.mm(item_genre_features, self.genre_projection)
        return self.item_embedding.weight + projected

def _operator_to_pyg_edges(operator: torch.Tensor):
    """
    Convert a torch sparse operator into PyG edge_index / edge_weight format.
    """
    operator = operator.coalesce()
    edge_index = operator.indices().long()
    edge_weight = operator.values().float()
    return edge_index, edge_weight

class NoGraphRecommender(nn.Module):
    def __init__(self, in_dim: int, n_items: int, genre_dim: int, hidden_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.item_module = ItemRepresentationModule(n_items=n_items, genre_dim=genre_dim, hidden_dim=hidden_dim)

    def get_user_embeddings(self, x_user: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.lin1(x_user))
        hidden = self.dropout(hidden)
        return self.lin2(hidden)

    def get_item_embeddings(self, item_genre_features: torch.Tensor) -> torch.Tensor:
        return self.item_module(item_genre_features)

    def get_alpha(self) -> float | None:
        return None


class GraphRecommender(nn.Module):
    def __init__(
        self,
        operator: torch.Tensor,
        in_dim: int,
        n_items: int,
        genre_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        fixed_alpha: float | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("operator", operator.coalesce())
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.item_module = ItemRepresentationModule(n_items=n_items, genre_dim=genre_dim, hidden_dim=hidden_dim)
        self.fixed_alpha = fixed_alpha

    def effective_operator(self) -> torch.Tensor:
        return self.operator

    def get_user_embeddings(self, x_user: torch.Tensor) -> torch.Tensor:
        operator = self.effective_operator()
        hidden = torch.sparse.mm(operator, x_user)
        hidden = torch.relu(self.lin1(hidden))
        hidden = self.dropout(hidden)
        hidden = torch.sparse.mm(operator, hidden)
        return self.lin2(hidden)

    def get_item_embeddings(self, item_genre_features: torch.Tensor) -> torch.Tensor:
        return self.item_module(item_genre_features)

    def get_alpha(self) -> float | None:
        return self.fixed_alpha


class LearnedAlphaGraphRecommender(nn.Module):
    def __init__(
        self,
        source_operator: torch.Tensor,
        target_operator: torch.Tensor,
        in_dim: int,
        n_items: int,
        genre_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.register_buffer("source_operator", source_operator.coalesce())
        self.register_buffer("target_operator", target_operator.coalesce())
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.item_module = ItemRepresentationModule(n_items=n_items, genre_dim=genre_dim, hidden_dim=hidden_dim)

    def alpha_tensor(self) -> torch.Tensor:
        return torch.sigmoid(self.theta)

    def effective_operator(self) -> torch.Tensor:
        alpha = self.alpha_tensor()
        return (alpha * self.source_operator) + ((1.0 - alpha) * self.target_operator)

    def get_user_embeddings(self, x_user: torch.Tensor) -> torch.Tensor:
        operator = self.effective_operator().coalesce()
        hidden = torch.sparse.mm(operator, x_user)
        hidden = torch.relu(self.lin1(hidden))
        hidden = self.dropout(hidden)
        hidden = torch.sparse.mm(operator, hidden)
        return self.lin2(hidden)

    def get_item_embeddings(self, item_genre_features: torch.Tensor) -> torch.Tensor:
        return self.item_module(item_genre_features)

    def get_alpha(self) -> float:
        return float(torch.sigmoid(self.theta.detach()).item())


class PerUserAlphaGraphRecommender(nn.Module):
    """Per-user learned alpha: α_i = σ(f(x_i, activity_i)).

    Instead of fusing adjacency matrices with a single scalar alpha,
    this model runs both source and target graph operators separately,
    then blends their outputs per-user based on each user's features
    plus an explicit activity signal (L1 norm of music profile).

    Users with strong music profiles can rely more on the source graph,
    while users with weak music signals lean on the target graph.
    """

    def __init__(
        self,
        source_operator: torch.Tensor,
        target_operator: torch.Tensor,
        in_dim: int,
        n_items: int,
        genre_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.register_buffer("source_operator", source_operator.coalesce())
        self.register_buffer("target_operator", target_operator.coalesce())

        # Per-user alpha network: [user_features | activity] → scalar alpha_i
        # +1 for the activity (L1 norm) signal appended to features
        alpha_in_dim = in_dim + 1
        self.alpha_net = nn.Sequential(
            nn.Linear(alpha_in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # Initialize so that alpha starts near 0.5 for all users
        nn.init.zeros_(self.alpha_net[2].weight)
        nn.init.zeros_(self.alpha_net[2].bias)

        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.item_module = ItemRepresentationModule(
            n_items=n_items, genre_dim=genre_dim, hidden_dim=hidden_dim
        )

    def _build_alpha_input(self, x_user: torch.Tensor) -> torch.Tensor:
        """Concatenate user features with their L1 activity norm."""
        activity = x_user.sum(dim=1, keepdim=True)  # (N, 1)
        return torch.cat([x_user, activity], dim=1)  # (N, in_dim+1)

    def compute_per_user_alpha(self, x_user: torch.Tensor) -> torch.Tensor:
        """Returns (N, 1) tensor of per-user alpha values in [0, 1]."""
        alpha_input = self._build_alpha_input(x_user)
        return torch.sigmoid(self.alpha_net(alpha_input))

    def get_user_embeddings(self, x_user: torch.Tensor) -> torch.Tensor:
        alpha = self.compute_per_user_alpha(x_user)  # (N, 1)

        # Run both operators, blend per-user
        source_out = torch.sparse.mm(self.source_operator, x_user)  # (N, in_dim)
        target_out = torch.sparse.mm(self.target_operator, x_user)  # (N, in_dim)
        fused = alpha * source_out + (1.0 - alpha) * target_out     # (N, in_dim)

        hidden = torch.relu(self.lin1(fused))
        hidden = self.dropout(hidden)

        # Second graph pass (same per-user blending)
        source_out2 = torch.sparse.mm(self.source_operator, hidden)
        target_out2 = torch.sparse.mm(self.target_operator, hidden)
        fused2 = alpha * source_out2 + (1.0 - alpha) * target_out2

        return self.lin2(fused2)

    def get_item_embeddings(self, item_genre_features: torch.Tensor) -> torch.Tensor:
        return self.item_module(item_genre_features)

    def get_alpha(self) -> float:
        """Return mean alpha across all users (for logging compatibility)."""
        return None  # actual per-user alphas logged separately

    def get_alpha_stats(self, x_user: torch.Tensor) -> dict[str, float]:
        """Return per-user alpha statistics for analysis."""
        with torch.no_grad():
            alpha = self.compute_per_user_alpha(x_user).squeeze(1)
            return {
                "alpha_mean": float(alpha.mean().item()),
                "alpha_std": float(alpha.std().item()),
                "alpha_min": float(alpha.min().item()),
                "alpha_max": float(alpha.max().item()),
                "alpha_median": float(alpha.median().item()),
            }


def score(user_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
    return torch.sum(user_embeddings * item_embeddings, dim=1)




class ChebNetRecommender(nn.Module):
    def __init__(
        self,
        operator: torch.Tensor,
        in_dim: int,
        n_items: int,
        genre_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        K: int = 3,
        fixed_alpha: Optional[float] = None,
    ):
        super().__init__()
        edge_index, edge_weight = _operator_to_pyg_edges(operator)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

        self.conv1 = ChebConv(in_dim, hidden_dim, K=K, normalization="sym")
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K, normalization="sym")
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.item_module = ItemRepresentationModule(n_items, genre_dim, hidden_dim)
        self.fixed_alpha = fixed_alpha

    def get_user_embeddings(self, x_user: Tensor) -> Tensor:
        h = self.conv1(x_user, self.edge_index, self.edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, self.edge_index, self.edge_weight)
        h = F.relu(h)
        h = self.fc(h)
        return h

    def get_item_embeddings(self, item_genre_features: Tensor) -> Tensor:
        return self.item_module(item_genre_features)

    def get_alpha(self):
        return self.fixed_alpha


class GraphSAGERecommender(nn.Module):
    def __init__(
        self,
        operator: torch.Tensor,
        in_dim: int,
        n_items: int,
        genre_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        fixed_alpha: Optional[float] = None,
    ):
        super().__init__()
        edge_index, edge_weight = _operator_to_pyg_edges(operator)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.item_module = ItemRepresentationModule(n_items, genre_dim, hidden_dim)
        self.fixed_alpha = fixed_alpha

    def get_user_embeddings(self, x_user: Tensor) -> Tensor:
        h = self.conv1(x_user, self.edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, self.edge_index)
        h = F.relu(h)
        h = self.fc(h)
        return h

    def get_item_embeddings(self, item_genre_features: Tensor) -> Tensor:
        return self.item_module(item_genre_features)

    def get_alpha(self):
        return self.fixed_alpha


class GATRecommender(nn.Module):
    def __init__(
        self,
        operator: torch.Tensor,
        in_dim: int,
        n_items: int,
        genre_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        heads: int = 1,
        attn_dropout: float = 0.0,
        fixed_alpha: Optional[float] = None,
    ):
        super().__init__()
        edge_index, edge_weight = _operator_to_pyg_edges(operator)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

        # 这里用 concat=False，保证输出维度就是 hidden_dim
        self.conv1 = GATConv(
            in_dim,
            hidden_dim,
            heads=heads,
            concat=False,
            dropout=attn_dropout,
            add_self_loops=False,
        )
        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=attn_dropout,
            add_self_loops=False,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.item_module = ItemRepresentationModule(n_items, genre_dim, hidden_dim)
        self.fixed_alpha = fixed_alpha

    def get_user_embeddings(self, x_user: Tensor) -> Tensor:
        h = self.conv1(x_user, self.edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.conv2(h, self.edge_index)
        h = F.elu(h)
        h = self.fc(h)
        return h

    def get_item_embeddings(self, item_genre_features: Tensor) -> Tensor:
        return self.item_module(item_genre_features)

    def get_alpha(self):
        return self.fixed_alpha
