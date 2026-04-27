from __future__ import annotations

import math

import torch
from torch import nn

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
\
\
\
\
\
\
\
\
\
       

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

        alpha_in_dim = in_dim + 1
        self.alpha_net = nn.Sequential(
            nn.Linear(alpha_in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.alpha_net[2].weight)
        nn.init.zeros_(self.alpha_net[2].bias)

        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.item_module = ItemRepresentationModule(
            n_items=n_items, genre_dim=genre_dim, hidden_dim=hidden_dim
        )

    def _build_alpha_input(self, x_user: torch.Tensor) -> torch.Tensor:
                                                                    
        activity = x_user.sum(dim=1, keepdim=True)
        return torch.cat([x_user, activity], dim=1)

    def compute_per_user_alpha(self, x_user: torch.Tensor) -> torch.Tensor:
                                                                       
        alpha_input = self._build_alpha_input(x_user)
        return torch.sigmoid(self.alpha_net(alpha_input))

    def get_user_embeddings(self, x_user: torch.Tensor) -> torch.Tensor:
        alpha = self.compute_per_user_alpha(x_user)

        source_out = torch.sparse.mm(self.source_operator, x_user)
        target_out = torch.sparse.mm(self.target_operator, x_user)
        fused = alpha * source_out + (1.0 - alpha) * target_out

        hidden = torch.relu(self.lin1(fused))
        hidden = self.dropout(hidden)

        source_out2 = torch.sparse.mm(self.source_operator, hidden)
        target_out2 = torch.sparse.mm(self.target_operator, hidden)
        fused2 = alpha * source_out2 + (1.0 - alpha) * target_out2

        return self.lin2(fused2)

    def get_item_embeddings(self, item_genre_features: torch.Tensor) -> torch.Tensor:
        return self.item_module(item_genre_features)

    def get_alpha(self) -> float:
                                                                             
        return None

    def get_alpha_stats(self, x_user: torch.Tensor) -> dict[str, float]:
                                                            
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
