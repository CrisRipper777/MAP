from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import scatter

from .common import get_activation, make_norm


def normalize_restart_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype,
    add_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the MAP-v2 GCN-normalized uniform adjacency exactly once."""
    return gcn_norm(
        edge_index,
        edge_weight=None,
        num_nodes=int(num_nodes),
        improved=False,
        add_self_loops=bool(add_self_loops),
        flow="source_to_target",
        dtype=dtype,
    )


def apply_restart_diffusion(
    h0: torch.Tensor,
    norm_edge_index: torch.Tensor,
    norm_edge_weight: torch.Tensor | None,
    num_hops: int,
    restart: float,
) -> torch.Tensor:
    """Apply MAP-v2's shallow restart recurrence to one or more streams."""
    if int(num_hops) <= 0:
        return h0
    if norm_edge_weight is None or norm_edge_index.numel() == 0:
        return h0

    row, col = norm_edge_index
    weight_shape = (-1,) + (1,) * (h0.dim() - 1)
    h = h0
    restart_value = min(max(float(restart), 0.0), 1.0)
    for _ in range(int(num_hops)):
        propagated = scatter(
            h[row] * norm_edge_weight.view(weight_shape),
            col,
            dim=0,
            dim_size=h0.size(0),
            reduce="sum",
        )
        h = (1.0 - restart_value) * propagated + restart_value * h0
    return h


class RestartDiffusion(nn.Module):
    """Parameter-free MAP-v2 shallow restart diffusion helper."""

    def __init__(self, num_hops: int = 2, restart: float = 0.15, add_self_loops: bool = True) -> None:
        super().__init__()
        if int(num_hops) < 0:
            raise ValueError(f"num_hops must be >= 0, got {num_hops}")
        self.num_hops = int(num_hops)
        self.restart = float(restart)
        self.add_self_loops = bool(add_self_loops)

    def forward(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        norm_edge_index: torch.Tensor | None = None,
        norm_edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.num_hops <= 0:
            return h0
        if norm_edge_index is None:
            if edge_index is None:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=h0.device)
            norm_edge_index, norm_edge_weight = normalize_restart_adjacency(
                edge_index.to(device=h0.device, dtype=torch.long),
                num_nodes=int(h0.size(0)),
                dtype=h0.dtype,
                add_self_loops=self.add_self_loops,
            )
        return apply_restart_diffusion(
            h0,
            norm_edge_index,
            norm_edge_weight,
            self.num_hops,
            self.restart,
        )


class OwnershipEdgeScorer(nn.Module):
    """Shared static scorer for ownership-conditioned original graph edges."""

    NUM_FACTORS = 3

    def __init__(self, factor_dim: int = 128, score_hidden_dim: int = 64) -> None:
        super().__init__()
        self.factor_dim = int(factor_dim)
        self.score_hidden_dim = int(score_hidden_dim)
        if self.factor_dim <= 0 or self.score_hidden_dim <= 0:
            raise ValueError(
                "OwnershipEdgeScorer dimensions must be positive, got "
                f"factor_dim={factor_dim}, score_hidden_dim={score_hidden_dim}"
            )
        self.query = nn.Linear(self.factor_dim, self.score_hidden_dim, bias=False)
        self.key = nn.Linear(self.factor_dim, self.score_hidden_dim, bias=False)
        self.ownership_embedding = nn.Parameter(
            torch.empty(self.NUM_FACTORS, self.score_hidden_dim)
        )
        self.score_out = nn.Linear(self.score_hidden_dim, 1, bias=True)
        nn.init.normal_(self.ownership_embedding, mean=0.0, std=0.02)
        # Zero output layer makes every initial composition weight exactly one.
        nn.init.zeros_(self.score_out.weight)
        nn.init.zeros_(self.score_out.bias)

    def forward(self, ownership0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return ``[E, 3]`` scores for original edges ``src=j, dst=i``."""
        if ownership0.dim() != 3 or ownership0.size(1) != self.NUM_FACTORS:
            raise ValueError(
                "ownership0 must have shape [N, 3, factor_dim], got "
                f"{tuple(ownership0.shape)}"
            )
        if ownership0.size(-1) != self.factor_dim:
            raise ValueError(
                f"ownership0 factor dimension must be {self.factor_dim}, "
                f"got {ownership0.size(-1)}"
            )
        if edge_index.numel() == 0:
            return ownership0.new_empty((0, self.NUM_FACTORS))

        src, dst = edge_index
        query = self.query(ownership0[dst])
        query = query + self.ownership_embedding.to(dtype=query.dtype).unsqueeze(0)
        key = self.key(ownership0[src])
        hidden = F.gelu(query + key)
        scores = self.score_out(hidden).squeeze(-1)
        return scores


class ModalityProjector(nn.Module):
    """Per-modality projection: Linear -> Norm -> Act -> Dropout -> Linear."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            make_norm(norm, int(hidden_dim)),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemanticFactorizer(nn.Module):
    """Factorize each node into common and modality-private representations."""

    def __init__(
        self,
        text_dim: int,
        visual_dim: int,
        hidden_dim: int = 256,
        factor_dim: int = 128,
        dropout: float = 0.2,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        self.text_projector = ModalityProjector(text_dim, hidden_dim, dropout, activation, norm)
        self.visual_projector = ModalityProjector(visual_dim, hidden_dim, dropout, activation, norm)
        self.common_encoder = self._build_factor_mlp(hidden_dim, factor_dim, activation, norm)
        self.private_text_encoder = self._build_factor_mlp(hidden_dim, factor_dim, activation, norm)
        self.private_visual_encoder = self._build_factor_mlp(hidden_dim, factor_dim, activation, norm)
        self.hidden_dim = int(hidden_dim)
        self.factor_dim = int(factor_dim)

    @staticmethod
    def _build_factor_mlp(in_dim: int, out_dim: int, activation: str, norm: str) -> nn.Module:
        return nn.Sequential(
            nn.Linear(int(in_dim), int(out_dim)),
            make_norm(norm, int(out_dim)),
            get_activation(activation),
            nn.Linear(int(out_dim), int(out_dim)),
        )

    def forward(self, x_t: torch.Tensor, x_v: torch.Tensor) -> dict[str, torch.Tensor]:
        h_t = self.text_projector(x_t)
        h_v = self.visual_projector(x_v)
        c_t = self.common_encoder(h_t)
        c_v = self.common_encoder(h_v)
        p_t = self.private_text_encoder(h_t)
        p_v = self.private_visual_encoder(h_v)
        c = 0.5 * (c_t + c_v)
        return {
            "h_t": h_t,
            "h_v": h_v,
            "c_t": c_t,
            "c_v": c_v,
            "c": c,
            "p_t": p_t,
            "p_v": p_v,
        }


class ReconstructionHead(nn.Module):
    """Reconstruct a projected modality embedding from its common/private pair."""

    def __init__(self, factor_dim: int, hidden_dim: int, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * int(factor_dim), int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )

    def forward(self, c_mod: torch.Tensor, p_mod: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([c_mod, p_mod], dim=-1))


class JointFusion(nn.Module):
    """Fuse the three local ownership states into one joint graph stream."""

    def __init__(
        self,
        factor_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * int(factor_dim), int(hidden_dim)),
            make_norm(norm, int(hidden_dim)),
            get_activation(activation),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        common: torch.Tensor,
        private_text: torch.Tensor,
        private_visual: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([common, private_text, private_visual], dim=-1))
