from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import scatter

from .common import make_norm


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, norm: str | None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            make_norm(norm, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
    norm: str | None,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        make_norm(norm, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class MAPMAG(nn.Module):
    """Modality-Adaptive Pathways for Multimodal Attributed Graphs."""

    def __init__(self, cfg, data_info: dict):
        super().__init__()
        input_dim = int(data_info["input_dim"])
        hidden_dim = int(cfg.model.get("hidden_dim", 256))
        dropout = float(cfg.model.get("dropout", 0.2))
        norm = cfg.model.get("norm", "layernorm")

        self.text_dim = int(data_info.get("text_dim", 0) or 0)
        self.visual_dim = int(data_info.get("visual_dim", 0) or 0)
        if self.text_dim <= 0 or self.visual_dim <= 0:
            self.text_dim = input_dim // 2
            self.visual_dim = input_dim - self.text_dim
        if self.text_dim + self.visual_dim > input_dim:
            raise ValueError(
                f"text_dim+visual_dim={self.text_dim + self.visual_dim} exceeds input_dim={input_dim}"
            )

        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.num_hops = int(cfg.model.get("num_hops", cfg.model.get("num_layers", 2)))
        self.restart = float(cfg.model.get("restart", 0.15))
        self.eps = float(cfg.model.get("eps", 1e-8))
        self.diffusion_add_self_loops = bool(cfg.model.get("diffusion_add_self_loops", True))
        self.requires_full_graph_training = bool(cfg.model.get("full_graph_training", False))

        self.use_self_path = bool(cfg.model.get("use_self_path", True))
        self.use_structure_path = bool(cfg.model.get("use_structure_path", True))
        self.use_prototype_path = bool(cfg.model.get("use_prototype_path", True))
        self.use_preference_router = bool(cfg.model.get("use_preference_router", True))
        self.structure_low_pass_only = bool(cfg.model.get("structure_low_pass_only", False))
        self.self_only = bool(cfg.model.get("self_only", False))
        if self.self_only:
            self.use_self_path = True
            self.use_structure_path = False
            self.use_prototype_path = False
            self.use_preference_router = False
        if not (self.use_self_path or self.use_structure_path or self.use_prototype_path):
            raise ValueError("At least one MAP-MAG path must be enabled")

        rel_hidden_dim = int(cfg.model.get("rel_hidden_dim", max(hidden_dim // 2, 16)))
        router_hidden_dim = int(cfg.model.get("router_hidden_dim", hidden_dim))
        freq_hidden_dim = int(cfg.model.get("freq_hidden_dim", max(hidden_dim // 2, 16)))
        self.num_prototypes = int(cfg.model.get("num_prototypes", 32))
        if self.num_prototypes < 1:
            raise ValueError(f"num_prototypes must be >= 1, got {self.num_prototypes}")

        self.lambda_proto = float(cfg.model.get("lambda_proto", 0.01))
        self.lambda_gate = float(cfg.model.get("lambda_gate", 0.001))
        self.use_gate_loss = bool(cfg.model.get("use_gate_loss", True))
        self.reliability_min = float(cfg.model.get("reliability_min", 0.0))
        if not (0.0 <= self.reliability_min <= 1.0):
            raise ValueError(f"reliability_min must be in [0, 1], got {self.reliability_min}")
        self.router_temperature = float(cfg.model.get("router_temperature", 2.0))
        if self.router_temperature <= 0.0:
            raise ValueError(f"router_temperature must be positive, got {self.router_temperature}")
        self.gate_warmup_epochs = int(cfg.model.get("gate_warmup_epochs", 0))
        if self.gate_warmup_epochs < 0:
            raise ValueError(f"gate_warmup_epochs must be >= 0, got {self.gate_warmup_epochs}")
        self.gamma_min = float(cfg.model.get("gamma_min", 0.05))
        self.gamma_max = float(cfg.model.get("gamma_max", 0.95))
        if not (0.0 <= self.gamma_min < self.gamma_max <= 1.0):
            raise ValueError(
                f"Expected 0 <= gamma_min < gamma_max <= 1, got "
                f"gamma_min={self.gamma_min}, gamma_max={self.gamma_max}"
            )
        self.gamma_temperature = float(cfg.model.get("gamma_temperature", 1.0))
        if self.gamma_temperature <= 0.0:
            raise ValueError(f"gamma_temperature must be positive, got {self.gamma_temperature}")
        self.gamma_warmup_epochs = int(cfg.model.get("gamma_warmup_epochs", 0))
        if self.gamma_warmup_epochs < 0:
            raise ValueError(f"gamma_warmup_epochs must be >= 0, got {self.gamma_warmup_epochs}")
        fixed_gamma = cfg.model.get("fixed_gamma", None)
        if isinstance(fixed_gamma, str) and fixed_gamma.strip().lower() in {"none", "null"}:
            fixed_gamma = None
        self.fixed_gamma = None if fixed_gamma is None else float(fixed_gamma)
        if self.fixed_gamma is not None and not (0.0 <= self.fixed_gamma <= 1.0):
            raise ValueError(f"fixed_gamma must be in [0, 1] or null, got {self.fixed_gamma}")

        self.text_proj = ProjectionMLP(self.text_dim, hidden_dim, dropout, norm)
        self.visual_proj = ProjectionMLP(self.visual_dim, hidden_dim, dropout, norm)

        rel_in_dim = hidden_dim * 4
        self.text_reliability = _make_mlp(rel_in_dim, rel_hidden_dim, 1, dropout, norm)
        self.visual_reliability = _make_mlp(rel_in_dim, rel_hidden_dim, 1, dropout, norm)

        router_in_dim = hidden_dim * 4 + 3
        self.router = _make_mlp(router_in_dim, router_hidden_dim, 3, dropout, norm)
        self.self_path = _make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, dropout, norm)
        self.freq_gate = _make_mlp(hidden_dim + 3, freq_hidden_dim, 1, dropout, norm)

        self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, hidden_dim))
        nn.init.xavier_normal_(self.prototypes)

        self.output_mlp = _make_mlp(hidden_dim, hidden_dim, hidden_dim, dropout, norm)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.register_buffer("_current_epoch", torch.zeros((), dtype=torch.long), persistent=True)

    def set_epoch(self, epoch: int) -> None:
        """Allow trainers to drive optional MAP-MAG warmup schedules."""
        self._current_epoch.fill_(max(int(epoch), 0))

    def _warmup_alpha(self, warmup_epochs: int) -> float:
        if warmup_epochs <= 0:
            return 1.0
        epoch = int(self._current_epoch.item())
        if epoch <= 0:
            return 1.0
        if epoch > warmup_epochs:
            return 1.0
        return max(0.0, min(1.0, float(epoch - 1) / float(warmup_epochs)))

    def _split_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dtype == torch.long:
            x = x.float()
        text_feat = x[:, : self.text_dim]
        visual_feat = x[:, self.text_dim : self.text_dim + self.visual_dim]
        return text_feat, visual_feat

    def _edge_index_or_empty(
        self,
        edge_index: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if edge_index is None:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return edge_index.to(device=device, dtype=torch.long)

    def _neighbor_mean(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(h)
        src, dst = edge_index
        return scatter(h[src], dst, dim=0, dim_size=h.size(0), reduce="mean")

    def _degree_feature(self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> torch.Tensor:
        if edge_index.numel() == 0:
            degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        else:
            ones = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
            degree = scatter(ones, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
        return torch.log1p(degree).unsqueeze(-1)

    def _edge_cosine_mean(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h.new_zeros((h.size(0), 1))
        src, dst = edge_index
        sim = F.cosine_similarity(h[src], h[dst], dim=-1, eps=self.eps)
        sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=-1.0)
        out = scatter(sim, dst, dim=0, dim_size=h.size(0), reduce="mean")
        return out.clamp(-1.0, 1.0).unsqueeze(-1)

    def _reliability_gate(
        self,
        h: torch.Tensor,
        mean_h: torch.Tensor,
        mlp: nn.Module,
    ) -> torch.Tensor:
        rel_input = torch.cat([h, mean_h, h - mean_h, h * mean_h], dim=-1)
        gate = torch.sigmoid(mlp(rel_input))
        if self.reliability_min <= 0.0:
            return gate
        return self.reliability_min + (1.0 - self.reliability_min) * gate

    def _diffuse(self, h0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.num_hops <= 0:
            return h0

        norm_edge_index, norm_edge_weight = gcn_norm(
            edge_index,
            edge_weight=None,
            num_nodes=int(h0.size(0)),
            improved=False,
            add_self_loops=self.diffusion_add_self_loops,
            flow="source_to_target",
            dtype=h0.dtype,
        )
        if norm_edge_weight is None or norm_edge_index.numel() == 0:
            return h0

        row, col = norm_edge_index
        h = h0
        restart = min(max(self.restart, 0.0), 1.0)
        for _ in range(self.num_hops):
            propagated = scatter(
                h[row] * norm_edge_weight.view(-1, 1),
                col,
                dim=0,
                dim_size=h0.size(0),
                reduce="sum",
            )
            h = (1.0 - restart) * propagated + restart * h0
        return h

    def _prototype_path(self, h0: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(float(self.hidden_dim))
        attn = torch.softmax(torch.matmul(h0, self.prototypes.t()) / scale, dim=-1)
        return torch.matmul(attn, self.prototypes)

    def _prototype_loss(self, ref: torch.Tensor) -> torch.Tensor:
        if not self.use_prototype_path or self.lambda_proto <= 0.0:
            return ref.new_tensor(0.0)
        proto = F.normalize(self.prototypes, p=2, dim=-1, eps=self.eps)
        similarity = torch.matmul(proto, proto.t())
        identity = torch.eye(self.num_prototypes, dtype=similarity.dtype, device=similarity.device)
        return (similarity - identity).pow(2).mean()

    def _active_path_mask(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [self.use_self_path, self.use_structure_path, self.use_prototype_path],
            dtype=dtype,
            device=device,
        ).view(1, 3)

    def _path_weights(self, router_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        active_count = int(mask.sum().item())
        if active_count <= 0:
            raise ValueError("At least one MAP-MAG path must be enabled")

        uniform = mask.expand(router_logits.size(0), -1) / float(active_count)
        if self.use_preference_router and active_count > 1:
            weights = torch.softmax(router_logits / self.router_temperature, dim=-1)
            weights = weights * mask
            learned = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            alpha = self._warmup_alpha(self.gate_warmup_epochs)
            if alpha < 1.0:
                return (1.0 - alpha) * uniform + alpha * learned
            return learned
        return uniform

    def _frequency_gate(self, gamma_input: torch.Tensor) -> torch.Tensor:
        if self.fixed_gamma is not None:
            return gamma_input.new_full((gamma_input.size(0), 1), self.fixed_gamma)

        gamma_raw = torch.sigmoid(self.freq_gate(gamma_input) / self.gamma_temperature)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * gamma_raw
        alpha = self._warmup_alpha(self.gamma_warmup_epochs)
        if alpha < 1.0:
            gamma = (1.0 - alpha) * 0.5 + alpha * gamma
        return gamma

    def _gate_loss(self, p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        active_count = int(mask.sum().item())
        if not self.use_gate_loss or self.lambda_gate <= 0.0 or active_count <= 1:
            return p.new_tensor(0.0)
        active = mask.view(-1).bool()
        mean_p = p[:, active].mean(dim=0).clamp_min(self.eps)
        return torch.sum(mean_p * torch.log(mean_p * float(active_count)))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        edge_index = self._edge_index_or_empty(edge_index, x.device)

        x_t, x_v = self._split_features(x)
        h_t = self.text_proj(x_t)
        h_v = self.visual_proj(x_v)

        mean_t = self._neighbor_mean(h_t, edge_index)
        mean_v = self._neighbor_mean(h_v, edge_index)
        r_t = self._reliability_gate(h_t, mean_t, self.text_reliability)
        r_v = self._reliability_gate(h_v, mean_v, self.visual_reliability)
        h_t_rel = r_t * h_t
        h_v_rel = r_v * h_v

        s_t = self._edge_cosine_mean(h_t, edge_index)
        s_v = self._edge_cosine_mean(h_v, edge_index)
        log_degree = self._degree_feature(edge_index, int(x.size(0)), h_t.dtype)

        z_self = self.self_path(torch.cat([h_t_rel, h_v_rel], dim=-1))

        lambda_denom = (r_t + r_v).clamp_min(self.eps)
        lambda_t = r_t / lambda_denom
        lambda_v = r_v / lambda_denom
        h0 = lambda_t * h_t_rel + lambda_v * h_v_rel

        z_low = self._diffuse(h0, edge_index)
        z_high = h0 - z_low
        gamma_input = torch.cat([h0, s_t, s_v, log_degree], dim=-1)
        gamma = self._frequency_gate(gamma_input)
        if self.structure_low_pass_only:
            z_struct = z_low
        else:
            z_struct = gamma * z_low + (1.0 - gamma) * z_high

        if self.use_prototype_path:
            z_proto = self._prototype_path(h0)
        else:
            z_proto = torch.zeros_like(z_self)

        router_input = torch.cat(
            [h_t_rel, h_v_rel, torch.abs(h_t_rel - h_v_rel), h_t_rel * h_v_rel, s_t, s_v, log_degree],
            dim=-1,
        )
        router_logits = self.router(router_input)
        path_mask = self._active_path_mask(x.device, h_t.dtype)
        p = self._path_weights(router_logits, path_mask)

        z = (
            p[:, 0:1] * z_self
            + p[:, 1:2] * z_struct
            + p[:, 2:3] * z_proto
        )
        z = self.output_norm(self.output_mlp(z) + z)
        z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)

        proto_loss = self._prototype_loss(z)
        gate_loss = self._gate_loss(p, path_mask)
        aux_loss = self.lambda_proto * proto_loss + self.lambda_gate * gate_loss
        aux_info = {
            "mean_r_text": r_t.detach().mean(),
            "mean_r_visual": r_v.detach().mean(),
            "mean_p_self": p[:, 0].detach().mean(),
            "mean_p_struct": p[:, 1].detach().mean(),
            "mean_p_proto": p[:, 2].detach().mean(),
            "mean_gamma": gamma.detach().mean(),
            "proto_loss": proto_loss.detach(),
            "gate_loss": gate_loss.detach(),
        }
        return z, None, None, aux_loss, aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        del batch_size
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        edge_index = edge_index.to(device) if edge_index is not None else None
        z, _, _, _, _ = self(x.to(device), edge_index)
        return z.detach().cpu()


Model = MAPMAG
