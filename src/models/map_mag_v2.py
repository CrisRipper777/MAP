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


class MAPMAGV2(nn.Module):
    """Experimental MAP-MAG v2 encoder.

    v2 keeps MAP-MAG as a task-agnostic node encoder while replacing v1's
    symmetric three-path fusion with reliability fusion, semantic edge
    reweighting, and a low-pass diffusion backbone.
    """

    EDGE_WEIGHT_MODES = {
        "avg_cos",
        "text_only",
        "visual_only",
        "reliability_aware",
        "learnable_scalar",
    }
    STRUCTURE_MODES = {"lowpass", "lowpass_residual", "original_v1_gamma"}

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
        if self.num_hops < 0:
            raise ValueError(f"num_hops must be >= 0, got {self.num_hops}")
        self.restart = float(cfg.model.get("restart", 0.15))
        self.eps = float(cfg.model.get("eps", 1e-8))
        self.diffusion_add_self_loops = bool(cfg.model.get("diffusion_add_self_loops", True))
        self.requires_full_graph_training = bool(cfg.model.get("full_graph_training", True))

        self.use_reliability = bool(cfg.model.get("use_reliability", True))
        self.reliability_min = float(cfg.model.get("reliability_min", 0.1))
        if not (0.0 <= self.reliability_min <= 1.0):
            raise ValueError(f"reliability_min must be in [0, 1], got {self.reliability_min}")
        self.reliability_mode = str(cfg.model.get("reliability_mode", "single")).strip().lower()
        if self.reliability_mode not in {"single", "double"}:
            raise ValueError(
                "model.reliability_mode must be one of {'single', 'double'}, "
                f"got {self.reliability_mode!r}"
            )

        self.use_semantic_edge_weight = bool(cfg.model.get("use_semantic_edge_weight", True))
        self.edge_weight_mode = str(cfg.model.get("edge_weight_mode", "avg_cos")).strip().lower()
        if self.edge_weight_mode not in self.EDGE_WEIGHT_MODES:
            valid = ", ".join(sorted(self.EDGE_WEIGHT_MODES))
            raise ValueError(f"model.edge_weight_mode must be one of [{valid}], got {self.edge_weight_mode!r}")
        self.edge_weight_min = float(cfg.model.get("edge_weight_min", 0.1))
        if not (0.0 <= self.edge_weight_min <= 1.0):
            raise ValueError(f"edge_weight_min must be in [0, 1], got {self.edge_weight_min}")
        self.edge_weight_temperature = float(cfg.model.get("edge_weight_temperature", 2.0))
        if self.edge_weight_temperature <= 0.0:
            raise ValueError(f"edge_weight_temperature must be positive, got {self.edge_weight_temperature}")

        self.structure_mode = str(cfg.model.get("structure_mode", "lowpass")).strip().lower()
        if self.structure_mode not in self.STRUCTURE_MODES:
            valid = ", ".join(sorted(self.STRUCTURE_MODES))
            raise ValueError(f"model.structure_mode must be one of [{valid}], got {self.structure_mode!r}")
        self.residual_eta_max = float(cfg.model.get("residual_eta_max", 0.2))
        if self.residual_eta_max < 0.0:
            raise ValueError(f"residual_eta_max must be >= 0, got {self.residual_eta_max}")

        self.use_self_residual = bool(cfg.model.get("use_self_residual", False))
        self.self_residual_max = float(cfg.model.get("self_residual_max", 0.2))
        if self.self_residual_max < 0.0:
            raise ValueError(f"self_residual_max must be >= 0, got {self.self_residual_max}")

        self.use_prototype_path = bool(cfg.model.get("use_prototype_path", False))
        self.num_prototypes = int(cfg.model.get("num_prototypes", 16))
        if self.num_prototypes < 1:
            raise ValueError(f"num_prototypes must be >= 1, got {self.num_prototypes}")
        self.prototype_residual_max = float(cfg.model.get("prototype_residual_max", 0.2))
        if self.prototype_residual_max < 0.0:
            raise ValueError(f"prototype_residual_max must be >= 0, got {self.prototype_residual_max}")

        self.lambda_proto = float(cfg.model.get("lambda_proto", 0.0))
        self.lambda_edge_reg = float(cfg.model.get("lambda_edge_reg", 0.0))
        if self.lambda_proto < 0.0:
            raise ValueError(f"lambda_proto must be >= 0, got {self.lambda_proto}")
        if self.lambda_edge_reg < 0.0:
            raise ValueError(f"lambda_edge_reg must be >= 0, got {self.lambda_edge_reg}")

        self.gamma_min = float(cfg.model.get("gamma_min", 0.05))
        self.gamma_max = float(cfg.model.get("gamma_max", 0.95))
        if not (0.0 <= self.gamma_min < self.gamma_max <= 1.0):
            raise ValueError(
                f"Expected 0 <= gamma_min < gamma_max <= 1, got "
                f"gamma_min={self.gamma_min}, gamma_max={self.gamma_max}"
            )
        self.gamma_temperature = float(cfg.model.get("gamma_temperature", 3.0))
        if self.gamma_temperature <= 0.0:
            raise ValueError(f"gamma_temperature must be positive, got {self.gamma_temperature}")
        fixed_gamma = cfg.model.get("fixed_gamma", None)
        if isinstance(fixed_gamma, str) and fixed_gamma.strip().lower() in {"none", "null"}:
            fixed_gamma = None
        self.fixed_gamma = None if fixed_gamma is None else float(fixed_gamma)
        if self.fixed_gamma is not None and not (0.0 <= self.fixed_gamma <= 1.0):
            raise ValueError(f"fixed_gamma must be in [0, 1] or null, got {self.fixed_gamma}")

        rel_hidden_dim = int(cfg.model.get("rel_hidden_dim", max(hidden_dim // 2, 16)))
        eta_hidden_dim = int(cfg.model.get("eta_hidden_dim", max(hidden_dim // 2, 16)))
        self_hidden_dim = int(cfg.model.get("self_hidden_dim", max(hidden_dim // 2, 16)))
        gamma_hidden_dim = int(cfg.model.get("gamma_hidden_dim", max(hidden_dim // 2, 16)))

        self.text_proj = ProjectionMLP(self.text_dim, hidden_dim, dropout, norm)
        self.visual_proj = ProjectionMLP(self.visual_dim, hidden_dim, dropout, norm)

        rel_in_dim = hidden_dim * 4
        self.text_reliability = _make_mlp(rel_in_dim, rel_hidden_dim, 1, dropout, norm)
        self.visual_reliability = _make_mlp(rel_in_dim, rel_hidden_dim, 1, dropout, norm)

        learnable_init = float(cfg.model.get("edge_weight_learnable_init", 1.0))
        self.edge_weight_text_scale = nn.Parameter(torch.tensor(learnable_init, dtype=torch.float32))
        self.edge_weight_visual_scale = nn.Parameter(torch.tensor(learnable_init, dtype=torch.float32))
        self.edge_weight_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

        self.residual_mlp = _make_mlp(hidden_dim, hidden_dim, hidden_dim, dropout, norm)
        self.eta_gate = _make_mlp(hidden_dim + 5, eta_hidden_dim, 1, dropout, norm)
        self.self_path = _make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, dropout, norm)
        self.self_gate = _make_mlp(hidden_dim + 3, self_hidden_dim, 1, dropout, norm)
        self.freq_gate = _make_mlp(hidden_dim + 3, gamma_hidden_dim, 1, dropout, norm)

        if self.use_prototype_path:
            self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, hidden_dim))
            nn.init.xavier_normal_(self.prototypes)

        self.output_mlp = _make_mlp(hidden_dim, hidden_dim, hidden_dim, dropout, norm)
        self.output_norm = nn.LayerNorm(hidden_dim)

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

    def _degree(self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        ones = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
        return scatter(ones, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")

    def _degree_feature(self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.log1p(self._degree(edge_index, num_nodes, dtype)).unsqueeze(-1)

    def _edge_cosine_values(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h.new_empty((0,))
        src, dst = edge_index
        sim = F.cosine_similarity(h[src], h[dst], dim=-1, eps=self.eps)
        sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=-1.0)
        return sim.clamp(-1.0, 1.0)

    def _edge_cosine_mean(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h.new_zeros((h.size(0), 1))
        sim = self._edge_cosine_values(h, edge_index)
        out = scatter(sim, edge_index[1], dim=0, dim_size=h.size(0), reduce="mean")
        return out.clamp(-1.0, 1.0).unsqueeze(-1)

    def _reliability_gate(
        self,
        h: torch.Tensor,
        mean_h: torch.Tensor,
        mlp: nn.Module,
    ) -> torch.Tensor:
        if not self.use_reliability:
            return torch.ones((h.size(0), 1), dtype=h.dtype, device=h.device)
        rel_input = torch.cat([h, mean_h, h - mean_h, h * mean_h], dim=-1)
        gate = torch.sigmoid(mlp(rel_input))
        if self.reliability_min <= 0.0:
            return gate
        return self.reliability_min + (1.0 - self.reliability_min) * gate

    def _reliability_fusion(
        self,
        h_t: torch.Tensor,
        h_v: torch.Tensor,
        r_t: torch.Tensor,
        r_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lambda_denom = (r_t + r_v).clamp_min(self.eps)
        lambda_t = r_t / lambda_denom
        lambda_v = r_v / lambda_denom
        if self.reliability_mode == "double":
            return lambda_t * (r_t * h_t) + lambda_v * (r_v * h_v), lambda_t, lambda_v
        return lambda_t * h_t + lambda_v * h_v, lambda_t, lambda_v

    def _zero_edge_stats(self, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = ref.new_tensor(0.0)
        return {
            "mean_edge_weight": zero,
            "std_edge_weight": zero,
            "min_edge_weight": zero,
            "max_edge_weight": zero,
            "mean_cos_text": zero,
            "mean_cos_visual": zero,
        }

    def _semantic_edge_weight(
        self,
        h_t: torch.Tensor,
        h_v: torch.Tensor,
        r_t: torch.Tensor,
        r_v: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if edge_index.numel() == 0:
            empty = h_t.new_empty((0,))
            return empty, empty, empty, self._zero_edge_stats(h_t)

        cos_t = self._edge_cosine_values(h_t, edge_index)
        cos_v = self._edge_cosine_values(h_v, edge_index)
        if not self.use_semantic_edge_weight:
            edge_weight = torch.ones(edge_index.size(1), dtype=h_t.dtype, device=h_t.device)
        elif self.edge_weight_mode == "avg_cos":
            sim = 0.5 * (cos_t + cos_v)
            edge_weight = self._edge_weight_from_similarity(sim)
        elif self.edge_weight_mode == "text_only":
            edge_weight = self._edge_weight_from_similarity(cos_t)
        elif self.edge_weight_mode == "visual_only":
            edge_weight = self._edge_weight_from_similarity(cos_v)
        elif self.edge_weight_mode == "reliability_aware":
            src, dst = edge_index
            rel_t = 0.5 * (r_t[src].squeeze(-1) + r_t[dst].squeeze(-1))
            rel_v = 0.5 * (r_v[src].squeeze(-1) + r_v[dst].squeeze(-1))
            sim = (rel_t * cos_t + rel_v * cos_v) / (rel_t + rel_v).clamp_min(self.eps)
            edge_weight = self._edge_weight_from_similarity(sim)
        elif self.edge_weight_mode == "learnable_scalar":
            sim = (
                self.edge_weight_text_scale.to(dtype=h_t.dtype) * cos_t
                + self.edge_weight_visual_scale.to(dtype=h_t.dtype) * cos_v
                + self.edge_weight_bias.to(dtype=h_t.dtype)
            )
            edge_weight = self._edge_weight_from_similarity(sim)
        else:
            raise RuntimeError(f"Unhandled edge_weight_mode: {self.edge_weight_mode}")

        edge_weight = torch.nan_to_num(
            edge_weight,
            nan=1.0,
            posinf=1.0,
            neginf=self.edge_weight_min,
        ).clamp(self.edge_weight_min, 1.0)
        stats = self._edge_stats(edge_weight, cos_t, cos_v)
        return edge_weight, cos_t, cos_v, stats

    def _edge_weight_from_similarity(self, sim: torch.Tensor) -> torch.Tensor:
        scaled = sim / self.edge_weight_temperature
        return self.edge_weight_min + (1.0 - self.edge_weight_min) * torch.sigmoid(scaled)

    def _edge_stats(
        self,
        edge_weight: torch.Tensor,
        cos_t: torch.Tensor,
        cos_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if edge_weight.numel() == 0:
            return self._zero_edge_stats(edge_weight)
        return {
            "mean_edge_weight": edge_weight.mean(),
            "std_edge_weight": edge_weight.std(unbiased=False),
            "min_edge_weight": edge_weight.min(),
            "max_edge_weight": edge_weight.max(),
            "mean_cos_text": cos_t.mean() if cos_t.numel() > 0 else edge_weight.new_tensor(0.0),
            "mean_cos_visual": cos_v.mean() if cos_v.numel() > 0 else edge_weight.new_tensor(0.0),
        }

    def _diffuse(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.num_hops <= 0:
            return h0

        norm_edge_index, norm_edge_weight = gcn_norm(
            edge_index,
            edge_weight=edge_weight,
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

    def _frequency_gate(self, gamma_input: torch.Tensor) -> torch.Tensor:
        if self.fixed_gamma is not None:
            return gamma_input.new_full((gamma_input.size(0), 1), self.fixed_gamma)
        gamma_raw = torch.sigmoid(self.freq_gate(gamma_input) / self.gamma_temperature)
        return self.gamma_min + (self.gamma_max - self.gamma_min) * gamma_raw

    def _prototype_path(self, h0: torch.Tensor) -> torch.Tensor:
        if not self.use_prototype_path:
            return torch.zeros_like(h0)
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

    def _edge_reg_loss(self, edge_weight: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if self.lambda_edge_reg <= 0.0 or edge_weight.numel() == 0:
            return ref.new_tensor(0.0)
        return (edge_weight - 1.0).pow(2).mean()

    def _encode_components(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor | dict]:
        x_t, x_v = self._split_features(x)
        h_t = self.text_proj(x_t)
        h_v = self.visual_proj(x_v)

        mean_t = self._neighbor_mean(h_t, edge_index)
        mean_v = self._neighbor_mean(h_v, edge_index)
        r_t = self._reliability_gate(h_t, mean_t, self.text_reliability)
        r_v = self._reliability_gate(h_v, mean_v, self.visual_reliability)
        h0, lambda_t, lambda_v = self._reliability_fusion(h_t, h_v, r_t, r_v)

        s_t = self._edge_cosine_mean(h_t, edge_index)
        s_v = self._edge_cosine_mean(h_v, edge_index)
        degree = self._degree(edge_index, int(x.size(0)), h_t.dtype)
        log_degree = torch.log1p(degree).unsqueeze(-1)
        edge_weight, cos_t_edges, cos_v_edges, edge_stats = self._semantic_edge_weight(
            h_t,
            h_v,
            r_t,
            r_v,
            edge_index,
        )

        z_low = self._diffuse(h0, edge_index, edge_weight=edge_weight)
        z_res = h0 - z_low
        eta = h0.new_zeros((h0.size(0), 1))
        gamma = h0.new_ones((h0.size(0), 1))
        if self.structure_mode == "lowpass":
            z_struct = z_low
        elif self.structure_mode == "lowpass_residual":
            eta_input = torch.cat([h0, s_t, s_v, log_degree, r_t, r_v], dim=-1)
            eta = self.residual_eta_max * torch.sigmoid(self.eta_gate(eta_input))
            z_struct = z_low + eta * self.residual_mlp(z_res)
        elif self.structure_mode == "original_v1_gamma":
            gamma_input = torch.cat([h0, s_t, s_v, log_degree], dim=-1)
            gamma = self._frequency_gate(gamma_input)
            z_struct = gamma * z_low + (1.0 - gamma) * z_res
        else:
            raise RuntimeError(f"Unhandled structure_mode: {self.structure_mode}")

        if self.use_prototype_path:
            z_proto = self._prototype_path(h0)
            z_struct = z_struct + self.prototype_residual_max * z_proto
        else:
            z_proto = torch.zeros_like(z_struct)

        alpha_self = h0.new_zeros((h0.size(0), 1))
        z_self = h0.new_zeros(h0.shape)
        z_input = z_struct
        if self.use_self_residual:
            z_self = self.self_path(torch.cat([h_t, h_v], dim=-1))
            alpha_input = torch.cat([h0, r_t, r_v, log_degree], dim=-1)
            alpha_self = self.self_residual_max * torch.sigmoid(self.self_gate(alpha_input))
            z_input = z_struct + alpha_self * z_self

        return {
            "h_t": h_t,
            "h_v": h_v,
            "h0": h0,
            "r_t": r_t,
            "r_v": r_v,
            "lambda_t": lambda_t,
            "lambda_v": lambda_v,
            "s_t": s_t,
            "s_v": s_v,
            "degree": degree,
            "log_degree": log_degree,
            "edge_weight": edge_weight,
            "cos_t_edges": cos_t_edges,
            "cos_v_edges": cos_v_edges,
            "edge_stats": edge_stats,
            "z_low": z_low,
            "z_res": z_res,
            "z_struct": z_struct,
            "z_proto": z_proto,
            "z_self": z_self,
            "z_input": z_input,
            "eta": eta,
            "gamma": gamma,
            "alpha_self": alpha_self,
        }

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        components = self._encode_components(x, edge_index)

        z_struct = components["z_struct"]
        z_input = components["z_input"]
        z = self.output_norm(self.output_mlp(z_input) + z_struct)
        z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)

        proto_loss = self._prototype_loss(z)
        edge_reg_loss = self._edge_reg_loss(components["edge_weight"], z)
        aux_loss = self.lambda_proto * proto_loss + self.lambda_edge_reg * edge_reg_loss

        edge_stats = components["edge_stats"]
        aux_info = {
            "mean_r_text": components["r_t"].detach().mean(),
            "mean_r_visual": components["r_v"].detach().mean(),
            "mean_edge_weight": edge_stats["mean_edge_weight"].detach(),
            "std_edge_weight": edge_stats["std_edge_weight"].detach(),
            "min_edge_weight": edge_stats["min_edge_weight"].detach(),
            "max_edge_weight": edge_stats["max_edge_weight"].detach(),
            "mean_cos_text": edge_stats["mean_cos_text"].detach(),
            "mean_cos_visual": edge_stats["mean_cos_visual"].detach(),
            "mean_lambda_text": components["lambda_t"].detach().mean(),
            "mean_lambda_visual": components["lambda_v"].detach().mean(),
            "mean_degree": components["degree"].detach().mean(),
            "structure_mode": self.structure_mode,
            "proto_loss": proto_loss.detach(),
            "prototype_loss": proto_loss.detach(),
            "edge_reg_loss": edge_reg_loss.detach(),
        }
        if self.use_self_residual:
            aux_info["mean_alpha_self"] = components["alpha_self"].detach().mean()
        if self.structure_mode == "lowpass_residual":
            aux_info["mean_eta_residual"] = components["eta"].detach().mean()
        if self.structure_mode == "original_v1_gamma":
            aux_info["mean_gamma"] = components["gamma"].detach().mean()
        if self.use_prototype_path:
            aux_info["mean_p_proto"] = z.new_tensor(self.prototype_residual_max)

        return z, None, None, aux_loss, aux_info

    @torch.no_grad()
    def node_aux_stats(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        x = x.to(device)
        edge_index = self._edge_index_or_empty(edge_index, device)
        components = self._encode_components(x, edge_index)
        h_t = components["h_t"]
        h_v = components["h_v"]
        text_visual_cosine = F.cosine_similarity(h_t, h_v, dim=-1, eps=self.eps)
        text_visual_cosine = torch.nan_to_num(text_visual_cosine, nan=0.0, posinf=1.0, neginf=-1.0)

        num_nodes = int(x.size(0))
        p_struct = torch.ones(num_nodes, dtype=h_t.dtype, device=h_t.device)
        p_self = components["alpha_self"].squeeze(-1)
        if self.use_prototype_path:
            p_proto = h_t.new_full((num_nodes,), self.prototype_residual_max)
        else:
            p_proto = h_t.new_zeros((num_nodes,))

        stats = {
            "degree": components["degree"].detach().cpu(),
            "s_text": components["s_t"].squeeze(-1).detach().cpu(),
            "s_visual": components["s_v"].squeeze(-1).detach().cpu(),
            "r_text": components["r_t"].squeeze(-1).detach().cpu(),
            "r_visual": components["r_v"].squeeze(-1).detach().cpu(),
            "p_self": p_self.detach().cpu(),
            "p_struct": p_struct.detach().cpu(),
            "p_proto": p_proto.detach().cpu(),
            "gamma": components["gamma"].squeeze(-1).detach().cpu(),
            "text_visual_cosine": text_visual_cosine.detach().cpu(),
            "lambda_text": components["lambda_t"].squeeze(-1).detach().cpu(),
            "lambda_visual": components["lambda_v"].squeeze(-1).detach().cpu(),
            "eta_residual": components["eta"].squeeze(-1).detach().cpu(),
            "alpha_self": components["alpha_self"].squeeze(-1).detach().cpu(),
        }
        if was_training:
            self.train()
        return stats

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


Model = MAPMAGV2
