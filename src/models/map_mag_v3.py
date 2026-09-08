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


class MAPMAGV3(nn.Module):
    """Modality-Preference Guided Dual Semantic Propagation.

    Text and visual features are propagated on separate sparse semantic graphs.
    A node-level modality router fuses the two propagated representations.
    Conflict propagation, self residuals, and modality-specific prototypes are
    implemented as controlled optional modules and are disabled in the core
    configuration.
    """

    EDGE_WEIGHT_MODES = {
        "separate_cos",
        "reliability_scaled",
        "learnable_scalar",
        "shared_avg_cos",
        "raw_uniform",
    }
    PROPAGATION_MODES = {"shared_fused", "modality_specific"}
    MODALITY_FUSION_MODES = {
        "reliability",
        "reliability_residual",
        "learned_router",
        "uniform",
    }
    RELIABILITY_MODES = {"single", "double", "none"}
    PROTOTYPE_INIT_MODES = {"random", "kmeans"}

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
        if not (0.0 <= self.restart <= 1.0):
            raise ValueError(f"restart must be in [0, 1], got {self.restart}")
        self.eps = float(cfg.model.get("eps", 1e-8))
        self.diffusion_add_self_loops = bool(cfg.model.get("diffusion_add_self_loops", True))
        self.requires_full_graph_training = bool(cfg.model.get("full_graph_training", True))

        self.use_reliability = bool(cfg.model.get("use_reliability", True))
        self.reliability_min = float(cfg.model.get("reliability_min", 0.1))
        if not (0.0 <= self.reliability_min <= 1.0):
            raise ValueError(f"reliability_min must be in [0, 1], got {self.reliability_min}")
        self.reliability_mode = str(
            cfg.model.get("reliability_mode", "single")
        ).strip().lower()
        if not self.use_reliability:
            self.reliability_mode = "none"
        if self.reliability_mode not in self.RELIABILITY_MODES:
            valid = ", ".join(sorted(self.RELIABILITY_MODES))
            raise ValueError(
                f"model.reliability_mode must be one of [{valid}], got {self.reliability_mode!r}"
            )

        self.use_modality_specific_edges = bool(
            cfg.model.get("use_modality_specific_edges", True)
        )
        self.edge_weight_mode = str(
            cfg.model.get("edge_weight_mode", "separate_cos")
        ).strip().lower()
        if self.edge_weight_mode not in self.EDGE_WEIGHT_MODES:
            valid = ", ".join(sorted(self.EDGE_WEIGHT_MODES))
            raise ValueError(
                f"model.edge_weight_mode must be one of [{valid}], got {self.edge_weight_mode!r}"
            )
        self.edge_weight_min = float(cfg.model.get("edge_weight_min", 0.1))
        if not (0.0 <= self.edge_weight_min <= 1.0):
            raise ValueError(f"edge_weight_min must be in [0, 1], got {self.edge_weight_min}")
        self.edge_weight_temperature = float(cfg.model.get("edge_weight_temperature", 2.0))
        if self.edge_weight_temperature <= 0.0:
            raise ValueError(
                f"edge_weight_temperature must be positive, got {self.edge_weight_temperature}"
            )

        self.propagation_mode = str(
            cfg.model.get("propagation_mode", "modality_specific")
        ).strip().lower()
        if self.propagation_mode not in self.PROPAGATION_MODES:
            valid = ", ".join(sorted(self.PROPAGATION_MODES))
            raise ValueError(
                f"model.propagation_mode must be one of [{valid}], got {self.propagation_mode!r}"
            )

        self.use_conflict_channel = bool(cfg.model.get("use_conflict_channel", False))
        self.conflict_min = float(cfg.model.get("conflict_min", 0.0))
        self.conflict_scale = float(cfg.model.get("conflict_scale", 1.0))
        self.conflict_eta_max = float(cfg.model.get("conflict_eta_max", 0.2))
        if min(self.conflict_min, self.conflict_scale, self.conflict_eta_max) < 0.0:
            raise ValueError("conflict_min, conflict_scale, and conflict_eta_max must be non-negative")

        self.modality_fusion_mode = str(
            cfg.model.get("modality_fusion_mode", "learned_router")
        ).strip().lower()
        if self.modality_fusion_mode not in self.MODALITY_FUSION_MODES:
            valid = ", ".join(sorted(self.MODALITY_FUSION_MODES))
            raise ValueError(
                f"model.modality_fusion_mode must be one of [{valid}], "
                f"got {self.modality_fusion_mode!r}"
            )
        self.modality_router_temperature = float(
            cfg.model.get("modality_router_temperature", 2.0)
        )
        if self.modality_router_temperature <= 0.0:
            raise ValueError(
                "modality_router_temperature must be positive, "
                f"got {self.modality_router_temperature}"
            )
        self.modality_router_residual_scale = float(
            cfg.model.get("modality_router_residual_scale", 1.0)
        )
        if self.modality_router_residual_scale < 0.0:
            raise ValueError(
                "modality_router_residual_scale must be non-negative, "
                f"got {self.modality_router_residual_scale}"
            )
        self.modality_router_residual_zero_init = bool(
            cfg.model.get("modality_router_residual_zero_init", True)
        )
        self.lambda_modality_balance = float(cfg.model.get("lambda_modality_balance", 0.0))

        self.use_self_residual = bool(cfg.model.get("use_self_residual", False))
        self.self_residual_max = float(cfg.model.get("self_residual_max", 0.1))
        if self.self_residual_max < 0.0:
            raise ValueError(f"self_residual_max must be non-negative, got {self.self_residual_max}")

        self.use_modality_prototypes = bool(cfg.model.get("use_modality_prototypes", False))
        self.num_text_prototypes = int(cfg.model.get("num_text_prototypes", 16))
        self.num_visual_prototypes = int(cfg.model.get("num_visual_prototypes", 16))
        self.num_common_prototypes = int(cfg.model.get("num_common_prototypes", 16))
        if min(
            self.num_text_prototypes,
            self.num_visual_prototypes,
            self.num_common_prototypes,
        ) < 1:
            raise ValueError("All modality prototype counts must be >= 1")
        self.prototype_residual_max = float(cfg.model.get("prototype_residual_max", 0.1))
        self.prototype_temperature = float(cfg.model.get("prototype_temperature", 1.0))
        if self.prototype_residual_max < 0.0:
            raise ValueError(
                f"prototype_residual_max must be non-negative, got {self.prototype_residual_max}"
            )
        if self.prototype_temperature <= 0.0:
            raise ValueError(
                f"prototype_temperature must be positive, got {self.prototype_temperature}"
            )
        self.prototype_init = str(cfg.model.get("prototype_init", "random")).strip().lower()
        if self.prototype_init not in self.PROTOTYPE_INIT_MODES:
            valid = ", ".join(sorted(self.PROTOTYPE_INIT_MODES))
            raise ValueError(
                f"model.prototype_init must be one of [{valid}], got {self.prototype_init!r}"
            )
        self.prototype_init_max_samples = int(
            cfg.model.get("prototype_init_max_samples", 10000)
        )
        self.prototype_init_iters = int(cfg.model.get("prototype_init_iters", 20))
        self.prototype_init_batch_size = int(
            cfg.model.get("prototype_init_batch_size", 4096)
        )
        if self.prototype_init_max_samples < 1:
            raise ValueError(
                "prototype_init_max_samples must be >= 1, got "
                f"{self.prototype_init_max_samples}"
            )
        if self.prototype_init_iters < 1:
            raise ValueError(
                f"prototype_init_iters must be >= 1, got {self.prototype_init_iters}"
            )
        if self.prototype_init_batch_size < 1:
            raise ValueError(
                "prototype_init_batch_size must be >= 1, got "
                f"{self.prototype_init_batch_size}"
            )

        self.lambda_proto_diversity = float(
            cfg.model.get("lambda_proto_diversity", 0.0)
        )
        self.lambda_edge_reg = float(cfg.model.get("lambda_edge_reg", 0.0))
        if min(
            self.lambda_modality_balance,
            self.lambda_proto_diversity,
            self.lambda_edge_reg,
        ) < 0.0:
            raise ValueError("MAP-MAG v3 auxiliary loss weights must be non-negative")

        rel_hidden_dim = int(cfg.model.get("rel_hidden_dim", max(hidden_dim // 2, 16)))
        router_hidden_dim = int(cfg.model.get("router_hidden_dim", hidden_dim))
        conflict_hidden_dim = int(
            cfg.model.get("conflict_hidden_dim", max(hidden_dim // 2, 16))
        )
        self_hidden_dim = int(cfg.model.get("self_hidden_dim", max(hidden_dim // 2, 16)))

        self.text_proj = ProjectionMLP(self.text_dim, hidden_dim, dropout, norm)
        self.visual_proj = ProjectionMLP(self.visual_dim, hidden_dim, dropout, norm)

        reliability_in_dim = hidden_dim * 4
        self.text_reliability = _make_mlp(
            reliability_in_dim,
            rel_hidden_dim,
            1,
            dropout,
            norm,
        )
        self.visual_reliability = _make_mlp(
            reliability_in_dim,
            rel_hidden_dim,
            1,
            dropout,
            norm,
        )

        learnable_init = float(cfg.model.get("edge_weight_learnable_init", 1.0))
        self.edge_weight_text_scale = nn.Parameter(
            torch.tensor(learnable_init, dtype=torch.float32)
        )
        self.edge_weight_visual_scale = nn.Parameter(
            torch.tensor(learnable_init, dtype=torch.float32)
        )
        self.edge_weight_text_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.edge_weight_visual_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

        conflict_gate_in_dim = hidden_dim + 3
        self.text_conflict_gate = _make_mlp(
            conflict_gate_in_dim,
            conflict_hidden_dim,
            1,
            dropout,
            norm,
        )
        self.visual_conflict_gate = _make_mlp(
            conflict_gate_in_dim,
            conflict_hidden_dim,
            1,
            dropout,
            norm,
        )
        self.text_conflict_mlp = _make_mlp(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            dropout,
            norm,
        )
        self.visual_conflict_mlp = _make_mlp(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            dropout,
            norm,
        )

        router_in_dim = hidden_dim * 4 + 5
        self.modality_router = _make_mlp(
            router_in_dim,
            router_hidden_dim,
            2,
            dropout,
            norm,
        )
        if (
            self.modality_fusion_mode == "reliability_residual"
            and self.modality_router_residual_zero_init
        ):
            output_layer = self.modality_router[-1]
            if not isinstance(output_layer, nn.Linear):
                raise TypeError("MAP-MAG v3 modality router must end with nn.Linear")
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

        self.self_path = _make_mlp(
            hidden_dim * 2,
            hidden_dim,
            hidden_dim,
            dropout,
            norm,
        )
        self.self_gate = _make_mlp(
            hidden_dim * 4 + 5,
            self_hidden_dim,
            1,
            dropout,
            norm,
        )

        if self.use_modality_prototypes:
            self.text_prototypes = nn.Parameter(
                torch.empty(self.num_text_prototypes, hidden_dim)
            )
            self.visual_prototypes = nn.Parameter(
                torch.empty(self.num_visual_prototypes, hidden_dim)
            )
            self.common_prototypes = nn.Parameter(
                torch.empty(self.num_common_prototypes, hidden_dim)
            )
            nn.init.xavier_normal_(self.text_prototypes)
            nn.init.xavier_normal_(self.visual_prototypes)
            nn.init.xavier_normal_(self.common_prototypes)
        prototypes_ready = (
            not self.use_modality_prototypes or self.prototype_init == "random"
        )
        self.register_buffer(
            "_prototypes_initialized",
            torch.tensor(prototypes_ready, dtype=torch.bool),
            persistent=True,
        )

        self.output_mlp = _make_mlp(hidden_dim, hidden_dim, hidden_dim, dropout, norm)
        self.output_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _run_kmeans(
        points: torch.Tensor,
        num_clusters: int,
        num_iters: int,
        seed: int,
    ) -> torch.Tensor:
        """Cluster projected frozen features without adding a new dependency."""
        if points.ndim != 2 or points.size(0) < 1:
            raise ValueError(
                f"K-means points must have shape [N, D] with N >= 1, got {tuple(points.shape)}"
            )
        points = torch.nan_to_num(
            points.detach().float().cpu(),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        num_points = int(points.size(0))

        first = int(torch.randint(num_points, (1,), generator=generator).item())
        centers = [points[first]]
        min_distance = (points - centers[0]).pow(2).sum(dim=-1)
        for cluster_id in range(1, num_clusters):
            distance_sum = min_distance.sum()
            if bool(torch.isfinite(distance_sum)) and float(distance_sum.item()) > 0.0:
                next_id = int(
                    torch.multinomial(
                        min_distance / distance_sum,
                        1,
                        generator=generator,
                    ).item()
                )
            else:
                next_id = cluster_id % num_points
            centers.append(points[next_id])
            next_distance = (points - centers[-1]).pow(2).sum(dim=-1)
            min_distance = torch.minimum(min_distance, next_distance)
        centers_tensor = torch.stack(centers, dim=0)

        for _ in range(num_iters):
            assignments = torch.cdist(points, centers_tensor).argmin(dim=1)
            sums = torch.zeros_like(centers_tensor)
            sums.index_add_(0, assignments, points)
            counts = torch.bincount(assignments, minlength=num_clusters).to(points.dtype)
            nonempty = counts > 0
            updated = centers_tensor.clone()
            updated[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(-1)
            if torch.allclose(updated, centers_tensor, rtol=1e-5, atol=1e-6):
                centers_tensor = updated
                break
            centers_tensor = updated
        return centers_tensor

    @torch.no_grad()
    def initialize_prototypes(
        self,
        x: torch.Tensor,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> bool:
        """Initialize P_t/P_v/P_c from projected frozen features once.

        Returns True only when K-means initialization was performed. Random
        initialization and already-restored checkpoints are left untouched.
        """
        if (
            not self.use_modality_prototypes
            or self.prototype_init != "kmeans"
            or bool(self._prototypes_initialized.item())
        ):
            return False
        if x.ndim != 2 or x.size(0) < 1:
            raise ValueError(
                f"Prototype initialization expects non-empty [N, D] features, got {tuple(x.shape)}"
            )
        if device is None:
            device = next(self.parameters()).device

        num_nodes = int(x.size(0))
        sample_size = min(num_nodes, self.prototype_init_max_samples)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        if sample_size < num_nodes:
            sample_ids = torch.randperm(num_nodes, generator=generator)[:sample_size]
            sampled_x = x[sample_ids]
        else:
            sampled_x = x

        was_training = self.training
        self.eval()
        text_chunks: list[torch.Tensor] = []
        visual_chunks: list[torch.Tensor] = []
        for start in range(0, sample_size, self.prototype_init_batch_size):
            end = min(start + self.prototype_init_batch_size, sample_size)
            batch = sampled_x[start:end].to(device)
            x_t, x_v = self._split_features(batch)
            text_chunks.append(self.text_proj(x_t).float().cpu())
            visual_chunks.append(self.visual_proj(x_v).float().cpu())
        h_t = torch.cat(text_chunks, dim=0)
        h_v = torch.cat(visual_chunks, dim=0)
        h_common = 0.5 * (h_t + h_v)

        centers_t = self._run_kmeans(
            h_t,
            self.num_text_prototypes,
            self.prototype_init_iters,
            seed,
        )
        centers_v = self._run_kmeans(
            h_v,
            self.num_visual_prototypes,
            self.prototype_init_iters,
            seed + 1,
        )
        centers_common = self._run_kmeans(
            h_common,
            self.num_common_prototypes,
            self.prototype_init_iters,
            seed + 2,
        )
        self.text_prototypes.copy_(
            centers_t.to(
                device=self.text_prototypes.device,
                dtype=self.text_prototypes.dtype,
            )
        )
        self.visual_prototypes.copy_(
            centers_v.to(
                device=self.visual_prototypes.device,
                dtype=self.visual_prototypes.dtype,
            )
        )
        self.common_prototypes.copy_(
            centers_common.to(
                device=self.common_prototypes.device,
                dtype=self.common_prototypes.dtype,
            )
        )
        self._prototypes_initialized.fill_(True)
        if was_training:
            self.train()
        return True

    def _ensure_prototypes_initialized(self) -> None:
        if (
            self.use_modality_prototypes
            and self.prototype_init == "kmeans"
            and not bool(self._prototypes_initialized.item())
        ):
            raise RuntimeError(
                "MAP-MAG v3 prototypes requested prototype_init=kmeans but have not "
                "been initialized. Call model.initialize_prototypes(x, device, seed) "
                "before training or inference."
            )

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

    def _degree(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        ones = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
        return scatter(ones, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")

    def _neighbor_mean(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(h)
        src, dst = edge_index
        return scatter(h[src], dst, dim=0, dim_size=h.size(0), reduce="mean")

    def _edge_cosine_values(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h.new_empty((0,))
        src, dst = edge_index
        similarity = F.cosine_similarity(h[src], h[dst], dim=-1, eps=self.eps)
        return torch.nan_to_num(
            similarity,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)

    def _edge_cosine_mean(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h.new_zeros((h.size(0), 1))
        similarity = self._edge_cosine_values(h, edge_index)
        mean = scatter(
            similarity,
            edge_index[1],
            dim=0,
            dim_size=h.size(0),
            reduce="mean",
        )
        return mean.clamp(-1.0, 1.0).unsqueeze(-1)

    def _reliability_gate(
        self,
        h: torch.Tensor,
        mean_h: torch.Tensor,
        mlp: nn.Module,
    ) -> torch.Tensor:
        if self.reliability_mode == "none":
            return torch.ones((h.size(0), 1), dtype=h.dtype, device=h.device)
        reliability_input = torch.cat([h, mean_h, h - mean_h, h * mean_h], dim=-1)
        gate = torch.sigmoid(mlp(reliability_input))
        return self.reliability_min + (1.0 - self.reliability_min) * gate

    def _reliability_weights(
        self,
        r_t: torch.Tensor,
        r_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        denominator = (r_t + r_v).clamp_min(self.eps)
        return r_t / denominator, r_v / denominator

    def _modality_inputs(
        self,
        h_t: torch.Tensor,
        h_v: torch.Tensor,
        r_t: torch.Tensor,
        r_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.reliability_mode == "double":
            return r_t * h_t, r_v * h_v
        return h_t, h_v

    def _fused_input(
        self,
        h_t: torch.Tensor,
        h_v: torch.Tensor,
        r_t: torch.Tensor,
        r_v: torch.Tensor,
        lambda_t: torch.Tensor,
        lambda_v: torch.Tensor,
    ) -> torch.Tensor:
        if self.reliability_mode == "double":
            return lambda_t * (r_t * h_t) + lambda_v * (r_v * h_v)
        return lambda_t * h_t + lambda_v * h_v

    def _edge_weight_from_similarity(self, similarity: torch.Tensor) -> torch.Tensor:
        weight = self.edge_weight_min + (1.0 - self.edge_weight_min) * torch.sigmoid(
            similarity / self.edge_weight_temperature
        )
        return torch.nan_to_num(
            weight,
            nan=1.0,
            posinf=1.0,
            neginf=self.edge_weight_min,
        ).clamp(self.edge_weight_min, 1.0)

    def _edge_stats(
        self,
        weight: torch.Tensor,
        ref: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if weight.numel() == 0:
            zero = ref.new_tensor(0.0)
            return {"mean": zero, "std": zero, "min": zero, "max": zero}
        return {
            "mean": weight.mean(),
            "std": weight.std(unbiased=False),
            "min": weight.min(),
            "max": weight.max(),
        }

    def _semantic_edge_weights(
        self,
        h_t: torch.Tensor,
        h_v: torch.Tensor,
        r_t: torch.Tensor,
        r_v: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if edge_index.numel() == 0:
            empty = h_t.new_empty((0,))
            return {
                "cos_t": empty,
                "cos_v": empty,
                "w_t": empty,
                "w_v": empty,
                "w_shared": empty,
                "stats_t": self._edge_stats(empty, h_t),
                "stats_v": self._edge_stats(empty, h_t),
                "stats_shared": self._edge_stats(empty, h_t),
            }

        cos_t = self._edge_cosine_values(h_t, edge_index)
        cos_v = self._edge_cosine_values(h_v, edge_index)
        mode = self.edge_weight_mode
        if not self.use_modality_specific_edges:
            mode = "shared_avg_cos"

        if mode == "raw_uniform":
            w_t = torch.ones_like(cos_t)
            w_v = torch.ones_like(cos_v)
        elif mode == "shared_avg_cos":
            shared = self._edge_weight_from_similarity(0.5 * (cos_t + cos_v))
            w_t = shared
            w_v = shared
        elif mode == "separate_cos":
            w_t = self._edge_weight_from_similarity(cos_t)
            w_v = self._edge_weight_from_similarity(cos_v)
        elif mode == "reliability_scaled":
            src, dst = edge_index
            rel_t = 0.5 * (r_t[src].squeeze(-1) + r_t[dst].squeeze(-1))
            rel_v = 0.5 * (r_v[src].squeeze(-1) + r_v[dst].squeeze(-1))
            w_t = self._edge_weight_from_similarity(rel_t * cos_t)
            w_v = self._edge_weight_from_similarity(rel_v * cos_v)
        elif mode == "learnable_scalar":
            sim_t = (
                self.edge_weight_text_scale.to(dtype=h_t.dtype) * cos_t
                + self.edge_weight_text_bias.to(dtype=h_t.dtype)
            )
            sim_v = (
                self.edge_weight_visual_scale.to(dtype=h_t.dtype) * cos_v
                + self.edge_weight_visual_bias.to(dtype=h_t.dtype)
            )
            w_t = self._edge_weight_from_similarity(sim_t)
            w_v = self._edge_weight_from_similarity(sim_v)
        else:
            raise RuntimeError(f"Unhandled edge_weight_mode: {mode}")

        w_shared = 0.5 * (w_t + w_v)
        return {
            "cos_t": cos_t,
            "cos_v": cos_v,
            "w_t": w_t,
            "w_v": w_v,
            "w_shared": w_shared,
            "stats_t": self._edge_stats(w_t, h_t),
            "stats_v": self._edge_stats(w_v, h_t),
            "stats_shared": self._edge_stats(w_shared, h_t),
        }

    def _diffuse(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
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
        for _ in range(self.num_hops):
            propagated = scatter(
                h[row] * norm_edge_weight.unsqueeze(-1),
                col,
                dim=0,
                dim_size=h0.size(0),
                reduce="sum",
            )
            h = (1.0 - self.restart) * propagated + self.restart * h0
        return h

    def _conflict_weight(self, positive_weight: torch.Tensor) -> torch.Tensor:
        return (self.conflict_min + self.conflict_scale * (1.0 - positive_weight)).clamp_min(0.0)

    def _conflict_gate(
        self,
        h: torch.Tensor,
        consistency: torch.Tensor,
        reliability: torch.Tensor,
        log_degree: torch.Tensor,
        gate: nn.Module,
    ) -> torch.Tensor:
        if not self.use_conflict_channel:
            return h.new_zeros((h.size(0), 1))
        gate_input = torch.cat([h, consistency, reliability, log_degree], dim=-1)
        return self.conflict_eta_max * torch.sigmoid(gate(gate_input))

    def _propagate_modalities(
        self,
        h_t: torch.Tensor,
        h_v: torch.Tensor,
        h0: torch.Tensor,
        r_t: torch.Tensor,
        r_v: torch.Tensor,
        s_t: torch.Tensor,
        s_v: torch.Tensor,
        log_degree: torch.Tensor,
        edge_index: torch.Tensor,
        edges: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        input_t, input_v = self._modality_inputs(h_t, h_v, r_t, r_v)
        w_t = edges["w_t"]
        w_v = edges["w_v"]
        w_shared = edges["w_shared"]

        if self.propagation_mode == "shared_fused":
            z_shared_pos = self._diffuse(h0, edge_index, w_shared)
            z_t_pos = z_shared_pos
            z_v_pos = z_shared_pos
        else:
            z_shared_pos = torch.zeros_like(h0)
            z_t_pos = self._diffuse(input_t, edge_index, w_t)
            z_v_pos = self._diffuse(input_v, edge_index, w_v)

        eta_t = self._conflict_gate(
            input_t,
            s_t,
            r_t,
            log_degree,
            self.text_conflict_gate,
        )
        eta_v = self._conflict_gate(
            input_v,
            s_v,
            r_v,
            log_degree,
            self.visual_conflict_gate,
        )
        conflict_w_t = self._conflict_weight(w_t)
        conflict_w_v = self._conflict_weight(w_v)

        if self.use_conflict_channel:
            if self.propagation_mode == "shared_fused":
                conflict_w_shared = 0.5 * (conflict_w_t + conflict_w_v)
                z_shared_conflict = self._diffuse(h0, edge_index, conflict_w_shared)
                delta = z_shared_conflict - z_shared_pos
                z_t = z_t_pos + eta_t * self.text_conflict_mlp(delta)
                z_v = z_v_pos + eta_v * self.visual_conflict_mlp(delta)
            else:
                z_t_conflict = self._diffuse(input_t, edge_index, conflict_w_t)
                z_v_conflict = self._diffuse(input_v, edge_index, conflict_w_v)
                z_t = z_t_pos + eta_t * self.text_conflict_mlp(z_t_conflict - z_t_pos)
                z_v = z_v_pos + eta_v * self.visual_conflict_mlp(z_v_conflict - z_v_pos)
        else:
            z_t = z_t_pos
            z_v = z_v_pos

        return {
            "z_t_pos": z_t_pos,
            "z_v_pos": z_v_pos,
            "z_t": z_t,
            "z_v": z_v,
            "z_shared_pos": z_shared_pos,
            "eta_t": eta_t,
            "eta_v": eta_v,
            "conflict_w_t": conflict_w_t,
            "conflict_w_v": conflict_w_v,
        }

    def _modality_weights(
        self,
        z_t: torch.Tensor,
        z_v: torch.Tensor,
        r_t: torch.Tensor,
        r_v: torch.Tensor,
        s_t: torch.Tensor,
        s_v: torch.Tensor,
        log_degree: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_input = torch.cat(
            [
                z_t,
                z_v,
                torch.abs(z_t - z_v),
                z_t * z_v,
                r_t,
                r_v,
                s_t,
                s_v,
                log_degree,
            ],
            dim=-1,
        )
        router_logits = self.modality_router(router_input)
        if self.modality_fusion_mode == "learned_router":
            alpha = torch.softmax(
                router_logits / self.modality_router_temperature,
                dim=-1,
            )
        elif self.modality_fusion_mode == "reliability_residual":
            lambda_t, lambda_v = self._reliability_weights(r_t, r_v)
            prior = torch.cat([lambda_t, lambda_v], dim=-1).clamp_min(self.eps)
            residual = self.modality_router_residual_scale * torch.tanh(
                router_logits / self.modality_router_temperature
            )
            alpha = torch.softmax(torch.log(prior) + residual, dim=-1)
        elif self.modality_fusion_mode == "reliability":
            lambda_t, lambda_v = self._reliability_weights(r_t, r_v)
            alpha = torch.cat([lambda_t, lambda_v], dim=-1)
        elif self.modality_fusion_mode == "uniform":
            alpha = z_t.new_full((z_t.size(0), 2), 0.5)
        else:
            raise RuntimeError(
                f"Unhandled modality_fusion_mode: {self.modality_fusion_mode}"
            )
        return alpha[:, 0:1], alpha[:, 1:2], router_logits

    def _modality_balance_loss(
        self,
        alpha_t: torch.Tensor,
        alpha_v: torch.Tensor,
    ) -> torch.Tensor:
        if self.lambda_modality_balance <= 0.0:
            return alpha_t.new_tensor(0.0)
        mean_alpha = torch.stack([alpha_t.mean(), alpha_v.mean()]).clamp_min(self.eps)
        return torch.sum(mean_alpha * torch.log(mean_alpha * 2.0))

    def _self_residual(
        self,
        h_t: torch.Tensor,
        h_v: torch.Tensor,
        r_t: torch.Tensor,
        r_v: torch.Tensor,
        s_t: torch.Tensor,
        s_v: torch.Tensor,
        log_degree: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_self_residual:
            return torch.zeros_like(h_t), h_t.new_zeros((h_t.size(0), 1))
        z_self = self.self_path(torch.cat([h_t, h_v], dim=-1))
        gate_input = torch.cat(
            [
                h_t,
                h_v,
                torch.abs(h_t - h_v),
                h_t * h_v,
                r_t,
                r_v,
                s_t,
                s_v,
                log_degree,
            ],
            dim=-1,
        )
        beta = self.self_residual_max * torch.sigmoid(self.self_gate(gate_input))
        return z_self, beta

    def _prototype_attention(
        self,
        h: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        scale = math.sqrt(float(self.hidden_dim)) * self.prototype_temperature
        attention = torch.softmax(torch.matmul(h, prototypes.t()) / scale, dim=-1)
        return torch.matmul(attention, prototypes)

    def _prototype_residual(
        self,
        h_t: torch.Tensor,
        h_v: torch.Tensor,
        alpha_t: torch.Tensor,
        alpha_v: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_modality_prototypes:
            return torch.zeros_like(h_t)
        proto_t = self._prototype_attention(h_t, self.text_prototypes)
        proto_v = self._prototype_attention(h_v, self.visual_prototypes)
        proto_common = self._prototype_attention(0.5 * (h_t + h_v), self.common_prototypes)
        return alpha_t * proto_t + alpha_v * proto_v + proto_common

    def _prototype_diversity(
        self,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        normalized = F.normalize(prototypes, p=2, dim=-1, eps=self.eps)
        similarity = torch.matmul(normalized, normalized.t())
        identity = torch.eye(
            prototypes.size(0),
            dtype=prototypes.dtype,
            device=prototypes.device,
        )
        return (similarity - identity).pow(2).mean()

    def _prototype_loss(self, ref: torch.Tensor) -> torch.Tensor:
        if not self.use_modality_prototypes or self.lambda_proto_diversity <= 0.0:
            return ref.new_tensor(0.0)
        return (
            self._prototype_diversity(self.text_prototypes)
            + self._prototype_diversity(self.visual_prototypes)
            + self._prototype_diversity(self.common_prototypes)
        )

    def _edge_reg_loss(
        self,
        w_t: torch.Tensor,
        w_v: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if self.lambda_edge_reg <= 0.0 or w_t.numel() == 0:
            return ref.new_tensor(0.0)
        return 0.5 * ((w_t - 1.0).pow(2).mean() + (w_v - 1.0).pow(2).mean())

    def _encode_components(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        self._ensure_prototypes_initialized()
        x_t, x_v = self._split_features(x)
        h_t = self.text_proj(x_t)
        h_v = self.visual_proj(x_v)

        mean_t = self._neighbor_mean(h_t, edge_index)
        mean_v = self._neighbor_mean(h_v, edge_index)
        r_t = self._reliability_gate(h_t, mean_t, self.text_reliability)
        r_v = self._reliability_gate(h_v, mean_v, self.visual_reliability)
        lambda_t, lambda_v = self._reliability_weights(r_t, r_v)
        h0 = self._fused_input(h_t, h_v, r_t, r_v, lambda_t, lambda_v)

        s_t = self._edge_cosine_mean(h_t, edge_index)
        s_v = self._edge_cosine_mean(h_v, edge_index)
        degree = self._degree(edge_index, int(x.size(0)), h_t.dtype)
        log_degree = torch.log1p(degree).unsqueeze(-1)
        edge_components = self._semantic_edge_weights(h_t, h_v, r_t, r_v, edge_index)
        propagated = self._propagate_modalities(
            h_t,
            h_v,
            h0,
            r_t,
            r_v,
            s_t,
            s_v,
            log_degree,
            edge_index,
            edge_components,
        )

        alpha_t, alpha_v, router_logits = self._modality_weights(
            propagated["z_t"],
            propagated["z_v"],
            r_t,
            r_v,
            s_t,
            s_v,
            log_degree,
        )
        z_struct = alpha_t * propagated["z_t"] + alpha_v * propagated["z_v"]
        z_self, beta_self = self._self_residual(
            h_t,
            h_v,
            r_t,
            r_v,
            s_t,
            s_v,
            log_degree,
        )
        z_proto = self._prototype_residual(h_t, h_v, alpha_t, alpha_v)
        z_input = z_struct + beta_self * z_self
        if self.use_modality_prototypes:
            z_input = z_input + self.prototype_residual_max * z_proto

        text_visual_cosine = F.cosine_similarity(h_t, h_v, dim=-1, eps=self.eps)
        text_visual_cosine = torch.nan_to_num(
            text_visual_cosine,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).unsqueeze(-1)

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
            "edges": edge_components,
            "propagated": propagated,
            "alpha_t": alpha_t,
            "alpha_v": alpha_v,
            "router_logits": router_logits,
            "z_struct": z_struct,
            "z_self": z_self,
            "beta_self": beta_self,
            "z_proto": z_proto,
            "z_input": z_input,
            "text_visual_cosine": text_visual_cosine,
        }

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        components = self._encode_components(x, edge_index)
        z_input = components["z_input"]
        z = self.output_norm(self.output_mlp(z_input) + z_input)
        z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)

        edges = components["edges"]
        propagated = components["propagated"]
        modality_balance_loss = self._modality_balance_loss(
            components["alpha_t"],
            components["alpha_v"],
        )
        proto_loss = self._prototype_loss(z)
        edge_reg_loss = self._edge_reg_loss(edges["w_t"], edges["w_v"], z)
        aux_loss = (
            self.lambda_modality_balance * modality_balance_loss
            + self.lambda_proto_diversity * proto_loss
            + self.lambda_edge_reg * edge_reg_loss
        )

        stats_t = edges["stats_t"]
        stats_v = edges["stats_v"]
        stats_shared = edges["stats_shared"]
        aux_info = {
            "mean_r_text": components["r_t"].detach().mean(),
            "mean_r_visual": components["r_v"].detach().mean(),
            "mean_lambda_text": components["lambda_t"].detach().mean(),
            "mean_lambda_visual": components["lambda_v"].detach().mean(),
            "mean_edge_weight_text": stats_t["mean"].detach(),
            "std_edge_weight_text": stats_t["std"].detach(),
            "mean_edge_weight_visual": stats_v["mean"].detach(),
            "std_edge_weight_visual": stats_v["std"].detach(),
            "mean_edge_weight": stats_shared["mean"].detach(),
            "std_edge_weight": stats_shared["std"].detach(),
            "min_edge_weight": stats_shared["min"].detach(),
            "max_edge_weight": stats_shared["max"].detach(),
            "mean_cos_text": edges["cos_t"].detach().mean()
            if edges["cos_t"].numel() > 0
            else z.new_tensor(0.0),
            "mean_cos_visual": edges["cos_v"].detach().mean()
            if edges["cos_v"].numel() > 0
            else z.new_tensor(0.0),
            "mean_alpha_text": components["alpha_t"].detach().mean(),
            "mean_alpha_visual": components["alpha_v"].detach().mean(),
            "mean_degree": components["degree"].detach().mean(),
            "modality_balance_loss": modality_balance_loss.detach(),
            "proto_loss": proto_loss.detach(),
            "edge_reg_loss": edge_reg_loss.detach(),
        }
        if self.use_conflict_channel:
            aux_info.update(
                {
                    "mean_conflict_weight_text": propagated["conflict_w_t"].detach().mean()
                    if propagated["conflict_w_t"].numel() > 0
                    else z.new_tensor(0.0),
                    "mean_conflict_weight_visual": propagated[
                        "conflict_w_v"
                    ].detach().mean()
                    if propagated["conflict_w_v"].numel() > 0
                    else z.new_tensor(0.0),
                    "mean_eta_conflict_text": propagated["eta_t"].detach().mean(),
                    "mean_eta_conflict_visual": propagated["eta_v"].detach().mean(),
                }
            )
        if self.use_self_residual:
            aux_info["mean_beta_self"] = components["beta_self"].detach().mean()
        if self.use_modality_prototypes:
            aux_info["mean_proto_residual"] = z.new_tensor(self.prototype_residual_max)
        if self.propagation_mode == "shared_fused":
            aux_info["mean_shared_edge_weight"] = stats_shared["mean"].detach()
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
        propagated = components["propagated"]
        num_nodes = int(x.size(0))
        proto_weight = (
            self.prototype_residual_max if self.use_modality_prototypes else 0.0
        )
        stats = {
            "degree": components["degree"].detach().cpu(),
            "s_text": components["s_t"].squeeze(-1).detach().cpu(),
            "s_visual": components["s_v"].squeeze(-1).detach().cpu(),
            "r_text": components["r_t"].squeeze(-1).detach().cpu(),
            "r_visual": components["r_v"].squeeze(-1).detach().cpu(),
            "p_self": components["beta_self"].squeeze(-1).detach().cpu(),
            "p_struct": torch.ones(num_nodes, dtype=components["h_t"].dtype),
            "p_proto": torch.full(
                (num_nodes,),
                proto_weight,
                dtype=components["h_t"].dtype,
            ),
            "gamma": torch.ones(num_nodes, dtype=components["h_t"].dtype),
            "text_visual_cosine": components["text_visual_cosine"].squeeze(-1).detach().cpu(),
            "lambda_text": components["lambda_t"].squeeze(-1).detach().cpu(),
            "lambda_visual": components["lambda_v"].squeeze(-1).detach().cpu(),
            "alpha_text": components["alpha_t"].squeeze(-1).detach().cpu(),
            "alpha_visual": components["alpha_v"].squeeze(-1).detach().cpu(),
            "beta_self": components["beta_self"].squeeze(-1).detach().cpu(),
            "eta_conflict_text": propagated["eta_t"].squeeze(-1).detach().cpu(),
            "eta_conflict_visual": propagated["eta_v"].squeeze(-1).detach().cpu(),
        }
        if was_training:
            self.train()
        return stats

    @torch.no_grad()
    def edge_aux_stats(
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
        edges = components["edges"]
        src, dst = edge_index
        degree = components["degree"]
        stats = {
            "src": src.detach().cpu(),
            "dst": dst.detach().cpu(),
            "cos_text": edges["cos_t"].detach().cpu(),
            "cos_visual": edges["cos_v"].detach().cpu(),
            "edge_weight": edges["w_shared"].detach().cpu(),
            "edge_weight_text": edges["w_t"].detach().cpu(),
            "edge_weight_visual": edges["w_v"].detach().cpu(),
            "edge_weight_shared": edges["w_shared"].detach().cpu(),
            "src_degree": degree[src].detach().cpu(),
            "dst_degree": degree[dst].detach().cpu(),
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


Model = MAPMAGV3
