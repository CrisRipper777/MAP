from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from .common import get_activation, make_norm
from .ored_components import (
    OwnershipEdgeScorer,
    ReconstructionHead,
    RestartDiffusion,
    SemanticFactorizer,
    apply_restart_diffusion,
    normalize_restart_adjacency,
)


class Model(nn.Module):
    """ORED-MAG semantic ownership encoder and ORED-2 matched variants.

    ``p0`` preserves the ORED-1 topology-free encoder.  The other variants keep
    its factorizer, fusion, auxiliary losses, and output dimensions fixed while
    adding only the pre-registered MAP-v2-style refinement and/or diffusion.
    """

    def __init__(self, cfg, data_info):
        super().__init__()
        variant = str(cfg.model.get("variant", "p0")).lower()
        valid_variants = {
            "p0",
            "p0_refine",
            "joint_rd",
            "ownership_rd",
            "comp_uniform",
            "comp_shared",
            "comp_ownership",
        }
        if variant not in valid_variants:
            raise ValueError(
                "ored_mag variant must be one of "
                f"{sorted(valid_variants)}, got {variant!r}"
            )

        text_dim = int(data_info["text_dim"])
        visual_dim = int(data_info["visual_dim"])
        input_dim = int(data_info["input_dim"])
        if text_dim <= 0 or visual_dim <= 0:
            raise ValueError(
                f"ored_mag requires both modalities, got text_dim={text_dim}, visual_dim={visual_dim}"
            )
        if input_dim < text_dim + visual_dim:
            raise ValueError(
                f"input_dim={input_dim} < text_dim+visual_dim={text_dim + visual_dim}; "
                "data.x must be [x_t | x_v] concatenated"
            )

        hidden_dim = int(cfg.model.hidden_dim)
        factor_dim = int(cfg.model.factor_dim)
        dropout = float(cfg.model.dropout)
        activation = str(cfg.model.get("activation", "gelu"))
        norm = str(cfg.model.get("norm", "layernorm"))

        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.factor_dim = factor_dim
        self.variant = variant

        self.factorizer = SemanticFactorizer(
            text_dim, visual_dim, hidden_dim, factor_dim, dropout, activation, norm
        )
        self.recon_text_head = ReconstructionHead(factor_dim, hidden_dim, activation)
        self.recon_visual_head = ReconstructionHead(factor_dim, hidden_dim, activation)
        self.fusion = nn.Sequential(
            nn.Linear(3 * factor_dim, hidden_dim),
            make_norm(norm, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
        )

        # Keep the original p0 module graph untouched.  All three ORED-2
        # controls share exactly these modules; diffusion itself has no params.
        self.output_mlp = None
        self.output_norm = None
        self.restart_diffusion = None
        self.edge_scorer = None
        if variant != "p0":
            self.output_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.output_norm = nn.LayerNorm(hidden_dim)
            num_hops = int(cfg.model.get("num_hops", 2))
            restart = float(cfg.model.get("restart", 0.15))
            add_self_loops = bool(cfg.model.get("diffusion_add_self_loops", True))
            self.restart_diffusion = RestartDiffusion(num_hops, restart, add_self_loops)
            if variant in {"comp_uniform", "comp_shared", "comp_ownership"}:
                # Construct after all ORED-2 parent modules so their
                # initialization order remains unchanged.
                self.edge_scorer = OwnershipEdgeScorer(
                    factor_dim,
                    int(cfg.model.get("score_hidden_dim", 64)),
                )

        # These are fixed by the ORED-3A protocol; reject accidental tuning.
        self.composition_strength = float(cfg.model.get("composition_strength", 0.5))
        self.composition_temperature = float(cfg.model.get("composition_temperature", 1.0))
        if self.composition_strength != 0.5 or self.composition_temperature != 1.0:
            raise ValueError("ORED-3A composition strength/temperature are fixed at 0.5/1.0")

        self.lambda_common = float(cfg.model.get("lambda_common", 0.1))
        self.lambda_orth = float(cfg.model.get("lambda_orth", 0.01))
        self.lambda_recon = float(cfg.model.get("lambda_recon", 0.1))
        self.orth_fallback_batch = int(cfg.model.get("orth_fallback_batch", 16))
        self.out_dim = hidden_dim

    def _split_modalities(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.size(-1) >= self.text_dim + self.visual_dim, (
            f"expected x.size(-1) >= {self.text_dim + self.visual_dim} ([x_t | x_v]), "
            f"got {x.size(-1)}"
        )
        x_t = x[:, : self.text_dim]
        x_v = x[:, self.text_dim : self.text_dim + self.visual_dim]
        return x_t, x_v

    def _encode_local(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        x_t, x_v = self._split_modalities(x)
        factors = self.factorizer(x_t, x_v)
        z_local = self.fusion(torch.cat([factors["c"], factors["p_t"], factors["p_v"]], dim=-1))
        return factors, z_local

    def _encode(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Preserve the ORED-1 local encoding API used by diagnostics/tests."""
        return self._encode_local(x)

    def _refine(self, value: torch.Tensor) -> torch.Tensor:
        if self.output_mlp is None or self.output_norm is None:
            return value
        return self.output_norm(self.output_mlp(value) + value)

    def _edge_index_or_empty(
        self, edge_index: torch.Tensor | None, device: torch.device
    ) -> torch.Tensor:
        if edge_index is None:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return edge_index.to(device=device, dtype=torch.long)

    def _composition_weights(
        self,
        ownership0: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score original edges once and map scores to factor edge weights."""
        if self.edge_scorer is None:
            raise RuntimeError("composition weights requested without an edge scorer")
        scores = self.edge_scorer(ownership0, edge_index)
        if self.variant == "comp_uniform":
            return scores.new_ones((scores.size(0), 3)), {"edge_scores": scores}

        if self.variant == "comp_shared":
            shared_score = scores.mean(dim=1)
            shared = 1.0 + self.composition_strength * torch.tanh(
                shared_score / self.composition_temperature
            )
            return shared.unsqueeze(-1).expand(-1, 3), {
                "edge_scores": scores,
                "edge_weight_shared": shared,
            }
        if self.variant == "comp_ownership":
            weights = 1.0 + self.composition_strength * torch.tanh(
                scores / self.composition_temperature
            )
            return weights, {"edge_scores": scores}
        raise RuntimeError(f"composition weights requested for {self.variant!r}")

    def _ownership_diffusion(
        self,
        ownership0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Diffuse each ownership payload with its own static normalized graph."""
        assert self.restart_diffusion is not None
        outputs: list[torch.Tensor] = []
        for factor in range(3):
            norm_edge_index, norm_edge_weight = gcn_norm(
                edge_index,
                edge_weight=edge_weights[:, factor],
                num_nodes=int(ownership0.size(0)),
                improved=False,
                add_self_loops=self.restart_diffusion.add_self_loops,
                flow="source_to_target",
                dtype=ownership0.dtype,
            )
            outputs.append(
                apply_restart_diffusion(
                    ownership0[:, factor],
                    norm_edge_index,
                    norm_edge_weight,
                    self.restart_diffusion.num_hops,
                    self.restart_diffusion.restart,
                )
            )
        return torch.stack(outputs, dim=1)

    def _variant_encoding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        factors, z_local = self._encode_local(x)
        c0, pt0, pv0 = factors["c"], factors["p_t"], factors["p_v"]
        states: dict[str, torch.Tensor] = {
            "C0": c0,
            "Pt0": pt0,
            "Pv0": pv0,
            "z_local": z_local,
        }

        if self.variant == "p0":
            return factors, states, z_local
        if self.variant == "p0_refine":
            return factors, states, self._refine(z_local)

        assert self.restart_diffusion is not None
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        norm_edge_index, norm_edge_weight = normalize_restart_adjacency(
            edge_index,
            num_nodes=int(x.size(0)),
            dtype=z_local.dtype,
            add_self_loops=self.restart_diffusion.add_self_loops,
        )

        if self.variant == "joint_rd":
            joint2 = self.restart_diffusion(
                z_local,
                norm_edge_index=norm_edge_index,
                norm_edge_weight=norm_edge_weight,
            )
            states.update({"joint0": z_local, "joint2": joint2})
            return factors, states, self._refine(joint2)

        # One normalized adjacency is reused for the three ownership streams.
        ownership0 = torch.stack([c0, pt0, pv0], dim=1)
        composition_info: dict[str, torch.Tensor] = {}
        if self.variant == "ownership_rd":
            ownership2 = self.restart_diffusion(
                ownership0,
                norm_edge_index=norm_edge_index,
                norm_edge_weight=norm_edge_weight,
            )
        else:
            assert self.edge_scorer is not None
            edge_weights, composition_info = self._composition_weights(ownership0, edge_index)
            if self.variant == "comp_uniform":
                ownership2 = self.restart_diffusion(
                    ownership0,
                    norm_edge_index=norm_edge_index,
                    norm_edge_weight=norm_edge_weight,
                )
            elif self.variant == "comp_shared":
                shared_norm_edge_index, shared_norm_edge_weight = gcn_norm(
                    edge_index,
                    edge_weight=edge_weights[:, 0],
                    num_nodes=int(x.size(0)),
                    improved=False,
                    add_self_loops=self.restart_diffusion.add_self_loops,
                    flow="source_to_target",
                    dtype=ownership0.dtype,
                )
                ownership2 = self.restart_diffusion(
                    ownership0,
                    norm_edge_index=shared_norm_edge_index,
                    norm_edge_weight=shared_norm_edge_weight,
                )
            else:
                ownership2 = self._ownership_diffusion(
                    ownership0,
                    edge_index,
                    edge_weights,
                )
        c2, pt2, pv2 = ownership2.unbind(dim=1)
        u = self.fusion(torch.cat([c2, pt2, pv2], dim=-1))
        states.update({"C2": c2, "Pt2": pt2, "Pv2": pv2, "u": u})
        states.update(composition_info)
        if self.variant in {"comp_uniform", "comp_shared", "comp_ownership"}:
            if self.variant == "comp_shared":
                shared = composition_info["edge_weight_shared"]
                states.update(
                    {
                        "edge_weight_C": shared,
                        "edge_weight_Pt": shared,
                        "edge_weight_Pv": shared,
                    }
                )
            else:
                factor_weights = (
                    states["edge_scores"].new_ones((edge_index.size(1), 3))
                    if self.variant == "comp_uniform"
                    else 1.0
                    + self.composition_strength
                    * torch.tanh(states["edge_scores"] / self.composition_temperature)
                )
                states.update(
                    {
                        "edge_weight_C": factor_weights[:, 0],
                        "edge_weight_Pt": factor_weights[:, 1],
                        "edge_weight_Pv": factor_weights[:, 2],
                    }
                )
        return factors, states, self._refine(u)

    def _orth_loss(self, c: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, d = c.size(0), c.size(1)
        if batch < self.orth_fallback_batch:
            c_n = F.normalize(c, dim=-1)
            p_n = F.normalize(p, dim=-1)
            cos2 = (c_n * p_n).sum(dim=-1).square().mean()
            return cos2, cos2.sqrt()
        c_c = c - c.mean(dim=0, keepdim=True)
        p_c = p - p.mean(dim=0, keepdim=True)
        cov = c_c.t() @ p_c / max(batch - 1, 1)
        overlap = cov.norm() / d
        return overlap.square(), overlap

    def _compute_aux(
        self, factors: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h_t, h_v = factors["h_t"], factors["h_v"]
        c_t, c_v = factors["c_t"], factors["c_v"]
        p_t, p_v = factors["p_t"], factors["p_v"]

        c_t_n = F.normalize(c_t, dim=-1)
        c_v_n = F.normalize(c_v, dim=-1)
        common_sim = (c_t_n * c_v_n).sum(dim=-1).mean()
        common_loss = 1.0 - common_sim

        orth_t, overlap_t = self._orth_loss(c_t, p_t)
        orth_v, overlap_v = self._orth_loss(c_v, p_v)
        orth_loss = orth_t + orth_v

        rec_t = F.mse_loss(self.recon_text_head(c_t, p_t), h_t)
        rec_v = F.mse_loss(self.recon_visual_head(c_v, p_v), h_v)
        rec_loss = rec_t + rec_v

        aux_loss = (
            self.lambda_common * common_loss
            + self.lambda_orth * orth_loss
            + self.lambda_recon * rec_loss
        )

        p_t_n = F.normalize(p_t, dim=-1)
        p_v_n = F.normalize(p_v, dim=-1)
        private_sim = (p_t_n * p_v_n).sum(dim=-1).mean()
        aux_info = {
            "p0_common_loss": common_loss.detach(),
            "p0_orth_loss": orth_loss.detach(),
            "p0_recon_loss": rec_loss.detach(),
            "p0_common_sim": common_sim.detach(),
            "p0_private_sim": private_sim.detach(),
            "p0_c_norm": factors["c"].norm(dim=-1).mean().detach(),
            "p0_pt_norm": p_t.norm(dim=-1).mean().detach(),
            "p0_pv_norm": p_v.norm(dim=-1).mean().detach(),
            "p0_cp_overlap_t": overlap_t.detach(),
            "p0_cp_overlap_v": overlap_v.detach(),
        }
        return aux_loss, aux_info

    def forward(self, x: torch.Tensor, edge_index=None):
        factors, _, z = self._variant_encoding(x, edge_index)
        if self.training:
            aux_loss, aux_info = self._compute_aux(factors)
        else:
            aux_loss = z.new_tensor(0.0)
            aux_info = {}
        return z, None, None, aux_loss, aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        if self.variant in {"joint_rd", "ownership_rd", "comp_uniform", "comp_shared", "comp_ownership"}:
            edge_index = self._edge_index_or_empty(edge_index, device)
            z, _, _, _, _ = self.forward(x.to(device), edge_index)
            return z.detach().cpu()

        outputs = torch.empty((x.size(0), self.out_dim), dtype=x.dtype, device="cpu")
        for start in range(0, x.size(0), batch_size):
            end = min(start + batch_size, x.size(0))
            z, _, _, _, _ = self.forward(x[start:end].to(device), None)
            outputs[start:end] = z.detach().cpu()
        return outputs

    @torch.no_grad()
    def encode_ored_states(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return full-graph ORED-2 states on CPU for offline diagnosis."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        _, states, z = self._variant_encoding(x.to(device), edge_index)
        states["z"] = z
        return {key: value.detach().cpu() for key, value in states.items()}

    @torch.no_grad()
    def encode_factors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return CPU factors, optionally in chunks, for offline diagnostics."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        num_nodes = int(x.size(0))
        keys = ("c", "c_t", "c_v", "p_t", "p_v", "z_local")
        if batch_size is None or num_nodes <= batch_size:
            factors, z = self._encode(x.to(device))
            return {
                "c": factors["c"].cpu(),
                "c_t": factors["c_t"].cpu(),
                "c_v": factors["c_v"].cpu(),
                "p_t": factors["p_t"].cpu(),
                "p_v": factors["p_v"].cpu(),
                "z_local": z.cpu(),
            }
        chunks: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)
            factors, z = self._encode(x[start:end].to(device))
            chunks["c"].append(factors["c"].cpu())
            chunks["c_t"].append(factors["c_t"].cpu())
            chunks["c_v"].append(factors["c_v"].cpu())
            chunks["p_t"].append(factors["p_t"].cpu())
            chunks["p_v"].append(factors["p_v"].cpu())
            chunks["z_local"].append(z.cpu())
        return {key: torch.cat(chunks[key], dim=0) for key in keys}

    @torch.no_grad()
    def edge_aux_stats(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return static original-edge composition diagnostics on CPU."""
        if self.variant not in {"comp_uniform", "comp_shared", "comp_ownership"}:
            raise RuntimeError(f"edge composition diagnostics unavailable for {self.variant!r}")
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        edge_index = self._edge_index_or_empty(edge_index, device)
        _, states, _ = self._variant_encoding(x.to(device), edge_index)
        src, dst = edge_index
        return {
            "src": src.detach().cpu(),
            "dst": dst.detach().cpu(),
            "edge_scores": states["edge_scores"].detach().cpu(),
            "edge_weight_C": states["edge_weight_C"].detach().cpu(),
            "edge_weight_Pt": states["edge_weight_Pt"].detach().cpu(),
            "edge_weight_Pv": states["edge_weight_Pv"].detach().cpu(),
        }
