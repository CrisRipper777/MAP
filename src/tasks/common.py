from __future__ import annotations

import copy
import math

import torch
from omegaconf import ListConfig


AUX_INFO_KEYS = (
    "mean_r_text",
    "mean_r_visual",
    "mean_p_self",
    "mean_p_struct",
    "mean_p_proto",
    "mean_gamma",
    "mean_edge_weight",
    "std_edge_weight",
    "min_edge_weight",
    "max_edge_weight",
    "mean_edge_weight_text",
    "std_edge_weight_text",
    "mean_edge_weight_visual",
    "std_edge_weight_visual",
    "mean_shared_edge_weight",
    "mean_cos_text",
    "mean_cos_visual",
    "mean_lambda_text",
    "mean_lambda_visual",
    "mean_alpha_text",
    "mean_alpha_visual",
    "mean_degree",
    "mean_eta_residual",
    "mean_alpha_self",
    "mean_conflict_weight_text",
    "mean_conflict_weight_visual",
    "mean_eta_conflict_text",
    "mean_eta_conflict_visual",
    "mean_beta_self",
    "mean_proto_residual",
    "proto_loss",
    "prototype_loss",
    "edge_reg_loss",
    "gate_loss",
    "modality_balance_loss",
)


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def load_state_dict_cpu(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(copy.deepcopy(state_dict))


def build_optimizer(parameters, cfg, model=None) -> torch.optim.Optimizer:
    """Build the 0901-style optimizer with optional model presets.

    Task-level values define the common protocol. A model config may provide
    an official baseline preset for lr/weight_decay, as in the reference
    project.
    """
    name = str(cfg.task.get("optimizer", "adamw")).strip().lower()
    lr = float(cfg.model.get("lr", cfg.task.lr))
    weight_decay = float(cfg.model.get("weight_decay", cfg.task.weight_decay))
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"task.optimizer must be adam|adamw, got {name!r}")


def scheduler_step(cfg, optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int) -> None:
    """Optional warmup-cosine schedule from the 0901 training protocol."""
    name = str(cfg.task.get("scheduler", "null")).strip().lower()
    if name in {"null", "", "none"}:
        return
    if name != "warmup_cosine":
        raise ValueError(f"task.scheduler must be null|warmup_cosine, got {name!r}")
    warmup = max(int(cfg.task.get("scheduler_warmup_epochs", 10)), 1)
    base_lr = float(cfg.model.get("lr", cfg.task.lr))
    final_lr = float(cfg.task.get("scheduler_min_lr", 1e-5))
    if epoch <= warmup:
        lr = base_lr * epoch / warmup
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        lr = final_lr + 0.5 * (base_lr - final_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr


def resolve_num_neighbors(cfg) -> list[int]:
    num_layers = int(cfg.model.get("num_layers", 1))
    if num_layers < 1:
        raise ValueError(f"model.num_layers must be >= 1, got {num_layers}")

    raw_neighbors = cfg.task.get("num_neighbors", -1)
    if isinstance(raw_neighbors, str):
        raw_neighbors = raw_neighbors.strip()
        if raw_neighbors.startswith("[") and raw_neighbors.endswith("]"):
            values = [int(item.strip()) for item in raw_neighbors[1:-1].split(",") if item.strip()]
        else:
            values = [int(raw_neighbors)]
    elif isinstance(raw_neighbors, (list, tuple, ListConfig)):
        values = [int(value) for value in raw_neighbors]
    else:
        values = [int(raw_neighbors)]

    if not values:
        raise ValueError("task.num_neighbors must contain at least one value")
    if len(values) >= num_layers:
        return values[:num_layers]
    return values + [values[-1]] * (num_layers - len(values))


def update_aux_info_stats(
    sums: dict[str, float],
    counts: dict[str, float],
    aux_info: dict | None,
    weight: float = 1.0,
) -> None:
    if not isinstance(aux_info, dict):
        return
    for key in AUX_INFO_KEYS:
        if key not in aux_info:
            continue
        value = aux_info[key]
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            scalar = float(value.detach().cpu().item())
        else:
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
        sums[key] = sums.get(key, 0.0) + scalar * weight
        counts[key] = counts.get(key, 0.0) + weight


def summarize_aux_info_stats(sums: dict[str, float], counts: dict[str, float]) -> dict[str, float]:
    return {
        key: sums[key] / max(counts.get(key, 0.0), 1e-12)
        for key in AUX_INFO_KEYS
        if key in sums
    }


def format_aux_info_stats(stats: dict[str, float]) -> str:
    return " | ".join(f"{key} {stats[key]:.4f}" for key in AUX_INFO_KEYS if key in stats)
