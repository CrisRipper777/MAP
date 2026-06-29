from __future__ import annotations

from dataclasses import replace
from typing import Iterator

import torch

from src.data import MAGData


def modality_mask_eval_enabled(cfg) -> bool:
    settings = cfg.task.get("eval_modality_masks")
    if settings is None:
        return False
    return bool(settings.get("enabled", False))


def modality_mask_modes(cfg) -> list[str]:
    settings = cfg.task.get("eval_modality_masks")
    if settings is None:
        return []
    modes = settings.get("modes", [])
    return [normalize_mask_mode(str(mode)) for mode in modes]


def modality_mask_key(mode: str) -> str:
    mode = normalize_mask_mode(mode)
    if mode == "mask_text":
        return "mask_text"
    if mode == "mask_visual":
        return "mask_visual"
    if mode == "mask_both":
        return "mask_both"
    if mode == "mask_both_partial":
        return "mask_both_partial"
    raise ValueError(f"Unsupported modality mask mode: {mode}")


def modality_drop_key(mode: str) -> str:
    mode = normalize_mask_mode(mode)
    if mode == "mask_text":
        return "drop_text"
    if mode == "mask_visual":
        return "drop_visual"
    if mode == "mask_both":
        return "drop_both"
    if mode == "mask_both_partial":
        return "drop_both_partial"
    raise ValueError(f"Unsupported modality mask mode: {mode}")


def normalize_mask_mode(mode: str) -> str:
    value = mode.strip().lower().replace("-", "_")
    aliases = {
        "text": "mask_text",
        "zero_text": "mask_text",
        "visual": "mask_visual",
        "image": "mask_visual",
        "zero_visual": "mask_visual",
        "both": "mask_both",
        "zero_both": "mask_both",
        "partial": "mask_both_partial",
        "both_partial": "mask_both_partial",
        "partial_both": "mask_both_partial",
    }
    return aliases.get(value, value)


def iter_masked_data(cfg, data: MAGData, seed: int) -> Iterator[tuple[str, MAGData]]:
    if not modality_mask_eval_enabled(cfg):
        return
    settings = cfg.task.eval_modality_masks
    ratio = float(settings.get("partial_ratio", 0.5))
    seed_offset = int(settings.get("partial_seed_offset", 100000))
    for mode in modality_mask_modes(cfg):
        yield mode, make_masked_data(data, mode, partial_ratio=ratio, seed=seed + seed_offset)


def make_masked_data(
    data: MAGData,
    mode: str,
    *,
    partial_ratio: float = 0.5,
    seed: int = 0,
) -> MAGData:
    mode = normalize_mask_mode(mode)
    if data.x_t is None or data.x_i is None:
        raise ValueError(
            f"Modality mask eval requires both text and visual features; "
            f"dataset={data.name}, mode={mode}"
        )
    if not 0.0 <= partial_ratio <= 1.0:
        raise ValueError(f"partial_ratio must be in [0, 1], got {partial_ratio}")

    x_t = data.x_t.clone()
    x_i = data.x_i.clone()
    mask_info: dict[str, object] = {"modality_mask_mode": mode}

    if mode == "mask_text":
        x_t.zero_()
    elif mode == "mask_visual":
        x_i.zero_()
    elif mode == "mask_both":
        x_t.zero_()
        x_i.zero_()
    elif mode == "mask_both_partial":
        generator = torch.Generator().manual_seed(int(seed))
        text_mask = torch.rand(x_t.size(0), generator=generator) < partial_ratio
        visual_mask = torch.rand(x_i.size(0), generator=generator) < partial_ratio
        x_t[text_mask] = 0
        x_i[visual_mask] = 0
        mask_info.update(
            {
                "partial_ratio": partial_ratio,
                "masked_text_nodes": int(text_mask.sum().item()),
                "masked_visual_nodes": int(visual_mask.sum().item()),
            }
        )
    else:
        raise ValueError(f"Unsupported modality mask mode: {mode}")

    x = torch.cat([x_t, x_i], dim=1).contiguous()
    return replace(data, x=x, x_t=x_t.contiguous(), x_i=x_i.contiguous(), info={**data.info, **mask_info})
