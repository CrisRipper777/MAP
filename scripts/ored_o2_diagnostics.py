from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.data import load_mag_data
from src.models import build_model


def _effective_rank(value: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(value.float())
    energy = singular.square()
    energy = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
    entropy = -(energy.clamp_min(torch.finfo(energy.dtype).eps) * energy.clamp_min(torch.finfo(energy.dtype).eps).log()).sum()
    return float(entropy.exp().item())


def _norm_stats(value: torch.Tensor) -> dict[str, object]:
    return {
        "finite": bool(torch.isfinite(value).all().item()),
        "mean_norm": float(value.norm(dim=-1).mean().item()),
        "effective_rank": _effective_rank(value),
    }


def _relative_update(after: torch.Tensor, before: torch.Tensor) -> float:
    return float(((after - before).norm(dim=-1) / (before.norm(dim=-1) + 1e-8)).mean().item())


def _mean_cos(after: torch.Tensor, before: torch.Tensor) -> float:
    return float(F.cosine_similarity(before, after, dim=-1).mean().item())


def _load_states(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    output_dir_candidates = [
        checkpoint_path.parent.parent / "formal" / checkpoint_path.stem,
        checkpoint_path.parent.parent / "o2a" / checkpoint_path.stem,
        checkpoint_path.parent.parent / "o2a" / checkpoint_path.stem.removesuffix("_best"),
    ]
    output_dir = next(
        candidate for candidate in output_dir_candidates if (candidate / ".hydra" / "config.yaml").is_file()
    )
    cfg = OmegaConf.load(output_dir / ".hydra" / "config.yaml")
    seed = int(cfg.seed)
    data = load_mag_data(cfg, "nc", seed)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model.encode_ored_states(data.x, data.edge_index.to(device), device=device)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ORED-2 Movies seed42 representation diagnostics.")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/ored/o2a"))
    parser.add_argument("--output", type=Path, default=Path("experiments/ored/o2/o2_movies_seed42_diagnostics.json"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")

    def checkpoint_for(variant: str) -> Path:
        formal = args.checkpoint_root / f"movies_seed42_{variant}.pt"
        smoke = args.checkpoint_root / f"movies_seed42_{variant}_best.pt"
        return formal if formal.is_file() else smoke

    p0_refine = _load_states(checkpoint_for("p0_refine"), device)
    joint = _load_states(checkpoint_for("joint_rd"), device)
    ownership = _load_states(checkpoint_for("ownership_rd"), device)

    output = {
        "stage": "ORED-2A",
        "dataset": "Movies",
        "seed": 42,
        "uses_test_labels_or_metrics": False,
        "p0_refine": {
            "z_local": _norm_stats(p0_refine["z_local"]),
            "z_final": _norm_stats(p0_refine["z"]),
            "relative_refinement_update_mean": _relative_update(p0_refine["z"], p0_refine["z_local"]),
        },
        "joint_rd": {
            "joint0": _norm_stats(joint["joint0"]),
            "joint2": _norm_stats(joint["joint2"]),
            "graph_change_ratio_mean": _relative_update(joint["joint2"], joint["joint0"]),
            "cos_joint0_joint2_mean": _mean_cos(joint["joint2"], joint["joint0"]),
            "graph_update_nonzero": bool((joint["joint2"] - joint["joint0"]).abs().max().item() > 0.0),
        },
        "ownership_rd": {
            "C": {
                "before": _norm_stats(ownership["C0"]),
                "after": _norm_stats(ownership["C2"]),
                "relative_update_mean": _relative_update(ownership["C2"], ownership["C0"]),
                "cos_before_after_mean": _mean_cos(ownership["C2"], ownership["C0"]),
            },
            "Pt": {
                "before": _norm_stats(ownership["Pt0"]),
                "after": _norm_stats(ownership["Pt2"]),
                "relative_update_mean": _relative_update(ownership["Pt2"], ownership["Pt0"]),
                "cos_before_after_mean": _mean_cos(ownership["Pt2"], ownership["Pt0"]),
            },
            "Pv": {
                "before": _norm_stats(ownership["Pv0"]),
                "after": _norm_stats(ownership["Pv2"]),
                "relative_update_mean": _relative_update(ownership["Pv2"], ownership["Pv0"]),
                "cos_before_after_mean": _mean_cos(ownership["Pv2"], ownership["Pv0"]),
            },
            "u": _norm_stats(ownership["u"]),
            "z": _norm_stats(ownership["z"]),
            "graph_update_nonzero": any(
                bool((ownership[after] - ownership[before]).abs().max().item() > 0.0)
                for before, after in (("C0", "C2"), ("Pt0", "Pt2"), ("Pv0", "Pv2"))
            ),
        },
        "interpretation": (
            "The contrast is ownership-preserving factor-wise contextualization versus "
            "pre-fused joint contextualization; uniform diffusion is linear node mixing, "
            "but Fusion nonlinearity occurs before versus after diffusion."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
