from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
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


def _finite_stats(value: torch.Tensor) -> dict[str, object]:
    finite = bool(torch.isfinite(value).all().item())
    norms = value.norm(dim=-1)
    std = value.std(dim=0, unbiased=False)
    return {
        "finite": finite,
        "mean_norm": float(norms.mean().item()),
        "min_norm": float(norms.min().item()),
        "max_norm": float(norms.max().item()),
        "per_dim_std_mean": float(std.mean().item()),
        "per_dim_std_min": float(std.min().item()),
        "per_dim_std_max": float(std.max().item()),
        "effective_rank": _effective_rank(value),
    }


def _load_cfg(repo_root: Path, device: str) -> object:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs")):
        return hydra.compose(
            config_name="config",
            overrides=[
                "dataset=Movies",
                "task=nc",
                "model=ored_mag",
                "num_runs=1",
                "seed=42",
                f"device={device}",
                "task.evaluate_test=false",
            ],
        )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="ORED-1 P0 feature-only diagnostics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ored/o1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = _load_cfg(repo_root, args.device)
    data = load_mag_data(cfg, "nc", 42)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")

    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    x = data.x
    x_device = x.to(device)
    edge_index = data.edge_index.to(device)

    edge_variants = {
        "original": edge_index,
        "empty": torch.empty((2, 0), dtype=torch.long, device=device),
        "shuffled": edge_index[:, torch.randperm(edge_index.size(1), device=device)],
        "random": torch.randint(0, data.num_nodes, edge_index.shape, dtype=torch.long, device=device),
    }
    with torch.no_grad():
        reference = model(x_device, edge_variants["original"])[0]
        topology_invariance: dict[str, dict[str, object]] = {}
        for name, variant_edges in edge_variants.items():
            candidate = model(x_device, variant_edges)[0]
            delta = (candidate - reference).abs()
            topology_invariance[name] = {
                "max_abs_delta": float(delta.max().item()),
                "mean_abs_delta": float(delta.mean().item()),
                "allclose": bool(torch.allclose(candidate, reference, rtol=1e-5, atol=1e-6)),
            }

    factors = model.encode_factors(
        x,
        edge_index=edge_index,
        batch_size=args.batch_size,
        device=device,
    )
    factor_stats = {name: _finite_stats(value) for name, value in factors.items() if name != "z_local"}
    factor_stats["z_local"] = _finite_stats(factors["z_local"])

    common_alignment = F.cosine_similarity(factors["c_t"], factors["c_v"], dim=-1)
    private_alignment = F.cosine_similarity(factors["p_t"], factors["p_v"], dim=-1)
    common_private_t = F.cosine_similarity(factors["c_t"], factors["p_t"], dim=-1)
    common_private_v = F.cosine_similarity(factors["c_v"], factors["p_v"], dim=-1)

    rec_sum = {"text": 0.0, "visual": 0.0}
    rec_count = {"text": 0, "visual": 0}
    for start in range(0, data.num_nodes, args.batch_size):
        end = min(start + args.batch_size, data.num_nodes)
        factors_batch = model.factorizer(
            *model._split_modalities(x[start:end].to(device))
        )
        rec_text = model.recon_text_head(factors_batch["c_t"], factors_batch["p_t"])
        rec_visual = model.recon_visual_head(factors_batch["c_v"], factors_batch["p_v"])
        rec_sum["text"] += float((rec_text - factors_batch["h_t"]).square().sum().item())
        rec_sum["visual"] += float((rec_visual - factors_batch["h_v"]).square().sum().item())
        rec_count["text"] += int(factors_batch["h_t"].numel())
        rec_count["visual"] += int(factors_batch["h_v"].numel())

    output = {
        "dataset": "Movies",
        "seed": 42,
        "checkpoint": str(args.checkpoint),
        "uses_test_labels_or_metrics": False,
        "topology_invariance": topology_invariance,
        "common_alignment": {
            "mean_cosine_c_t_c_v": float(common_alignment.mean().item()),
            "min_cosine_c_t_c_v": float(common_alignment.min().item()),
            "max_cosine_c_t_c_v": float(common_alignment.max().item()),
        },
        "factor_pair_cosines": {
            "mean_cosine_p_t_p_v": float(private_alignment.mean().item()),
            "mean_cosine_c_t_p_t": float(common_private_t.mean().item()),
            "mean_cosine_c_v_p_v": float(common_private_v.mean().item()),
        },
        "reconstruction_mse": {
            "text": rec_sum["text"] / max(rec_count["text"], 1),
            "visual": rec_sum["visual"] / max(rec_count["visual"], 1),
        },
        "factor_stats": factor_stats,
        "nan_or_inf_detected": not all(bool(stats["finite"]) for stats in factor_stats.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "factor_diagnostics.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
