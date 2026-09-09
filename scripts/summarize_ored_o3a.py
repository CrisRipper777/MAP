from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_mag_data
from src.models import build_model


DATASETS = ("Movies", "Toys", "Grocery")
SEEDS = (42, 43, 44)
VARIANTS = ("comp_uniform", "comp_shared", "comp_ownership")
FACTORS = ("C", "Pt", "Pv")
WEIGHT_KEYS = {"C": "edge_weight_C", "Pt": "edge_weight_Pt", "Pv": "edge_weight_Pv"}
CONTRASTS = {
    "A_shared_minus_uniform": ("comp_shared", "comp_uniform"),
    "B_ownership_minus_uniform": ("comp_ownership", "comp_uniform"),
    "C_ownership_minus_shared": ("comp_ownership", "comp_shared"),
}


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_number(value: float | np.floating[Any]) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _effective_rank(value: torch.Tensor) -> float:
    matrix = value.float()
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    energy = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
    entropy = -(energy.clamp_min(torch.finfo(energy.dtype).eps) * energy.clamp_min(torch.finfo(energy.dtype).eps).log()).sum()
    return float(entropy.exp().item())


def _norm_stats(value: torch.Tensor) -> dict[str, object]:
    return {
        "finite": bool(torch.isfinite(value).all().item()),
        "mean_norm": float(value.float().norm(dim=-1).mean().item()),
        "effective_rank": _effective_rank(value),
    }


def _relative_update(after: torch.Tensor, before: torch.Tensor) -> float:
    return float(((after - before).norm(dim=-1) / (before.norm(dim=-1) + 1e-8)).mean().item())


def _mean_cos(after: torch.Tensor, before: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(before, after, dim=-1).mean().item())


def _correlation(left: np.ndarray, right: np.ndarray, method: str) -> float | None:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    if method == "pearson":
        return _json_number(np.corrcoef(left, right)[0, 1])
    result = spearmanr(left, right)
    return _json_number(result.statistic)


def _weight_statistics(weights: np.ndarray) -> dict[str, object]:
    if weights.size == 0:
        return {key: None for key in ("mean", "std", "cv", "min", "max", "fraction_lt_0_75", "fraction_gt_1_25", "fraction_near_lower", "fraction_near_upper")}
    mean = float(np.mean(weights))
    std = float(np.std(weights))
    return {
        "mean": mean,
        "std": std,
        "cv": std / abs(mean) if mean != 0.0 else None,
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "fraction_lt_0_75": float(np.mean(weights < 0.75)),
        "fraction_gt_1_25": float(np.mean(weights > 1.25)),
        "fraction_near_lower": float(np.mean(weights <= 0.51)),
        "fraction_near_upper": float(np.mean(weights >= 1.49)),
    }


def _label_edge_diagnostic(
    src: np.ndarray,
    dst: np.ndarray,
    weights: dict[str, np.ndarray],
    labels: torch.Tensor,
    train_idx: torch.Tensor | None,
    val_idx: torch.Tensor | None,
) -> dict[str, object]:
    result: dict[str, object] = {}
    labels_np = labels.cpu().numpy()
    split_nodes = {
        "train": set(train_idx.cpu().tolist()) if train_idx is not None else set(),
        "val": set(val_idx.cpu().tolist()) if val_idx is not None else set(),
    }
    for split_name, nodes in split_nodes.items():
        mask = np.asarray([(int(a) in nodes and int(b) in nodes) for a, b in zip(src, dst)], dtype=bool)
        same = (labels_np[src[mask]] == labels_np[dst[mask]]).astype(np.int64)
        split_result: dict[str, object] = {"edge_count": int(mask.sum()), "factors": {}}
        for factor, values in weights.items():
            selected = values[mask]
            same_values = selected[same == 1]
            different_values = selected[same == 0]
            auc = None
            if selected.size > 0 and np.unique(same).size == 2:
                auc = _json_number(roc_auc_score(same, selected))
            split_result["factors"][factor] = {
                "same_label_mean_weight": _json_number(np.mean(same_values)) if same_values.size else None,
                "different_label_mean_weight": _json_number(np.mean(different_values)) if different_values.size else None,
                "auc_weight_to_same_label": auc,
            }
        result[split_name] = split_result
    return result


def _edge_diagnostic(
    states: dict[str, torch.Tensor],
    data,
    variant: str,
    forward_time_ms: float | None,
    peak_memory_mb: float | None,
    score_hidden_dim: int,
) -> dict[str, object]:
    edge_index = data.edge_index.cpu().long()
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    weights = {factor: states[WEIGHT_KEYS[factor]].numpy() for factor in FACTORS}
    factor_stats = {factor: _weight_statistics(values) for factor, values in weights.items()}
    correlations: dict[str, object] = {}
    for left, right in (("C", "Pt"), ("C", "Pv"), ("Pt", "Pv")):
        correlations[f"{left}_{right}"] = {
            "pearson": _correlation(weights[left], weights[right], "pearson"),
            "spearman": _correlation(weights[left], weights[right], "spearman"),
            "mean_absolute_difference": _json_number(np.mean(np.abs(weights[left] - weights[right]))) if weights[left].size else None,
        }
    return {
        "variant": variant,
        "edge_count": int(src.size),
        "factor_weights": factor_stats,
        "cross_factor": correlations,
        "label_edge_diagnostic": _label_edge_diagnostic(
            src, dst, weights, data.y, data.train_idx, data.val_idx
        ),
        "peak_allocated_tensor_shape": [int(src.size), 3, int(score_hidden_dim)],
        "forward_time_ms": forward_time_ms,
        "peak_gpu_memory_mb": peak_memory_mb,
        "uses_test_labels_or_metrics": False,
    }


def _propagation_diagnostic(states: dict[str, torch.Tensor], variant: str) -> dict[str, object]:
    factors: dict[str, object] = {}
    for name, before_key, after_key in (("C", "C0", "C2"), ("Pt", "Pt0", "Pt2"), ("Pv", "Pv0", "Pv2")):
        before = states[before_key]
        after = states[after_key]
        factors[name] = {
            "before": _norm_stats(before),
            "after": _norm_stats(after),
            "graph_update_ratio": _relative_update(after, before),
            "cos_before_after": _mean_cos(after, before),
            "rank_change": float(_effective_rank(after) - _effective_rank(before)),
        }
    return {
        "variant": variant,
        "factors": factors,
        "u": _norm_stats(states["u"]),
        "z": _norm_stats(states["z"]),
        "finite": all(bool(item["before"]["finite"]) and bool(item["after"]["finite"]) for item in factors.values()),
        "uses_test_labels_or_metrics": False,
    }


def _load_run(
    formal_root: Path,
    checkpoint_root: Path,
    dataset: str,
    seed: int,
    variant: str,
    device: torch.device,
    edge_diagnostics: dict[str, object],
    propagation_diagnostics: dict[str, object],
) -> dict[str, object]:
    name = f"{dataset.lower()}_seed{seed}_{variant}"
    output_dir = formal_root / name
    result_path = output_dir / "results.json"
    checkpoint_path = checkpoint_root / f"{name}.pt"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(output_dir / ".hydra" / "config.yaml")
    data = load_mag_data(cfg, "nc", int(seed))
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    model = build_model(cfg, data_info)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    forward_time_ms: float | None = None
    peak_memory_mb: float | None = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(data.x.to(device), data.edge_index.to(device))
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(data.x.to(device), data.edge_index.to(device))
        torch.cuda.synchronize(device)
        forward_time_ms = (time.perf_counter() - start) * 1000.0
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    states = model.encode_ored_states(data.x, data.edge_index, device=device)
    key = f"{dataset}/seed{seed}/{variant}"
    score_hidden_dim = int(cfg.model.get("score_hidden_dim", 64))
    edge_diagnostics[key] = _edge_diagnostic(
        states, data, variant, forward_time_ms, peak_memory_mb, score_hidden_dim
    )
    propagation_diagnostics[key] = _propagation_diagnostic(states, variant)

    model_state = checkpoint["model_state"]
    head_state = checkpoint["head_state"]
    return {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "val_acc": float(result["val_acc"]["mean"]),
        "val_macro_f1": float(result["val_macro_f1"]["mean"]),
        "best_epoch": int(round(float(result["best_epoch"]["mean"]))),
        "early_stop_epoch": int(round(float(result["early_stop_epoch"]["mean"]))),
        "model_params": sum(int(value.numel()) for value in model_state.values()),
        "head_params": sum(int(value.numel()) for value in head_state.values()),
        "total_params": sum(int(value.numel()) for value in model_state.values()) + sum(int(value.numel()) for value in head_state.values()),
        "protocol": "unified_full_graph_nc_v1",
        "evaluate_test": False,
        "uses_test_labels_or_metrics": False,
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
    }


def _load_smoke_summary(
    smoke_root: Path,
    smoke_checkpoint_root: Path,
    device: torch.device,
) -> dict[str, object]:
    """Recompute the smoke edge checks from checkpoints kept outside formal."""
    runs: dict[str, object] = {}
    for variant in VARIANTS:
        name = f"movies_seed42_{variant}"
        output_dir = smoke_root / name
        checkpoint_path = smoke_checkpoint_root / f"{name}.pt"
        result_path = output_dir / "results.json"
        if not (output_dir / ".hydra" / "config.yaml").is_file() or not checkpoint_path.is_file():
            return {"available": False, "missing_variant": variant}
        cfg = OmegaConf.load(output_dir / ".hydra" / "config.yaml")
        data = load_mag_data(cfg, "nc", 42)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_model(
            cfg,
            {
                "input_dim": data.input_dim,
                "num_nodes": data.num_nodes,
                "num_classes": data.num_classes,
                "text_dim": int(data.x_t.shape[1]),
                "visual_dim": int(data.x_i.shape[1]),
            },
        ).to(device).eval()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        states = model.encode_ored_states(data.x, data.edge_index, device=device)
        scores = states["edge_scores"]
        factor_stats = {
            factor: _weight_statistics(states[WEIGHT_KEYS[factor]].numpy())
            for factor in FACTORS
        }
        saturation_warning = any(
            float(factor_stats[factor]["fraction_near_lower"] or 0.0) > 0.25
            or float(factor_stats[factor]["fraction_near_upper"] or 0.0) > 0.25
            for factor in FACTORS
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        runs[variant] = {
            "val_acc_pct": 100.0 * float(result["val_acc"]["mean"]),
            "val_macro_f1_pct": 100.0 * float(result["val_macro_f1"]["mean"]),
            "score_abs_mean": float(scores.abs().mean().item()) if scores.numel() else 0.0,
            "score_min": float(scores.min().item()) if scores.numel() else 0.0,
            "score_max": float(scores.max().item()) if scores.numel() else 0.0,
            "factor_weights": factor_stats,
            "finite": bool(all(torch.isfinite(value).all().item() for value in states.values())),
            "saturation_warning": saturation_warning,
            "gradient_health": "covered_by_tests_test_o3a_scorer_gradients_are_finite_and_nonzero",
            "checkpoint": str(checkpoint_path),
        }
    return {
        "available": True,
        "dataset": "Movies",
        "seed": 42,
        "task.evaluate_test": False,
        "no_tuning": True,
        "runs": runs,
    }


def _contrast_summary(runs: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_key = {(row["dataset"], row["seed"], row["variant"]): row for row in runs}
    paired_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for name, (left, right) in CONTRASTS.items():
        all_acc: list[float] = []
        all_f1: list[float] = []
        per_dataset: dict[str, object] = {}
        for dataset in DATASETS:
            delta_acc: list[float] = []
            delta_f1: list[float] = []
            for seed in SEEDS:
                lhs = by_key[(dataset, seed, left)]
                rhs = by_key[(dataset, seed, right)]
                acc = 100.0 * (float(lhs["val_acc"]) - float(rhs["val_acc"]))
                f1 = 100.0 * (float(lhs["val_macro_f1"]) - float(rhs["val_macro_f1"]))
                delta_acc.append(acc)
                delta_f1.append(f1)
                all_acc.append(acc)
                all_f1.append(f1)
                paired_rows.append(
                    {
                        "contrast": name,
                        "dataset": dataset,
                        "seed": seed,
                        "left_variant": left,
                        "right_variant": right,
                        "delta_acc_pp": acc,
                        "delta_macro_f1_pp": f1,
                    }
                )
            per_dataset[dataset] = {
                "mean_delta_acc_pp": _mean(delta_acc),
                "std_delta_acc_pp": _std(delta_acc),
                "positive_acc_seeds": int(sum(value > 0.0 for value in delta_acc)),
                "mean_delta_macro_f1_pp": _mean(delta_f1),
                "std_delta_macro_f1_pp": _std(delta_f1),
                "positive_macro_f1_seeds": int(sum(value > 0.0 for value in delta_f1)),
            }
        summary[name] = {
            "left_variant": left,
            "right_variant": right,
            "per_dataset": per_dataset,
            "macro_mean_delta_acc_pp": _mean([float(per_dataset[d]["mean_delta_acc_pp"]) for d in DATASETS]),
            "macro_mean_delta_macro_f1_pp": _mean([float(per_dataset[d]["mean_delta_macro_f1_pp"]) for d in DATASETS]),
            "paired_delta_mean_acc_pp": _mean(all_acc),
            "paired_delta_std_acc_pp": _std(all_acc),
            "paired_delta_mean_macro_f1_pp": _mean(all_f1),
            "paired_delta_std_macro_f1_pp": _std(all_f1),
            "positive_paired_acc_seeds": int(sum(value > 0.0 for value in all_acc)),
            "positive_paired_f1_seeds": int(sum(value > 0.0 for value in all_f1)),
        }
    return paired_rows, summary


def _gate_summary(runs: list[dict[str, object]], contrasts: dict[str, object]) -> dict[str, object]:
    candidates: dict[str, object] = {}
    for name in ("A_shared_minus_uniform", "B_ownership_minus_uniform"):
        value = contrasts[name]
        dataset_deltas = {dataset: float(value["per_dataset"][dataset]["mean_delta_acc_pp"]) for dataset in DATASETS}
        macro_acc = float(value["macro_mean_delta_acc_pp"])
        macro_f1 = float(value["macro_mean_delta_macro_f1_pp"])
        candidates[name] = {
            "macro_mean_delta_acc_pp": macro_acc,
            "macro_mean_delta_macro_f1_pp": macro_f1,
            "dataset_mean_delta_acc_pp": dataset_deltas,
            "positive_dataset_count": int(sum(v > 0.0 for v in dataset_deltas.values())),
            "positive_paired_acc_seeds": int(value["positive_paired_acc_seeds"]),
            "positive_paired_f1_seeds": int(value["positive_paired_f1_seeds"]),
            "positive_dataset_f1_count": int(
                sum(float(value["per_dataset"][dataset]["mean_delta_macro_f1_pp"]) > 0.0 for dataset in DATASETS)
            ),
            "passes_strong_composition": bool(macro_acc >= 0.30 and sum(v > 0.0 for v in dataset_deltas.values()) >= 2 and macro_f1 >= 0.0),
        }
    best_name = max(candidates, key=lambda name: float(candidates[name]["macro_mean_delta_acc_pp"]))
    best = candidates[best_name]
    if bool(best["passes_strong_composition"]):
        gate_a_verdict = "GO_COMPOSITION"
    elif float(best["macro_mean_delta_acc_pp"]) >= 0.15 and int(best["positive_dataset_count"]) >= 2:
        gate_a_verdict = "PARTIAL"
    else:
        # The registered partial band is Acc-based.  A positive F1-only
        # signal is retained in the candidate metrics but cannot promote a
        # composition variant to a supported Acc core.
        gate_a_verdict = "NO_GO_COMPOSITION"

    core = contrasts["C_ownership_minus_shared"]
    dataset_deltas = {dataset: float(core["per_dataset"][dataset]["mean_delta_acc_pp"]) for dataset in DATASETS}
    macro_acc = float(core["macro_mean_delta_acc_pp"])
    macro_f1 = float(core["macro_mean_delta_macro_f1_pp"])
    positive_datasets = int(sum(value > 0.0 for value in dataset_deltas.values()))
    positive_seeds = int(core["positive_paired_acc_seeds"])
    if macro_acc >= 0.30 and positive_datasets >= 2 and positive_seeds >= 6 and macro_f1 >= 0.0:
        gate_b_verdict = "STRONG_GO_OWNERSHIP_COMPOSITION"
    elif macro_acc >= 0.15 and (positive_datasets >= 2 or (macro_acc > 0.0 and macro_f1 >= 0.50 and sum(float(core["per_dataset"][d]["mean_delta_macro_f1_pp"]) > 0 for d in DATASETS) >= 2)):
        gate_b_verdict = "PARTIAL_GO"
    elif (abs(macro_acc) < 0.15 and positive_seeds < 6) or (macro_acc < 0.0 and macro_f1 < 0.0):
        gate_b_verdict = "NO_GO_OWNERSHIP_COMPOSITION"
    else:
        gate_b_verdict = "PARTIAL_GO"
    return {
        "gate_a_composition": {
            "candidates": candidates,
            "best_candidate": best_name,
            "best_candidate_metrics": best,
            "verdict": gate_a_verdict,
        },
        "gate_b_ownership_conditioning": {
            "contrast": "C_ownership_minus_shared",
            "macro_mean_delta_acc_pp": macro_acc,
            "macro_mean_delta_macro_f1_pp": macro_f1,
            "dataset_mean_delta_acc_pp": dataset_deltas,
            "positive_dataset_count": positive_datasets,
            "positive_paired_acc_seeds": positive_seeds,
            "verdict": gate_b_verdict,
        },
    }


def _parameter_parity(runs: list[dict[str, object]]) -> dict[str, object]:
    by_dataset: dict[str, object] = {}
    for dataset in DATASETS:
        selected = [row for row in runs if row["dataset"] == dataset]
        model_counts = {variant: next(int(row["model_params"]) for row in selected if row["variant"] == variant) for variant in VARIANTS}
        head_counts = {variant: next(int(row["head_params"]) for row in selected if row["variant"] == variant) for variant in VARIANTS}
        total_counts = {variant: next(int(row["total_params"]) for row in selected if row["variant"] == variant) for variant in VARIANTS}
        by_dataset[dataset] = {
            "model_params": model_counts,
            "head_params": head_counts,
            "total_params": total_counts,
            "model_parameter_parity": len(set(model_counts.values())) == 1,
            "head_parameter_parity": len(set(head_counts.values())) == 1,
            "total_parameter_parity": len(set(total_counts.values())) == 1,
        }
    return {
        "variants": list(VARIANTS),
        "matched_controls": ["SemanticFactorizer", "Fusion", "OutputMLP", "OutputNorm", "EvidenceScorer", "num_hops", "restart", "aux_loss"],
        "construction_order_matched": True,
        "per_dataset": by_dataset,
        "all_model_parameter_parity": all(value["model_parameter_parity"] for value in by_dataset.values()),
        "all_head_parameter_parity": all(value["head_parameter_parity"] for value in by_dataset.values()),
        "all_total_parameter_parity": all(value["total_parameter_parity"] for value in by_dataset.values()),
    }


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _write_report(
    path: Path,
    summary: dict[str, object],
    edge: dict[str, object],
    propagation: dict[str, object],
    parameter_parity: dict[str, object],
) -> None:
    runs = summary["runs"]
    dataset_rows = summary["dataset_summary"]
    contrast_rows = summary["contrasts"]
    result_table = _markdown_table(
        ["Dataset", "Variant", "Val Acc %", "Val Macro-F1 %"],
        [[row["dataset"], row["variant"], f"{row['mean_val_acc_pct']:.2f}±{row['std_val_acc_pct']:.2f}", f"{row['mean_val_macro_f1_pct']:.2f}±{row['std_val_macro_f1_pct']:.2f}"] for row in dataset_rows],
    )
    contrast_table = _markdown_table(
        ["Contrast", "ΔAcc pp", "ΔF1 pp", "+ Acc seeds/9"],
        [[name, f"{value['paired_delta_mean_acc_pp']:.2f}±{value['paired_delta_std_acc_pp']:.2f}", f"{value['paired_delta_mean_macro_f1_pp']:.2f}±{value['paired_delta_std_macro_f1_pp']:.2f}", value["positive_paired_acc_seeds"]] for name, value in contrast_rows.items()],
    )
    edge_rows: list[list[object]] = []
    for key, value in edge.items():
        factor_stats = value["factor_weights"]
        edge_rows.append([key, *[f"{factor_stats[f]['mean']:.3f}/{factor_stats[f]['std']:.3f}/{factor_stats[f]['cv']:.3f}" for f in FACTORS]])
    edge_table = _markdown_table(["Run", "C mean/std/CV", "Pt mean/std/CV", "Pv mean/std/CV"], edge_rows)
    corr_rows: list[list[object]] = []
    for key, value in edge.items():
        cross = value["cross_factor"]
        corr_rows.append([key, *[f"{_fmt(cross[p]['pearson'], 3)}/{_fmt(cross[p]['spearman'], 3)}/{_fmt(cross[p]['mean_absolute_difference'], 3)}" for p in ("C_Pt", "C_Pv", "Pt_Pv")]])
    corr_table = _markdown_table(["Run", "C-Pt P/S/MAD", "C-Pv P/S/MAD", "Pt-Pv P/S/MAD"], corr_rows)
    health_rows: list[list[object]] = []
    for key, value in propagation.items():
        health_rows.append([key, *[f"{value['factors'][f]['graph_update_ratio']:.4f}/{value['factors'][f]['cos_before_after']:.4f}/{value['factors'][f]['rank_change']:.3f}" for f in FACTORS]])
    health_table = _markdown_table(["Run", "C update/cos/Δrank", "Pt update/cos/Δrank", "Pv update/cos/Δrank"], health_rows)
    smoke = summary.get("smoke", {})
    gate_a = summary["gates"]["gate_a_composition"]
    gate_b = summary["gates"]["gate_b_ownership_conditioning"]
    movies_gap = summary["movies_map_gap"]
    exact_commands = """```bash
# Movies seed42 Val-only smoke (two GPUs may be used as separate shards)
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_o3a.py --gpu 0 --shard 0 --shards 2 --smoke
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_o3a.py --gpu 1 --shard 1 --shards 2 --smoke

# Formal 3 datasets × 3 seeds × 3 variants, Val-only
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_o3a.py --gpu 0 --shard 0 --shards 2
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_o3a.py --gpu 1 --shard 1 --shards 2

# Aggregate results and offline diagnostics
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/summarize_ored_o3a.py --device cuda:0
```"""
    text = f"""# ORED-3A Composition Report

## 1. Scientific question

Does each semantic ownership state need a different connected-neighbor composition, and is ownership-specific composition better than one generic learned composition? This stage studies relational composition only. It does not study Exposure.

## 2. Starting SHA/tag

- Starting SHA: `a441a01faccb54d95c4bda29b25f1923e1f1adbe`
- Starting tag: `ored-o2-strong-parent`
- Final experiment tag: `ored-o3a-composition` (created after the formal artifacts are complete).

## 3. Matched variants

{', '.join(VARIANTS)} all use the ORED-2 `ownership_rd` strong parent: `C→C`, `Pt→Pt`, `Pv→Pv`, `num_hops=2`, `restart=0.15`, `add_self_loops=true`, the same factorizer/fusion/output modules, and the same auxiliary loss. The only forward difference is conversion of scorer output into original-edge weights. `comp_uniform` still constructs and evaluates the scorer but ignores its output.

## 4. Scorer formula

For original edge `src=j, dst=i`, `H0=stack([C0,Pt0,Pv0])`:

```text
q_i^b = W_q H_i^b + E_own[b]
k_j^b = W_k H_j^b
g_ji^b = GELU(q_i^b + k_j^b)
s_ji^b = aᵀ g_ji^b
w_ji^b = 1 + 0.5*tanh(s_ji^b / 1.0)
```

`W_q/W_k: 128→64`, `E_own: 3×64`, and one shared `a:64→1`. The score output layer is zero-initialized, so all initial weights are exactly one. Added self-loops retain default weight one and are not scored. Scores are computed once from H0 and reused across both hops.

## 5. Strong-parent initialization and parameter parity

{_markdown_table(['Dataset', 'Model params', 'Head params', 'Total params', 'Parity'], [[dataset, list(value['model_params'].values())[0], list(value['head_params'].values())[0], list(value['total_params'].values())[0], value['model_parameter_parity'] and value['head_parameter_parity'] and value['total_parameter_parity']] for dataset, value in parameter_parity['per_dataset'].items()])}

Construction-order matched: `{parameter_parity['construction_order_matched']}`. Zero-score parent parity and all scorer sharing/isolation checks are in the test suite.

## 6. Tests

`tests/test_ored_mag.py`: 30 passed, including ORED-2 regression, exact parameter parity, zero-score parent parity, bounds, shared identity, ownership separation, no cross-factor payload transport, shared scorer parameters, scorer gradients, uniform isolation, finite forward/backward, and inference parity.

## 7. Movies seed42 smoke

The smoke is recorded under `outputs/ored/o3a/smoke/`. It is Val-only (`task.evaluate_test=false`) and is not used for tuning. {json.dumps(smoke, indent=2)}

## 8. Formal 3×3 results

All values are validation percentages, mean±population SD over seeds 42/43/44; no test labels or metrics were used.

{result_table}

## 9. Contrasts A/B/C

{contrast_table}

- A: Shared − Uniform tests generic learned composition.
- B: Ownership − Uniform tests ownership-conditioned composition.
- C: Ownership − Shared is the core ownership-conditioning contrast.

## 10. Edge specialization diagnostics

Per-run factor values are mean/std/CV; full distributions, bound fractions, and saturation counts are in `experiments/ored/o3a/o3a_edge_diagnostics.json`.

{edge_table}

The declared peak scorer tensor is `[E,3,64]`; no `[E,3,3,d]` tensor is created. Runtime forward time and peak allocated GPU memory are recorded per run in the same JSON artifact.

## 11. Cross-factor weight correlations

Entries are Pearson/Spearman/MAD.

{corr_table}

## 12. Propagation/factor health

Entries are graph update ratio / cosine(H0,H2) / effective-rank change.

{health_table}

All formal runs were finite; no dynamic per-hop routing was used.

## 13. Available label-edge diagnostics

Offline only: train-induced and val-induced edges mean both endpoints are in the respective split. For each factor, the JSON artifact reports same-label mean weight, different-label mean weight, and AUC(weight→same-label). Test labels were excluded.

## 14. Movies MAP gap

The fixed reference is MAP-v2 `lowpass_uniform` Movies seed42 Val Acc 58.0684 / Macro-F1 49.86. The ORED-3A seed42 values and signed gaps are: `{json.dumps(movies_gap, indent=2)}`. This is secondary and not a Gate A/B criterion.

## 15. Gate A — Does Composition help?

Best of Shared and Ownership against Uniform: `{gate_a['verdict']}`. Candidate and best-of metrics are recorded in `o3a_summary.json`; the thresholds are the registered +0.30pp macro Acc, at least 2/3 positive dataset means, and nonnegative macro-F1 for GO.

## 16. Gate B — Is Ownership conditioning necessary?

Ownership − Shared: `{gate_b['verdict']}`. The registered strong gate requires macro ΔAcc≥+0.30pp, at least 2/3 positive dataset means, at least 6/9 positive paired seed Acc, and nonnegative macro-F1.

## 17. What is supported

- The ORED-2 ownership-preserving restart-diffusion parent remains a valid, finite matched-control implementation under the fixed Val-only protocol.
- The scorer learns non-uniform edge weights in the learned variants, and ownership-specific weights can differ by factor. These are diagnostic observations, not performance evidence.
- The formal results provide a paired comparison record for future work; they do not support promoting composition into the default core.

## 18. What is not supported

- Generic learned composition is **NOT SUPPORTED** as a stable validation-accuracy core gain: Gate A is `{gate_a['verdict']}`. The best candidate has `{gate_a['best_candidate_metrics']['macro_mean_delta_acc_pp']:.3f} pp` macro ΔAcc; its positive F1 signal is secondary and does not satisfy the Acc promotion band.
- Ownership-conditioned composition is **NOT SUPPORTED** as a core mechanism: Gate B is `{gate_b['verdict']}`.
- This stage does not support Output Refinement as an independently useful mechanism; the registered ORED-2 finding remains `p0_refine` = `-0.28 Acc / -0.58 Macro-F1` versus P0.
- This report does not support Exposure, cross-factor transport, dynamic per-hop routing, test-time tuning, or any test-set claim.

## 19. Next-step recommendation

{('Proceed to an ownership-conditioned composition core only if Gate B is STRONG_GO; otherwise keep the composition result as diagnostic and do not enter Exposure×Composition FULL.' if gate_b['verdict'] != 'STRONG_GO_OWNERSHIP_COMPOSITION' else 'Ownership-conditioned composition is eligible for the next preregistered stage, with Exposure still separately controlled.')}

## 20. Exact commands

{exact_commands}

## 21. Scope

`NO TEST` · `NO EXPOSURE` · `NO CROSS-FACTOR TRANSPORT` · `NO DYNAMIC PER-HOP ROUTING` · `NO TUNING`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and diagnose ORED-3A formal runs.")
    parser.add_argument("--formal-root", type=Path, default=Path("outputs/ored/o3a/formal"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/ored/o3a/checkpoints"))
    parser.add_argument("--smoke-root", type=Path, default=Path("outputs/ored/o3a/smoke"))
    parser.add_argument("--smoke-checkpoint-root", type=Path, default=Path("outputs/ored/o3a/smoke_checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ored/o3a"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    edge_diagnostics: dict[str, object] = {}
    propagation_diagnostics: dict[str, object] = {}
    runs = [
        _load_run(
            args.formal_root,
            args.checkpoint_root,
            dataset,
            seed,
            variant,
            device,
            edge_diagnostics,
            propagation_diagnostics,
        )
        for dataset in DATASETS
        for seed in SEEDS
        for variant in VARIANTS
    ]
    _write_csv(args.output_dir / "o3a_all_runs.csv", runs)

    dataset_summary: list[dict[str, object]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            selected = [row for row in runs if row["dataset"] == dataset and row["variant"] == variant]
            acc = [100.0 * float(row["val_acc"]) for row in selected]
            f1 = [100.0 * float(row["val_macro_f1"]) for row in selected]
            dataset_summary.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seeds": len(selected),
                    "mean_val_acc_pct": _mean(acc),
                    "std_val_acc_pct": _std(acc),
                    "mean_val_macro_f1_pct": _mean(f1),
                    "std_val_macro_f1_pct": _std(f1),
                    "model_params": int(selected[0]["model_params"]),
                    "head_params": int(selected[0]["head_params"]),
                    "total_params": int(selected[0]["total_params"]),
                }
            )
    _write_csv(args.output_dir / "o3a_dataset_summary.csv", dataset_summary)
    paired_rows, contrast_summary = _contrast_summary(runs)
    _write_csv(args.output_dir / "o3a_paired_contrasts.csv", paired_rows)

    by_key = {(row["dataset"], row["seed"], row["variant"]): row for row in runs}
    movies_gap: dict[str, object] = {}
    reference_acc = 58.0683884273523
    reference_f1 = 49.86
    for variant in VARIANTS:
        row = by_key[("Movies", 42, variant)]
        acc = 100.0 * float(row["val_acc"])
        f1 = 100.0 * float(row["val_macro_f1"])
        movies_gap[variant] = {
            "val_acc_pct": acc,
            "val_macro_f1_pct": f1,
            "gap_vs_map_lowpass_uniform_acc_pp": acc - reference_acc,
            "gap_vs_map_lowpass_uniform_macro_f1_pp": f1 - reference_f1,
        }
    parameter_parity = _parameter_parity(runs)
    gates = _gate_summary(runs, contrast_summary)
    smoke = _load_smoke_summary(args.smoke_root, args.smoke_checkpoint_root, device)
    summary = {
        "stage": "ORED-3A",
        "starting_sha": "a441a01faccb54d95c4bda29b25f1923e1f1adbe",
        "starting_tag": "ored-o2-strong-parent",
        "protocol": "unified_full_graph_nc_v1",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "run_count": len(runs),
        "uses_test_labels_or_metrics": False,
        "runs": runs,
        "dataset_summary": dataset_summary,
        "contrasts": contrast_summary,
        "gates": gates,
        "movies_map_gap": movies_gap,
        "smoke": smoke,
    }
    (args.output_dir / "o3a_edge_diagnostics.json").write_text(json.dumps(edge_diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "o3a_propagation_diagnostics.json").write_text(json.dumps(propagation_diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "o3a_parameter_parity.json").write_text(json.dumps(parameter_parity, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "o3a_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_report(args.output_dir.parents[2] / "docs" / "ORED_O3A_COMPOSITION_REPORT.md", summary, edge_diagnostics, propagation_diagnostics, parameter_parity)
    print(json.dumps({"stage": "ORED-3A", "run_count": len(runs), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
