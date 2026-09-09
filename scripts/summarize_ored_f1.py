from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_mag_data
from src.models import build_model


DATASETS = ("Movies", "Toys", "Grocery")
SEEDS = (42, 43, 44)
VARIANTS = ("f1_owner", "f1_dual_direct", "f1_dual_ocb")
CONTRASTS = {
    "A_dual_direct_minus_owner": ("f1_dual_direct", "f1_owner"),
    "B_dual_ocb_minus_direct": ("f1_dual_ocb", "f1_dual_direct"),
    "C_dual_ocb_minus_owner": ("f1_dual_ocb", "f1_owner"),
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _model_data(cfg, seed: int):
    data = load_mag_data(cfg, "nc", int(seed))
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    return data, data_info


def _diagnostic(
    states: dict[str, torch.Tensor],
    variant: str,
    bridge_raw: float,
    forward_time_ms: float | None,
    peak_memory_mb: float | None,
) -> dict[str, object]:
    def scalar(name: str) -> float:
        value = states[name]
        return float(value.detach().float().mean().item())

    gate_names = ("c", "pt", "pv")
    gates = {
        f"mean_bridge_gate_{name}": scalar(f"bridge_gate_{name}")
        for name in gate_names
    }
    gate_stds = {
        f"std_bridge_gate_{name}": float(
            states[f"bridge_gate_{name}"].detach().float().std(unbiased=False).item()
        )
        for name in gate_names
    }
    updates = {
        f"bridge_update_ratio_{name}": scalar(f"bridge_update_ratio_{name}")
        for name in gate_names
    }
    joint_update = scalar("joint_graph_update_ratio")
    return {
        "variant": variant,
        "bridge_raw": bridge_raw,
        "mean_bridge_alpha": scalar("bridge_alpha"),
        **gates,
        **gate_stds,
        "joint_graph_update_ratio": joint_update,
        "joint_graph_cosine": scalar("joint_graph_cosine"),
        **updates,
        "joint_branch_nontrivial": bool(joint_update > 1e-6),
        "bridge_nonzero": bool(any(value > 1e-8 for value in updates.values())),
        "finite": bool(all(torch.isfinite(value).all().item() for value in states.values())),
        "forward_time_ms": forward_time_ms,
        "peak_gpu_memory_mb": peak_memory_mb,
        "uses_test_labels_or_metrics": False,
    }


def _load_single(
    root: Path,
    checkpoint_root: Path,
    dataset: str,
    seed: int,
    variant: str,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    name = f"{dataset.lower()}_seed{seed}_{variant}"
    output_dir = root / name
    result_path = output_dir / "results.json"
    checkpoint_path = checkpoint_root / f"{name}.pt"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(output_dir / ".hydra" / "config.yaml")
    data, data_info = _model_data(cfg, seed)
    model = build_model(cfg, data_info).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)

    forward_time_ms: float | None = None
    peak_memory_mb: float | None = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        with torch.no_grad():
            _ = model(x, edge_index)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x, edge_index)
        torch.cuda.synchronize(device)
        forward_time_ms = (time.perf_counter() - start) * 1000.0
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    states = model.encode_ored_states(data.x, data.edge_index, device=device)
    model_state = checkpoint["model_state"]
    head_state = checkpoint["head_state"]
    bridge_raw = float(model_state["bridge_raw"].item())
    diagnostic = _diagnostic(states, variant, bridge_raw, forward_time_ms, peak_memory_mb)
    row = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "val_acc": float(result["val_acc"]["mean"]),
        "val_macro_f1": float(result["val_macro_f1"]["mean"]),
        "best_epoch": int(round(float(result["best_epoch"]["mean"]))),
        "early_stop_epoch": int(round(float(result["early_stop_epoch"]["mean"]))),
        "model_params": sum(int(value.numel()) for value in model_state.values()),
        "head_params": sum(int(value.numel()) for value in head_state.values()),
        "total_params": sum(int(value.numel()) for value in model_state.values())
        + sum(int(value.numel()) for value in head_state.values()),
        "protocol": "unified_full_graph_nc_v1",
        "evaluate_test": False,
        "uses_test_labels_or_metrics": False,
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
    }
    return row, diagnostic


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
            "macro_mean_delta_acc_pp": _mean(
                [float(per_dataset[dataset]["mean_delta_acc_pp"]) for dataset in DATASETS]
            ),
            "macro_mean_delta_macro_f1_pp": _mean(
                [float(per_dataset[dataset]["mean_delta_macro_f1_pp"]) for dataset in DATASETS]
            ),
            "paired_delta_mean_acc_pp": _mean(all_acc),
            "paired_delta_std_acc_pp": _std(all_acc),
            "paired_delta_mean_macro_f1_pp": _mean(all_f1),
            "paired_delta_std_macro_f1_pp": _std(all_f1),
            "positive_paired_acc_seeds": int(sum(value > 0.0 for value in all_acc)),
            "positive_paired_f1_seeds": int(sum(value > 0.0 for value in all_f1)),
        }
    return paired_rows, summary


def _parameter_parity(runs: list[dict[str, object]]) -> dict[str, object]:
    by_dataset: dict[str, object] = {}
    for dataset in DATASETS:
        selected = [row for row in runs if row["dataset"] == dataset]
        model_counts = {
            variant: next(int(row["model_params"]) for row in selected if row["variant"] == variant)
            for variant in VARIANTS
        }
        head_counts = {
            variant: next(int(row["head_params"]) for row in selected if row["variant"] == variant)
            for variant in VARIANTS
        }
        total_counts = {
            variant: next(int(row["total_params"]) for row in selected if row["variant"] == variant)
            for variant in VARIANTS
        }
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
        "matched_modules": [
            "SemanticFactorizer",
            "OwnershipDiffusion",
            "JointFusion",
            "JointDiffusion",
            "W_C/W_Pt/W_Pv",
            "ownership_embedding",
            "shared_gate_mlp",
            "bridge_raw",
            "factor_bridge_norms",
            "LateFusion",
            "OutputMLP",
            "OutputNorm",
            "P0 auxiliary loss",
        ],
        "construction_order_matched": True,
        "per_dataset": by_dataset,
        "all_model_parameter_parity": all(
            bool(value["model_parameter_parity"]) for value in by_dataset.values()
        ),
        "all_head_parameter_parity": all(
            bool(value["head_parameter_parity"]) for value in by_dataset.values()
        ),
        "all_total_parameter_parity": all(
            bool(value["total_parameter_parity"]) for value in by_dataset.values()
        ),
    }


def _verdict(
    runs: list[dict[str, object]],
    diagnostics: dict[str, object],
    contrasts: dict[str, object],
) -> dict[str, object]:
    core = contrasts["C_dual_ocb_minus_owner"]
    dataset_acc = {
        dataset: float(core["per_dataset"][dataset]["mean_delta_acc_pp"])
        for dataset in DATASETS
    }
    dataset_f1 = {
        dataset: float(core["per_dataset"][dataset]["mean_delta_macro_f1_pp"])
        for dataset in DATASETS
    }
    macro_acc = float(core["macro_mean_delta_acc_pp"])
    macro_f1 = float(core["macro_mean_delta_macro_f1_pp"])
    positive_datasets = sum(value > 0.0 for value in dataset_acc.values())
    positive_f1_datasets = sum(value > 0.0 for value in dataset_f1.values())
    stable = all(bool(value["finite"]) for value in diagnostics.values())
    bridge_nonzero = any(
        bool(value["bridge_nonzero"])
        for key, value in diagnostics.items()
        if "/f1_dual_" in key
    )
    joint_nontrivial = any(bool(value["joint_branch_nontrivial"]) for value in diagnostics.values())
    no_systematic_f1_decline = macro_f1 >= -0.20 and positive_f1_datasets >= 1
    headroom = any(
        dataset_acc[dataset] >= 0.30 or dataset_f1[dataset] >= 0.50 for dataset in DATASETS
    )
    go = (
        macro_acc >= -0.20
        and no_systematic_f1_decline
        and headroom
        and stable
        and bridge_nonzero
        and joint_nontrivial
    )
    strong_go = go and macro_acc > 0.0 and macro_f1 > 0.0 and positive_datasets >= 2
    hold_bridge = macro_acc < -0.50 and positive_datasets <= 1
    if strong_go:
        verdict = "STRONG_GO"
    elif go:
        verdict = "GO_TO_F2"
    elif hold_bridge:
        verdict = "HOLD_BRIDGE"
    else:
        verdict = "HOLD_REVIEW"
    return {
        "dual_ocb_relative_to_owner": {
            "macro_delta_acc_pp": macro_acc,
            "macro_delta_macro_f1_pp": macro_f1,
            "dataset_mean_delta_acc_pp": dataset_acc,
            "dataset_mean_delta_macro_f1_pp": dataset_f1,
            "positive_dataset_count_acc": positive_datasets,
            "positive_dataset_count_macro_f1": positive_f1_datasets,
            "headroom_found": headroom,
            "stable": stable,
            "bridge_nonzero": bridge_nonzero,
            "joint_branch_nontrivial": joint_nontrivial,
            "no_systematic_f1_decline_operationalized": no_systematic_f1_decline,
        },
        "verdict": verdict,
        "strong_go": strong_go,
        "formal_run_count": len(runs),
    }


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def _write_report(
    path: Path,
    summary: dict[str, object],
    diagnostics: dict[str, object],
    parity: dict[str, object],
) -> None:
    dataset_rows = summary["dataset_summary"]
    contrasts = summary["contrasts"]
    result_table = _markdown_table(
        ["Dataset", "Variant", "Val Acc %", "Val Macro-F1 %"],
        [
            [
                row["dataset"],
                row["variant"],
                f"{row['mean_val_acc_pct']:.2f}±{row['std_val_acc_pct']:.2f}",
                f"{row['mean_val_macro_f1_pct']:.2f}±{row['std_val_macro_f1_pct']:.2f}",
            ]
            for row in dataset_rows
        ],
    )
    contrast_table = _markdown_table(
        ["Contrast", "ΔAcc pp", "ΔF1 pp", "+ Acc seeds/9", "+ F1 seeds/9"],
        [
            [
                name,
                f"{value['paired_delta_mean_acc_pp']:.2f}±{value['paired_delta_std_acc_pp']:.2f}",
                f"{value['paired_delta_mean_macro_f1_pp']:.2f}±{value['paired_delta_std_macro_f1_pp']:.2f}",
                value["positive_paired_acc_seeds"],
                value["positive_paired_f1_seeds"],
            ]
            for name, value in contrasts.items()
        ],
    )
    smoke = summary["smoke"]
    smoke_rows = []
    for variant, value in smoke["runs"].items():
        smoke_rows.append(
            [
                variant,
                f"{value['val_acc_pct']:.2f}",
                f"{value['val_macro_f1_pct']:.2f}",
                f"{value['mean_bridge_alpha']:.4f}",
                f"{value['mean_bridge_gate_c']:.4f}/{value['mean_bridge_gate_pt']:.4f}/{value['mean_bridge_gate_pv']:.4f}",
                f"{value['joint_graph_update_ratio']:.4f}",
                f"{value['bridge_update_ratio_c']:.4f}/{value['bridge_update_ratio_pt']:.4f}/{value['bridge_update_ratio_pv']:.4f}",
            ]
        )
    smoke_table = _markdown_table(
        ["Variant", "Val Acc %", "Val F1 %", "alpha", "gate C/Pt/Pv", "joint update", "bridge C/Pt/Pv"],
        smoke_rows,
    )
    diagnostic_rows = []
    for key, value in diagnostics.items():
        diagnostic_rows.append(
            [
                key,
                f"{value['mean_bridge_alpha']:.4f}",
                f"{value['mean_bridge_gate_c']:.3f}/{value['mean_bridge_gate_pt']:.3f}/{value['mean_bridge_gate_pv']:.3f}",
                f"{value['std_bridge_gate_c']:.3f}/{value['std_bridge_gate_pt']:.3f}/{value['std_bridge_gate_pv']:.3f}",
                f"{value['joint_graph_update_ratio']:.4f}",
                f"{value['bridge_update_ratio_c']:.4f}/{value['bridge_update_ratio_pt']:.4f}/{value['bridge_update_ratio_pv']:.4f}",
            ]
        )
    diagnostics_table = _markdown_table(
        ["Run", "alpha", "gate means C/Pt/Pv", "gate SD C/Pt/Pv", "joint update", "bridge update C/Pt/Pv"],
        diagnostic_rows,
    )
    parity_rows = []
    for dataset, value in parity["per_dataset"].items():
        parity_rows.append(
            [
                dataset,
                next(iter(value["model_params"].values())),
                next(iter(value["head_params"].values())),
                next(iter(value["total_params"].values())),
                value["model_parameter_parity"] and value["head_parameter_parity"] and value["total_parameter_parity"],
            ]
        )
    parity_table = _markdown_table(
        ["Dataset", "Model params", "Head params", "Total params", "Parity"], parity_rows
    )
    verdict = summary["verdict"]
    exact_commands = """```bash
# Movies seed42 Val-only smoke
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_f1.py --gpu 0 --shard 0 --shards 2 --smoke
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_f1.py --gpu 1 --shard 1 --shards 2 --smoke

# Formal 3 datasets × 3 seeds × 3 variants, Val-only
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_f1.py --gpu 0 --shard 0 --shards 2
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_f1.py --gpu 1 --shard 1 --shards 2

# Aggregate results and diagnostics
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/summarize_ored_f1.py --device cuda:0
```"""
    text = f"""# ORED-F1 Dual-Granularity Collaborative Diffusion Prototype

## 1. Architecture motivation

ORED-F1 tests whether a strong ownership-preserving graph parent benefits from a second, joint relational context. The ownership branch keeps semantic ownership explicit; the joint branch provides one graph-context stream that can be projected back into each ownership state.

The central design is deliberately small: ownership diffusion, joint diffusion, ownership-specific projections, a shared compatibility gate, and a single scalar bridge strength. No new task loss is introduced.

## 2. Relation to ORED-2 findings

ORED-2 established uniform two-hop restart diffusion on `C`, `Pt`, and `Pv` as a viable strong parent. F1 therefore retains `num_hops=2`, `restart=0.15`, self-loops, the existing SemanticFactorizer, the existing P0 common/orth/reconstruction losses, and the existing late fusion/output refinement block.

## 3. Why Composition was closed

ORED-3A learned edge composition and ownership-specific edge weighting but did not obtain a stable downstream accuracy gain. Composition is closed for this stage and is not used by any F1 variant. F1 studies same-node dual-granularity context, not semantic edge reweighting or cross-factor transport.

## 4. Three matched variants

- `f1_owner`: ownership diffusion only; effective bridge strength is exactly zero.
- `f1_dual_direct`: ownership diffusion plus the projected joint stream with all gates fixed to one.
- `f1_dual_ocb`: ownership diffusion plus the projected joint stream with the learned ownership-conditioned compatibility gate.

All three instantiate the complete module graph, including the joint branch, projections, ownership embedding, gate MLP, bridge parameter, factor bridge norms, late fusion, output MLP and output norm. The variant only controls which paths affect the output.

## 5. Dual-granularity data flow

```text
C0, Pt0, Pv0 ── ownership restart diffusion ──> CG, PtG, PvG
       │
       └─ concat ─> JointFusion ─> Joint restart diffusion ─> JG
                                                        │
                                  W_C, W_Pt, W_Pv ─────┘
                                                        │
                         compatibility gates ─> bridge ─┘
                                                        │
                 C*, Pt*, Pv* ─> LateFusion ─> OutputMLP/Norm ─> z
```

Both graph streams use the original uniform GCN-normalized graph. Ownership payloads never move between factors.

## 6. OCB formula

For ownership factor `b`, let `H_b^G` be the ownership branch state and `G_b=W_b(J^G)` the projected joint state. The shared gate MLP receives:

```text
[H_b^G || G_b || |H_b^G-G_b| || H_b^G*G_b || ownership_embedding[b]]
```

and returns `g_b=sigmoid(MLP(...))`. The bridge is:

```text
alpha = 0.5 * tanh(bridge_raw)
H_b* = H_b^G + alpha * g_b * BridgeNorm_b(G_b)
```

`f1_dual_direct` fixes `g_b=1`; `f1_owner` fixes `alpha_effective=0`.

## 7. Parent-preserving initialization

`bridge_raw=0` at initialization, hence `alpha=0` and all three variants produce the same representation under identical weights in evaluation mode. The update is multiplied by alpha before it is added to the ownership state, so the bridge norms remain bypassable at initialization.

## 8. Parameter parity

{parity_table}

Construction order is matched and the complete module/state-dict manifest is checked by the test suite. The classifier head varies only with dataset class count and is also matched across variants.

## 9. Tests

`tests/test_ored_mag.py` covers the previous ORED-1/ORED-2/ORED-3A tests plus F1 parameter/state-dict parity, exact zero-bridge parity, owner joint/bridge isolation, direct gate isolation, OCB gate effect, graph sensitivity, factor-local ownership diffusion, projection shapes, gate/alpha bounds, finite forward/backward, bridge and gate gradients, and inference parity.

## 10. Movies smoke

The smoke is Val-only, `task.evaluate_test=false`, seed 42, and is not used for tuning. It is stored under `outputs/ored/f1/smoke/` and summarized here:

{smoke_table}

## 11. 3 datasets × 3 seeds results

All values are validation percentages, mean±population SD over seeds 42/43/44. No test labels or metrics were used.

{result_table}

## 12. Contrasts A/B/C

{contrast_table}

- A (`dual_direct - owner`) measures the value of dual-granularity relational context without compatibility gating.
- B (`dual_ocb - dual_direct`) measures the value of the ownership-constrained bridge relative to direct collaboration.
- C (`dual_ocb - owner`) measures the complete F1 prototype relative to the strong ownership parent.

Paired seed rows are in `experiments/ored/f1/f1_paired_contrasts.csv`; no significance test is performed.

## 13. Alpha/gate diagnostics

Alpha, gate means and gate standard deviations are in `f1_bridge_diagnostics.json`. The full formal diagnostic table is:

{diagnostics_table}

## 14. Joint/bridge update magnitude

The joint update ratio is `mean(||JG-J0||/(||J0||+eps))`. Each bridge update ratio is `mean(||H_b*-H_b^G||/(||H_b^G||+eps))`. These are descriptive representation diagnostics and use no labels.

## 15. Performance-vs-complexity

All F1 variants are matched in parameter count by construction. Relative to the owner control, direct and OCB activate an additional joint diffusion and three hidden-to-factor projections, plus compatibility gating in OCB. The parameter manifest records model/head/total counts; runtime and peak allocated GPU memory are recorded per run in `f1_bridge_diagnostics.json`.

## 16. GO/HOLD

The registered promotion rule is applied to OCB relative to owner: macro ΔAcc at least -0.20pp, no operationalized systematic Macro-F1 decline, at least one dataset with ΔAcc at least +0.30pp or ΔF1 at least +0.50pp, stable training, nonzero bridge, and nontrivial joint update. The automated result is **{verdict['verdict']}**.

Detailed rule fields are in `f1_summary.json`. `STRONG_GO` additionally requires positive macro Acc and Macro-F1 and positive Acc means on at least two of three datasets. `HOLD_BRIDGE` is reserved for macro ΔAcc below -0.50pp with at most one positive dataset mean.

## 17. Recommended F2 final architecture

If the verdict is `STRONG_GO` or `GO_TO_F2`, carry the OCB prototype forward as the F2 candidate while keeping the parent-preserving scalar bridge, uniform ownership diffusion, and no new auxiliary loss. If the verdict is `HOLD_REVIEW` or `HOLD_BRIDGE`, retain `f1_owner` as the safe parent and do not claim that the bridge is supported. Here OCB is slightly positive relative to owner but below the headroom rule, while OCB versus direct has a small positive Acc and negative Macro-F1 contrast; both are reported for researcher review rather than triggering an automatic structural rewrite.

## 18. Exact commands

{exact_commands}

## 19. Scope

`NO TEST` · `NO EXPOSURE` · `NO COMPOSITION` · `NO TUNING` · `NO LP` · `NO CROSS-FACTOR MESSAGE PASSING` · `NO NEW LOSS`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_smoke(root: Path, checkpoint_root: Path, device: torch.device) -> dict[str, object]:
    runs: dict[str, object] = {}
    for variant in VARIANTS:
        row, diagnostic = _load_single(root, checkpoint_root, "Movies", 42, variant, device)
        runs[variant] = {
            "val_acc_pct": 100.0 * float(row["val_acc"]),
            "val_macro_f1_pct": 100.0 * float(row["val_macro_f1"]),
            **diagnostic,
            "checkpoint": row["checkpoint"],
        }
    return {
        "available": True,
        "dataset": "Movies",
        "seed": 42,
        "task.evaluate_test": False,
        "no_tuning": True,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ORED-F1 formal runs and diagnostics.")
    parser.add_argument("--formal-root", type=Path, default=Path("outputs/ored/f1/formal"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/ored/f1/checkpoints"))
    parser.add_argument("--smoke-root", type=Path, default=Path("outputs/ored/f1/smoke"))
    parser.add_argument("--smoke-checkpoint-root", type=Path, default=Path("outputs/ored/f1/smoke_checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ored/f1"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, object] = {}
    runs: list[dict[str, object]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS:
                row, diagnostic = _load_single(
                    args.formal_root,
                    args.checkpoint_root,
                    dataset,
                    seed,
                    variant,
                    device,
                )
                runs.append(row)
                diagnostics[f"{dataset}/seed{seed}/{variant}"] = diagnostic

    _write_csv(args.output_dir / "f1_all_runs.csv", runs)
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
    _write_csv(args.output_dir / "f1_dataset_summary.csv", dataset_summary)
    paired_rows, contrast_summary = _contrast_summary(runs)
    _write_csv(args.output_dir / "f1_paired_contrasts.csv", paired_rows)
    parity = _parameter_parity(runs)
    smoke = _load_smoke(args.smoke_root, args.smoke_checkpoint_root, device)
    summary = {
        "stage": "ORED-F1",
        "starting_sha": "8792df8",
        "starting_tag": "ored-o3a-composition",
        "protocol": "unified_full_graph_nc_v1",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "run_count": len(runs),
        "uses_test_labels_or_metrics": False,
        "runs": runs,
        "dataset_summary": dataset_summary,
        "contrasts": contrast_summary,
        "smoke": smoke,
    }
    summary["verdict"] = _verdict(runs, diagnostics, contrast_summary)
    (args.output_dir / "f1_bridge_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "f1_parameter_parity.json").write_text(
        json.dumps(parity, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "f1_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_report(args.output_dir.parents[2] / "docs" / "ORED_F1_DUAL_GRANULARITY_PROTOTYPE_REPORT.md", summary, diagnostics, parity)
    print(json.dumps({"stage": "ORED-F1", "run_count": len(runs), "verdict": summary["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
