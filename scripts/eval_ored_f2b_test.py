from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_mag_data
from src.models import build_model


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANT = "f1_dual_direct"
FROZEN_COMMIT = "5a53d1d"
FROZEN_TAG = "ored-f2a-architecture-freeze"
F1_COMMIT = "901f254"
F1_TAG = "ored-f1-dual-prototype"
FROZEN_CONFIG = {
    "hidden_dim": 256,
    "factor_dim": 128,
    "num_hops": 2,
    "restart": 0.15,
}
REFERENCE_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "sage": "GraphSAGE",
    "mmgcn": "MMGCN",
    "dgf": "DGF",
    "lgmrec": "LGMRec",
    "dip": "DiP",
    "map_mag": "MAP",
    "map_mag_v2": "MAP-v2",
    "map_mag_v3": "MAP-v3",
}


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _pop_sd(values: list[float]) -> float:
    if not values:
        return float("nan")
    avg = _mean(values)
    return float(math.sqrt(sum((value - avg) ** 2 for value in values) / len(values)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _job_paths(dataset: str, seed: int) -> dict[str, Path]:
    slug = dataset.lower()
    stage = "f1" if dataset in {"Movies", "Toys", "Grocery"} else "f2a"
    root = Path("outputs/ored") / stage
    name = f"{slug}_seed{seed}_{VARIANT}"
    return {
        "stage": Path(stage),
        "source_commit": Path(F1_COMMIT if stage == "f1" else FROZEN_COMMIT),
        "source_tag": Path(F1_TAG if stage == "f1" else FROZEN_TAG),
        "output_dir": root / "formal" / name,
        "checkpoint": root / "checkpoints" / f"{name}.pt",
    }


def _data_info(data) -> dict[str, int]:
    return {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def _load_source_metadata(dataset: str, seed: int, checkpoint: Path) -> tuple[dict[str, Any], Any, Any]:
    paths = _job_paths(dataset, seed)
    output_dir = paths["output_dir"]
    config_path = output_dir / ".hydra" / "config.yaml"
    result_path = output_dir / "results.json"
    required = (config_path, result_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source metadata: " + ", ".join(missing))
    cfg = OmegaConf.load(config_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = {
        "source_stage": str(paths["stage"]),
        "source_config": str(config_path),
        "source_result": str(result_path),
        "source_commit": str(paths["source_commit"]),
        "source_tag": str(paths["source_tag"]),
        "best_val_acc": float(result["val_acc"]["mean"]),
        "val_macro_f1": float(result["val_macro_f1"]["mean"]),
        "best_epoch": int(round(float(result["best_epoch"]["mean"]))),
        "source_checkpoint": str(checkpoint),
    }
    return metadata, cfg, result


def _integrity_checks(
    dataset: str,
    seed: int,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    cfg,
    data,
    model,
    data_info: dict[str, int],
) -> list[str]:
    errors: list[str] = []
    if checkpoint.get("task") != "nc":
        errors.append(f"task={checkpoint.get('task')!r}, expected 'nc'")
    if int(checkpoint.get("seed", -1)) != seed:
        errors.append(f"checkpoint seed={checkpoint.get('seed')!r}, expected {seed}")
    stored_info = checkpoint.get("data_info")
    if stored_info != data_info:
        errors.append(f"data_info mismatch: checkpoint={stored_info!r}, loaded={data_info!r}")
    if str(cfg.model.name) != "ored_mag":
        errors.append(f"model.name={cfg.model.name!r}, expected 'ored_mag'")
    if str(cfg.model.get("variant", "")) != VARIANT:
        errors.append(f"model.variant={cfg.model.get('variant')!r}, expected {VARIANT!r}")
    for key, expected in FROZEN_CONFIG.items():
        actual = getattr(cfg.model, key, None)
        if actual is None or float(actual) != float(expected):
            errors.append(f"model.{key}={actual!r}, expected {expected!r}")
    for key, expected in FROZEN_CONFIG.items():
        actual = getattr(model, key, None)
        if key == "restart" and hasattr(model, "restart_diffusion"):
            actual = model.restart_diffusion.restart
        if key == "num_hops" and hasattr(model, "restart_diffusion"):
            actual = model.restart_diffusion.num_hops
        if actual is None or float(actual) != float(expected):
            errors.append(f"instantiated model {key}={actual!r}, expected {expected!r}")
    if not checkpoint_path.is_file():
        errors.append("checkpoint does not exist")
    if not data.test_idx.numel():
        errors.append("test_idx is empty")
    return errors


@torch.no_grad()
def _evaluate_checkpoint(
    dataset: str,
    seed: int,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    metadata, cfg, _ = _load_source_metadata(dataset, seed, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = load_mag_data(cfg, "nc", seed)
    data_info = _data_info(data)
    model = build_model(cfg, data_info)
    classifier = nn.Linear(model.out_dim, int(data.num_classes))
    errors = _integrity_checks(
        dataset, seed, checkpoint_path, checkpoint, cfg, data, model, data_info
    )
    if errors:
        raise RuntimeError(f"integrity check failed for {dataset} seed={seed}: " + "; ".join(errors))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    classifier.load_state_dict(checkpoint["head_state"], strict=True)
    model = model.to(device).eval()
    classifier = classifier.to(device).eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    z, _, _, _, _ = model(x, edge_index)

    def metrics(idx: torch.Tensor) -> tuple[float, float]:
        logits = classifier(z[idx.to(device)])
        pred = logits.argmax(dim=-1).cpu()
        target = data.y[idx].cpu()
        return (
            float((pred == target).float().mean().item()),
            float(f1_score(target.numpy(), pred.numpy(), average="macro", zero_division=0)),
        )

    test_acc, test_f1 = metrics(data.test_idx)
    row = {
        "dataset": dataset,
        "seed": seed,
        "variant": VARIANT,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "best_val_acc": metadata["best_val_acc"],
        "val_macro_f1": metadata["val_macro_f1"],
        "best_epoch": metadata["best_epoch"],
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "test_minus_val_acc": test_acc - metadata["best_val_acc"],
        "source_stage": metadata["source_stage"],
        "source_config": metadata["source_config"],
        "source_commit": metadata["source_commit"],
        "source_tag": metadata["source_tag"],
        "task": "nc",
        "model_variant": VARIANT,
        "hidden_dim": FROZEN_CONFIG["hidden_dim"],
        "factor_dim": FROZEN_CONFIG["factor_dim"],
        "num_hops": FROZEN_CONFIG["num_hops"],
        "restart": FROZEN_CONFIG["restart"],
        "optimizer_step": False,
        "checkpoint_selected_after_test": False,
    }
    del z, x, edge_index, model, classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _make_manifest(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            paths = _job_paths(dataset, seed)
            checkpoint = paths["checkpoint"]
            row: dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "variant": VARIANT,
                "checkpoint_path": str(checkpoint),
                "checkpoint_exists": checkpoint.is_file(),
                "source_stage": str(paths["stage"]),
                "source_config": str(paths["output_dir"] / ".hydra" / "config.yaml"),
                "source_commit": str(paths["source_commit"]),
                "source_tag": str(paths["source_tag"]),
                "retrained": False,
            }
            if checkpoint.is_file():
                row["checkpoint_sha256"] = _sha256(checkpoint)
                try:
                    metadata, cfg, _ = _load_source_metadata(dataset, seed, checkpoint)
                    row.update(
                        {
                            "best_val_acc": metadata["best_val_acc"],
                            "val_macro_f1": metadata["val_macro_f1"],
                            "best_epoch": metadata["best_epoch"],
                            "source_result": metadata["source_result"],
                            "source_config": metadata["source_config"],
                            "config_model_variant": str(cfg.model.get("variant", "")),
                            "config_hidden_dim": int(cfg.model.hidden_dim),
                            "config_factor_dim": int(cfg.model.factor_dim),
                            "config_num_hops": int(cfg.model.num_hops),
                            "config_restart": float(cfg.model.restart),
                        }
                    )
                except Exception as exc:  # manifest must preserve the audit failure
                    row["metadata_error"] = f"{type(exc).__name__}: {exc}"
            else:
                row["missing_reason"] = "checkpoint file not found; no retraining attempted"
                missing.append(row.copy())
            manifest.append(row)
    payload = {
        "architecture": "ORED-MAG / f1_dual_direct",
        "frozen_commit": FROZEN_COMMIT,
        "frozen_tag": FROZEN_TAG,
        "test_driven_tuning": False,
        "test_driven_model_selection": False,
        "ensemble": False,
        "seed_filtering": False,
        "retrained_missing_checkpoints": False,
        "entries": manifest,
        "missing": missing,
    }
    (output_root / "checkpoint_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest, missing


def _parse_benchmark(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_text(encoding="utf-8")
    marker = "## 跨数据集平均"
    if marker not in text:
        raise ValueError(f"could not find cross-dataset benchmark table in {path}")
    table = text[text.index(marker) :]
    refs: dict[str, dict[str, float]] = {}
    for line in table.splitlines():
        if not line.startswith("|") or line.startswith("| Model") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        key = cells[0]
        try:
            # The benchmark uses Markdown emphasis around a few best values.
            numeric = [float(cell.replace("**", "").split()[0]) for cell in cells[1:4]]
            refs[key] = {
                "mean_test_acc_pct": numeric[0],
                "mean_test_macro_f1_pct": numeric[1],
                "mean_val_acc_pct": numeric[2],
            }
        except ValueError:
            continue
    missing = sorted(set(REFERENCE_LABELS) - set(refs))
    if missing:
        raise ValueError(f"benchmark table missing models: {missing}")
    return {REFERENCE_LABELS[key]: value for key, value in refs.items() if key in REFERENCE_LABELS}


def _make_dataset_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        val_acc = [100.0 * float(row["best_val_acc"]) for row in selected]
        val_f1 = [100.0 * float(row["val_macro_f1"]) for row in selected]
        test_acc = [100.0 * float(row["test_acc"]) for row in selected]
        test_f1 = [100.0 * float(row["test_macro_f1"]) for row in selected]
        delta = [100.0 * float(row["test_minus_val_acc"]) for row in selected]
        summary.append(
            {
                "dataset": dataset,
                "seeds": len(selected),
                "mean_val_acc_pct": _mean(val_acc),
                "std_val_acc_pct": _pop_sd(val_acc),
                "mean_val_macro_f1_pct": _mean(val_f1),
                "std_val_macro_f1_pct": _pop_sd(val_f1),
                "mean_test_acc_pct": _mean(test_acc),
                "std_test_acc_pct": _pop_sd(test_acc),
                "mean_test_macro_f1_pct": _mean(test_f1),
                "std_test_macro_f1_pct": _pop_sd(test_f1),
                "mean_test_minus_val_acc_pp": _mean(delta),
                "std_test_minus_val_acc_pp": _pop_sd(delta),
            }
        )
    return summary


def _make_benchmark_comparison(summary: list[dict[str, Any]], benchmark_path: Path) -> list[dict[str, Any]]:
    refs = _parse_benchmark(benchmark_path)
    ored_acc = _mean([float(row["mean_test_acc_pct"]) for row in summary])
    ored_f1 = _mean([float(row["mean_test_macro_f1_pct"]) for row in summary])
    rows: list[dict[str, Any]] = []
    for name in ("MLP", "GCN", "GraphSAGE", "MMGCN", "DGF", "LGMRec", "DiP", "MAP", "MAP-v2", "MAP-v3"):
        ref = refs[name]
        rows.append(
            {
                "model": name,
                "mean_test_acc_pct": ref["mean_test_acc_pct"],
                "mean_test_macro_f1_pct": ref["mean_test_macro_f1_pct"],
                "ored_minus_test_acc_pp": ored_acc - ref["mean_test_acc_pct"],
                "ored_minus_test_macro_f1_pp": ored_f1 - ref["mean_test_macro_f1_pct"],
                "source": str(benchmark_path),
            }
        )
    rows.append(
        {
            "model": "ORED-MAG",
            "mean_test_acc_pct": ored_acc,
            "mean_test_macro_f1_pct": ored_f1,
            "ored_minus_test_acc_pp": 0.0,
            "ored_minus_test_macro_f1_pp": 0.0,
            "source": "f2b frozen checkpoint evaluation",
        }
    )
    return rows


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    mean_acc = _mean([float(row["mean_test_acc_pct"]) for row in summary])
    mean_f1 = _mean([float(row["mean_test_macro_f1_pct"]) for row in summary])
    tier = "TIER_A" if mean_acc >= 80.4 and mean_f1 >= 73.8 else (
        "TIER_B" if mean_acc >= 80.0 and mean_f1 >= 73.3 else "TIER_C"
    )
    lines = [
        "# ORED-F2B Final Test Benchmark",
        "",
        "## Frozen protocol",
        "",
        f"- Architecture: ORED-MAG, variant `{VARIANT}`.",
        "- Frozen architecture: Semantic Ownership Decomposition; Dual-Granularity Relational Diffusion; Ownership-Projected Residual Collaboration.",
        f"- Architecture-freeze commit/tag: `{FROZEN_COMMIT}` / `{FROZEN_TAG}`.",
        "- Checkpoint-only evaluation; validation-accuracy-selected checkpoints from F1/F2A.",
        "- ARCHITECTURE FROZEN BEFORE TEST; NO TEST-DRIVEN TUNING; NO TEST-DRIVEN MODEL SELECTION; NO ENSEMBLE; NO SEED FILTERING.",
        "- Exact full-graph inference on the fixed test split; no optimizer step was executed.",
        "",
        "## Checkpoint manifest",
        "",
        f"The manifest contains {len(rows)} expected entries and {len(missing)} missing entries. No missing checkpoint was retrained.",
        "See `experiments/ored/f2b/checkpoint_manifest.json` for paths, source configs, validation metadata, and SHA256 values.",
        "",
        "## Per-dataset results",
        "",
        "All values are percentages; `±` is population SD over seeds 42/43/44.",
        "",
        "| Dataset | Val Acc | Test Acc | Test Macro-F1 | Test Acc − Val Acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['dataset']} | {_fmt(item['mean_val_acc_pct'])}±{_fmt(item['std_val_acc_pct'])} | "
            f"{_fmt(item['mean_test_acc_pct'])}±{_fmt(item['std_test_acc_pct'])} | "
            f"{_fmt(item['mean_test_macro_f1_pct'])}±{_fmt(item['std_test_macro_f1_pct'])} | "
            f"{_fmt(item['mean_test_minus_val_acc_pp'])}±{_fmt(item['std_test_minus_val_acc_pp'])} pp |"
        )
    lines += [
        "",
        "## Five-dataset macro mean",
        "",
        f"- ORED-MAG Test Accuracy: **{_fmt(mean_acc, 4)}%**.",
        f"- ORED-MAG Test Macro-F1: **{_fmt(mean_f1, 4)}%**.",
        "",
        "| Reference | ORED minus Test Acc (pp) | ORED minus Test Macro-F1 (pp) |",
        "|---|---:|---:|",
    ]
    for item in comparison:
        if item["model"] == "ORED-MAG":
            continue
        lines.append(
            f"| {item['model']} | {_fmt(item['ored_minus_test_acc_pp'], 4)} | {_fmt(item['ored_minus_test_macro_f1_pp'], 4)} |"
        )
    lines += [
        "",
        "## Unified NC benchmark comparison",
        "",
        "The reference values are read from `docs/nc_benchmark_results.md`; no baseline was rerun.",
        "",
        "| Model | Mean Test Accuracy | Mean Test Macro-F1 |",
        "|---|---:|---:|",
    ]
    for item in comparison:
        lines.append(
            f"| {item['model']} | {_fmt(item['mean_test_acc_pct'], 4)} | {_fmt(item['mean_test_macro_f1_pct'], 4)} |"
        )
    lines += [
        "",
        "## Performance tier",
        "",
        f"**{tier}**: mean Test Accuracy={_fmt(mean_acc, 4)} and mean Test Macro-F1={_fmt(mean_f1, 4)}.",
        "",
        "## Stability and anomalies",
        "",
        "- All 15 expected frozen checkpoints were present and evaluated.",
        "- Every checkpoint passed task, seed, data-info, strict model-state/head-state, variant, and frozen hyperparameter checks.",
        "- The reported validation values are the source checkpoint-selection values; this evaluation computed metrics only on `test_idx`.",
        "- No checkpoint was selected using Test metrics, no seed was filtered, and no ensemble was formed.",
        "",
        "## Next paper-validation steps",
        "",
        "Freeze these Test numbers as final evidence, preserve the manifest and source checkpoint hashes, and proceed to paper-level error analysis and reproducibility packaging. Do not reopen OCB, Composition, or Exposure based on this Test result.",
        "",
        "## Evidence boundary",
        "",
        "Final Dual ORED architecture = FROZEN BEFORE TEST. Test benchmark = FINAL PERFORMANCE EVIDENCE.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen ORED-F2B checkpoints on NC test splits.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/ored/f2b"))
    parser.add_argument("--benchmark", type=Path, default=Path("docs/nc_benchmark_results.md"))
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    _, missing = _make_manifest(args.output_root)
    if args.manifest_only:
        print(f"manifest written; missing={len(missing)}")
        return
    if missing:
        raise RuntimeError(
            "frozen checkpoint set is incomplete; see checkpoint_manifest.json; "
            "no retraining was attempted"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            checkpoint = _job_paths(dataset, seed)["checkpoint"]
            print(f"EVAL {dataset} seed={seed} checkpoint={checkpoint} device={device}", flush=True)
            row = _evaluate_checkpoint(dataset, seed, checkpoint, device)
            rows.append(row)
            print(
                f"  Test Acc={100.0 * row['test_acc']:.4f}% "
                f"Test Macro-F1={100.0 * row['test_macro_f1']:.4f}%",
                flush=True,
            )
    _write_csv(args.output_root / "f2b_test_runs.csv", rows)
    summary = _make_dataset_summary(rows)
    _write_csv(args.output_root / "f2b_test_dataset_summary.csv", summary)
    comparison = _make_benchmark_comparison(summary, args.benchmark)
    _write_csv(args.output_root / "f2b_benchmark_comparison.csv", comparison)
    mean_acc = _mean([float(row["mean_test_acc_pct"]) for row in summary])
    mean_f1 = _mean([float(row["mean_test_macro_f1_pct"]) for row in summary])
    tier = "TIER_A" if mean_acc >= 80.4 and mean_f1 >= 73.8 else (
        "TIER_B" if mean_acc >= 80.0 and mean_f1 >= 73.3 else "TIER_C"
    )
    summary_payload = {
        "architecture": "ORED-MAG",
        "variant": VARIANT,
        "frozen_commit": FROZEN_COMMIT,
        "frozen_tag": FROZEN_TAG,
        "protocol": "checkpoint_only_exact_full_graph_nc_test",
        "architecture_frozen_before_test": True,
        "no_test_driven_tuning": True,
        "no_test_driven_model_selection": True,
        "no_ensemble": True,
        "no_seed_filtering": True,
        "num_evaluations": len(rows),
        "mean_test_accuracy_pct": mean_acc,
        "mean_test_macro_f1_pct": mean_f1,
        "performance_tier": tier,
        "datasets": summary,
        "benchmark_comparison": comparison,
    }
    (args.output_root / "f2b_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(Path("docs/ORED_F2B_FINAL_TEST_BENCHMARK.md"), rows, summary, comparison, missing)
    print(f"Wrote F2B artifacts under {args.output_root}", flush=True)
    print(f"Mean Test Acc={mean_acc:.4f}% | Mean Test Macro-F1={mean_f1:.4f}% | {tier}", flush=True)


if __name__ == "__main__":
    main()
