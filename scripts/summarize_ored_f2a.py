from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


NEW_DATASETS = ("ele-fashion", "Reddit-S")
ALL_DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("f1_owner", "f1_dual_direct")
REFERENCE_MODELS = ("dip", "map_mag", "map_mag_v2", "map_mag_v3")
EXPECTED_REFERENCE_MEANS = {
    "dip": 80.9594,
    "map_mag": 80.4363,
    "map_mag_v2": 80.8712,
    "map_mag_v3": 80.8441,
}


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def pop_sd(values: list[float]) -> float:
    if not values:
        return float("nan")
    avg = mean(values)
    return float(math.sqrt(sum((value - avg) ** 2 for value in values) / len(values)))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_new_row(root: Path, checkpoint_root: Path, dataset: str, seed: int, variant: str) -> dict[str, object]:
    name = f"{dataset.lower()}_seed{seed}_{variant}"
    output_dir = root / name
    result_path = output_dir / "results.json"
    checkpoint_path = checkpoint_root / f"{name}.pt"
    config_path = output_dir / ".hydra" / "config.yaml"
    for path in (result_path, checkpoint_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(config_path)
    model_state = checkpoint["model_state"]
    head_state = checkpoint["head_state"]
    log_path = output_dir / "main.log"
    log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    bad_log = bool(re.search(r"\b(?:nan|inf|infinity|out of memory|traceback)\b", log_text, re.IGNORECASE))
    row: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "val_acc": float(result["val_acc"]["mean"]),
        "val_macro_f1": float(result["val_macro_f1"]["mean"]),
        "best_epoch": int(round(float(result["best_epoch"]["mean"]))),
        "early_stop_epoch": int(round(float(result["early_stop_epoch"]["mean"]))),
        "model_params": sum(int(value.numel()) for value in model_state.values()),
        "head_params": sum(int(value.numel()) for value in head_state.values()),
        "protocol": str(cfg.task.get("protocol_version", "")),
        "training_mode": str(cfg.task.get("training_mode", "")),
        "evaluate_test": bool(cfg.task.get("evaluate_test", True)),
        "uses_test_labels_or_metrics": False,
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
    }
    row["total_params"] = int(row["model_params"]) + int(row["head_params"])
    row["finite"] = all(finite(row[key]) for key in ("val_acc", "val_macro_f1", "best_epoch", "early_stop_epoch"))
    row["bad_log_markers"] = bad_log
    row["stable"] = bool(
        row["finite"]
        and not bad_log
        and row["protocol"] == "unified_full_graph_nc_v1"
        and row["training_mode"] == "full_graph"
        and row["evaluate_test"] is False
        and checkpoint.get("task") == "nc"
        and int(checkpoint.get("seed", -1)) == seed
    )
    return row


def load_f1_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["variant"] not in VARIANTS:
                continue
            row: dict[str, object] = dict(raw)
            row["seed"] = int(raw["seed"])
            for key in ("val_acc", "val_macro_f1"):
                row[key] = float(raw[key])
            for key in ("best_epoch", "early_stop_epoch", "model_params", "head_params", "total_params"):
                row[key] = int(float(raw[key]))
            row["evaluate_test"] = raw["evaluate_test"].lower() == "true"
            row["uses_test_labels_or_metrics"] = raw["uses_test_labels_or_metrics"].lower() == "true"
            row["training_mode"] = "full_graph"
            row["finite"] = all(finite(row[key]) for key in ("val_acc", "val_macro_f1"))
            row["bad_log_markers"] = False
            row["stable"] = bool(
                row["finite"]
                and raw["protocol"] == "unified_full_graph_nc_v1"
                and row["evaluate_test"] is False
                and row["uses_test_labels_or_metrics"] is False
            )
            rows.append(row)
    expected = 3 * len(SEEDS) * len(VARIANTS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} F1 owner/direct rows, found {len(rows)}")
    return rows


def make_dataset_summary(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for dataset in ALL_DATASETS:
        for variant in VARIANTS:
            selected = [row for row in runs if row["dataset"] == dataset and row["variant"] == variant]
            if len(selected) != len(SEEDS):
                raise ValueError(f"{dataset}/{variant}: expected 3 rows, found {len(selected)}")
            acc = [100.0 * float(row["val_acc"]) for row in selected]
            f1 = [100.0 * float(row["val_macro_f1"]) for row in selected]
            result.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seeds": len(selected),
                    "mean_val_acc_pct": mean(acc),
                    "std_val_acc_pct": pop_sd(acc),
                    "mean_val_macro_f1_pct": mean(f1),
                    "std_val_macro_f1_pct": pop_sd(f1),
                    "model_params": int(selected[0]["model_params"]),
                    "head_params": int(selected[0]["head_params"]),
                    "total_params": int(selected[0]["total_params"]),
                    "all_stable": all(bool(row["stable"]) for row in selected),
                }
            )
    return result


def make_paired(runs: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_key = {(row["dataset"], int(row["seed"]), row["variant"]): row for row in runs}
    rows: list[dict[str, object]] = []
    per_dataset: dict[str, object] = {}
    all_acc: list[float] = []
    all_f1: list[float] = []
    for dataset in ALL_DATASETS:
        acc_values: list[float] = []
        f1_values: list[float] = []
        for seed in SEEDS:
            dual = by_key[(dataset, seed, "f1_dual_direct")]
            owner = by_key[(dataset, seed, "f1_owner")]
            delta_acc = 100.0 * (float(dual["val_acc"]) - float(owner["val_acc"]))
            delta_f1 = 100.0 * (float(dual["val_macro_f1"]) - float(owner["val_macro_f1"]))
            acc_values.append(delta_acc)
            f1_values.append(delta_f1)
            all_acc.append(delta_acc)
            all_f1.append(delta_f1)
            rows.append(
                {
                    "contrast": "f1_dual_direct_minus_owner",
                    "dataset": dataset,
                    "seed": seed,
                    "left_variant": "f1_dual_direct",
                    "right_variant": "f1_owner",
                    "delta_acc_pp": delta_acc,
                    "delta_macro_f1_pp": delta_f1,
                }
            )
        per_dataset[dataset] = {
            "mean_delta_acc_pp": mean(acc_values),
            "std_delta_acc_pp": pop_sd(acc_values),
            "positive_acc_seeds": int(sum(value > 0.0 for value in acc_values)),
            "mean_delta_macro_f1_pp": mean(f1_values),
            "std_delta_macro_f1_pp": pop_sd(f1_values),
            "positive_macro_f1_seeds": int(sum(value > 0.0 for value in f1_values)),
        }
    summary = {
        "left_variant": "f1_dual_direct",
        "right_variant": "f1_owner",
        "per_dataset": per_dataset,
        "macro_mean_delta_acc_pp": mean([float(per_dataset[d]["mean_delta_acc_pp"]) for d in ALL_DATASETS]),
        "macro_mean_delta_macro_f1_pp": mean([float(per_dataset[d]["mean_delta_macro_f1_pp"]) for d in ALL_DATASETS]),
        "paired_delta_mean_acc_pp": mean(all_acc),
        "paired_delta_std_acc_pp": pop_sd(all_acc),
        "paired_delta_mean_macro_f1_pp": mean(all_f1),
        "paired_delta_std_macro_f1_pp": pop_sd(all_f1),
        "positive_paired_acc_seeds": int(sum(value > 0.0 for value in all_acc)),
        "positive_paired_f1_seeds": int(sum(value > 0.0 for value in all_f1)),
    }
    return rows, summary


def load_benchmark(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_text(encoding="utf-8")
    sections = list(re.finditer(r"^## (.+?)\s*$", text, re.MULTILINE))
    result: dict[str, dict[str, float]] = {}
    for index, match in enumerate(sections):
        dataset = match.group(1)
        if dataset not in ALL_DATASETS:
            continue
        end = sections[index + 1].start() if index + 1 < len(sections) else len(text)
        values: dict[str, float] = {}
        for line in text[match.end() : end].splitlines():
            if not line.startswith("|") or line.startswith("| Model") or line.startswith("|---"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[0] in REFERENCE_MODELS:
                value = cells[1].split("±", 1)[0].strip().strip("*")
                values[cells[0]] = float(value)
        if set(values) != set(REFERENCE_MODELS):
            raise ValueError(f"benchmark rows missing for {dataset}")
        result[dataset] = values
    if set(result) != set(ALL_DATASETS):
        raise ValueError("benchmark dataset sections are incomplete")
    for model, expected in EXPECTED_REFERENCE_MEANS.items():
        actual = mean([result[dataset][model] for dataset in ALL_DATASETS])
        if abs(actual - expected) > 1e-4:
            raise ValueError(f"{model} mean {actual:.4f} != expected {expected:.4f}")
    return result


def make_benchmark_rows(dataset_summary: list[dict[str, object]], benchmark: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in ALL_DATASETS:
        dual = next(row for row in dataset_summary if row["dataset"] == dataset and row["variant"] == "f1_dual_direct")
        best_model = max(REFERENCE_MODELS, key=lambda model: benchmark[dataset][model])
        best_value = benchmark[dataset][best_model]
        rows.append(
            {
                "dataset": dataset,
                "dual_mean_val_acc_pct": float(dual["mean_val_acc_pct"]),
                "dual_std_val_acc_pct": float(dual["std_val_acc_pct"]),
                "best_map_dip_model": best_model,
                "best_map_dip_val_acc_pct": best_value,
                "dual_minus_best_pp": float(dual["mean_val_acc_pct"]) - best_value,
            }
        )
    dual_mean = mean([float(row["dual_mean_val_acc_pct"]) for row in rows])
    for model in REFERENCE_MODELS:
        baseline_mean = mean([benchmark[dataset][model] for dataset in ALL_DATASETS])
        rows.append(
            {
                "dataset": f"macro_mean_{model}",
                "dual_mean_val_acc_pct": dual_mean,
                "dual_std_val_acc_pct": "",
                "best_map_dip_model": model,
                "best_map_dip_val_acc_pct": baseline_mean,
                "dual_minus_best_pp": dual_mean - baseline_mean,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ORED-F2A validation runs.")
    parser.add_argument("--new-root", type=Path, default=Path("outputs/ored/f2a/formal"))
    parser.add_argument("--new-checkpoint-root", type=Path, default=Path("outputs/ored/f2a/checkpoints"))
    parser.add_argument("--f1-csv", type=Path, default=Path("experiments/ored/f1/f1_all_runs.csv"))
    parser.add_argument("--benchmark", type=Path, default=Path("docs/nc_benchmark_results.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ored/f2a"))
    args = parser.parse_args()

    f1_rows = load_f1_rows(args.f1_csv)
    new_rows = [
        load_new_row(args.new_root, args.new_checkpoint_root, dataset, seed, variant)
        for dataset in NEW_DATASETS
        for seed in SEEDS
        for variant in VARIANTS
    ]
    runs = f1_rows + new_rows
    if len(runs) != 30:
        raise ValueError(f"expected 30 merged owner/direct rows, found {len(runs)}")
    dataset_summary = make_dataset_summary(runs)
    paired_rows, paired = make_paired(runs)
    benchmark = load_benchmark(args.benchmark)
    benchmark_rows = make_benchmark_rows(dataset_summary, benchmark)

    owner_acc = mean([float(row["mean_val_acc_pct"]) for row in dataset_summary if row["variant"] == "f1_owner"])
    dual_acc = mean([float(row["mean_val_acc_pct"]) for row in dataset_summary if row["variant"] == "f1_dual_direct"])
    owner_f1 = mean([float(row["mean_val_macro_f1_pct"]) for row in dataset_summary if row["variant"] == "f1_owner"])
    dual_f1 = mean([float(row["mean_val_macro_f1_pct"]) for row in dataset_summary if row["variant"] == "f1_dual_direct"])
    delta_acc = {dataset: float(paired["per_dataset"][dataset]["mean_delta_acc_pp"]) for dataset in ALL_DATASETS}
    delta_f1 = {dataset: float(paired["per_dataset"][dataset]["mean_delta_macro_f1_pp"]) for dataset in ALL_DATASETS}
    stable_count = sum(bool(row["stable"]) for row in runs)
    severe_count = sum(value <= -0.80 for value in delta_acc.values())
    # “No systematic material decline” is made explicit as macro F1 >= -0.20 pp
    # and no majority of dataset-level means being negative.
    macro_f1_pass = float(paired["macro_mean_delta_macro_f1_pp"]) >= -0.20 and sum(value < 0.0 for value in delta_f1.values()) <= 2
    decision = {
        "macro_acc_pass": float(paired["macro_mean_delta_acc_pp"]) >= -0.20,
        "macro_f1_pass": macro_f1_pass,
        "severe_acc_regression_pass": severe_count < 2,
        "stable_pass": stable_count == len(runs),
        "macro_acc_threshold_pp": -0.20,
        "macro_delta_acc_pp": float(paired["macro_mean_delta_acc_pp"]),
        "macro_f1_threshold_pp": -0.20,
        "macro_delta_f1_pp": float(paired["macro_mean_delta_macro_f1_pp"]),
        "datasets_acc_le_minus_0_80pp": severe_count,
        "stable_run_count": stable_count,
        "verdict": "FREEZE_DUAL_FINAL" if all((
            float(paired["macro_mean_delta_acc_pp"]) >= -0.20,
            macro_f1_pass,
            severe_count < 2,
            stable_count == len(runs),
        )) else "FREEZE_OWNER_FINAL",
    }
    summary = {
        "stage": "ORED-F2A",
        "starting_sha": "901f254",
        "starting_tag": "ored-f1-dual-prototype",
        "protocol": "unified_full_graph_nc_v1",
        "task": "nc",
        "training_mode": "full_graph",
        "evaluate_test": False,
        "no_test": True,
        "no_tuning": True,
        "no_new_module": True,
        "no_ocb_formal_rerun": True,
        "datasets": list(ALL_DATASETS),
        "new_datasets": list(NEW_DATASETS),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "run_count": len(runs),
        "new_run_count": len(new_rows),
        "stable_run_count": stable_count,
        "uses_test_labels_or_metrics": False,
        "dataset_summary": dataset_summary,
        "paired_contrast": paired,
        "positive_dataset_means": {
            "acc": int(sum(value > 0.0 for value in delta_acc.values())),
            "f1": int(sum(value > 0.0 for value in delta_f1.values())),
        },
        "dataset_delta_acc_pp": delta_acc,
        "dataset_delta_macro_f1_pp": delta_f1,
        "macro": {
            "owner_acc_pct": owner_acc,
            "dual_acc_pct": dual_acc,
            "delta_acc_pp": dual_acc - owner_acc,
            "owner_f1_pct": owner_f1,
            "dual_f1_pct": dual_f1,
            "delta_f1_pp": dual_f1 - owner_f1,
        },
        "benchmark_reference": {
            "source": str(args.benchmark),
            "models": list(REFERENCE_MODELS),
            "mean_val_acc_pct": EXPECTED_REFERENCE_MEANS,
            "per_dataset_val_acc_pct": benchmark,
            "comparison_rows": benchmark_rows,
        },
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "f2a_new_runs.csv", new_rows)
    write_csv(args.output_dir / "f2a_five_dataset_summary.csv", dataset_summary)
    write_csv(args.output_dir / "f2a_paired_contrasts.csv", paired_rows)
    write_csv(args.output_dir / "f2a_benchmark_val_comparison.csv", benchmark_rows)
    (args.output_dir / "f2a_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "ORED-F2A", "new_run_count": len(new_rows), "decision": decision["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
