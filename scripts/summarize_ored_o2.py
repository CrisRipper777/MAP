from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DATASETS = ("Movies", "Toys", "Grocery")
SEEDS = (42, 43, 44)
VARIANTS = ("p0", "p0_refine", "joint_rd", "ownership_rd")
CONTRASTS = {
    "A_refinement_gain": ("p0_refine", "p0"),
    "B_joint_graph_gain": ("joint_rd", "p0_refine"),
    "C_ownership_graph_gain": ("ownership_rd", "p0_refine"),
    "D_ownership_minus_joint": ("ownership_rd", "joint_rd"),
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _load_run(formal_root: Path, dataset: str, seed: int, variant: str) -> dict[str, object]:
    output_dir = formal_root / f"{dataset.lower()}_seed{seed}_{variant}"
    result_path = output_dir / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_path = formal_root.parent / "checkpoints" / f"{dataset.lower()}_seed{seed}_{variant}.pt"
    import torch

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params = sum(int(value.numel()) for value in checkpoint_data["model_state"].values())
    head_params = sum(int(value.numel()) for value in checkpoint_data["head_state"].values())
    return {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "val_acc": float(result["val_acc"]["mean"]),
        "val_macro_f1": float(result["val_macro_f1"]["mean"]),
        "best_epoch": int(round(float(result["best_epoch"]["mean"]))),
        "early_stop_epoch": int(round(float(result["early_stop_epoch"]["mean"]))),
        "model_params": model_params,
        "head_params": head_params,
        "total_params": model_params + head_params,
        "protocol": "unified_full_graph_nc_v1",
        "evaluate_test": False,
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the fixed ORED-2 Val-only grid.")
    parser.add_argument("--formal-root", type=Path, default=Path("outputs/ored/o2/formal"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ored/o2"))
    args = parser.parse_args()

    runs = [
        _load_run(args.formal_root, dataset, seed, variant)
        for dataset in DATASETS
        for seed in SEEDS
        for variant in VARIANTS
    ]
    _write_csv(args.output_dir / "o2_all_runs.csv", runs)

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
    _write_csv(args.output_dir / "o2_dataset_summary.csv", dataset_summary)

    by_key = {(row["dataset"], row["seed"], row["variant"]): row for row in runs}
    paired_rows: list[dict[str, object]] = []
    contrast_summary: dict[str, object] = {}
    for name, (left, right) in CONTRASTS.items():
        per_dataset: dict[str, object] = {}
        all_acc: list[float] = []
        all_f1: list[float] = []
        for dataset in DATASETS:
            delta_acc: list[float] = []
            delta_f1: list[float] = []
            for seed in SEEDS:
                lhs = by_key[(dataset, seed, left)]
                rhs = by_key[(dataset, seed, right)]
                da = 100.0 * (float(lhs["val_acc"]) - float(rhs["val_acc"]))
                df = 100.0 * (float(lhs["val_macro_f1"]) - float(rhs["val_macro_f1"]))
                delta_acc.append(da)
                delta_f1.append(df)
                all_acc.append(da)
                all_f1.append(df)
                paired_rows.append(
                    {
                        "contrast": name,
                        "dataset": dataset,
                        "seed": seed,
                        "left_variant": left,
                        "right_variant": right,
                        "delta_acc_pp": da,
                        "delta_macro_f1_pp": df,
                    }
                )
            per_dataset[dataset] = {
                "mean_delta_acc_pp": _mean(delta_acc),
                "std_delta_acc_pp": _std(delta_acc),
                "positive_acc_seeds": sum(value > 0.0 for value in delta_acc),
                "mean_delta_macro_f1_pp": _mean(delta_f1),
                "std_delta_macro_f1_pp": _std(delta_f1),
                "positive_macro_f1_seeds": sum(value > 0.0 for value in delta_f1),
            }
        contrast_summary[name] = {
            "left_variant": left,
            "right_variant": right,
            "per_dataset": per_dataset,
            "macro_mean_delta_acc_pp": _mean([float(per_dataset[d]["mean_delta_acc_pp"]) for d in DATASETS]),
            "macro_mean_delta_macro_f1_pp": _mean(
                [float(per_dataset[d]["mean_delta_macro_f1_pp"]) for d in DATASETS]
            ),
            "paired_delta_mean_acc_pp": _mean(all_acc),
            "paired_delta_std_acc_pp": _std(all_acc),
            "paired_delta_mean_macro_f1_pp": _mean(all_f1),
            "paired_delta_std_macro_f1_pp": _std(all_f1),
        }
    _write_csv(args.output_dir / "o2_paired_contrasts.csv", paired_rows)

    p0_regression = next(
        row for row in dataset_summary if row["dataset"] == "Movies" and row["variant"] == "p0"
    )
    joint = contrast_summary["B_joint_graph_gain"]
    ownership = contrast_summary["C_ownership_graph_gain"]
    core = contrast_summary["D_ownership_minus_joint"]
    graph_candidates = {"joint_rd": joint, "ownership_rd": ownership}
    gate1_candidates = {
        variant: {
            "macro_mean_delta_acc_pp": float(value["macro_mean_delta_acc_pp"]),
            "dataset_mean_acc_pp": {
                dataset: float(value["per_dataset"][dataset]["mean_delta_acc_pp"])
                for dataset in DATASETS
            },
            "passes": float(value["macro_mean_delta_acc_pp"]) >= 1.5
            and all(float(value["per_dataset"][dataset]["mean_delta_acc_pp"]) > 0.0 for dataset in DATASETS),
        }
        for variant, value in graph_candidates.items()
    }
    core_dataset_acc = [float(core["per_dataset"][dataset]["mean_delta_acc_pp"]) for dataset in DATASETS]
    core_macro_acc = float(core["macro_mean_delta_acc_pp"])
    core_macro_f1 = float(core["macro_mean_delta_macro_f1_pp"])
    gate2 = {
        "macro_mean_delta_acc_pp": core_macro_acc,
        "macro_mean_delta_macro_f1_pp": core_macro_f1,
        "dataset_mean_delta_acc_pp": dict(zip(DATASETS, core_dataset_acc)),
        "positive_dataset_count": sum(value > 0.0 for value in core_dataset_acc),
        "datasets_at_or_below_minus_0_5pp": sum(value <= -0.5 for value in core_dataset_acc),
        "strong_go": core_macro_acc >= 0.30 and sum(value > 0.0 for value in core_dataset_acc) >= 2 and core_macro_f1 >= 0.0,
        "hold_separate_propagation": core_macro_acc < -0.50 or sum(value <= -0.5 for value in core_dataset_acc) >= 2,
    }
    if gate2["strong_go"]:
        gate2["verdict"] = "STRONG_GO"
    elif gate2["hold_separate_propagation"]:
        gate2["verdict"] = "HOLD_SEPARATE_PROPAGATION"
    elif core_macro_acc >= -0.30 and sum(value <= -0.5 for value in core_dataset_acc) < 2 and core_macro_f1 >= -0.30:
        gate2["verdict"] = "GO_VIABLE"
    else:
        gate2["verdict"] = "HOLD"

    summary = {
        "stage": "ORED-2",
        "protocol": "unified_full_graph_nc_v1",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "run_count": len(runs),
        "uses_test_labels_or_metrics": False,
        "p0_movies_summary": p0_regression,
        "contrasts": contrast_summary,
        "gates": {
            "gate1_graph_computation": {
                "candidates": gate1_candidates,
                "verdict": "GO_GRAPH" if any(item["passes"] for item in gate1_candidates.values()) else "HOLD_GRAPH_PARENT",
            },
            "gate2_ownership_preserving": gate2,
            "gate3_absolute_parent_quality": {
                "verdict": "CONDITIONAL_PARENT_PERFORMANCE_GAP",
                "Movies": {
                    "reference": "MAP-v2 lowpass_uniform, ORED-0 Movies seed42 Val reference",
                    "reference_val_acc_pct": 58.0683884273523,
                    "reference_val_macro_f1_pct": 49.86,
                    "ored_ownership_val_acc_pct": 100.0 * float(by_key[("Movies", 42, "ownership_rd")]["val_acc"]),
                    "ored_ownership_val_macro_f1_pct": 100.0 * float(by_key[("Movies", 42, "ownership_rd")]["val_macro_f1"]),
                    "gap_acc_pp": 100.0 * float(by_key[("Movies", 42, "ownership_rd")]["val_acc"]) - 58.0683884273523,
                    "gap_macro_f1_pp": 100.0 * float(by_key[("Movies", 42, "ownership_rd")]["val_macro_f1"]) - 49.86,
                },
                "Toys": "NO COMPARABLE VAL REFERENCE",
                "Grocery": "NO COMPARABLE VAL REFERENCE",
            },
        },
    }
    (args.output_dir / "o2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "o2_parameter_parity.json").write_text(
        json.dumps(
            {
                "variants": {
                    variant: {
                        "model_params": next(row["model_params"] for row in runs if row["variant"] == variant),
                        "head_params": next(row["head_params"] for row in runs if row["variant"] == variant),
                        "total_params": next(row["total_params"] for row in runs if row["variant"] == variant),
                    }
                    for variant in VARIANTS
                },
                "matched_control_variants": list(VARIANTS[1:]),
                "parity_by_dataset": {
                    dataset: {
                        "model_parameter_parity": len(
                            {row["model_params"] for row in runs if row["dataset"] == dataset and row["variant"] != "p0"}
                        ) == 1,
                        "head_parameter_parity": len(
                            {row["head_params"] for row in runs if row["dataset"] == dataset and row["variant"] != "p0"}
                        ) == 1,
                        "total_parameter_parity": len(
                            {row["total_params"] for row in runs if row["dataset"] == dataset and row["variant"] != "p0"}
                        ) == 1,
                    }
                    for dataset in DATASETS
                },
                "model_parameter_parity": all(
                    len({row["model_params"] for row in runs if row["dataset"] == dataset and row["variant"] != "p0"}) == 1
                    for dataset in DATASETS
                ),
                "head_parameter_parity": all(
                    len({row["head_params"] for row in runs if row["dataset"] == dataset and row["variant"] != "p0"}) == 1
                    for dataset in DATASETS
                ),
                "total_parameter_parity": all(
                    len({row["total_params"] for row in runs if row["dataset"] == dataset and row["variant"] != "p0"}) == 1
                    for dataset in DATASETS
                ),
                "diffusion_parameter_count": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
