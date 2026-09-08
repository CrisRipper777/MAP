from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


SEED_RE = re.compile(r"seed(?P<seed>\d+)_runs(?P<runs>\d+)")
RELIABILITY_MIN_RE = re.compile(r"reliability_min_(?P<value>\d+)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MAP-MAG v1 reliability validation outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/map_mag_v1_reliability_validation"),
        help="Reliability-validation output root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Summary output directory. Defaults to <input-root>.",
    )
    return parser.parse_args()


def _metric(results: dict, key: str, stat: str) -> float | None:
    value = results.get(key)
    if not isinstance(value, dict):
        return None
    metric = value.get(stat)
    return None if metric is None else float(metric)


def _seed_info(seed_spec: str) -> tuple[int | None, int | None]:
    match = SEED_RE.search(seed_spec)
    if match is None:
        return None, None
    return int(match.group("seed")), int(match.group("runs"))


def _reliability_min(setting: str) -> float | None:
    match = RELIABILITY_MIN_RE.search(setting)
    if match is None:
        return None
    return int(match.group("value")) / 100.0


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return sum(clean) / len(clean) if clean else None


def _std(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return None
    return statistics.pstdev(clean) if len(clean) > 1 else 0.0


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _read_float_column(path: Path, column: str) -> list[float]:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            return values
        for row in reader:
            value = row.get(column)
            if value not in (None, ""):
                values.append(float(value))
    return values


def _aux_summary(run_dir: Path, reliability_min: float | None) -> dict[str, float | None]:
    files = sorted((run_dir / "node_aux").glob("run_*_best_val_node_aux.csv"))
    if not files:
        return {}

    output: dict[str, float | None] = {"aux_runs": float(len(files))}
    for column in ("r_text", "r_visual"):
        values: list[float] = []
        for path in files:
            values.extend(_read_float_column(path, column))
        output[f"{column}_mean"] = _mean(values)
        output[f"{column}_std"] = _std(values)
        output[f"{column}_q05"] = _quantile(values, 0.05)
        output[f"{column}_q50"] = _quantile(values, 0.50)
        output[f"{column}_q95"] = _quantile(values, 0.95)
        if reliability_min is not None and values:
            eps = 1e-4
            output[f"{column}_near_min_frac"] = sum(value <= reliability_min + eps for value in values) / len(values)
    return output


def _row_from_results(path: Path, input_root: Path) -> dict[str, object] | None:
    rel = path.relative_to(input_root)
    parts = rel.parts
    if len(parts) < 6:
        return None
    dataset, phase, setting, seed_spec = parts[0], parts[1], parts[2], parts[3]
    timestamp = "/".join(parts[4:-1])
    base_seed, num_runs = _seed_info(seed_spec)
    results = json.loads(path.read_text(encoding="utf-8"))
    row: dict[str, object] = {
        "dataset": dataset,
        "phase": phase,
        "setting": setting,
        "seed_spec": seed_spec,
        "base_seed": base_seed,
        "num_runs": num_runs,
        "timestamp": timestamp,
        "results_path": str(path),
    }

    metric_names = sorted(results)
    for metric in metric_names:
        row[f"{metric}_mean"] = _metric(results, metric, "mean")
        row[f"{metric}_std"] = _metric(results, metric, "std")

    rel_min = _reliability_min(setting)
    if rel_min is not None:
        row["reliability_min"] = rel_min
    row.update(_aux_summary(path.parent, rel_min))
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _main_metric(dataset: str) -> str:
    return "test_mrr_mean" if dataset.endswith("-LP") else "test_acc_mean"


def _support_metric(dataset: str) -> str:
    return "test_hits@10_mean" if dataset.endswith("-LP") else "test_macro_f1_mean"


def _single_modality_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    datasets = sorted({str(row["dataset"]) for row in rows if row.get("phase") == "single_modality"})
    settings = ["mlp_text", "mlp_visual", "gcn_text", "gcn_visual", "map_mag_full"]
    for dataset in datasets:
        ds_rows = [row for row in rows if row.get("dataset") == dataset and row.get("phase") == "single_modality"]
        metric = _main_metric(dataset)
        support = _support_metric(dataset)
        record: dict[str, object] = {"dataset": dataset, "main_metric": metric, "support_metric": support}
        for setting in settings:
            row = next((item for item in ds_rows if item.get("setting") == setting), None)
            record[f"{setting}_main"] = row.get(metric) if row else None
            record[f"{setting}_support"] = row.get(support) if row else None
        full = next((item for item in ds_rows if item.get("setting") == "map_mag_full"), None)
        if full:
            record["map_mag_r_text_mean"] = full.get("r_text_mean")
            record["map_mag_r_visual_mean"] = full.get("r_visual_mean")
        output.append(record)
    return output


def _mask_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if row.get("phase") != "mask_robustness":
            continue
        dataset = str(row["dataset"])
        if dataset.endswith("-LP"):
            text_drop = row.get("drop_text_mrr_mean")
            visual_drop = row.get("drop_visual_mrr_mean")
            partial_drop = row.get("drop_both_partial_mrr_mean")
            full = row.get("test_mrr_mean")
        else:
            text_drop = row.get("drop_text_acc_mean")
            visual_drop = row.get("drop_visual_acc_mean")
            partial_drop = row.get("drop_both_partial_acc_mean")
            full = row.get("test_acc_mean")
        output.append(
            {
                "dataset": dataset,
                "setting": row.get("setting"),
                "num_runs": row.get("num_runs"),
                "full_metric": full,
                "mean_r_text": row.get("r_text_mean"),
                "mean_r_visual": row.get("r_visual_mean"),
                "drop_text": text_drop,
                "drop_visual": visual_drop,
                "drop_both_partial": partial_drop,
                "results_path": row.get("results_path"),
            }
        )
    return output


def _reliability_min_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = [row for row in rows if row.get("phase") == "reliability_min"]
    return sorted(output, key=lambda row: (str(row.get("dataset")), float(row.get("reliability_min", -1))))


def _fmt(value: object, pct: bool = False) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pct:
        number *= 100.0
    return f"{number:.2f}" if pct else f"{number:.4f}"


def _write_digest(output_dir: Path, single_rows: list[dict[str, object]], mask_rows: list[dict[str, object]], rel_rows: list[dict[str, object]]) -> None:
    lines: list[str] = ["# MAP-MAG v1 Reliability Validation Digest", ""]
    lines.append("## Single-Modality Performance")
    lines.append("| dataset | MLP-text | MLP-visual | GCN-text | GCN-visual | MAP-MAG full | r_text | r_visual |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in single_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["dataset"],
                _fmt(row.get("mlp_text_main"), True),
                _fmt(row.get("mlp_visual_main"), True),
                _fmt(row.get("gcn_text_main"), True),
                _fmt(row.get("gcn_visual_main"), True),
                _fmt(row.get("map_mag_full_main"), True),
                _fmt(row.get("map_mag_r_text_mean")),
                _fmt(row.get("map_mag_r_visual_mean")),
            )
        )

    lines.extend(["", "## Mask Robustness"])
    lines.append("| dataset | full | r_text | r_visual | drop_text | drop_visual | drop_both_partial |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in mask_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                row["dataset"],
                _fmt(row.get("full_metric"), True),
                _fmt(row.get("mean_r_text")),
                _fmt(row.get("mean_r_visual")),
                _fmt(row.get("drop_text"), True),
                _fmt(row.get("drop_visual"), True),
                _fmt(row.get("drop_both_partial"), True),
            )
        )

    lines.extend(["", "## Reliability-Min Ablation"])
    lines.append("| dataset | reliability_min | main metric | r_text | r_visual | r_text near min | r_visual near min |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rel_rows:
        metric = _main_metric(str(row["dataset"]))
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                row["dataset"],
                _fmt(row.get("reliability_min")),
                _fmt(row.get(metric), True),
                _fmt(row.get("r_text_mean")),
                _fmt(row.get("r_visual_mean")),
                _fmt(row.get("r_text_near_min_frac"), True),
                _fmt(row.get("r_visual_near_min_frac"), True),
            )
        )
    (output_dir / "reliability_validation_digest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    input_root = args.input_root
    output_dir = args.output_dir or input_root
    rows = [
        row
        for path in sorted(input_root.glob("*/*/*/*/**/results.json"))
        if (row := _row_from_results(path, input_root)) is not None
    ]
    if not rows:
        raise RuntimeError(f"No results.json files found under {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "reliability_validation_summary.csv"
    single_path = output_dir / "single_modality_performance.csv"
    mask_path = output_dir / "mask_robustness_summary.csv"
    rel_path = output_dir / "reliability_min_summary.csv"

    single_rows = _single_modality_rows(rows)
    mask_rows = _mask_rows(rows)
    rel_rows = _reliability_min_rows(rows)
    _write_csv(all_path, rows)
    _write_csv(single_path, single_rows)
    _write_csv(mask_path, mask_rows)
    _write_csv(rel_path, rel_rows)
    _write_digest(output_dir, single_rows, mask_rows, rel_rows)
    print(f"Summarized {len(rows)} results file(s): {all_path}", flush=True)


if __name__ == "__main__":
    main()
