from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SEED_RE = re.compile(r"seed(?P<seed>\d+)_runs(?P<runs>\d+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MAP-MAG v1 fusion ablation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/map_mag_v1_fusion_ablation"),
        help="Fusion ablation output root.",
    )
    parser.add_argument(
        "--core-baseline-root",
        type=Path,
        default=Path("outputs/map_mag_v1_core_ablation"),
        help="Core ablation root containing weighted_sum full baselines.",
    )
    parser.add_argument(
        "--include-core-baseline",
        action="store_true",
        help="Include latest core_ablation/<dataset>/full runs as weighted_sum baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to <input-root>/fusion_ablation_summary.csv.",
    )
    return parser.parse_args()


def _seed_info(seed_spec: str) -> tuple[int | None, int | None]:
    match = SEED_RE.search(seed_spec)
    if match is None:
        return None, None
    return int(match.group("seed")), int(match.group("runs"))


def _metric(results: dict, key: str, stat: str) -> float | None:
    value = results.get(key)
    if not isinstance(value, dict):
        return None
    metric = value.get(stat)
    return None if metric is None else float(metric)


def _row_from_results(path: Path, input_root: Path) -> dict[str, object] | None:
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 5:
        return None
    dataset, fusion_type, seed_spec = parts[0], parts[1], parts[2]
    seed, num_runs = _seed_info(seed_spec)
    timestamp = "/".join(parts[3:-1])
    results = json.loads(path.read_text(encoding="utf-8"))
    row: dict[str, object] = {
        "dataset": dataset,
        "fusion_type": fusion_type,
        "seed_spec": seed_spec,
        "base_seed": seed,
        "num_runs": num_runs,
        "timestamp": timestamp,
        "results_path": str(path),
    }
    for metric in (
        "val_acc",
        "test_acc",
        "test_macro_f1",
        "val_mrr",
        "test_mrr",
        "test_hits@1",
        "test_hits@3",
        "test_hits@10",
    ):
        row[f"{metric}_mean"] = _metric(results, metric, "mean")
        row[f"{metric}_std"] = _metric(results, metric, "std")
    return row


def _latest_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    latest: dict[tuple[object, object, object], str] = {}
    for row in rows:
        key = (row["dataset"], row["fusion_type"], row["seed_spec"])
        latest[key] = max(latest.get(key, ""), str(row["timestamp"]))
    return [row for row in rows if row["timestamp"] == latest[(row["dataset"], row["fusion_type"], row["seed_spec"])]]


def _discover_fusion_rows(input_root: Path) -> list[dict[str, object]]:
    rows = [
        row
        for path in sorted(input_root.glob("*/*/*/**/results.json"))
        if (row := _row_from_results(path, input_root)) is not None
    ]
    return _latest_rows(rows)


def _discover_core_baseline_rows(core_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(core_root.glob("*/full/*/**/results.json")):
        row = _row_from_results(path, core_root)
        if row is None:
            continue
        row["fusion_type"] = "weighted_sum"
        rows.append(row)
    return _latest_rows(rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No fusion ablation rows to summarize")
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


def _val_metric(dataset: str) -> str:
    return "val_mrr_mean" if dataset.endswith("-LP") else "val_acc_mean"


def _support_metric(dataset: str) -> str:
    return "test_hits@10_mean" if dataset.endswith("-LP") else "test_macro_f1_mean"


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


def _write_digest(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["# MAP-MAG v1 Fusion Ablation Summary", ""]
    lines.append("| dataset | fusion_type | runs | val | test | support |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        dataset = str(row["dataset"])
        lines.append(
            "| {} | {} | {} | {}±{} | {}±{} | {}±{} |".format(
                dataset,
                row["fusion_type"],
                row.get("num_runs", ""),
                _fmt(row.get(_val_metric(dataset)), True),
                _fmt(row.get(_val_metric(dataset).replace("_mean", "_std")), True),
                _fmt(row.get(_main_metric(dataset)), True),
                _fmt(row.get(_main_metric(dataset).replace("_mean", "_std")), True),
                _fmt(row.get(_support_metric(dataset)), True),
                _fmt(row.get(_support_metric(dataset).replace("_mean", "_std")), True),
            )
        )

    lines.extend(["", "## concat_mlp - weighted_sum Delta"])
    lines.append("| dataset | delta val pp | delta test pp | delta support pp |")
    lines.append("| --- | ---: | ---: | ---: |")
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        weighted = next((row for row in rows if row["dataset"] == dataset and row["fusion_type"] == "weighted_sum"), None)
        concat = next((row for row in rows if row["dataset"] == dataset and row["fusion_type"] == "concat_mlp"), None)
        if weighted is None or concat is None:
            continue
        val_key = _val_metric(dataset)
        test_key = _main_metric(dataset)
        support_key = _support_metric(dataset)
        deltas = []
        for key in (val_key, test_key, support_key):
            if weighted.get(key) is None or concat.get(key) is None:
                deltas.append("")
            else:
                deltas.append(_fmt((float(concat[key]) - float(weighted[key])) * 100.0))
        lines.append(f"| {dataset} | {deltas[0]} | {deltas[1]} | {deltas[2]} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    rows = _discover_fusion_rows(args.input_root)
    if args.include_core_baseline:
        rows.extend(_discover_core_baseline_rows(args.core_baseline_root))
        rows = _latest_rows(rows)
    if not rows:
        raise RuntimeError(f"No results.json files found under {args.input_root}")
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["fusion_type"])))

    output = args.output or args.input_root / "fusion_ablation_summary.csv"
    _write_csv(output, rows)
    _write_digest(output.with_suffix(".md"), rows)
    print(f"Summarized {len(rows)} result row(s): {output}", flush=True)


if __name__ == "__main__":
    main()
