from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


GAMMA_RE = re.compile(r"fixed_gamma_(?P<digits>\d+)$")
SEED_RE = re.compile(r"seed(?P<seed>\d+)_runs(?P<runs>\d+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MAP-MAG frequency-gamma results from results.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/map_mag_v1_frequency_gamma"),
        help="Frequency-gamma output root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to <input-root>/frequency_gamma_summary.csv.",
    )
    return parser.parse_args()


def _gamma_value(gamma_key: str) -> str:
    if gamma_key == "learned_gamma":
        return "learned"
    match = GAMMA_RE.match(gamma_key)
    if match is None:
        return gamma_key
    return f"{int(match.group('digits')) / 100:.2f}"


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
    rel = path.relative_to(input_root)
    parts = rel.parts
    if len(parts) < 5:
        return None
    dataset, gamma_key, seed_spec = parts[0], parts[1], parts[2]
    seed, num_runs = _seed_info(seed_spec)
    timestamp = "/".join(parts[3:-1])
    results = json.loads(path.read_text(encoding="utf-8"))
    row: dict[str, object] = {
        "dataset": dataset,
        "gamma_key": gamma_key,
        "gamma": _gamma_value(gamma_key),
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


def main() -> None:
    args = _parse_args()
    input_root = args.input_root
    output = args.output or input_root / "frequency_gamma_summary.csv"
    rows = [
        row
        for path in sorted(input_root.glob("*/*/*/**/results.json"))
        if (row := _row_from_results(path, input_root)) is not None
    ]
    if not rows:
        raise RuntimeError(f"No results.json files found under {input_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summarized {len(rows)} results file(s): {output}", flush=True)


if __name__ == "__main__":
    main()
