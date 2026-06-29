from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


CHECKPOINTS = {"best_val", "final_epoch"}
RUN_RE = re.compile(r"run_(?P<run_id>\d+)_(?P<checkpoint>best_val|final_epoch)_prototype_aux\.csv$")
SEED_RE = re.compile(r"seed(?P<seed>\d+)_runs(?P<runs>\d+)")
TERCILE_LABELS = ("low", "middle", "high")


@dataclass(frozen=True)
class PrototypeFile:
    path: Path
    dataset: str
    ablation: str
    seed_spec: str
    timestamp: str
    run_id: int
    seed: int | None
    checkpoint: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MAP-MAG v1 prototype assignment semanticity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=Path("outputs/map_mag_v1_prototype_analysis"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/map_mag_v1_prototype_semantics/<timestamp>.",
    )
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), default="best_val")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset labels to include.")
    parser.add_argument("--ablations", nargs="+", default=["full"], help="Ablations to include, or 'all'.")
    parser.add_argument("--use-all-timestamps", action="store_true")
    return parser.parse_args()


def _infer_seed(seed_spec: str, run_id: int) -> int | None:
    match = SEED_RE.search(seed_spec)
    return None if match is None else int(match.group("seed")) + int(run_id) - 1


def _parse_file(path: Path, input_root: Path) -> PrototypeFile | None:
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 6 or parts[-2] != "prototype_aux":
        return None
    match = RUN_RE.match(parts[-1])
    if match is None:
        return None
    run_id = int(match.group("run_id"))
    seed_spec = parts[2]
    return PrototypeFile(
        path=path,
        dataset=parts[0],
        ablation=parts[1],
        seed_spec=seed_spec,
        timestamp=parts[3],
        run_id=run_id,
        seed=_infer_seed(seed_spec, run_id),
        checkpoint=match.group("checkpoint"),
    )


def _discover(input_root: Path, checkpoint: str) -> list[PrototypeFile]:
    files: list[PrototypeFile] = []
    for path in sorted(input_root.glob(f"*/*/*/*/prototype_aux/run_*_{checkpoint}_prototype_aux.csv")):
        parsed = _parse_file(path, input_root)
        if parsed is not None:
            files.append(parsed)
    return files


def _latest_timestamp_files(files: list[PrototypeFile]) -> list[PrototypeFile]:
    latest: dict[tuple[str, str, str], str] = {}
    for item in files:
        key = (item.dataset, item.ablation, item.seed_spec)
        latest[key] = max(latest.get(key, ""), item.timestamp)
    return [item for item in files if item.timestamp == latest[(item.dataset, item.ablation, item.seed_spec)]]


def _filter_files(files: list[PrototypeFile], datasets: list[str] | None, ablations: list[str]) -> list[PrototypeFile]:
    if datasets is not None:
        allowed = set(datasets)
        files = [item for item in files if item.dataset in allowed]
    if "all" not in ablations:
        allowed = set(ablations)
        files = [item for item in files if item.ablation in allowed]
    return files


def _float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(row: dict[str, object], key: str) -> int | None:
    value = _float(row, key)
    return None if value is None else int(value)


def _read_file(item: PrototypeFile) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with item.path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["dataset"] = item.dataset
            row["ablation"] = item.ablation
            row["seed_spec"] = item.seed_spec
            row["timestamp"] = item.timestamp
            row["run_id"] = item.run_id
            row["seed"] = "" if item.seed is None else item.seed
            row["checkpoint"] = item.checkpoint
            s_text = _float(row, "s_text") or 0.0
            s_visual = _float(row, "s_visual") or 0.0
            row["consistency"] = (s_text + s_visual) / 2.0
            rows.append(row)
    return rows


def _load(files: list[PrototypeFile]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in files:
        rows.extend(_read_file(item))
    if not rows:
        raise RuntimeError("No prototype aux CSV rows loaded.")
    return rows


def _group(rows: Iterable[dict[str, object]], keys: tuple[str, ...]):
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return groups


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return sum(clean) / len(clean) if clean else None


def _std(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None
    mu = sum(clean) / len(clean)
    return math.sqrt(sum((value - mu) ** 2 for value in clean) / len(clean))


def _median(values: list[float]) -> float | None:
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def _entropy_from_counts(counts: list[int]) -> float | None:
    total = sum(counts)
    if total <= 0:
        return None
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    return entropy


def _normalized_entropy_from_counts(counts: list[int]) -> float | None:
    if len(counts) <= 1:
        return 0.0
    entropy = _entropy_from_counts(counts)
    return None if entropy is None else entropy / math.log(len(counts))


def _comb2(n: int) -> float:
    return n * (n - 1) / 2.0


def _nmi(labels: list[int], clusters: list[int]) -> float | None:
    if len(labels) != len(clusters) or not labels:
        return None
    n = len(labels)
    label_counts = Counter(labels)
    cluster_counts = Counter(clusters)
    joint = Counter(zip(labels, clusters, strict=True))
    mi = 0.0
    for (label, cluster), count in joint.items():
        mi += (count / n) * math.log((count * n) / (label_counts[label] * cluster_counts[cluster]))
    h_label = _entropy_from_counts(list(label_counts.values())) or 0.0
    h_cluster = _entropy_from_counts(list(cluster_counts.values())) or 0.0
    if h_label <= 0.0 or h_cluster <= 0.0:
        return 0.0
    return mi / math.sqrt(h_label * h_cluster)


def _ari(labels: list[int], clusters: list[int]) -> float | None:
    if len(labels) != len(clusters) or not labels:
        return None
    n = len(labels)
    label_counts = Counter(labels)
    cluster_counts = Counter(clusters)
    joint = Counter(zip(labels, clusters, strict=True))
    sum_joint = sum(_comb2(count) for count in joint.values())
    sum_label = sum(_comb2(count) for count in label_counts.values())
    sum_cluster = sum(_comb2(count) for count in cluster_counts.values())
    total = _comb2(n)
    if total <= 0:
        return 0.0
    expected = sum_label * sum_cluster / total
    maximum = 0.5 * (sum_label + sum_cluster)
    denom = maximum - expected
    return 0.0 if denom == 0.0 else (sum_joint - expected) / denom


def _prototype_usage_summary(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    global_rows: list[dict[str, object]] = []
    usage_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for (dataset, ablation, checkpoint), group in _group(rows, ("dataset", "ablation", "checkpoint")).items():
        proto_ids = [_int(row, "top1_proto") for row in group]
        proto_ids = [proto for proto in proto_ids if proto is not None]
        if not proto_ids:
            continue
        max_proto = max(proto_ids)
        for key in ("top2_proto", "top3_proto"):
            values = [_int(row, key) for row in group]
            values = [value for value in values if value is not None]
            if values:
                max_proto = max(max_proto, max(values))
        num_proto = max_proto + 1
        counts = [0] * num_proto
        for proto in proto_ids:
            counts[proto] += 1
        total = sum(counts)
        top_proto = max(range(num_proto), key=lambda idx: counts[idx])
        labels: list[int] = []
        clusters: list[int] = []
        purity_numer = 0

        for proto in range(num_proto):
            proto_group = [row for row in group if _int(row, "top1_proto") == proto]
            label_counts: Counter[int] = Counter()
            for row in proto_group:
                label = _int(row, "label")
                if label is not None:
                    label_counts[label] += 1
                    labels.append(label)
                    clusters.append(proto)
            top_label = ""
            purity = None
            label_entropy = None
            if label_counts:
                top_label, top_count = label_counts.most_common(1)[0]
                purity = top_count / sum(label_counts.values())
                label_entropy = _entropy_from_counts(list(label_counts.values()))
                purity_numer += top_count
                for label, count in sorted(label_counts.items()):
                    label_rows.append(
                        {
                            "dataset": dataset,
                            "ablation": ablation,
                            "checkpoint": checkpoint,
                            "prototype": proto,
                            "label": label,
                            "count": count,
                            "fraction_in_prototype": count / sum(label_counts.values()),
                        }
                    )

            usage_rows.append(
                {
                    "dataset": dataset,
                    "ablation": ablation,
                    "checkpoint": checkpoint,
                    "prototype": proto,
                    "count": len(proto_group),
                    "fraction": len(proto_group) / max(total, 1),
                    "top_label": top_label,
                    "purity": purity,
                    "label_entropy": label_entropy,
                    "mean_p_proto": _mean([_float(row, "p_proto") for row in proto_group if _float(row, "p_proto") is not None]),
                    "mean_degree": _mean([_float(row, "degree") for row in proto_group if _float(row, "degree") is not None]),
                    "mean_consistency": _mean([_float(row, "consistency") for row in proto_group if _float(row, "consistency") is not None]),
                    "correct_rate": _mean([_float(row, "correct") for row in proto_group if _float(row, "correct") is not None]),
                }
            )

        global_rows.append(
            {
                "dataset": dataset,
                "ablation": ablation,
                "checkpoint": checkpoint,
                "num_nodes": total,
                "num_prototypes": num_proto,
                "active_prototypes": sum(count > 0 for count in counts),
                "active_prototypes_1pct": sum(count / max(total, 1) >= 0.01 for count in counts),
                "top_prototype": top_proto,
                "top_prototype_count": counts[top_proto],
                "top_prototype_fraction": counts[top_proto] / max(total, 1),
                "usage_entropy": _entropy_from_counts(counts),
                "usage_entropy_norm": _normalized_entropy_from_counts(counts),
                "mean_assignment_entropy": _mean([_float(row, "prototype_entropy") for row in group if _float(row, "prototype_entropy") is not None]),
                "mean_assignment_entropy_norm": _mean([_float(row, "prototype_entropy_norm") for row in group if _float(row, "prototype_entropy_norm") is not None]),
                "weighted_purity": purity_numer / max(len(labels), 1) if labels else None,
                "nmi_top1_label": _nmi(labels, clusters),
                "ari_top1_label": _ari(labels, clusters),
            }
        )
    return global_rows, usage_rows, label_rows


def _assign_terciles(rows: list[dict[str, object]], column: str) -> list[tuple[dict[str, object], str]]:
    valid = [(idx, _float(row, column)) for idx, row in enumerate(rows)]
    valid = [(idx, value) for idx, value in valid if value is not None]
    labels = ["unknown"] * len(rows)
    if len(valid) < 3:
        for idx, _ in valid:
            labels[idx] = "middle"
        return list(zip(rows, labels, strict=True))
    valid.sort(key=lambda item: item[1])
    n = len(valid)
    for rank, (idx, _) in enumerate(valid):
        if rank < n / 3:
            labels[idx] = "low"
        elif rank < 2 * n / 3:
            labels[idx] = "middle"
        else:
            labels[idx] = "high"
    return list(zip(rows, labels, strict=True))


def _split_scopes(group: list[dict[str, object]]) -> list[tuple[str, list[dict[str, object]]]]:
    scopes = [("all_nodes", group)]
    test_rows = [row for row in group if str(row.get("split", "")) == "test"]
    if test_rows:
        scopes.append(("test_nodes", test_rows))
    return scopes


def _describe_column(rows: list[dict[str, object]], column: str) -> dict[str, float | None]:
    values = [_float(row, column) for row in rows]
    values = [value for value in values if value is not None]
    return {"mean": _mean(values), "std": _std(values), "median": _median(values)}


def _p_proto_group_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    columns = ["p_proto", "degree", "consistency", "prototype_entropy_norm", "pred_confidence", "true_prob", "correct"]
    for (dataset, ablation, checkpoint), group in _group(rows, ("dataset", "ablation", "checkpoint")).items():
        for split_scope, scoped in _split_scopes(group):
            assigned = _assign_terciles(scoped, "p_proto")
            by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row, label in assigned:
                by_label[label].append(row)
            for label in TERCILE_LABELS:
                label_rows = by_label.get(label, [])
                record: dict[str, object] = {
                    "dataset": dataset,
                    "ablation": ablation,
                    "checkpoint": checkpoint,
                    "split_scope": split_scope,
                    "p_proto_group": label,
                    "n": len(label_rows),
                }
                for column in columns:
                    desc = _describe_column(label_rows, column)
                    for stat, value in desc.items():
                        record[f"{column}_{stat}"] = value
                output.append(record)
    return output


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / math.sqrt(vx * vy)


def _corr(rows: list[dict[str, object]], x_col: str, y_col: str, method: str) -> float | None:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        x = _float(row, x_col)
        y = _float(row, y_col)
        if x is not None and y is not None:
            pairs.append((x, y))
    if len(pairs) < 3:
        return None
    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    if method == "spearman":
        xs = _rank(xs)
        ys = _rank(ys)
    return _pearson(xs, ys)


def _p_proto_relation_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    columns = ["degree", "consistency", "prototype_entropy_norm", "pred_confidence", "true_prob", "correct"]
    for (dataset, ablation, checkpoint), group in _group(rows, ("dataset", "ablation", "checkpoint")).items():
        for split_scope, scoped in _split_scopes(group):
            record: dict[str, object] = {
                "dataset": dataset,
                "ablation": ablation,
                "checkpoint": checkpoint,
                "split_scope": split_scope,
                "n": len(scoped),
            }
            for column in columns:
                record[f"pearson_p_proto_{column}"] = _corr(scoped, "p_proto", column, "pearson")
                record[f"spearman_p_proto_{column}"] = _corr(scoped, "p_proto", column, "spearman")
            output.append(record)
    return output


def _write_manifest(files: list[PrototypeFile], output_dir: Path) -> None:
    rows = [
        {
            "dataset": item.dataset,
            "ablation": item.ablation,
            "seed_spec": item.seed_spec,
            "timestamp": item.timestamp,
            "run_id": item.run_id,
            "seed": item.seed,
            "checkpoint": item.checkpoint,
            "path": str(item.path),
        }
        for item in files
    ]
    _write_csv(output_dir / "manifest.csv", rows)


def _fmt(value: object, pct: bool = False) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    if pct:
        number *= 100.0
    return f"{number:.2f}" if pct else f"{number:.4f}"


def _markdown_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in fields) + " |")
    return "\n".join(lines)


def _write_digest(
    output_dir: Path,
    global_rows: list[dict[str, object]],
    relation_rows: list[dict[str, object]],
    group_rows: list[dict[str, object]],
) -> None:
    lines = ["# MAP-MAG v1 Prototype Semanticity Digest", ""]
    lines.append("## Prototype Assignment Usage")
    lines.append(
        _markdown_table(
            global_rows,
            [
                "dataset",
                "active_prototypes",
                "active_prototypes_1pct",
                "top_prototype_fraction",
                "usage_entropy_norm",
                "mean_assignment_entropy_norm",
                "weighted_purity",
                "nmi_top1_label",
                "ari_top1_label",
            ],
        )
    )
    lines.extend(["", "## p_proto Relation Correlations"])
    lines.append(
        _markdown_table(
            [row for row in relation_rows if row["split_scope"] in {"all_nodes", "test_nodes"}],
            [
                "dataset",
                "split_scope",
                "spearman_p_proto_degree",
                "spearman_p_proto_consistency",
                "spearman_p_proto_correct",
                "spearman_p_proto_pred_confidence",
            ],
        )
    )
    lines.extend(["", "## p_proto Tercile Summary"])
    lines.append(
        _markdown_table(
            [row for row in group_rows if row["split_scope"] in {"all_nodes", "test_nodes"}],
            [
                "dataset",
                "split_scope",
                "p_proto_group",
                "degree_mean",
                "consistency_mean",
                "correct_mean",
                "pred_confidence_mean",
                "true_prob_mean",
            ],
        )
    )
    (output_dir / "prototype_semanticity_digest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    files = _discover(args.input_root, args.checkpoint)
    files = _filter_files(files, args.datasets, args.ablations)
    if not args.use_all_timestamps:
        files = _latest_timestamp_files(files)
    if not files:
        raise RuntimeError(f"No prototype aux CSV files found under {args.input_root}")

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("outputs") / "map_mag_v1_prototype_semantics" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load(files)
    global_rows, usage_rows, label_rows = _prototype_usage_summary(rows)
    group_rows = _p_proto_group_summary(rows)
    relation_rows = _p_proto_relation_summary(rows)

    _write_manifest(files, output_dir)
    _write_csv(output_dir / "node_prototype_assignments.csv", rows)
    _write_csv(output_dir / "prototype_global_summary.csv", global_rows)
    _write_csv(output_dir / "prototype_usage_summary.csv", usage_rows)
    _write_csv(output_dir / "prototype_label_distribution.csv", label_rows)
    _write_csv(output_dir / "p_proto_group_summary.csv", group_rows)
    _write_csv(output_dir / "p_proto_relation_summary.csv", relation_rows)
    _write_digest(output_dir, global_rows, relation_rows, group_rows)

    print(f"Analyzed {len(files)} prototype aux file(s): {output_dir}", flush=True)


if __name__ == "__main__":
    main()
