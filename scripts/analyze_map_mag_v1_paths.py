from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CHECKPOINTS = {"best_val", "final_epoch"}
DISTRIBUTION_COLUMNS = ("p_self", "p_struct", "p_proto", "gamma", "r_text", "r_visual")
DEGREE_GROUP_COLUMNS = ("p_self", "p_struct", "p_proto", "gamma")
CONSISTENCY_GROUP_COLUMNS = ("gamma", "p_struct", "p_self", "p_proto")
CLASS_COLUMNS = ("r_text", "r_visual", "p_self", "p_struct", "p_proto", "gamma")
TERCILE_LABELS = ("low", "middle", "high")
RUN_RE = re.compile(r"run_(?P<run_id>\d+)_(?P<checkpoint>best_val|final_epoch)_node_aux\.csv$")
SEED_RE = re.compile(r"seed(?P<seed>\d+)_runs(?P<runs>\d+)")


@dataclass(frozen=True)
class AuxFile:
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
        description="Analyze MAP-MAG v1 node-level aux CSVs for path preference interpretation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/map_mag_v1_core_ablation"),
        help="Root produced by scripts/run_map_mag_v1_core_ablation.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for analysis outputs. Defaults to outputs/map_mag_v1_path_analysis/<timestamp>.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=sorted(CHECKPOINTS),
        default="best_val",
        help="Which node aux checkpoint to analyze.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset labels such as Movies-NC Toys-NC sports-copurchase-LP. Default: all discovered.",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["full"],
        help="Ablation keys to analyze. Use 'all' to analyze every discovered ablation.",
    )
    parser.add_argument(
        "--use-all-timestamps",
        action="store_true",
        help="Analyze every matching timestamp instead of only the latest timestamp per dataset/ablation/seed spec.",
    )
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV summaries only.")
    return parser.parse_args()


def _infer_seed(seed_spec: str, run_id: int) -> int | None:
    match = SEED_RE.search(seed_spec)
    if not match:
        return None
    return int(match.group("seed")) + int(run_id) - 1


def _parse_aux_file(path: Path, input_root: Path) -> AuxFile | None:
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 6 or parts[-2] != "node_aux":
        return None
    match = RUN_RE.match(parts[-1])
    if match is None:
        return None
    run_id = int(match.group("run_id"))
    seed_spec = parts[2]
    return AuxFile(
        path=path,
        dataset=parts[0],
        ablation=parts[1],
        seed_spec=seed_spec,
        timestamp=parts[3],
        run_id=run_id,
        seed=_infer_seed(seed_spec, run_id),
        checkpoint=match.group("checkpoint"),
    )


def _discover_aux_files(input_root: Path, checkpoint: str) -> list[AuxFile]:
    files: list[AuxFile] = []
    for path in sorted(input_root.glob(f"*/*/*/*/node_aux/run_*_{checkpoint}_node_aux.csv")):
        parsed = _parse_aux_file(path, input_root)
        if parsed is not None:
            files.append(parsed)
    return files


def _latest_timestamp_files(files: list[AuxFile]) -> list[AuxFile]:
    latest: dict[tuple[str, str, str], str] = {}
    for item in files:
        key = (item.dataset, item.ablation, item.seed_spec)
        latest[key] = max(latest.get(key, ""), item.timestamp)
    return [
        item
        for item in files
        if item.timestamp == latest[(item.dataset, item.ablation, item.seed_spec)]
    ]


def _filter_files(files: list[AuxFile], datasets: list[str] | None, ablations: list[str]) -> list[AuxFile]:
    if datasets is not None:
        allowed_datasets = set(datasets)
        files = [item for item in files if item.dataset in allowed_datasets]
    if "all" not in ablations:
        allowed_ablations = set(ablations)
        files = [item for item in files if item.ablation in allowed_ablations]
    return files


def _read_aux(item: AuxFile) -> pd.DataFrame:
    df = pd.read_csv(item.path)
    df["dataset"] = item.dataset
    df["ablation"] = item.ablation
    df["seed_spec"] = item.seed_spec
    df["timestamp"] = item.timestamp
    df["run_id"] = item.run_id
    df["seed"] = item.seed if item.seed is not None else np.nan
    df["checkpoint"] = item.checkpoint
    df["consistency"] = (df["s_text"].astype(float) + df["s_visual"].astype(float)) / 2.0
    return df


def _load_group(files: list[AuxFile]) -> pd.DataFrame:
    frames = [_read_aux(item) for item in files]
    if not frames:
        raise RuntimeError("No node aux CSV files matched the requested filters.")
    return pd.concat(frames, ignore_index=True)


def _add_tercile_group(df: pd.DataFrame, column: str, group_column: str) -> None:
    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.notna()
    df[group_column] = "unknown"
    if int(valid.sum()) < 3:
        df.loc[valid, group_column] = "middle"
        return
    ranks = values[valid].rank(method="first")
    groups = pd.qcut(ranks, q=3, labels=TERCILE_LABELS)
    df.loc[valid, group_column] = groups.astype(str).to_numpy()


def _describe_series(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "q05": float(clean.quantile(0.05)),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.quantile(0.50)),
        "q75": float(clean.quantile(0.75)),
        "q95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def _distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, ablation, checkpoint), group in df.groupby(["dataset", "ablation", "checkpoint"], sort=True):
        for column in DISTRIBUTION_COLUMNS:
            row: dict[str, object] = {
                "dataset": dataset,
                "ablation": ablation,
                "checkpoint": checkpoint,
                "variable": column,
            }
            row.update(_describe_series(group[column]))
            rows.append(row)
    return pd.DataFrame(rows)


def _histogram_bins(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, ablation, checkpoint), group in df.groupby(["dataset", "ablation", "checkpoint"], sort=True):
        for column in DISTRIBUTION_COLUMNS:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy()
            counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
            total = max(int(counts.sum()), 1)
            for index, count in enumerate(counts):
                rows.append(
                    {
                        "dataset": dataset,
                        "ablation": ablation,
                        "checkpoint": checkpoint,
                        "variable": column,
                        "bin_left": float(edges[index]),
                        "bin_right": float(edges[index + 1]),
                        "count": int(count),
                        "fraction": float(count / total),
                    }
                )
    return pd.DataFrame(rows)


def _group_summary(
    df: pd.DataFrame,
    group_column: str,
    value_columns: Iterable[str],
    extra_columns: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_keys = ["dataset", "ablation", "checkpoint", group_column]
    for key, group in df.groupby(group_keys, sort=True):
        row: dict[str, object] = dict(zip(group_keys, key, strict=True))
        row["n"] = int(len(group))
        for column in extra_columns:
            desc = _describe_series(group[column])
            row[f"{column}_min"] = desc["min"]
            row[f"{column}_median"] = desc["median"]
            row[f"{column}_max"] = desc["max"]
        for column in value_columns:
            desc = _describe_series(group[column])
            for stat in ("mean", "std", "q25", "median", "q75"):
                row[f"{column}_{stat}"] = desc[stat]
        rows.append(row)
    return pd.DataFrame(rows)


def _class_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = pd.to_numeric(df["label"], errors="coerce")
    labeled = df[labels.notna()].copy()
    if labeled.empty:
        return pd.DataFrame()
    labeled["label"] = labels[labels.notna()].astype(int).to_numpy()
    for (dataset, ablation, checkpoint, label), group in labeled.groupby(
        ["dataset", "ablation", "checkpoint", "label"],
        sort=True,
    ):
        row: dict[str, object] = {
            "dataset": dataset,
            "ablation": ablation,
            "checkpoint": checkpoint,
            "label": int(label),
            "n": int(len(group)),
        }
        for column in CLASS_COLUMNS:
            desc = _describe_series(group[column])
            row[f"mean_{column}"] = desc["mean"]
            row[f"std_{column}"] = desc["std"]
        rows.append(row)
    return pd.DataFrame(rows)


def _degree_consistency_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["dataset", "ablation", "checkpoint", "degree_group", "consistency_group"]
    for key, group in df.groupby(keys, sort=True):
        row: dict[str, object] = dict(zip(keys, key, strict=True))
        row["n"] = int(len(group))
        for column in ("degree", "consistency", "p_self", "p_struct", "p_proto", "gamma"):
            desc = _describe_series(group[column])
            row[f"{column}_mean"] = desc["mean"]
            row[f"{column}_median"] = desc["median"]
        rows.append(row)
    return pd.DataFrame(rows)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{value:.3f}"


def _color01(value: float) -> str:
    value = max(0.0, min(1.0, float(value)))
    r = int(247 - 190 * value)
    g = int(251 - 115 * value)
    b = int(255 - 45 * value)
    return f"#{r:02x}{g:02x}{b:02x}"


def _write_svg(path: Path, body: str, width: int, height: int) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<style>text{font-family:Arial,sans-serif;} .small{font-size:11px;} '
        '.title{font-size:16px;font-weight:700;} .axis{stroke:#333;stroke-width:1;} '
        '.grid{stroke:#ddd;stroke-width:1;} .box{fill:#d9e8fb;stroke:#2f5f9f;stroke-width:1.5;} '
        '.median{stroke:#d1495b;stroke-width:2;} .whisker{stroke:#2f5f9f;stroke-width:1.5;}'
        '</style>\n'
        f"{body}\n</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def _histogram_svg(df: pd.DataFrame, path: Path, title: str, bins: int) -> None:
    width, height = 1180, 760
    panel_w, panel_h = 350, 270
    margin_x, margin_y = 60, 70
    gap_x, gap_y = 35, 55
    body = [f'<text x="30" y="32" class="title">{escape(title)}</text>']
    for index, column in enumerate(DISTRIBUTION_COLUMNS):
        row, col = divmod(index, 3)
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = margin_y + row * (panel_h + gap_y)
        values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy()
        counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
        max_count = max(int(counts.max()) if counts.size else 0, 1)
        plot_h = panel_h - 45
        plot_w = panel_w - 40
        body.append(f'<text x="{x0}" y="{y0 - 12}" class="small">{escape(column)}</text>')
        body.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" class="axis"/>')
        body.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" class="axis"/>')
        for tick in (0.0, 0.5, 1.0):
            tx = x0 + tick * plot_w
            body.append(f'<line x1="{tx}" y1="{y0 + plot_h}" x2="{tx}" y2="{y0 + plot_h + 4}" class="axis"/>')
            body.append(f'<text x="{tx - 8}" y="{y0 + plot_h + 18}" class="small">{tick:g}</text>')
        bar_w = plot_w / bins
        for bin_index, count in enumerate(counts):
            bar_h = (float(count) / max_count) * (plot_h - 5)
            bx = x0 + bin_index * bar_w
            by = y0 + plot_h - bar_h
            body.append(
                f'<rect x="{bx:.2f}" y="{by:.2f}" width="{max(bar_w - 1, 0.5):.2f}" '
                f'height="{bar_h:.2f}" fill="#4c78a8"/>'
            )
        body.append(f'<text x="{x0 + plot_w - 72}" y="{y0 + 12}" class="small">n={len(values)}</text>')
    _write_svg(path, "\n".join(body), width, height)


def _box_stats(values: pd.Series) -> tuple[float, float, float, float, float] | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return tuple(float(clean.quantile(q)) for q in (0.05, 0.25, 0.50, 0.75, 0.95))


def _distribution_boxplot_svg(df: pd.DataFrame, path: Path, title: str) -> None:
    width, height = 980, 520
    x0, y0 = 80, 60
    plot_w, plot_h = 820, 360
    body = [f'<text x="30" y="32" class="title">{escape(title)}</text>']
    body.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" class="axis"/>')
    body.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" class="axis"/>')
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        ty = y0 + plot_h - tick * plot_h
        body.append(f'<line x1="{x0}" y1="{ty}" x2="{x0 + plot_w}" y2="{ty}" class="grid"/>')
        body.append(f'<text x="{x0 - 42}" y="{ty + 4}" class="small">{tick:g}</text>')
    step = plot_w / len(DISTRIBUTION_COLUMNS)
    for index, column in enumerate(DISTRIBUTION_COLUMNS):
        stats = _box_stats(df[column])
        cx = x0 + step * (index + 0.5)
        body.append(f'<text x="{cx - 34}" y="{y0 + plot_h + 28}" class="small">{escape(column)}</text>')
        if stats is None:
            continue
        q05, q25, q50, q75, q95 = stats
        y05, y25, y50, y75, y95 = [y0 + plot_h - q * plot_h for q in stats]
        body.append(f'<line x1="{cx}" y1="{y95}" x2="{cx}" y2="{y05}" class="whisker"/>')
        body.append(f'<line x1="{cx - 18}" y1="{y95}" x2="{cx + 18}" y2="{y95}" class="whisker"/>')
        body.append(f'<line x1="{cx - 18}" y1="{y05}" x2="{cx + 18}" y2="{y05}" class="whisker"/>')
        body.append(f'<rect x="{cx - 26}" y="{y75}" width="52" height="{max(y25 - y75, 1)}" class="box"/>')
        body.append(f'<line x1="{cx - 28}" y1="{y50}" x2="{cx + 28}" y2="{y50}" class="median"/>')
    _write_svg(path, "\n".join(body), width, height)


def _group_boxplot_svg(
    df: pd.DataFrame,
    path: Path,
    title: str,
    group_column: str,
    value_columns: Iterable[str],
) -> None:
    value_columns = tuple(value_columns)
    width, height = 1120, 280 * len(value_columns) + 70
    body = [f'<text x="30" y="32" class="title">{escape(title)}</text>']
    labels = list(TERCILE_LABELS)
    for metric_index, metric in enumerate(value_columns):
        x0, y0 = 80, 70 + metric_index * 280
        plot_w, plot_h = 900, 190
        body.append(f'<text x="{x0}" y="{y0 - 18}" class="small">{escape(metric)}</text>')
        body.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" class="axis"/>')
        body.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" class="axis"/>')
        for tick in (0.0, 0.5, 1.0):
            ty = y0 + plot_h - tick * plot_h
            body.append(f'<line x1="{x0}" y1="{ty}" x2="{x0 + plot_w}" y2="{ty}" class="grid"/>')
            body.append(f'<text x="{x0 - 35}" y="{ty + 4}" class="small">{tick:g}</text>')
        step = plot_w / len(labels)
        for index, group_name in enumerate(labels):
            values = df.loc[df[group_column] == group_name, metric]
            stats = _box_stats(values)
            cx = x0 + step * (index + 0.5)
            body.append(f'<text x="{cx - 28}" y="{y0 + plot_h + 22}" class="small">{escape(group_name)}</text>')
            if stats is None:
                continue
            q05, q25, q50, q75, q95 = stats
            y05, y25, y50, y75, y95 = [y0 + plot_h - q * plot_h for q in stats]
            body.append(f'<line x1="{cx}" y1="{y95}" x2="{cx}" y2="{y05}" class="whisker"/>')
            body.append(f'<line x1="{cx - 22}" y1="{y95}" x2="{cx + 22}" y2="{y95}" class="whisker"/>')
            body.append(f'<line x1="{cx - 22}" y1="{y05}" x2="{cx + 22}" y2="{y05}" class="whisker"/>')
            body.append(f'<rect x="{cx - 34}" y="{y75}" width="68" height="{max(y25 - y75, 1)}" class="box"/>')
            body.append(f'<line x1="{cx - 36}" y1="{y50}" x2="{cx + 36}" y2="{y50}" class="median"/>')
            body.append(f'<text x="{cx - 18}" y="{y0 + plot_h + 38}" class="small">n={len(values.dropna())}</text>')
    _write_svg(path, "\n".join(body), width, height)


def _class_heatmap_svg(class_df: pd.DataFrame, path: Path, title: str) -> None:
    if class_df.empty:
        return
    metrics = [f"mean_{column}" for column in CLASS_COLUMNS]
    labels = list(class_df["label"].astype(int))
    cell_w, cell_h = 95, 24
    left, top = 90, 70
    width = left + cell_w * len(metrics) + 30
    height = top + cell_h * len(labels) + 45
    body = [f'<text x="30" y="32" class="title">{escape(title)}</text>']
    for col_idx, metric in enumerate(metrics):
        x = left + col_idx * cell_w
        body.append(f'<text x="{x + 5}" y="{top - 15}" class="small">{escape(metric.removeprefix("mean_"))}</text>')
    for row_idx, (_, row) in enumerate(class_df.iterrows()):
        y = top + row_idx * cell_h
        body.append(f'<text x="35" y="{y + 16}" class="small">class {int(row["label"])}</text>')
        for col_idx, metric in enumerate(metrics):
            value = float(row[metric])
            x = left + col_idx * cell_w
            body.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{_color01(value)}" stroke="#ffffff"/>'
            )
            body.append(f'<text x="{x + 30}" y="{y + 16}" class="small">{_fmt(value)}</text>')
    _write_svg(path, "\n".join(body), width, height)


def _write_manifest(files: list[AuxFile], output_dir: Path) -> None:
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
    pd.DataFrame(rows).to_csv(output_dir / "manifest.csv", index=False)


def _write_dataset_outputs(df: pd.DataFrame, output_dir: Path, bins: int, no_plots: bool) -> None:
    for (dataset, ablation), group in df.groupby(["dataset", "ablation"], sort=True):
        target = output_dir / dataset / ablation
        _ensure_dir(target)
        group = group.copy()
        _add_tercile_group(group, "degree", "degree_group")
        _add_tercile_group(group, "consistency", "consistency_group")

        _distribution_summary(group).to_csv(target / "distribution_summary.csv", index=False)
        _histogram_bins(group, bins).to_csv(target / "histogram_bins.csv", index=False)
        _group_summary(
            group,
            "degree_group",
            DEGREE_GROUP_COLUMNS,
            extra_columns=("degree",),
        ).to_csv(target / "degree_group_summary.csv", index=False)
        _group_summary(
            group,
            "consistency_group",
            CONSISTENCY_GROUP_COLUMNS,
            extra_columns=("consistency",),
        ).to_csv(target / "consistency_group_summary.csv", index=False)
        _degree_consistency_summary(group).to_csv(target / "degree_consistency_group_summary.csv", index=False)
        class_df = _class_summary(group)
        if not class_df.empty:
            class_df.to_csv(target / "class_group_summary.csv", index=False)

        if no_plots:
            continue
        _histogram_svg(group, target / "distribution_histograms.svg", f"{dataset} / {ablation}: distributions", bins)
        _distribution_boxplot_svg(group, target / "distribution_boxplots.svg", f"{dataset} / {ablation}: boxplots")
        _group_boxplot_svg(
            group,
            target / "degree_group_boxplots.svg",
            f"{dataset} / {ablation}: degree groups",
            "degree_group",
            DEGREE_GROUP_COLUMNS,
        )
        _group_boxplot_svg(
            group,
            target / "consistency_group_boxplots.svg",
            f"{dataset} / {ablation}: consistency groups",
            "consistency_group",
            CONSISTENCY_GROUP_COLUMNS,
        )
        if not class_df.empty:
            _class_heatmap_svg(class_df, target / "class_group_heatmap.svg", f"{dataset} / {ablation}: class means")


def main() -> None:
    args = _parse_args()
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("outputs/map_mag_v1_path_analysis") / timestamp
    else:
        output_dir = args.output_dir
    _ensure_dir(output_dir)

    files = _discover_aux_files(args.input_root, args.checkpoint)
    files = _filter_files(files, args.datasets, args.ablations)
    if not args.use_all_timestamps:
        files = _latest_timestamp_files(files)
    files = sorted(files, key=lambda item: (item.dataset, item.ablation, item.seed_spec, item.timestamp, item.run_id))
    if not files:
        raise RuntimeError(
            f"No node aux files found under {args.input_root} for checkpoint={args.checkpoint}, "
            f"datasets={args.datasets or 'all'}, ablations={args.ablations}."
        )

    df = _load_group(files)
    _write_manifest(files, output_dir)
    _write_dataset_outputs(df, output_dir, bins=int(args.bins), no_plots=bool(args.no_plots))

    print(f"Analyzed {len(files)} node aux file(s)")
    print(f"Datasets: {', '.join(sorted(df['dataset'].unique()))}")
    print(f"Ablations: {', '.join(sorted(df['ablation'].unique()))}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
