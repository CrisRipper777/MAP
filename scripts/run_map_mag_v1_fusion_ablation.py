from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    task: str
    label: str


DATASETS = (
    DatasetSpec(key="Movies", task="nc", label="Movies-NC"),
    DatasetSpec(key="Toys", task="nc", label="Toys-NC"),
    DatasetSpec(key="ele-fashion", task="nc", label="ele-fashion-NC"),
    DatasetSpec(key="sports-copurchase", task="lp", label="sports-copurchase-LP"),
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run MAP-MAG v1 path-fusion ablation. Defaults to concat_mlp only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default=None, help="Hydra device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--base-seed", type=int, default=42, help="Hydra cfg.seed. With num_runs=3 this uses 42/43/44.")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of seeds per dataset/fusion setting.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[item.key for item in DATASETS],
        default=[item.key for item in DATASETS],
        help="Datasets to run.",
    )
    parser.add_argument(
        "--fusion-types",
        nargs="+",
        choices=["concat_mlp", "weighted_sum"],
        default=["concat_mlp"],
        help="Fusion types to run. weighted_sum is included only for sanity reruns.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/map_mag_v1_fusion_ablation",
        help="Hydra run directory root.",
    )
    parser.add_argument(
        "--no-node-aux",
        action="store_true",
        help="Disable node-level aux CSV export for faster metric-only runs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args, extra_overrides = parser.parse_known_args()
    if extra_overrides and extra_overrides[0] == "--":
        extra_overrides = extra_overrides[1:]
    return args, extra_overrides


def _selected_datasets(keys: list[str]) -> list[DatasetSpec]:
    selected = set(keys)
    return [item for item in DATASETS if item.key in selected]


def _build_command(
    dataset: DatasetSpec,
    fusion_type: str,
    args: argparse.Namespace,
    extra_overrides: list[str],
) -> list[str]:
    run_dir = (
        f"{args.output_root}/{dataset.label}/{fusion_type}/"
        f"seed{args.base_seed}_runs{args.num_runs}/${{now:%Y-%m-%d_%H-%M-%S}}"
    )
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset.key}",
        f"task={dataset.task}",
        "model=map_mag_v1",
        f"seed={args.base_seed}",
        f"num_runs={args.num_runs}",
        f"hydra.run.dir={run_dir}",
        f"model.fusion_type={fusion_type}",
    ]
    if args.device:
        cmd.append(f"device={args.device}")
    if args.no_node_aux:
        cmd.append("model.export_node_aux=false")
    cmd.extend(extra_overrides)
    return cmd


def main() -> None:
    args, extra_overrides = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    datasets = _selected_datasets(args.datasets)
    fusion_types = list(args.fusion_types)
    total = len(datasets) * len(fusion_types)
    if total <= 0:
        raise RuntimeError("No dataset/fusion jobs selected")

    print(
        "MAP-MAG v1 fusion ablation: "
        f"{len(datasets)} dataset(s) x {len(fusion_types)} fusion setting(s) = {total} Hydra job(s)",
        flush=True,
    )
    print(
        f"Seeds per job: {args.base_seed}..{args.base_seed + args.num_runs - 1} "
        f"(num_runs={args.num_runs})",
        flush=True,
    )
    if extra_overrides:
        print(f"Extra Hydra overrides: {' '.join(extra_overrides)}", flush=True)

    job_id = 0
    for dataset in datasets:
        for fusion_type in fusion_types:
            job_id += 1
            cmd = _build_command(dataset, fusion_type, args, extra_overrides)
            print(f"\n[{job_id}/{total}] {dataset.label} | fusion_type={fusion_type}", flush=True)
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
