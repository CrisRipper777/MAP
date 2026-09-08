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
    DatasetSpec(key="sports-copurchase", task="lp", label="sports-copurchase-LP"),
    DatasetSpec(key="ele-fashion", task="nc", label="ele-fashion-NC"),
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run MAP-MAG v1 full model with prototype semantic analysis export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default=None, help="Hydra device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--base-seed", type=int, default=42, help="Hydra cfg.seed. With num_runs=3 this uses 42/43/44.")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of seeds per dataset.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[item.key for item in DATASETS],
        default=[item.key for item in DATASETS],
        help="Datasets to run.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/map_mag_v1_prototype_analysis",
        help="Hydra run directory root.",
    )
    parser.add_argument(
        "--no-node-aux",
        action="store_true",
        help="Disable regular node aux CSV export; prototype aux remains enabled.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args, extra_overrides = parser.parse_known_args()
    if extra_overrides and extra_overrides[0] == "--":
        extra_overrides = extra_overrides[1:]
    return args, extra_overrides


def _selected_datasets(keys: list[str]) -> list[DatasetSpec]:
    selected = set(keys)
    return [item for item in DATASETS if item.key in selected]


def _build_command(dataset: DatasetSpec, args: argparse.Namespace, extra_overrides: list[str]) -> list[str]:
    run_dir = (
        f"{args.output_root}/{dataset.label}/full/"
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
        "model.export_prototype_aux=true",
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
    if not datasets:
        raise RuntimeError("No dataset jobs selected")

    print(
        "MAP-MAG v1 prototype semantic analysis export: "
        f"{len(datasets)} dataset(s) x 1 full setting = {len(datasets)} Hydra job(s)",
        flush=True,
    )
    print(
        f"Seeds per job: {args.base_seed}..{args.base_seed + args.num_runs - 1} "
        f"(num_runs={args.num_runs})",
        flush=True,
    )
    if extra_overrides:
        print(f"Extra Hydra overrides: {' '.join(extra_overrides)}", flush=True)

    for job_id, dataset in enumerate(datasets, start=1):
        cmd = _build_command(dataset, args, extra_overrides)
        print(f"\n[{job_id}/{len(datasets)}] {dataset.label} | MAP-MAG full + prototype export", flush=True)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
