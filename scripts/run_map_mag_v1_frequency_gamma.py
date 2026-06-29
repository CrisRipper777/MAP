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


@dataclass(frozen=True)
class GammaSpec:
    key: str
    label: str
    overrides: tuple[str, ...]


DATASETS = (
    DatasetSpec(key="Movies", task="nc", label="Movies-NC"),
    DatasetSpec(key="Toys", task="nc", label="Toys-NC"),
    DatasetSpec(key="ele-fashion", task="nc", label="ele-fashion-NC"),
)


GAMMA_SPECS = (
    GammaSpec(key="learned_gamma", label="learned gamma", overrides=("model.fixed_gamma=null",)),
    GammaSpec(key="fixed_gamma_005", label="fixed gamma=0.05", overrides=("model.fixed_gamma=0.05",)),
    GammaSpec(key="fixed_gamma_050", label="fixed gamma=0.50", overrides=("model.fixed_gamma=0.5",)),
    GammaSpec(key="fixed_gamma_075", label="fixed gamma=0.75", overrides=("model.fixed_gamma=0.75",)),
    GammaSpec(key="fixed_gamma_095", label="fixed gamma=0.95", overrides=("model.fixed_gamma=0.95",)),
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run MAP-MAG v1 frequency-gamma diagnostics on Movies/Toys/ele-fashion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default=None, help="Hydra device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--base-seed", type=int, default=42, help="Hydra cfg.seed. With num_runs=3 this uses 42/43/44.")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of seeds per dataset/gamma setting.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[item.key for item in DATASETS],
        default=[item.key for item in DATASETS],
        help="Datasets to run.",
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        choices=[item.key for item in GAMMA_SPECS],
        default=[item.key for item in GAMMA_SPECS],
        help="Gamma settings to run.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/map_mag_v1_frequency_gamma",
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


def _selected_gammas(keys: list[str]) -> list[GammaSpec]:
    selected = set(keys)
    return [item for item in GAMMA_SPECS if item.key in selected]


def _build_command(
    dataset: DatasetSpec,
    gamma: GammaSpec,
    args: argparse.Namespace,
    extra_overrides: list[str],
) -> list[str]:
    run_dir = (
        f"{args.output_root}/{dataset.label}/{gamma.key}/"
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
    ]
    if args.device:
        cmd.append(f"device={args.device}")
    if args.no_node_aux:
        cmd.append("model.export_node_aux=false")
    cmd.extend(gamma.overrides)
    cmd.extend(extra_overrides)
    return cmd


def main() -> None:
    args, extra_overrides = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    datasets = _selected_datasets(args.datasets)
    gammas = _selected_gammas(args.gammas)
    total = len(datasets) * len(gammas)
    if total <= 0:
        raise RuntimeError("No dataset/gamma jobs selected")

    print(
        "MAP-MAG v1 frequency-gamma diagnostics: "
        f"{len(datasets)} dataset(s) x {len(gammas)} gamma setting(s) = {total} Hydra job(s)",
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
        for gamma in gammas:
            job_id += 1
            cmd = _build_command(dataset, gamma, args, extra_overrides)
            print(f"\n[{job_id}/{total}] {dataset.label} | {gamma.label}", flush=True)
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
