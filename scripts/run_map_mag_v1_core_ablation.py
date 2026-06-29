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
class AblationSpec:
    key: str
    label: str
    purpose: str
    overrides: tuple[str, ...]


DATASETS = (
    DatasetSpec(key="Movies", task="nc", label="Movies-NC"),
    DatasetSpec(key="Toys", task="nc", label="Toys-NC"),
    DatasetSpec(key="sports-copurchase", task="lp", label="sports-copurchase-LP"),
    DatasetSpec(key="ele-fashion", task="nc", label="ele-fashion-NC"),
)


ABLATIONS = (
    AblationSpec(
        key="full",
        label="Full MAP-MAG",
        purpose="main model",
        overrides=(),
    ),
    AblationSpec(
        key="wo_reliability",
        label="w/o reliability",
        purpose="modality reliability gates fixed to 1",
        overrides=("model.use_reliability=false",),
    ),
    AblationSpec(
        key="wo_router",
        label="w/o router",
        purpose="uniform fusion over active paths",
        overrides=("model.use_preference_router=false",),
    ),
    AblationSpec(
        key="wo_prototype",
        label="w/o prototype",
        purpose="remove global prototype path",
        overrides=("model.use_prototype_path=false",),
    ),
    AblationSpec(
        key="wo_structure",
        label="w/o structure",
        purpose="remove local structure path",
        overrides=("model.use_structure_path=false",),
    ),
    AblationSpec(
        key="wo_self",
        label="w/o self",
        purpose="remove self modality preservation path",
        overrides=("model.use_self_path=false",),
    ),
    AblationSpec(
        key="self_only",
        label="self only",
        purpose="pure attribute pathway ceiling",
        overrides=("model.self_only=true",),
    ),
    AblationSpec(
        key="struct_only",
        label="struct only",
        purpose="structure path independent capacity",
        overrides=(
            "model.use_self_path=false",
            "model.use_structure_path=true",
            "model.use_prototype_path=false",
            "model.use_preference_router=false",
        ),
    ),
    AblationSpec(
        key="proto_only",
        label="proto only",
        purpose="prototype path independent capacity",
        overrides=(
            "model.use_self_path=false",
            "model.use_structure_path=false",
            "model.use_prototype_path=true",
            "model.use_preference_router=false",
        ),
    ),
    AblationSpec(
        key="low_pass_only",
        label="low-pass only",
        purpose="remove high-frequency residual in structure path",
        overrides=("model.structure_low_pass_only=true",),
    ),
    AblationSpec(
        key="fixed_gamma_095",
        label="fixed gamma=0.95",
        purpose="mostly low-frequency structure diffusion",
        overrides=("model.fixed_gamma=0.95",),
    ),
    AblationSpec(
        key="fixed_gamma_05",
        label="fixed gamma=0.5",
        purpose="fixed middle low/high frequency mixture",
        overrides=("model.fixed_gamma=0.5",),
    ),
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run MAP-MAG v1 core ablations on Movies/Toys/sports/ele-fashion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default=None, help="Hydra device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--base-seed", type=int, default=42, help="Hydra cfg.seed. With num_runs=3 this uses 42/43/44.")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of seeds per dataset/ablation.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[item.key for item in DATASETS],
        default=[item.key for item in DATASETS],
        help="Datasets to run.",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=[item.key for item in ABLATIONS],
        default=[item.key for item in ABLATIONS],
        help="Ablations to run.",
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


def _selected_ablations(keys: list[str]) -> list[AblationSpec]:
    selected = set(keys)
    return [item for item in ABLATIONS if item.key in selected]


def _build_command(
    dataset: DatasetSpec,
    ablation: AblationSpec,
    args: argparse.Namespace,
    extra_overrides: list[str],
) -> list[str]:
    run_dir = (
        f"outputs/map_mag_v1_core_ablation/{dataset.label}/{ablation.key}/"
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
    cmd.extend(ablation.overrides)
    cmd.extend(extra_overrides)
    return cmd


def main() -> None:
    args, extra_overrides = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    datasets = _selected_datasets(args.datasets)
    ablations = _selected_ablations(args.ablations)
    total = len(datasets) * len(ablations)
    if total <= 0:
        raise RuntimeError("No dataset/ablation jobs selected")

    print(
        "MAP-MAG v1 core ablations: "
        f"{len(datasets)} dataset(s) x {len(ablations)} ablation(s) = {total} Hydra job(s)",
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
        for ablation in ablations:
            job_id += 1
            cmd = _build_command(dataset, ablation, args, extra_overrides)
            print(f"\n[{job_id}/{total}] {dataset.label} | {ablation.label}", flush=True)
            print(f"Purpose: {ablation.purpose}", flush=True)
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
