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
class JobSpec:
    phase: str
    key: str
    label: str
    model: str
    overrides: tuple[str, ...]


DATASETS = (
    DatasetSpec(key="Movies", task="nc", label="Movies-NC"),
    DatasetSpec(key="Toys", task="nc", label="Toys-NC"),
    DatasetSpec(key="Grocery", task="nc", label="Grocery-NC"),
    DatasetSpec(key="sports-copurchase", task="lp", label="sports-copurchase-LP"),
    DatasetSpec(key="ele-fashion", task="nc", label="ele-fashion-NC"),
)

DEFAULT_DATASETS = ("Movies", "Toys", "sports-copurchase", "ele-fashion")

SINGLE_MODALITY_SPECS = (
    JobSpec(
        phase="single_modality",
        key="mlp_text",
        label="MLP-text",
        model="mlp",
        overrides=("+dataset.feature_mode=text",),
    ),
    JobSpec(
        phase="single_modality",
        key="mlp_visual",
        label="MLP-visual",
        model="mlp",
        overrides=("+dataset.feature_mode=visual",),
    ),
    JobSpec(
        phase="single_modality",
        key="gcn_text",
        label="GCN-text",
        model="gcn",
        overrides=("+dataset.feature_mode=text",),
    ),
    JobSpec(
        phase="single_modality",
        key="gcn_visual",
        label="GCN-visual",
        model="gcn",
        overrides=("+dataset.feature_mode=visual",),
    ),
    JobSpec(
        phase="single_modality",
        key="map_mag_full",
        label="MAP-MAG full",
        model="map_mag_v1",
        overrides=(),
    ),
)

PHASES = ("single_modality", "mask_robustness", "reliability_min")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run MAP-MAG v1 modality reliability validation experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default=None, help="Hydra device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--base-seed", type=int, default=42, help="Hydra cfg.seed. With num_runs=3 this uses 42/43/44.")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of seeds per job.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[item.key for item in DATASETS],
        default=list(DEFAULT_DATASETS),
        help="Datasets to run. Grocery is available but not in the default four-core set.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=PHASES,
        default=list(PHASES),
        help="Experiment phases to run.",
    )
    parser.add_argument(
        "--single-settings",
        nargs="+",
        choices=[item.key for item in SINGLE_MODALITY_SPECS],
        default=[item.key for item in SINGLE_MODALITY_SPECS],
        help="Single-modality settings to run.",
    )
    parser.add_argument(
        "--reliability-mins",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2],
        help="model.reliability_min values for the reliability-min ablation.",
    )
    parser.add_argument(
        "--mask-modes",
        nargs="+",
        default=["mask_text", "mask_visual", "mask_both_partial"],
        help="Best-checkpoint test-time mask modes.",
    )
    parser.add_argument(
        "--mask-partial-ratio",
        type=float,
        default=0.5,
        help="Node-level per-modality mask ratio for mask_both_partial.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/map_mag_v1_reliability_validation",
        help="Hydra run directory root.",
    )
    parser.add_argument(
        "--no-node-aux",
        action="store_true",
        help="Disable MAP-MAG node-level aux CSV export for faster metric-only runs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args, extra_overrides = parser.parse_known_args()
    if extra_overrides and extra_overrides[0] == "--":
        extra_overrides = extra_overrides[1:]
    return args, extra_overrides


def _selected_datasets(keys: list[str]) -> list[DatasetSpec]:
    selected = set(keys)
    return [item for item in DATASETS if item.key in selected]


def _selected_single_specs(keys: list[str]) -> list[JobSpec]:
    selected = set(keys)
    return [item for item in SINGLE_MODALITY_SPECS if item.key in selected]


def _reliability_min_key(value: float) -> str:
    return f"reliability_min_{int(round(value * 100)):03d}"


def _reliability_min_specs(values: list[float]) -> list[JobSpec]:
    return [
        JobSpec(
            phase="reliability_min",
            key=_reliability_min_key(value),
            label=f"reliability_min={value:g}",
            model="map_mag_v1",
            overrides=(f"model.reliability_min={value:g}",),
        )
        for value in values
    ]


def _mask_robustness_spec(args: argparse.Namespace) -> JobSpec:
    modes = ",".join(args.mask_modes)
    return JobSpec(
        phase="mask_robustness",
        key=f"map_mag_full_masks_p{int(round(args.mask_partial_ratio * 100)):02d}",
        label="MAP-MAG full + test-time modality masks",
        model="map_mag_v1",
        overrides=(
            "task.eval_modality_masks.enabled=true",
            f"task.eval_modality_masks.modes=[{modes}]",
            f"task.eval_modality_masks.partial_ratio={args.mask_partial_ratio:g}",
        ),
    )


def _jobs(args: argparse.Namespace) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    phases = set(args.phases)
    if "single_modality" in phases:
        jobs.extend(_selected_single_specs(args.single_settings))
    if "mask_robustness" in phases:
        jobs.append(_mask_robustness_spec(args))
    if "reliability_min" in phases:
        jobs.extend(_reliability_min_specs(args.reliability_mins))
    return jobs


def _build_command(
    dataset: DatasetSpec,
    job: JobSpec,
    args: argparse.Namespace,
    extra_overrides: list[str],
) -> list[str]:
    run_dir = (
        f"{args.output_root}/{dataset.label}/{job.phase}/{job.key}/"
        f"seed{args.base_seed}_runs{args.num_runs}/${{now:%Y-%m-%d_%H-%M-%S}}"
    )
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset.key}",
        f"task={dataset.task}",
        f"model={job.model}",
        f"seed={args.base_seed}",
        f"num_runs={args.num_runs}",
        f"hydra.run.dir={run_dir}",
    ]
    if args.device:
        cmd.append(f"device={args.device}")
    if args.no_node_aux and job.model == "map_mag_v1":
        cmd.append("model.export_node_aux=false")
    cmd.extend(job.overrides)
    cmd.extend(extra_overrides)
    return cmd


def main() -> None:
    args, extra_overrides = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    datasets = _selected_datasets(args.datasets)
    jobs = _jobs(args)
    total = len(datasets) * len(jobs)
    if total <= 0:
        raise RuntimeError("No reliability-validation jobs selected")

    print(
        "MAP-MAG v1 reliability validation: "
        f"{len(datasets)} dataset(s) x {len(jobs)} setting(s) = {total} Hydra job(s)",
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
        for job in jobs:
            job_id += 1
            cmd = _build_command(dataset, job, args, extra_overrides)
            print(f"\n[{job_id}/{total}] {dataset.label} | {job.phase} | {job.label}", flush=True)
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
