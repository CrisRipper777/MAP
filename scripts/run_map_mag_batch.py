from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    dataset: str
    task: str
    label: str


EXPERIMENTS = [
    Experiment(dataset="Movies", task="nc", label="Movies-NC"),
    Experiment(dataset="Reddit-S", task="nc", label="Reddit-S-NC"),
    Experiment(dataset="ele-fashion", task="nc", label="ele-fashion-NC"),
    Experiment(dataset="sports-copurchase", task="lp", label="sports-copurchase-LP"),
    Experiment(dataset="Grocery", task="lp", label="Grocery-LP"),
]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run MAP-MAG batch experiments on selected MAG datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        choices=(1, 3),
        default=3,
        help="Number of independent runs per experiment. Seeds are base_seed + run_id.",
    )
    parser.add_argument("--base-seed", type=int, default=42, help="Hydra seed passed as cfg.seed.")
    parser.add_argument("--device", default=None, help="Optional Hydra device override, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--only",
        choices=("all", "nc", "lp"),
        default="all",
        help="Run all experiments, only node classification, or only link prediction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args, extra_overrides = parser.parse_known_args()
    if extra_overrides and extra_overrides[0] == "--":
        extra_overrides = extra_overrides[1:]
    return args, extra_overrides


def _selected_experiments(kind: str) -> list[Experiment]:
    if kind == "all":
        return EXPERIMENTS
    return [exp for exp in EXPERIMENTS if exp.task == kind]


def _build_command(exp: Experiment, args: argparse.Namespace, extra_overrides: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={exp.dataset}",
        f"task={exp.task}",
        "model=map_mag",
        f"num_runs={args.num_runs}",
        f"seed={args.base_seed}",
    ]
    if args.device:
        cmd.append(f"device={args.device}")
    cmd.extend(extra_overrides)
    return cmd


def main() -> None:
    args, extra_overrides = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    experiments = _selected_experiments(args.only)
    if not experiments:
        raise RuntimeError(f"No experiments selected for --only={args.only!r}")

    print(
        f"Running MAP-MAG batch: {len(experiments)} experiment(s), "
        f"num_runs={args.num_runs}, seeds={args.base_seed}..{args.base_seed + args.num_runs - 1}",
        flush=True,
    )
    if extra_overrides:
        print(f"Extra Hydra overrides: {' '.join(extra_overrides)}", flush=True)

    for index, exp in enumerate(experiments, start=1):
        cmd = _build_command(exp, args, extra_overrides)
        print(f"\n[{index}/{len(experiments)}] {exp.label}", flush=True)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
