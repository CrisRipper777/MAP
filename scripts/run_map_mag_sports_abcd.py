from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentGroup:
    name: str
    description: str
    overrides: tuple[str, ...]


BASE_OVERRIDES = (
    "model=map_mag",
    "task=lp",
    "dataset=sports-copurchase",
    "num_runs=3",
    "seed=42",
    "task.epochs=150",
    "task.lr=1e-5",
    "model.num_prototypes=16",
    "model.lambda_gate=0.01",
    "model.lambda_proto=0.1",
    "model.router_temperature=3",
)


GROUPS = (
    ExperimentGroup(
        name="A",
        description="current MAP-MAG config",
        overrides=(),
    ),
    ExperimentGroup(
        name="B",
        description="A + gamma_temperature=3",
        overrides=("model.gamma_temperature=3",),
    ),
    ExperimentGroup(
        name="C",
        description="B + reliability_min=0.1",
        overrides=("model.gamma_temperature=3", "model.reliability_min=0.1"),
    ),
    ExperimentGroup(
        name="D",
        description="C + gate/gamma warmup=10",
        overrides=(
            "model.gamma_temperature=3",
            "model.reliability_min=0.1",
            "model.gate_warmup_epochs=10",
            "model.gamma_warmup_epochs=10",
        ),
    ),
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run MAP-MAG A/B/C/D sports-copurchase LP experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cuda:1", help="Hydra device override.")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=[group.name for group in GROUPS],
        default=[group.name for group in GROUPS],
        help="Experiment groups to run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args, extra_overrides = parser.parse_known_args()
    if extra_overrides and extra_overrides[0] == "--":
        extra_overrides = extra_overrides[1:]
    return args, extra_overrides


def _build_command(group: ExperimentGroup, args: argparse.Namespace, extra_overrides: list[str]) -> list[str]:
    cmd = [sys.executable, "src/main.py"]
    cmd.extend(BASE_OVERRIDES)
    cmd.append(f"device={args.device}")
    cmd.extend(group.overrides)
    cmd.extend(extra_overrides)
    return cmd


def main() -> None:
    args, extra_overrides = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    selected = [group for group in GROUPS if group.name in set(args.groups)]

    print("MAP-MAG sports-copurchase LP A/B/C/D experiments", flush=True)
    print("Seeds: 42/43/44 via num_runs=3 and seed=42", flush=True)
    if extra_overrides:
        print(f"Extra Hydra overrides: {' '.join(extra_overrides)}", flush=True)

    for index, group in enumerate(selected, start=1):
        cmd = _build_command(group, args, extra_overrides)
        print(f"\n[{index}/{len(selected)}] Group {group.name}: {group.description}", flush=True)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
