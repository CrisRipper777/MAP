from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_OVERRIDES = (
    "model=map_mag",
    "task=lp",
    "dataset=sports-copurchase",
    "num_runs=1",
    "seed=42",
    "task.epochs=150",
    "task.lr=1e-5",
    "model.num_prototypes=16",
    "model.lambda_gate=0.01",
    "model.lambda_proto=0.1",
    "model.router_temperature=3",
)

FIXED_GAMMA_VALUES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run MAP-MAG fixed-gamma diagnostics on sports-copurchase LP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cuda:1", help="Hydra device override.")
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        default=list(FIXED_GAMMA_VALUES),
        help="Fixed gamma values to evaluate.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args, extra_overrides = parser.parse_known_args()
    if extra_overrides and extra_overrides[0] == "--":
        extra_overrides = extra_overrides[1:]
    return args, extra_overrides


def _build_command(fixed_gamma: float, args: argparse.Namespace, extra_overrides: list[str]) -> list[str]:
    cmd = [sys.executable, "src/main.py"]
    cmd.extend(BASE_OVERRIDES)
    cmd.append(f"device={args.device}")
    cmd.append(f"model.fixed_gamma={fixed_gamma:g}")
    cmd.extend(extra_overrides)
    return cmd


def main() -> None:
    args, extra_overrides = _parse_args()
    project_root = Path(__file__).resolve().parents[1]

    print("MAP-MAG sports-copurchase LP fixed-gamma diagnostics", flush=True)
    print("Seed: 42 via num_runs=1 and seed=42", flush=True)
    if extra_overrides:
        print(f"Extra Hydra overrides: {' '.join(extra_overrides)}", flush=True)

    for index, fixed_gamma in enumerate(args.values, start=1):
        cmd = _build_command(float(fixed_gamma), args, extra_overrides)
        print(f"\n[{index}/{len(args.values)}] fixed_gamma={fixed_gamma:g}", flush=True)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
