from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DATASETS = ("Movies", "Toys", "Grocery")
SEEDS = (42, 43, 44)
VARIANTS = ("f1_owner", "f1_dual_direct", "f1_dual_ocb")


def _jobs(smoke: bool) -> list[tuple[str, int, str]]:
    if smoke:
        return [("Movies", 42, variant) for variant in VARIANTS]
    return [
        (dataset, seed, variant)
        for dataset in DATASETS
        for seed in SEEDS
        for variant in VARIANTS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed ORED-F1 dual-granularity NC grid.")
    parser.add_argument("--gpu", required=True, help="Physical GPU id exposed through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="Run Movies seed42 only.")
    parser.add_argument("--quiet", action="store_true", help="Keep child training logs in each output directory only.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ored/f1/formal"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/ored/f1/checkpoints"))
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError(f"invalid shard {args.shard}/{args.shards}")

    jobs = _jobs(args.smoke)
    selected = [job for index, job in enumerate(jobs) if index % args.shards == args.shard]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTHONPATH", ".")

    stage = "smoke" if args.smoke else "formal"
    output_root = args.output_root if not args.smoke else args.output_root.parent / "smoke"
    checkpoint_root = args.checkpoint_root if not args.smoke else args.checkpoint_root.parent / "smoke_checkpoints"
    for dataset, seed, variant in selected:
        name = f"{dataset.lower()}_seed{seed}_{variant}"
        output_dir = output_root / name
        checkpoint = checkpoint_root / f"{name}.pt"
        if output_dir.joinpath("results.json").is_file() and checkpoint.is_file():
            print(f"SKIP existing {stage} {dataset} seed={seed} variant={variant}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "src.main",
            f"dataset={dataset}",
            "task=nc",
            "model=ored_mag",
            f"model.variant={variant}",
            "num_runs=1",
            f"seed={seed}",
            "device=cuda:0",
            "task.evaluate_test=false",
            "model.hidden_dim=256",
            "model.factor_dim=128",
            "model.dropout=0.2",
            "model.num_hops=2",
            "model.restart=0.15",
            "model.diffusion_add_self_loops=true",
            "model.bridge_gate_hidden_dim=128",
            "model.bridge_ownership_dim=16",
            f"task.save_ckpt_path={checkpoint}",
            f"hydra.run.dir={output_dir}",
        ]
        print(f"RUN {stage} {dataset} seed={seed} variant={variant}", flush=True)
        subprocess.run(
            command,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL if args.quiet else None,
            stderr=subprocess.DEVNULL if args.quiet else None,
        )


if __name__ == "__main__":
    main()
