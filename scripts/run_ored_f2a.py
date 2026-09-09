from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DATASETS = ("ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("f1_owner", "f1_dual_direct")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed ORED-F2A NC validation grid.")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ored/f2a/formal"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/ored/f2a/checkpoints"))
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError(f"invalid shard {args.shard}/{args.shards}")

    jobs = [
        (dataset, seed, variant)
        for dataset in DATASETS
        for seed in SEEDS
        for variant in VARIANTS
    ]
    selected = [job for index, job in enumerate(jobs) if index % args.shards == args.shard]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTHONPATH", ".")

    for dataset, seed, variant in selected:
        name = f"{dataset.lower()}_seed{seed}_{variant}"
        output_dir = args.output_root / name
        checkpoint = args.checkpoint_root / f"{name}.pt"
        if output_dir.joinpath("results.json").is_file() and checkpoint.is_file():
            print(f"SKIP existing {dataset} seed={seed} variant={variant}", flush=True)
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
            "task.training_mode=full_graph",
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
        print(f"RUN {dataset} seed={seed} variant={variant} on GPU {args.gpu}", flush=True)
        subprocess.run(
            command,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL if args.quiet else None,
            stderr=subprocess.DEVNULL if args.quiet else None,
        )


if __name__ == "__main__":
    main()
