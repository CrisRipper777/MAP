from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DATASETS = ("Movies", "Toys", "Grocery")
SEEDS = (42, 43, 44)
VARIANTS = ("p0", "p0_refine", "joint_rd", "ownership_rd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed ORED-2 paired NC replication grid.")
    parser.add_argument("--gpu", required=True, help="Physical GPU id exposed through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ored/o2/formal"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/ored/o2/checkpoints"))
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
            f"task.save_ckpt_path={checkpoint}",
            f"hydra.run.dir={output_dir}",
        ]
        print("RUN", dataset, seed, variant, flush=True)
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
