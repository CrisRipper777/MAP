"""Run the complete MAP-MAG benchmark with one process per GPU.

The default plan is the main benchmark requested for this repository:

* NC: Movies, Toys, Grocery, ele-fashion, Reddit-S with seeds 42/43/44.
* LP: sports-copurchase with seed 42.
* All models listed as implemented in README.md.

Each worker owns one device and runs at most one Hydra job at a time.  This is
deliberately a small scheduler rather than a shell loop: it makes two-GPU
parallelism safe, also works with ``--devices cuda:0`` on one GPU, gives every
job a unique output directory, and can resume completed jobs.

Run from the project root, for example:

    conda run --no-capture-output -n yhf_env \
      python scripts/run_full_benchmark.py

Use ``--dry-run`` first to inspect the 78 commands.  Extra Hydra overrides can
be appended after ``--`` (or are accepted as unknown arguments), for example
``task.epochs=1`` for a smoke run.  Such overrides are intended for debugging,
not for the formal benchmark protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NC_DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
LP_DATASETS = ("sports-copurchase",)

# Keep this list aligned with the implemented-model list in README.md.  The
# v3 LP preset is selected below without presenting it as a second model.
MODEL_KEYS = (
    "mlp",
    "gcn",
    "sage",
    "mmgcn",
    "mgat",
    "dip",
    "dgf",
    "dmgc",
    "lgmrec",
    "map_mag",
    "map_mag_v1",
    "map_mag_v2",
    "map_mag_v3",
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    nc_config: str
    lp_config: str
    is_map_mag: bool = False


MODEL_SPECS = tuple(
    ModelSpec(
        key=model,
        nc_config=model,
        lp_config=model,
        is_map_mag=model.startswith("map_mag"),
    )
    for model in MODEL_KEYS
)

# v3 has a task-specific LP preset.  It still imports src.models.map_mag_v3,
# but turns on the v3 LP encoder controls while retaining the shared decoder.
MODEL_SPECS = tuple(
    ModelSpec(
        key=spec.key,
        nc_config=spec.nc_config,
        lp_config="map_mag_v3_lp" if spec.key == "map_mag_v3" else spec.lp_config,
        is_map_mag=spec.is_map_mag,
    )
    for spec in MODEL_SPECS
)


@dataclass(frozen=True)
class Job:
    index: int
    dataset: str
    task: str
    model: str
    model_config: str
    base_seed: int
    num_runs: int
    run_dir: Path


@dataclass(frozen=True)
class JobResult:
    index: int
    dataset: str
    task: str
    model: str
    device: str
    status: str
    returncode: int | None
    seconds: float
    run_dir: str
    log_path: str
    message: str = ""


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full MAP-MAG NC/LP benchmark with one process per device. "
            "Default: 78 jobs, two GPU workers."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0", "cuda:1"],
        help=(
            "Devices used as independent worker lanes. One job at a time is "
            "launched on each device; use one value for single-card mode."
        ),
    )
    parser.add_argument(
        "--only",
        choices=("all", "nc", "lp"),
        default="all",
        help="Run both tasks, only NC, or only LP.",
    )
    parser.add_argument(
        "--nc-datasets",
        nargs="+",
        choices=NC_DATASETS,
        default=list(NC_DATASETS),
        help="NC datasets in the benchmark.",
    )
    parser.add_argument(
        "--lp-datasets",
        nargs="+",
        choices=LP_DATASETS,
        default=list(LP_DATASETS),
        help="LP datasets in the benchmark.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KEYS + ("map_mag_v3_full",),
        default=list(MODEL_KEYS),
        help=(
            "Models to run. map_mag_v3_full is an optional controlled ablation; "
            "it is not part of the default main-model list."
        ),
    )
    parser.add_argument(
        "--include-v3-full",
        action="store_true",
        help="Append map_mag_v3_full to the model list as an extra ablation.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed. NC uses base_seed..base_seed+2 by default.",
    )
    parser.add_argument(
        "--nc-num-runs",
        type=int,
        default=3,
        help="Number of NC runs/seeds per dataset and model.",
    )
    parser.add_argument(
        "--lp-num-runs",
        type=int,
        default=1,
        help="Number of LP runs/seeds per dataset and model.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/full_benchmark",
        help="Root directory for Hydra runs, manifest, and summary.",
    )
    parser.add_argument(
        "--export-aux",
        action="store_true",
        help="Keep MAP-MAG node auxiliary CSV export (disabled by default).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Run jobs even when their results.json already exists.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop assigning new jobs after the first failed job.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scheduled commands without launching them.",
    )
    args, extra_overrides = parser.parse_known_args()
    if extra_overrides and extra_overrides[0] == "--":
        extra_overrides = extra_overrides[1:]
    return args, extra_overrides


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.nc_num_runs < 1 or args.lp_num_runs < 1:
        parser.error("--nc-num-runs and --lp-num-runs must be >= 1")
    if not args.devices:
        parser.error("at least one --devices value is required")
    if len(set(args.devices)) != len(args.devices):
        parser.error("duplicate devices would create unsafe same-device concurrency")
    if args.only in {"all", "nc"} and not args.nc_datasets:
        parser.error("NC was selected but --nc-datasets is empty")
    if args.only in {"all", "lp"} and not args.lp_datasets:
        parser.error("LP was selected but --lp-datasets is empty")


def _selected_models(args: argparse.Namespace) -> list[str]:
    selected = list(dict.fromkeys(args.models))
    if args.include_v3_full and "map_mag_v3_full" not in selected:
        selected.append("map_mag_v3_full")
    return selected


def _model_spec(model: str) -> ModelSpec:
    if model == "map_mag_v3_full":
        return ModelSpec(
            key=model,
            nc_config=model,
            lp_config=model,
            is_map_mag=True,
        )
    for spec in MODEL_SPECS:
        if spec.key == model:
            return spec
    raise KeyError(f"Unknown model {model!r}")


def _safe_path_component(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _build_jobs(args: argparse.Namespace) -> list[Job]:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    models = _selected_models(args)
    jobs: list[Job] = []
    index = 0

    if args.only in {"all", "nc"}:
        for dataset in args.nc_datasets:
            for model in models:
                spec = _model_spec(model)
                index += 1
                run_dir = (
                    output_root
                    / "nc"
                    / _safe_path_component(dataset)
                    / _safe_path_component(model)
                    / f"seed{args.base_seed}_runs{args.nc_num_runs}"
                )
                jobs.append(
                    Job(
                        index=index,
                        dataset=dataset,
                        task="nc",
                        model=model,
                        model_config=spec.nc_config,
                        base_seed=args.base_seed,
                        num_runs=args.nc_num_runs,
                        run_dir=run_dir,
                    )
                )

    if args.only in {"all", "lp"}:
        for dataset in args.lp_datasets:
            for model in models:
                spec = _model_spec(model)
                index += 1
                run_dir = (
                    output_root
                    / "lp"
                    / _safe_path_component(dataset)
                    / _safe_path_component(model)
                    / f"seed{args.base_seed}_runs{args.lp_num_runs}"
                )
                jobs.append(
                    Job(
                        index=index,
                        dataset=dataset,
                        task="lp",
                        model=model,
                        model_config=spec.lp_config,
                        base_seed=args.base_seed,
                        num_runs=args.lp_num_runs,
                        run_dir=run_dir,
                    )
                )
    return jobs


def _job_overrides(job: Job, args: argparse.Namespace) -> list[str]:
    overrides: list[str] = []
    if not args.export_aux and job.model.startswith("map_mag"):
        # map_mag and v1 export by default.  The key is present in the v2/v3
        # configs as well, so this is safe for every MAP-MAG config.
        overrides.append("model.export_node_aux=false")
    return overrides


def _build_command(
    job: Job,
    device: str,
    args: argparse.Namespace,
    extra_overrides: Iterable[str],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={job.dataset}",
        f"task={job.task}",
        f"model={job.model_config}",
        f"seed={job.base_seed}",
        f"num_runs={job.num_runs}",
        f"device={device}",
        f"hydra.run.dir={job.run_dir}",
    ]
    command.extend(_job_overrides(job, args))
    command.extend(extra_overrides)
    return command


def _format_command(command: Iterable[str]) -> str:
    # repr-style quoting is enough for readable logs and handles absolute paths
    # without relying on a shell.  The actual subprocess call uses argv.
    return " ".join(repr(part) if any(ch in part for ch in " \t\n") else part for part in command)


def _manifest_payload(
    jobs: list[Job],
    args: argparse.Namespace,
    extra_overrides: list[str],
) -> dict:
    return {
        "script": Path(__file__).name,
        "project_root": str(PROJECT_ROOT),
        "base_seed": args.base_seed,
        "nc_num_runs": args.nc_num_runs,
        "lp_num_runs": args.lp_num_runs,
        "devices": list(args.devices),
        "extra_overrides": list(extra_overrides),
        "jobs": [
            {
                "index": job.index,
                "dataset": job.dataset,
                "task": job.task,
                "model": job.model,
                "model_config": job.model_config,
                "base_seed": job.base_seed,
                "num_runs": job.num_runs,
                "run_dir": str(job.run_dir),
            }
            for job in jobs
        ],
    }


def _run_one_job(
    job: Job,
    device: str,
    args: argparse.Namespace,
    extra_overrides: list[str],
    print_lock: threading.Lock,
) -> JobResult:
    start = time.monotonic()
    log_path = job.run_dir / "launcher.log"
    result_path = job.run_dir / "results.json"

    if not args.no_resume and result_path.is_file():
        with print_lock:
            print(
                f"[skip] [{job.index}] {job.task}/{job.dataset}/{job.model} "
                f"already has {result_path}",
                flush=True,
            )
        return JobResult(
            index=job.index,
            dataset=job.dataset,
            task=job.task,
            model=job.model,
            device=device,
            status="skipped",
            returncode=0,
            seconds=0.0,
            run_dir=str(job.run_dir),
            log_path=str(log_path),
            message="results.json already exists",
        )

    job.run_dir.mkdir(parents=True, exist_ok=True)
    command = _build_command(job, device, args, extra_overrides)
    with print_lock:
        print(
            f"[start] [{job.index}] {job.task}/{job.dataset}/{job.model} on {device}",
            flush=True,
        )
        print(f"        {_format_command(command)}", flush=True)

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    returncode: int | None = None
    message = ""
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        returncode = int(completed.returncode)
        if returncode == 0:
            status = "completed"
        else:
            status = "failed"
            message = (
                "child process failed; inspect launcher.log and main.log"
            )
    except OSError as exc:
        status = "failed"
        message = f"could not launch child process: {exc}"

    seconds = time.monotonic() - start
    with print_lock:
        marker = "done" if status == "completed" else "FAIL"
        print(
            f"[{marker}] [{job.index}] {job.task}/{job.dataset}/{job.model} "
            f"on {device} ({seconds / 60.0:.1f} min)",
            flush=True,
        )
        if message:
            print(f"       {message}: {log_path}", flush=True)

    return JobResult(
        index=job.index,
        dataset=job.dataset,
        task=job.task,
        model=job.model,
        device=device,
        status=status,
        returncode=returncode,
        seconds=seconds,
        run_dir=str(job.run_dir),
        log_path=str(log_path),
        message=message,
    )


def _run_parallel(
    jobs: list[Job],
    args: argparse.Namespace,
    extra_overrides: list[str],
) -> list[JobResult]:
    pending: queue.Queue[Job | None] = queue.Queue()
    for job in jobs:
        pending.put(job)
    for _ in args.devices:
        pending.put(None)

    print_lock = threading.Lock()
    stop_assigning = threading.Event()
    results: list[JobResult] = []
    results_lock = threading.Lock()

    def worker(device: str) -> None:
        while not stop_assigning.is_set():
            job = pending.get()
            if job is None:
                return
            result = _run_one_job(job, device, args, extra_overrides, print_lock)
            with results_lock:
                results.append(result)
            if args.fail_fast and result.status == "failed":
                stop_assigning.set()
                return

    threads = [
        threading.Thread(target=worker, args=(device,), name=f"worker-{device}")
        for device in args.devices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Jobs left in the queue are explicitly reported when --fail-fast is used.
    if stop_assigning.is_set():
        assigned = {result.index for result in results}
        for job in jobs:
            if job.index not in assigned:
                results.append(
                    JobResult(
                        index=job.index,
                        dataset=job.dataset,
                        task=job.task,
                        model=job.model,
                        device="",
                        status="not_started",
                        returncode=None,
                        seconds=0.0,
                        run_dir=str(job.run_dir),
                        log_path=str(job.run_dir / "launcher.log"),
                        message="not started because --fail-fast stopped scheduling",
                    )
                )
    return sorted(results, key=lambda result: result.index)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    # Parse through the real parser once so parser.error has the expected CLI
    # behavior without duplicating the complete argument definition here.
    args, extra_overrides = _parse_args()
    _validate_args(args, parser)

    jobs = _build_jobs(args)
    if not jobs:
        parser.error("no jobs selected")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Benchmark plan: {len(jobs)} jobs | devices={','.join(args.devices)} | "
        f"NC runs={args.nc_num_runs} | LP runs={args.lp_num_runs}",
        flush=True,
    )
    if extra_overrides:
        print(f"Extra Hydra overrides: {' '.join(extra_overrides)}", flush=True)

    _write_json(
        output_root / "benchmark_manifest.json",
        _manifest_payload(jobs, args, extra_overrides),
    )

    if args.dry_run:
        for offset, job in enumerate(jobs):
            device = args.devices[offset % len(args.devices)]
            command = _build_command(job, device, args, extra_overrides)
            print(
                f"[{job.index}/{len(jobs)}] {job.task}/{job.dataset}/{job.model} "
                f"on {device}: {_format_command(command)}",
                flush=True,
            )
        print(f"Manifest: {output_root / 'benchmark_manifest.json'}", flush=True)
        return 0

    results = _run_parallel(jobs, args, extra_overrides)
    summary = {
        "total": len(jobs),
        "completed": sum(result.status == "completed" for result in results),
        "skipped": sum(result.status == "skipped" for result in results),
        "failed": sum(result.status == "failed" for result in results),
        "not_started": sum(result.status == "not_started" for result in results),
        "results": [asdict(result) for result in results],
    }
    _write_json(output_root / "benchmark_summary.json", summary)

    print(
        "Benchmark finished: "
        f"completed={summary['completed']} skipped={summary['skipped']} "
        f"failed={summary['failed']} not_started={summary['not_started']}",
        flush=True,
    )
    print(f"Summary: {output_root / 'benchmark_summary.json'}", flush=True)
    return 1 if summary["failed"] or summary["not_started"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
