"""MAP-MAG v2 mechanism experiments and diagnostics.

This is intentionally an external analysis runner.  It does not modify the
MAP source code or the checked-in Hydra configurations.  It can launch the
existing NC trainer on one job per device, save best-validation checkpoints,
and run post-hoc diagnostics on the learned projected features and semantic
edge weights.

Examples (from the MAP/MAP project root)::

    # Inspect the five NC graphs without training.
    python scripts/map_v2_analyze.py audit

    # Five core v2 decompositions, three paired runs, one job per GPU.
    python scripts/map_v2_analyze.py run --devices cuda:0 cuda:1 \
        --experiments core_no_graph lowpass_uniform reliability_uniform semantic_uniform core

    # Two concurrent jobs on one GPU (use only when memory permits).
    python scripts/map_v2_analyze.py run --all-experiments \
        --devices cuda:1 --parallel-per-device 2 \
        --exclusive-datasets ele-fashion

    # Analyze checkpoints produced above (edge/gate/representation metrics).
    python scripts/map_v2_analyze.py diagnose --fit-probes

    # Do the audit, training and diagnostics in one invocation.
    python scripts/map_v2_analyze.py all --devices cuda:0 cuda:1

Extra Hydra overrides may be appended after ``--`` to ``run``/``all``, for
example ``-- task.epochs=3 task.patience=3`` for a smoke run.  Smoke results
must not be used as performance results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from torch_geometric.utils import scatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "map_v2_analysis"
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    purpose: str
    overrides: tuple[str, ...]


# These are the minimum useful v2 decomposition.  All settings use the same
# MAPMAGV2 class; only the computation path changes.
EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "core_no_graph",
        "no graph/no reliability",
        "attribute-only reference inside the v2 scaffold",
        (
            "model.num_hops=0",
            "model.use_reliability=false",
            "model.use_semantic_edge_weight=false",
        ),
    ),
    Experiment(
        "lowpass_uniform",
        "uniform graph low-pass",
        "isolates restart low-pass diffusion on the original graph",
        (
            "model.use_reliability=false",
            "model.use_semantic_edge_weight=false",
        ),
    ),
    Experiment(
        "reliability_uniform",
        "reliability + uniform graph",
        "isolates reliability fusion from semantic edge weights",
        ("model.use_semantic_edge_weight=false",),
    ),
    Experiment(
        "semantic_uniform",
        "semantic graph + uniform fusion",
        "isolates semantic edge weighting with equal modality fusion",
        ("model.use_reliability=false",),
    ),
    Experiment(
        "core",
        "v2 core",
        "single reliability + shared semantic graph + low-pass backbone",
        (),
    ),
    Experiment(
        "double_reliability",
        "double reliability",
        "tests whether v1-style second reliability multiplication helps",
        ("model.reliability_mode=double",),
    ),
    Experiment(
        "edge_text_only",
        "text-only semantic edges",
        "tests whether text controls the useful graph signal",
        ("model.edge_weight_mode=text_only",),
    ),
    Experiment(
        "edge_visual_only",
        "visual-only semantic edges",
        "tests whether visual controls the useful graph signal",
        ("model.edge_weight_mode=visual_only",),
    ),
    Experiment(
        "edge_reliability_aware",
        "reliability-aware edges",
        "tests endpoint-reliability modulation of semantic edges",
        ("model.edge_weight_mode=reliability_aware",),
    ),
    Experiment(
        "lowpass_residual",
        "low-pass + high-pass residual",
        "tests the optional bounded high-frequency correction",
        ("model.structure_mode=lowpass_residual",),
    ),
    Experiment(
        "original_v1_gamma",
        "v1 gamma diagnostic",
        "tests symmetric low/high frequency mixing as a diagnostic only",
        ("model.structure_mode=original_v1_gamma",),
    ),
    Experiment(
        "self_residual",
        "low-pass + self residual",
        "tests whether attribute preservation is useful after propagation",
        ("model.use_self_residual=true",),
    ),
    Experiment(
        "prototype_residual",
        "low-pass + shared prototype",
        "tests the optional shared prototype residual (without changing loss)",
        ("model.use_prototype_path=true",),
    ),
)

EXPERIMENT_BY_KEY = {item.key: item for item in EXPERIMENTS}
DEFAULT_EXPERIMENT_KEYS = (
    "core_no_graph",
    "lowpass_uniform",
    "reliability_uniform",
    "semantic_uniform",
    "core",
)


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _normalise_device(value: str) -> str:
    value = str(value).strip()
    if value.isdigit():
        return f"cuda:{value}"
    return value


def _selected_experiments(values: list[str] | None, all_experiments: bool) -> list[Experiment]:
    if all_experiments or values == ["all"]:
        return list(EXPERIMENTS)
    keys = values or list(DEFAULT_EXPERIMENT_KEYS)
    unknown = [key for key in keys if key not in EXPERIMENT_BY_KEY]
    if unknown:
        raise ValueError(f"Unknown experiment(s): {', '.join(unknown)}")
    return [EXPERIMENT_BY_KEY[key] for key in dict.fromkeys(keys)]


def _compose_cfg(dataset: str, experiment: Experiment | None = None, seed: int = 42):
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=map_mag_v2",
        f"seed={seed}",
    ]
    if experiment is not None:
        overrides.extend(experiment.overrides)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _load_dataset(dataset: str, seed: int = 42):
    # Imported lazily so --help and --dry-run remain usable without loading
    # DGL/PyG data files.
    from src.data import load_mag_data

    cfg = _compose_cfg(dataset, seed=seed)
    return cfg, load_mag_data(cfg, "nc", seed)


def _edge_cos(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return h.new_empty((0,))
    src, dst = edge_index
    return F.cosine_similarity(h[src], h[dst], dim=-1, eps=1e-8).nan_to_num(0.0).clamp(-1, 1)


def _binary_auc(scores: torch.Tensor, positives: torch.Tensor) -> float:
    scores = scores.detach().float().cpu()
    positives = positives.detach().bool().cpu()
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    rank_sum = ranks[positives].sum().item()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=-1).detach().cpu().long()
    labels = labels.detach().cpu().long()
    valid = (labels >= 0) & (labels < num_classes)
    pred, labels = pred[valid], labels[valid]
    if labels.numel() == 0:
        return float("nan")
    cm = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    cm.index_put_((labels, pred), torch.ones_like(labels, dtype=torch.float64), accumulate=True)
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp_min(1.0)
    recall = tp / cm.sum(1).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    present = cm.sum(1) > 0
    return float(f1[present].mean())


def _classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, index: torch.Tensor, num_classes: int
) -> dict[str, float]:
    index = index.to(logits.device)
    selected = labels.to(logits.device)[index]
    pred_logits = logits[index]
    valid = selected >= 0
    if not bool(valid.any()):
        return {"acc": float("nan"), "macro_f1": float("nan")}
    pred_logits, selected = pred_logits[valid], selected[valid]
    return {
        "acc": float((pred_logits.argmax(-1) == selected).float().mean().item()),
        "macro_f1": _macro_f1(pred_logits, selected, num_classes),
    }


def _feature_effective_rank(x: torch.Tensor, max_samples: int = 5000) -> float:
    if x.size(0) > max_samples:
        generator = torch.Generator(device=x.device).manual_seed(20260908)
        ids = torch.randperm(x.size(0), generator=generator, device=x.device)[:max_samples]
        x = x[ids]
    x = x.float() - x.float().mean(0, keepdim=True)
    # The covariance is small (hidden_dim x hidden_dim), unlike an N x N SVD.
    cov = x.t().matmul(x) / max(x.size(0) - 1, 1)
    values = torch.linalg.eigvalsh(cov).clamp_min(0)
    probs = values / values.sum().clamp_min(1e-12)
    return float(torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum()).item())


def _dirichlet_energy(z: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None) -> float:
    if edge_index.numel() == 0:
        return 0.0
    src, dst = edge_index
    delta = (z[src] - z[dst]).pow(2).sum(-1)
    if edge_weight is not None and edge_weight.numel() == delta.numel():
        delta = delta * edge_weight
    return float(delta.mean().item())


def _weighted_message_stats(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    labels: torch.Tensor,
    num_nodes: int,
) -> dict[str, float]:
    if edge_index.numel() == 0:
        return {"weighted_homophily": float("nan"), "harmful_message_mass": float("nan"), "mean_incoming_entropy": 0.0}
    src, dst = edge_index
    labels = labels.to(edge_index.device)
    valid = (labels[src] >= 0) & (labels[dst] >= 0)
    same = (labels[src] == labels[dst]) & valid
    w = edge_weight.float().clamp_min(0)
    weighted_hom = float((w[same].sum() / w[valid].sum().clamp_min(1e-12)).item())
    total = scatter(w, dst, dim=0, dim_size=num_nodes, reduce="sum")
    harmful = scatter(w * (valid & ~same).float(), dst, dim=0, dim_size=num_nodes, reduce="sum")
    nonzero = total > 0
    harmful_mass = float((harmful[nonzero] / total[nonzero].clamp_min(1e-12)).mean().item())
    p = w / scatter(w, dst, dim=0, dim_size=num_nodes, reduce="sum")[dst].clamp_min(1e-12)
    entropy = scatter(-p * p.clamp_min(1e-12).log(), dst, dim=0, dim_size=num_nodes, reduce="sum")
    entropy = entropy[nonzero]
    return {
        "weighted_homophily": weighted_hom,
        "harmful_message_mass": harmful_mass,
        "mean_incoming_entropy": float(entropy.mean().item()) if entropy.numel() else 0.0,
    }


def _raw_audit(dataset: str, seed: int = 42, edge_sample: int = 200_000) -> dict[str, Any]:
    _, data = _load_dataset(dataset, seed)
    edge_index = data.edge_index.cpu()
    labels = data.y.cpu()
    src, dst = edge_index
    valid = (labels[src] >= 0) & (labels[dst] >= 0)
    same = (labels[src] == labels[dst]) & valid
    generator = torch.Generator().manual_seed(20260908)
    if edge_index.size(1) > edge_sample:
        ids = torch.randperm(edge_index.size(1), generator=generator)[:edge_sample]
    else:
        ids = torch.arange(edge_index.size(1))
    sampled_src, sampled_dst = src[ids], dst[ids]
    x_t, x_v = data.x_t.float(), data.x_i.float()
    cos_t = _edge_cos(x_t, torch.stack([sampled_src, sampled_dst]))
    cos_v = _edge_cos(x_v, torch.stack([sampled_src, sampled_dst]))
    sampled_valid = (labels[sampled_src] >= 0) & (labels[sampled_dst] >= 0)
    sampled_same = (labels[sampled_src] == labels[sampled_dst]) & sampled_valid
    sampled_diff = sampled_valid & ~sampled_same
    degree = torch.bincount(dst, minlength=data.num_nodes).float()
    result: dict[str, Any] = {
        "dataset": dataset,
        "num_nodes": int(data.num_nodes),
        "num_edges_directed": int(edge_index.size(1)),
        "text_dim": int(x_t.size(1)),
        "visual_dim": int(x_v.size(1)),
        "num_classes": int(data.num_classes),
        "labelled_nodes": int((labels >= 0).sum()),
        "degree_mean": float(degree.mean()),
        "degree_median": float(degree.median()),
        "isolated_fraction": float((degree == 0).float().mean()),
        "edge_homophily": float(same[valid].float().mean()) if bool(valid.any()) else float("nan"),
        "sampled_edges": int(ids.numel()),
        "raw_text_edge_cos_mean": float(cos_t.mean()),
        "raw_visual_edge_cos_mean": float(cos_v.mean()),
        "raw_text_same_minus_diff": float(cos_t[sampled_same].mean() - cos_t[sampled_diff].mean()) if bool(sampled_same.any() and sampled_diff.any()) else float("nan"),
        "raw_visual_same_minus_diff": float(cos_v[sampled_same].mean() - cos_v[sampled_diff].mean()) if bool(sampled_same.any() and sampled_diff.any()) else float("nan"),
        "raw_text_edge_auc": _binary_auc(cos_t[sampled_valid], sampled_same[sampled_valid]),
        "raw_visual_edge_auc": _binary_auc(cos_v[sampled_valid], sampled_same[sampled_valid]),
    }
    return result


def run_audit(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = [_raw_audit(dataset, args.base_seed, args.edge_sample) for dataset in args.datasets]
    _json_write(output_root / "dataset_audit.json", {"seed": args.base_seed, "datasets": results})
    print(f"Wrote {output_root / 'dataset_audit.json'}", flush=True)
    for item in results:
        print(
            f"{item['dataset']:12s} N={item['num_nodes']:7d} E={item['num_edges_directed']:7d} "
            f"hom={item['edge_homophily']:.3f} Δcos(text/visual)="
            f"{item['raw_text_same_minus_diff']:.4f}/{item['raw_visual_same_minus_diff']:.4f}",
            flush=True,
        )
    return 0


@dataclass(frozen=True)
class Job:
    index: int
    dataset: str
    experiment: Experiment
    base_seed: int
    num_runs: int
    run_dir: Path


@dataclass(frozen=True)
class JobResult:
    index: int
    dataset: str
    experiment: str
    device: str
    status: str
    returncode: int | None
    seconds: float
    run_dir: str
    log_path: str
    message: str = ""


def _build_jobs(args: argparse.Namespace, experiments: list[Experiment]) -> list[Job]:
    root = Path(args.output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    jobs: list[Job] = []
    index = 0
    for dataset in args.datasets:
        for experiment in experiments:
            index += 1
            jobs.append(
                Job(
                    index=index,
                    dataset=dataset,
                    experiment=experiment,
                    base_seed=args.base_seed,
                    num_runs=args.num_runs,
                    run_dir=root / "nc" / dataset / experiment.key / f"seed{args.base_seed}_runs{args.num_runs}",
                )
            )
    return jobs


def _command(job: Job, device: str, extra_overrides: list[str], args: argparse.Namespace) -> list[str]:
    checkpoint = job.run_dir / "checkpoint.pt"
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={job.dataset}",
        "task=nc",
        "model=map_mag_v2",
        f"seed={job.base_seed}",
        f"num_runs={job.num_runs}",
        f"device={device}",
        f"hydra.run.dir={job.run_dir}",
        f"task.save_ckpt_path={checkpoint}",
        "model.export_node_aux=false",
        "model.export_aux_stats=false",
    ]
    if args.validation_only:
        command.append("task.evaluate_test=false")
    command.extend(job.experiment.overrides)
    command.extend(extra_overrides)
    return command


def _fmt_command(command: Iterable[str]) -> str:
    return " ".join(repr(part) if any(char in part for char in " \t\n") else part for part in command)


def _run_one(job: Job, device: str, extra_overrides: list[str], args: argparse.Namespace, lock: threading.Lock) -> JobResult:
    start = time.monotonic()
    log_path = job.run_dir / "launcher.log"
    result_path = job.run_dir / "results.json"
    if not args.no_resume and result_path.is_file():
        with lock:
            print(f"[skip] {job.dataset}/{job.experiment.key}: {result_path}", flush=True)
        return JobResult(job.index, job.dataset, job.experiment.key, device, "skipped", 0, 0.0, str(job.run_dir), str(log_path), "results.json exists")
    job.run_dir.mkdir(parents=True, exist_ok=True)
    command = _command(job, device, extra_overrides, args)
    with lock:
        print(f"[start] [{job.index}] {job.dataset}/{job.experiment.key} on {device}", flush=True)
        if args.dry_run:
            print(f"        {_fmt_command(command)}", flush=True)
    if args.dry_run:
        return JobResult(job.index, job.dataset, job.experiment.key, device, "dry_run", 0, 0.0, str(job.run_dir), str(log_path))
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
        code = int(completed.returncode)
        status = "completed" if code == 0 else "failed"
        message = "" if code == 0 else "inspect launcher.log/main.log"
    except OSError as exc:
        code, status, message = None, "failed", str(exc)
    seconds = time.monotonic() - start
    with lock:
        print(f"[{'done' if status == 'completed' else 'FAIL'}] [{job.index}] {job.dataset}/{job.experiment.key} ({seconds / 60:.1f} min)", flush=True)
    return JobResult(job.index, job.dataset, job.experiment.key, device, status, code, seconds, str(job.run_dir), str(log_path), message)


def _run_parallel(jobs: list[Job], args: argparse.Namespace, extra_overrides: list[str]) -> list[JobResult]:
    base_devices = [_normalise_device(value) for value in args.devices]
    slots = int(getattr(args, "parallel_per_device", 1))
    if slots < 1:
        raise ValueError("parallel_per_device must be >= 1")
    exclusive_datasets = {str(item) for item in getattr(args, "exclusive_datasets", [])}

    # A condition-based scheduler is needed when several workers share one
    # card.  An exclusive dataset (ele-fashion by default) is only assigned
    # when the card is completely idle; ordinary jobs can fill the other slots
    # concurrently.  This also prevents an ele-fashion job from starting
    # beside another ele-fashion job or beside any ordinary job.
    pending = list(jobs)
    condition = threading.Condition()
    print_lock = threading.Lock()
    results: list[JobResult] = []
    stop = threading.Event()
    active: dict[str, int] = {device: 0 for device in base_devices}
    exclusive_active: dict[str, bool] = {device: False for device in base_devices}

    def acquire_job(device: str) -> Job | None:
        with condition:
            while True:
                if stop.is_set() and args.fail_fast:
                    return None
                candidate: int | None = None
                # Give an exclusive job priority whenever this card is idle;
                # otherwise it could wait behind a long ordinary queue.
                if active[device] == 0 and not exclusive_active[device]:
                    for index, job in enumerate(pending):
                        if job.dataset in exclusive_datasets:
                            candidate = index
                            break
                if candidate is None and not exclusive_active[device] and active[device] < slots:
                    for index, job in enumerate(pending):
                        if job.dataset not in exclusive_datasets:
                            candidate = index
                            break
                if candidate is not None:
                    job = pending.pop(candidate)
                    active[device] += 1
                    if job.dataset in exclusive_datasets:
                        exclusive_active[device] = True
                    return job
                if not pending:
                    return None
                condition.wait()

    def release_job(device: str, job: Job) -> None:
        with condition:
            active[device] -= 1
            if job.dataset in exclusive_datasets:
                exclusive_active[device] = False
            condition.notify_all()

    def worker(device: str, slot: int) -> None:
        while True:
            job = acquire_job(device)
            if job is None:
                return
            result = _run_one(job, device, extra_overrides, args, print_lock)
            release_job(device, job)
            with condition:
                results.append(result)
                condition.notify_all()
            if args.fail_fast and result.status == "failed":
                stop.set()
                with condition:
                    condition.notify_all()
                return

    threads = [
        threading.Thread(target=worker, args=(device, slot), name=f"map-v2-{device}-slot{slot}")
        for device in base_devices
        for slot in range(1, slots + 1)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if stop.is_set():
        assigned = {item.index for item in results}
        for job in jobs:
            if job.index not in assigned:
                results.append(JobResult(job.index, job.dataset, job.experiment.key, "", "not_started", None, 0.0, str(job.run_dir), str(job.run_dir / "launcher.log"), "fail-fast"))
    return sorted(results, key=lambda item: item.index)


def run_training(args: argparse.Namespace, extra_overrides: list[str]) -> int:
    experiments = _selected_experiments(args.experiments, args.all_experiments)
    jobs = _build_jobs(args, experiments)
    root = Path(args.output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": Path(__file__).name,
        "mode": "nc",
        "datasets": list(args.datasets),
        "base_seed": args.base_seed,
        "num_runs": args.num_runs,
        "devices": [_normalise_device(item) for item in args.devices],
        "parallel_per_device": int(args.parallel_per_device),
        "exclusive_datasets": list(args.exclusive_datasets),
        "extra_overrides": extra_overrides,
        "experiments": [
            {
                "key": item.experiment.key,
                "label": item.experiment.label,
                "purpose": item.experiment.purpose,
                "overrides": list(item.experiment.overrides),
                "extra_overrides": list(extra_overrides),
                "dataset": item.dataset,
                "run_dir": str(item.run_dir),
            }
            for item in jobs
        ],
    }
    _json_write(root / "manifest.json", manifest)
    print(f"Training plan: {len(jobs)} jobs, {len(experiments)} experiment(s) x {len(args.datasets)} dataset(s), devices={','.join(manifest['devices'])}", flush=True)
    if args.dry_run:
        _run_parallel(jobs, args, extra_overrides)
        return 0
    results = _run_parallel(jobs, args, extra_overrides)
    _json_write(root / "training_summary.json", {"results": [asdict(item) for item in results]})
    failed = sum(item.status in {"failed", "not_started"} for item in results)
    print(f"Training finished: completed={sum(item.status == 'completed' for item in results)} skipped={sum(item.status == 'skipped' for item in results)} failed={failed}", flush=True)
    return 1 if failed else 0


def _checkpoint_entries(root: Path) -> list[tuple[dict[str, Any], Path, int]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}; run the training subcommand first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[tuple[dict[str, Any], Path, int]] = []
    for item in manifest["experiments"]:
        run_dir = Path(item["run_dir"])
        for checkpoint in sorted(run_dir.glob("checkpoint_run*.pt")):
            stem = checkpoint.stem
            run_id = int(stem.split("_run")[-1])
            entries.append((item, checkpoint, run_id))
        # num_runs=1 is still allowed to save checkpoint.pt in older runs.
        plain = run_dir / "checkpoint.pt"
        if plain.is_file() and not list(run_dir.glob("checkpoint_run*.pt")):
            entries.append((item, plain, 1))
    return entries


def _load_checkpoint_model(
    dataset: str,
    experiment: Experiment,
    checkpoint: Path,
    device: torch.device,
    base_seed: int,
    extra_overrides: list[str] | None = None,
):
    from src.models.factory import build_model

    cfg = _compose_cfg(dataset, experiment, base_seed)
    if extra_overrides:
        # Re-compose with the exact model/task overrides used by the training
        # launcher.  This matters for smoke runs that change hidden_dim or
        # num_hops; otherwise strict checkpoint loading should fail loudly.
        overrides = [
            f"dataset={dataset}",
            "task=nc",
            "model=map_mag_v2",
            f"seed={base_seed}",
            *experiment.overrides,
            *extra_overrides,
        ]
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="config", overrides=overrides)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data_info = payload["data_info"]
    model = build_model(cfg, data_info).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    head = nn.Linear(int(model.out_dim), int(data_info["num_classes"]))
    head.load_state_dict(payload["head_state"], strict=True)
    head.to(device).eval()
    return cfg, model, head, int(payload.get("seed", base_seed))


def _node_split(data, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "train": data.train_idx.to(device),
        "val": data.val_idx.to(device),
        "test": data.test_idx.to(device),
    }


def _counterfactuals(model, components: dict[str, Any], edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return inference-only v2 counterfactual structural representations.

    These are deliberately post-hoc interventions.  They estimate reliance on
    a mechanism and are not substitutes for retraining the corresponding
    ablation.
    """
    h0 = components["h0"]
    weights = components["edge_weight"]
    cos_t = components["cos_t_edges"]
    cos_v = components["cos_v_edges"]
    def as_struct(z_low: torch.Tensor) -> torch.Tensor:
        # Reapply optional v2 residuals so an intervention remains comparable
        # for --all-experiments.  The gates are held fixed at their learned
        # values; this is intentionally an inference-only sensitivity test.
        if model.structure_mode == "lowpass_residual":
            z_struct = z_low + components["eta"] * model.residual_mlp(h0 - z_low)
        elif model.structure_mode == "original_v1_gamma":
            z_res = h0 - z_low
            z_struct = components["gamma"] * z_low + (1.0 - components["gamma"]) * z_res
        else:
            z_struct = z_low
        if model.use_prototype_path:
            z_struct = z_struct + model.prototype_residual_max * components["z_proto"]
        if model.use_self_residual:
            z_struct = z_struct + components["alpha_self"] * components["z_self"]
        return z_struct

    out = {
        "no_graph": as_struct(h0),
        "uniform_edges": as_struct(model._diffuse(h0, edge_index, torch.ones_like(weights))),
    }
    if weights.numel():
        generator = torch.Generator(device=weights.device).manual_seed(20260908)
        out["shuffled_edges"] = as_struct(model._diffuse(h0, edge_index, weights[torch.randperm(weights.numel(), generator=generator, device=weights.device)]))
        out["inverted_edges"] = as_struct(model._diffuse(h0, edge_index, (model.edge_weight_min + 1.0 - weights).clamp(model.edge_weight_min, 1.0)))
        out["text_only_edges"] = as_struct(model._diffuse(h0, edge_index, model._edge_weight_from_similarity(cos_t)))
        out["visual_only_edges"] = as_struct(model._diffuse(h0, edge_index, model._edge_weight_from_similarity(cos_v)))
    else:
        empty = weights
        for key in ("shuffled_edges", "inverted_edges", "text_only_edges", "visual_only_edges"):
            out[key] = as_struct(model._diffuse(h0, edge_index, empty))
    r_t, r_v = components["r_t"], components["r_v"]
    uniform_h0 = 0.5 * (components["h_t"] + components["h_v"])
    out["uniform_modality_fusion"] = as_struct(model._diffuse(uniform_h0, edge_index, weights))
    double_h0, _, _ = model._reliability_fusion(components["h_t"], components["h_v"], r_t, r_v)
    out["double_reliability_input"] = as_struct(model._diffuse(double_h0, edge_index, weights))
    return out


def _fit_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
) -> dict[str, float]:
    """Small post-hoc linear probe; labels are used only for diagnosis."""
    # diagnose_one is no-grad by design; explicitly re-enable autograd for the
    # independent probe, whose parameters must not affect the MAP checkpoint.
    with torch.enable_grad():
        x = F.normalize(features.float(), dim=-1).detach()
        y = labels.long()
        probe = nn.Linear(x.size(1), num_classes).to(x.device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(x[train_idx]), y[train_idx])
            loss.backward()
            optimizer.step()
        logits = probe(x).detach()
    result: dict[str, float] = {}
    for name, ids in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        metrics = _classification_metrics(logits, y, ids, num_classes)
        result[f"{name}_acc"] = metrics["acc"]
        result[f"{name}_macro_f1"] = metrics["macro_f1"]
    return result


def _write_node_csv(path: Path, data, components: dict[str, Any], logits: torch.Tensor, labels: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_nodes = int(data.num_nodes)
    pred = logits.argmax(-1).detach().cpu()
    labels_cpu = labels.detach().cpu()
    split = ["none"] * num_nodes
    for name, idx in (("train", data.train_idx), ("val", data.val_idx), ("test", data.test_idx)):
        for node in idx.cpu().tolist():
            split[int(node)] = name
    fields = [
        "node_id", "label", "split", "pred", "correct", "degree", "s_text", "s_visual",
        "r_text", "r_visual", "lambda_text", "lambda_visual", "delta_z_low", "delta_z_final",
    ]
    h0, zlow, zfinal = components["h0"], components["z_low"], logits.new_zeros((num_nodes, logits.size(1)))
    # The caller stores final embeddings in components to keep this writer
    # independent of model internals.
    zfinal = components["z_final"]
    delta_low = (zlow - h0).norm(dim=-1) / h0.norm(dim=-1).clamp_min(1e-8)
    delta_final = (zfinal - h0).norm(dim=-1) / h0.norm(dim=-1).clamp_min(1e-8)
    columns = {
        "node_id": range(num_nodes),
        "label": labels_cpu.tolist(),
        "split": split,
        "pred": pred.tolist(),
        "correct": ((pred == labels_cpu) & (labels_cpu >= 0)).int().tolist(),
        "degree": components["degree"].detach().cpu().tolist(),
        "s_text": components["s_t"].squeeze(-1).detach().cpu().tolist(),
        "s_visual": components["s_v"].squeeze(-1).detach().cpu().tolist(),
        "r_text": components["r_t"].squeeze(-1).detach().cpu().tolist(),
        "r_visual": components["r_v"].squeeze(-1).detach().cpu().tolist(),
        "lambda_text": components["lambda_t"].squeeze(-1).detach().cpu().tolist(),
        "lambda_visual": components["lambda_v"].squeeze(-1).detach().cpu().tolist(),
        "delta_z_low": delta_low.detach().cpu().tolist(),
        "delta_z_final": delta_final.detach().cpu().tolist(),
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for i in range(num_nodes):
            writer.writerow([columns[field][i] for field in fields])


@torch.no_grad()
def diagnose_one(
    item: dict[str, Any], checkpoint: Path, run_id: int, args: argparse.Namespace,
    data_cache: dict[str, Any], output_root: Path,
) -> dict[str, Any]:
    dataset = str(item["dataset"])
    experiment = EXPERIMENT_BY_KEY[str(item["key"])]
    if dataset not in data_cache:
        _, data_cache[dataset] = _load_dataset(dataset, args.base_seed)
    data = data_cache[dataset]
    device = torch.device(args.diagnose_device)
    _, model, head, actual_seed = _load_checkpoint_model(
        dataset,
        experiment,
        checkpoint,
        device,
        args.base_seed,
        list(item.get("extra_overrides", [])),
    )
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    labels = data.y.to(device)
    components = model._encode_components(x, edge_index)
    z_final = model.output_norm(model.output_mlp(components["z_input"]) + components["z_struct"])
    z_final = torch.nan_to_num(z_final)
    components["z_final"] = z_final
    logits = head(z_final)
    splits = _node_split(data, device)
    metrics: dict[str, Any] = {
        "dataset": dataset,
        "experiment": experiment.key,
        "checkpoint": str(checkpoint),
        "run_id": run_id,
        "seed": actual_seed,
        "num_nodes": int(data.num_nodes),
        "num_edges_directed": int(edge_index.size(1)),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) + sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)),
    }
    for name, idx in splits.items():
        split_metrics = _classification_metrics(logits, labels, idx, int(data.num_classes))
        metrics[f"{name}_acc"] = split_metrics["acc"]
        metrics[f"{name}_macro_f1"] = split_metrics["macro_f1"]
    edges = components["edge_weight"]
    cos_t, cos_v = components["cos_t_edges"], components["cos_v_edges"]
    same = (labels[edge_index[0]] == labels[edge_index[1]]) & (labels[edge_index[0]] >= 0) & (labels[edge_index[1]] >= 0)
    valid = (labels[edge_index[0]] >= 0) & (labels[edge_index[1]] >= 0)
    metrics.update({
        "mean_r_text": float(components["r_t"].mean()),
        "mean_r_visual": float(components["r_v"].mean()),
        "mean_lambda_text": float(components["lambda_t"].mean()),
        "mean_lambda_visual": float(components["lambda_v"].mean()),
        "mean_edge_weight": float(edges.mean()) if edges.numel() else 0.0,
        "std_edge_weight": float(edges.std(unbiased=False)) if edges.numel() else 0.0,
        "edge_weight_cv": float(edges.std(unbiased=False) / edges.mean().clamp_min(1e-8)) if edges.numel() else 0.0,
        "edge_weight_same_minus_diff": float(edges[same].mean() - edges[valid & ~same].mean()) if bool(same.any() and (valid & ~same).any()) else float("nan"),
        "edge_weight_auc_same_vs_diff": _binary_auc(edges[valid], same[valid]),
        "cos_text_auc_same_vs_diff": _binary_auc(cos_t[valid], same[valid]),
        "cos_visual_auc_same_vs_diff": _binary_auc(cos_v[valid], same[valid]),
        "dirichlet_h0": _dirichlet_energy(components["h0"], edge_index, None),
        "dirichlet_z_low": _dirichlet_energy(components["z_low"], edge_index, edges),
        "effective_rank_h0": _feature_effective_rank(components["h0"]),
        "effective_rank_z_low": _feature_effective_rank(components["z_low"]),
        "relative_change_low": float((components["z_low"] - components["h0"]).norm(dim=-1).mean() / components["h0"].norm(dim=-1).mean().clamp_min(1e-8)),
        "relative_change_final": float((z_final - components["h0"]).norm(dim=-1).mean() / components["h0"].norm(dim=-1).mean().clamp_min(1e-8)),
    })
    metrics.update(_weighted_message_stats(edge_index, edges, labels, int(data.num_nodes)))
    if args.fit_probes:
        probe_outputs = {}
        for key in ("h_t", "h_v", "h0", "z_low", "z_struct", "z_final"):
            probe_outputs[key] = _fit_probe(components[key], labels, splits["train"], splits["val"], splits["test"], int(data.num_classes), args.probe_epochs, args.probe_lr)
        metrics["linear_probes"] = probe_outputs
    if args.counterfactuals:
        counterfactuals = _counterfactuals(model, components, edge_index)
        cf_metrics = {}
        for key, z_struct in counterfactuals.items():
            z_cf = model.output_norm(model.output_mlp(z_struct) + z_struct)
            logits_cf = head(torch.nan_to_num(z_cf))
            values = {}
            for name, idx in splits.items():
                values[name] = _classification_metrics(logits_cf, labels, idx, int(data.num_classes))
            cf_metrics[key] = values
        metrics["counterfactuals"] = cf_metrics
    out_dir = output_root / "diagnostics" / dataset / experiment.key / f"run{run_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_write(out_dir / "diagnostics.json", metrics)
    if args.export_nodes:
        _write_node_csv(out_dir / "nodes.csv", data, components, logits, labels)
    if args.export_edges:
        edge_path = out_dir / "edges.csv"
        ids = torch.arange(edge_index.size(1), device=edge_index.device)
        if ids.numel() > args.edge_sample:
            generator = torch.Generator(device=edge_index.device).manual_seed(20260908)
            ids = ids[torch.randperm(ids.numel(), generator=generator, device=edge_index.device)[: args.edge_sample]]
        src, dst = edge_index[:, ids].detach().cpu(), edge_index[:, ids].detach().cpu()
        with edge_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["src", "dst", "label_src", "label_dst", "same_label", "cos_text", "cos_visual", "edge_weight"])
            labels_cpu = labels.detach().cpu()
            for j, edge_id in enumerate(ids.detach().cpu().tolist()):
                s, d = int(src[0, j]), int(src[1, j])
                writer.writerow([s, d, int(labels_cpu[s]), int(labels_cpu[d]), int(bool(same[edge_id])), float(cos_t[edge_id]), float(cos_v[edge_id]), float(edges[edge_id])])
    return metrics


def run_diagnose(args: argparse.Namespace) -> int:
    root = Path(args.output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    entries = _checkpoint_entries(root)
    if not entries:
        raise FileNotFoundError(f"No checkpoints found under {root}; run the run subcommand first")
    data_cache: dict[str, Any] = {}
    summary: list[dict[str, Any]] = []
    for offset, (item, checkpoint, run_id) in enumerate(entries, 1):
        print(f"[diagnose {offset}/{len(entries)}] {item['dataset']}/{item['key']}/run{run_id}", flush=True)
        summary.append(diagnose_one(item, checkpoint, run_id, args, data_cache, root))
    _json_write(root / "diagnostics_summary.json", {"results": summary})
    scalar_rows = []
    counterfactual_rows = []
    for result in summary:
        scalar = {key: value for key, value in result.items() if not isinstance(value, (dict, list))}
        scalar_rows.append(scalar)
        for name, split_values in result.get("counterfactuals", {}).items():
            for split, values in split_values.items():
                counterfactual_rows.append(
                    {
                        "dataset": result["dataset"],
                        "experiment": result["experiment"],
                        "run_id": result["run_id"],
                        "seed": result["seed"],
                        "counterfactual": name,
                        "split": split,
                        **values,
                    }
                )
    scalar_fields = sorted({key for row in scalar_rows for key in row})
    with (root / "diagnostics_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows(scalar_rows)
    if counterfactual_rows:
        cf_fields = sorted({key for row in counterfactual_rows for key in row})
        with (root / "counterfactual_table.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cf_fields)
            writer.writeheader()
            writer.writerows(counterfactual_rows)
    print(f"Wrote {root / 'diagnostics_summary.json'}", flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAP-MAG v2 NC mechanism runner and diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
        command.add_argument("--base-seed", type=int, default=42)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        command.add_argument("--edge-sample", type=int, default=200_000)

    audit = sub.add_parser("audit", help="read-only graph and raw modality audit")
    common(audit)

    run = sub.add_parser("run", help="launch existing NC trainer for v2 experiments")
    common(run)
    run.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENT_KEYS), help="experiment keys or all")
    run.add_argument("--all-experiments", action="store_true")
    run.add_argument("--devices", nargs="+", default=["cuda:0"], help="devices used by independent job lanes; use cpu for a CPU smoke run")
    run.add_argument("--parallel-per-device", type=int, default=1, help="number of simultaneous jobs allowed on each listed device (2 can reduce runtime but may cause OOM)")
    run.add_argument("--exclusive-datasets", nargs="+", default=["ele-fashion"], choices=DATASETS, help="datasets that require the entire device; they never run beside another job")
    run.add_argument("--num-runs", type=int, default=3)
    run.add_argument("--validation-only", action="store_true")
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    diagnose = sub.add_parser("diagnose", help="analyze saved v2 checkpoints")
    common(diagnose)
    diagnose.add_argument("--diagnose-device", default="cpu")
    diagnose.add_argument("--fit-probes", action="store_true", help="fit post-hoc linear probes for projected/intermediate features")
    diagnose.add_argument("--probe-epochs", type=int, default=60)
    diagnose.add_argument("--probe-lr", type=float, default=3e-3)
    diagnose.add_argument("--counterfactuals", action="store_true", help="evaluate inference-only uniform/shuffled/inverted edge interventions")
    diagnose.add_argument("--export-nodes", action="store_true")
    diagnose.add_argument("--export-edges", action="store_true")

    all_cmd = sub.add_parser("all", help="audit, run and diagnose")
    common(all_cmd)
    all_cmd.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENT_KEYS))
    all_cmd.add_argument("--all-experiments", action="store_true")
    all_cmd.add_argument("--devices", nargs="+", default=["cuda:0"])
    all_cmd.add_argument("--parallel-per-device", type=int, default=1, help="simultaneous jobs per listed device; increase only if the GPU has enough memory")
    all_cmd.add_argument("--exclusive-datasets", nargs="+", default=["ele-fashion"], choices=DATASETS, help="datasets that require the entire device")
    all_cmd.add_argument("--num-runs", type=int, default=3)
    all_cmd.add_argument("--validation-only", action="store_true")
    all_cmd.add_argument("--no-resume", action="store_true")
    all_cmd.add_argument("--fail-fast", action="store_true")
    all_cmd.add_argument("--dry-run", action="store_true")
    all_cmd.add_argument("--diagnose-device", default="cpu")
    all_cmd.add_argument("--fit-probes", action="store_true")
    all_cmd.add_argument("--probe-epochs", type=int, default=60)
    all_cmd.add_argument("--probe-lr", type=float, default=3e-3)
    all_cmd.add_argument("--counterfactuals", action="store_true")
    all_cmd.add_argument("--export-nodes", action="store_true")
    all_cmd.add_argument("--export-edges", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if args.command in {"run", "all"}:
        if args.num_runs < 1:
            parser.error("--num-runs must be >= 1")
        if not args.devices:
            parser.error("at least one device is required")
        if len(set(args.devices)) != len(args.devices):
            parser.error("duplicate devices are not allowed; use --parallel-per-device for same-card concurrency")
        if args.parallel_per_device < 1:
            parser.error("--parallel-per-device must be >= 1")
    if args.command == "audit":
        return run_audit(args)
    if args.command == "run":
        return run_training(args, extra)
    if args.command == "diagnose":
        return run_diagnose(args)
    # all: keep the same output root and manifest so diagnosis can discover
    # exactly the checkpoints just produced.
    status = run_audit(args)
    if status:
        return status
    status = run_training(args, extra)
    if status and args.fail_fast:
        return status
    if args.dry_run:
        return 0
    return run_diagnose(args)


if __name__ == "__main__":
    raise SystemExit(main())
