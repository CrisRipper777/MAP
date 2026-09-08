from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from omegaconf import ListConfig
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from src.data import EdgeSplit, MAGData
from src.data.graph_utils import edge_dict_to_index
from src.models import LinkPredictor, build_model
from src.tasks.common import (
    build_optimizer,
    clone_state_dict,
    format_aux_info_stats,
    load_state_dict_cpu,
    scheduler_step,
    summarize_aux_info_stats,
    update_aux_info_stats,
)
from src.tasks.analysis import export_edge_aux_stats, export_node_aux_stats
from src.tasks.inference import infer_all_embeddings, resolve_inference_mode
from src.tasks.modality import iter_masked_data, modality_drop_key, modality_mask_key, modality_mask_eval_enabled
from src.tasks.prototype_analysis import export_node_prototype_stats
from src.utils.metrics import format_pct
from src.utils.seeds import set_seed
from src.utils.summary import count_parameters, mean_std


def _uses_graph_encoder(cfg) -> bool:
    return str(cfg.model.name).lower() != "mlp"


def _resolve_lp_training_mode(cfg) -> str:
    mode = str(cfg.task.get("training_mode", "sampled")).strip().lower()
    if mode != "sampled":
        raise ValueError(
            "The frozen unified LP protocol requires task.training_mode='sampled'; "
            f"got {mode!r}"
        )
    return mode


def _edge_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return edge_index[0].long() * int(num_nodes) + edge_index[1].long()


def _all_positive_edge_index(edge_split: EdgeSplit) -> torch.Tensor:
    return torch.cat(
        [
            edge_dict_to_index(edge_split.train),
            edge_dict_to_index(edge_split.valid),
            edge_dict_to_index(edge_split.test),
        ],
        dim=1,
    ).cpu()


def _build_forbidden_edge_keys(edge_split: EdgeSplit, num_nodes: int, undirected: bool) -> torch.Tensor:
    edge_index = _all_positive_edge_index(edge_split)
    if undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return torch.unique(_edge_keys(edge_index, num_nodes).contiguous(), sorted=True)


def _is_forbidden_edge(src: torch.Tensor, dst: torch.Tensor, num_nodes: int, forbidden_keys: torch.Tensor) -> torch.Tensor:
    keys = src.long() * int(num_nodes) + dst.long()
    positions = torch.searchsorted(forbidden_keys, keys)
    in_bounds = positions < forbidden_keys.numel()
    matches = torch.zeros_like(in_bounds, dtype=torch.bool)
    if bool(in_bounds.any()):
        matches[in_bounds] = forbidden_keys[positions[in_bounds]] == keys[in_bounds]
    return matches


def _sample_filtered_negative_targets(
    src: torch.Tensor,
    num_nodes: int,
    num_neg: int,
    forbidden_keys: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    if num_neg <= 0:
        raise ValueError(f"num_neg must be positive, got {num_neg}")

    src = src.cpu().long().contiguous()
    forbidden_keys = forbidden_keys.cpu().long().contiguous()
    negatives = torch.empty((src.numel(), num_neg), dtype=torch.long)
    row_index = torch.arange(src.numel(), dtype=torch.long)

    for neg_col in range(num_neg):
        pending = row_index
        attempts = 0
        while pending.numel() > 0:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError(
                    "Unable to sample filtered train negatives after 1000 attempts; "
                    "the graph may be too dense for the requested num_train_neg."
                )
            pending_src = src[pending]
            candidate = torch.randint(0, num_nodes, (pending.numel(),), generator=generator)
            ok = candidate != pending_src
            ok &= ~_is_forbidden_edge(pending_src, candidate, num_nodes, forbidden_keys)
            if neg_col > 0:
                ok &= ~(negatives[pending, :neg_col] == candidate.view(-1, 1)).any(dim=1)

            accepted = pending[ok]
            if accepted.numel() > 0:
                negatives[accepted, neg_col] = candidate[ok]
            pending = pending[~ok]

    return negatives


def _build_epoch_train_labels(
    train_pos_edge_index: torch.Tensor | EdgeSplit,
    num_nodes: int,
    num_neg: int,
    forbidden_keys: torch.Tensor,
    generator: torch.Generator,
    train_pos_per_epoch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(train_pos_edge_index, EdgeSplit):
        pos_edge_index = edge_dict_to_index(train_pos_edge_index.train).cpu()
    else:
        pos_edge_index = train_pos_edge_index
    if train_pos_per_epoch is not None and train_pos_per_epoch < pos_edge_index.size(1):
        perm = torch.randperm(pos_edge_index.size(1), generator=generator)[:train_pos_per_epoch]
        pos_edge_index = pos_edge_index[:, perm]
    pos_src = pos_edge_index[0]
    neg_dst = _sample_filtered_negative_targets(pos_src, num_nodes, num_neg, forbidden_keys, generator)
    neg_edge_index = torch.stack(
        [pos_src.repeat_interleave(num_neg), neg_dst.reshape(-1)],
        dim=0,
    )
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1).contiguous()
    edge_label = torch.cat(
        [
            torch.ones(pos_edge_index.size(1), dtype=torch.float32),
            torch.zeros(neg_edge_index.size(1), dtype=torch.float32),
        ],
        dim=0,
    ).contiguous()
    return edge_label_index, edge_label


def _exclude_positive_label_edges_from_message_graph(
    edge_index: torch.Tensor,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    positive_mask = edge_label > 0.5
    if not bool(positive_mask.any()):
        return edge_index.contiguous()

    positive_edges = edge_label_index[:, positive_mask].long()
    forbidden_edges = torch.cat([positive_edges, positive_edges.flip(0)], dim=1)
    forbidden_keys = torch.unique(_edge_keys(forbidden_edges, num_nodes), sorted=True)
    edge_keys = _edge_keys(edge_index, num_nodes)
    positions = torch.searchsorted(forbidden_keys, edge_keys)
    in_bounds = positions < forbidden_keys.numel()
    drop_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
    if bool(in_bounds.any()):
        drop_mask[in_bounds] = forbidden_keys[positions[in_bounds]] == edge_keys[in_bounds]
    return edge_index[:, ~drop_mask].contiguous()


@dataclass
class _MessageEdgeLookup:
    """CPU lookup from a directed global edge code to every matching edge ID."""

    num_nodes: int
    sorted_codes: torch.Tensor
    sorted_edge_ids: torch.Tensor
    removal_scratch: torch.Tensor


def _build_message_edge_lookup(
    message_edge_index: torch.Tensor,
    num_nodes: int,
) -> _MessageEdgeLookup:
    """Build the reusable global ``e_id`` lookup used by sampled LP masking.

    Left/right ``searchsorted`` bounds make the lookup duplicate-safe: one
    directed endpoint pair may map to multiple global edge IDs, all of which
    must be removed. The lookup intentionally lives on CPU because masking
    happens before a sampled batch is copied to the accelerator.
    """
    edge_index = message_edge_index.detach().cpu().long().contiguous()
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            "message_edge_index must have shape [2, num_edges], "
            f"got {tuple(edge_index.shape)}"
        )
    if edge_index.numel() > 0:
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= int(num_nodes):
            raise ValueError("message_edge_index contains an out-of-range node ID")

    directed_codes = _edge_keys(edge_index, num_nodes)
    sorted_codes, sorted_edge_ids = directed_codes.sort()
    return _MessageEdgeLookup(
        num_nodes=int(num_nodes),
        sorted_codes=sorted_codes,
        sorted_edge_ids=sorted_edge_ids,
        removal_scratch=torch.zeros(edge_index.size(1), dtype=torch.bool),
    )


def _lookup_all_message_edge_ids(
    lookup: _MessageEdgeLookup,
    query_codes: torch.Tensor,
) -> torch.Tensor:
    """Return every global edge ID matching any query, including duplicates."""
    query_codes = torch.unique(query_codes.detach().cpu().long())
    if query_codes.numel() == 0 or lookup.sorted_codes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    starts = torch.searchsorted(lookup.sorted_codes, query_codes, right=False)
    ends = torch.searchsorted(lookup.sorted_codes, query_codes, right=True)
    lengths = ends - starts
    present = lengths > 0
    starts = starts[present]
    lengths = lengths[present]
    if starts.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    group_ids = torch.repeat_interleave(torch.arange(starts.numel()), lengths)
    flat_group_starts = torch.repeat_interleave(lengths.cumsum(dim=0) - lengths, lengths)
    sorted_positions = (
        starts[group_ids]
        + torch.arange(int(lengths.sum().item()), dtype=torch.long)
        - flat_group_starts
    )
    return lookup.sorted_edge_ids[sorted_positions]


def _exclude_positive_label_edges_by_global_eid(
    batch: Data,
    message_lookup: _MessageEdgeLookup,
) -> int:
    """Delete both orientations of batch positives via global ``batch.e_id``.

    This mutates ``batch.edge_index`` and ``batch.e_id`` in lockstep and must
    run before ``batch.to(device)``. The retained edge order is unchanged.
    """
    positive_mask = batch.edge_label > 0.5
    if not bool(positive_mask.any()) or batch.edge_index.numel() == 0:
        return 0
    if not hasattr(batch, "n_id") or not hasattr(batch, "e_id"):
        raise ValueError("global_eid edge masking requires sampled batch n_id and e_id")
    if batch.edge_index.device.type != "cpu" or batch.e_id.device.type != "cpu":
        raise ValueError("global_eid edge masking must run on CPU before batch.to(device)")
    if batch.e_id.dim() != 1 or batch.e_id.numel() != batch.edge_index.size(1):
        raise ValueError(
            "batch.e_id must align one-to-one with batch.edge_index columns; "
            f"got {batch.e_id.numel()} IDs for {batch.edge_index.size(1)} edges"
        )

    edge_ids = batch.e_id.long()
    if edge_ids.numel() > 0:
        if int(edge_ids.min()) < 0 or int(edge_ids.max()) >= message_lookup.removal_scratch.numel():
            raise ValueError("batch.e_id contains an ID outside the global message graph")

    supervision = batch.edge_label_index[:, positive_mask].long()
    global_source = batch.n_id[supervision[0]]
    global_target = batch.n_id[supervision[1]]
    query_codes = torch.cat(
        [
            global_source * message_lookup.num_nodes + global_target,
            global_target * message_lookup.num_nodes + global_source,
        ]
    )
    removal_ids = _lookup_all_message_edge_ids(message_lookup, query_codes)
    if removal_ids.numel() == 0:
        return 0

    message_lookup.removal_scratch[removal_ids] = True
    try:
        keep = ~message_lookup.removal_scratch[edge_ids]
        removed = int((~keep).sum())
        batch.edge_index = batch.edge_index[:, keep].contiguous()
        batch.e_id = batch.e_id[keep].contiguous()
    finally:
        message_lookup.removal_scratch[removal_ids] = False
    return removed


def _build_link_loader(
    cfg,
    pyg_data: Data,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    batch_generator: torch.Generator,
    neighbor_seed: int,
) -> LinkNeighborLoader:
    num_workers = int(cfg.task.get("loader_num_workers", 0))
    return LinkNeighborLoader(
        pyg_data,
        num_neighbors=_resolve_lp_num_neighbors(cfg),
        batch_size=int(cfg.task.batch_size),
        shuffle=True,
        subgraph_type=str(cfg.task.get("subgraph_type", "bidirectional")),
        num_workers=num_workers,
        generator=batch_generator,
        worker_init_fn=(
            partial(_seed_neighbor_worker, base_seed=int(neighbor_seed))
            if num_workers > 0
            else None
        ),
        **(
            {"prefetch_factor": int(cfg.task.get("loader_prefetch_factor", 2))}
            if int(cfg.task.get("loader_num_workers", 0)) > 0
            else {}
        ),
        edge_label_index=edge_label_index,
        edge_label=edge_label,
    )


def _resolve_lp_num_neighbors(cfg) -> list[int]:
    """Resolve the sampler depth independently from encoder depth.

    The unified LP protocol is explicitly two-hop. Some baselines (e.g. DGF)
    use a larger internal filtering iteration count, which must not silently
    turn the link sampler into a ten-hop sampler.
    """
    raw = cfg.task.get("num_neighbors", [5, 5])
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            values = [int(item.strip()) for item in raw[1:-1].split(",") if item.strip()]
        else:
            values = [int(raw)]
    else:
        values = [int(value) for value in raw] if isinstance(raw, (list, tuple, ListConfig)) else [int(raw)]
    if not values:
        raise ValueError("task.num_neighbors must contain at least one value")
    return values


def _seed_neighbor_worker(worker_id: int, base_seed: int) -> None:
    """Give worker-side neighbor sampling its own reproducible RNG stream."""
    worker_seed = (int(base_seed) + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _build_edge_loader(
    cfg,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    batch_generator: torch.Generator,
) -> TorchDataLoader:
    return TorchDataLoader(
        TensorDataset(edge_label_index.t().contiguous(), edge_label.contiguous()),
        batch_size=int(cfg.task.batch_size),
        shuffle=True,
        generator=batch_generator,
    )


def _prepare_eval_embeddings(
    z: torch.Tensor,
    device: torch.device,
    preload: bool,
    logger: logging.Logger,
) -> torch.Tensor:
    z_eval = z.to(device, non_blocking=True) if preload else z
    logger.info("Eval node embeddings: preload=%s | device=%s | shape=%s", preload, z_eval.device, tuple(z_eval.shape))
    return z_eval


def _evaluate_modality_masks(
    cfg,
    model,
    predictor: LinkPredictor,
    projection: nn.Module | None,
    data: MAGData,
    device: torch.device,
    uses_graph: bool,
    inference_mode: str,
    inference_batch_size: int,
    eval_preload_node_emb: bool,
    seed: int,
    full_test: dict[str, float],
    logger: logging.Logger,
) -> dict[str, float]:
    if not modality_mask_eval_enabled(cfg):
        return {}

    output: dict[str, float] = {}
    for mode, masked_data in iter_masked_data(cfg, data, seed):
        mask_key = modality_mask_key(mode)
        drop_key = modality_drop_key(mode)
        z = infer_all_embeddings(model, masked_data, device, uses_graph, inference_batch_size, inference_mode)
        if projection is not None:
            projection.eval()
            with torch.no_grad():
                z = projection(z.to(device)).detach().cpu()
        z_eval = _prepare_eval_embeddings(z, device, eval_preload_node_emb, logger)
        metrics = _evaluate_split(
            z_eval,
            predictor,
            data.edge_split.test,
            device,
            int(cfg.task.eval_edge_batch_size),
        )
        output[f"{mask_key}_test_mrr"] = metrics["mrr"]
        output[f"{mask_key}_test_hits@1"] = metrics["hits@1"]
        output[f"{mask_key}_test_hits@3"] = metrics["hits@3"]
        output[f"{mask_key}_test_hits@10"] = metrics["hits@10"]
        output[f"{drop_key}_mrr"] = full_test["test_mrr"] - metrics["mrr"]
        output[f"{drop_key}_hits@1"] = full_test["test_hits@1"] - metrics["hits@1"]
        output[f"{drop_key}_hits@3"] = full_test["test_hits@3"] - metrics["hits@3"]
        output[f"{drop_key}_hits@10"] = full_test["test_hits@10"] - metrics["hits@10"]
        logger.info(
            "Modality mask [%s] Test MRR %.2f | H@1 %.2f | H@3 %.2f | H@10 %.2f | Drop MRR %.2f",
            mode,
            format_pct(metrics["mrr"]),
            format_pct(metrics["hits@1"]),
            format_pct(metrics["hits@3"]),
            format_pct(metrics["hits@10"]),
            format_pct(output[f"{drop_key}_mrr"]),
        )
        del z_eval
        del z
        torch.cuda.empty_cache()
    return output


@torch.no_grad()
def _evaluate_split(
    z: torch.Tensor,
    predictor: LinkPredictor,
    split: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    predictor.eval()
    src_all = split["source_node"]
    dst_all = split["target_node"]
    neg_all = split["target_node_neg"]
    total = int(src_all.numel())
    mrr_sum = 0.0
    hits1_sum = 0.0
    hits3_sum = 0.0
    hits10_sum = 0.0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        if z.device.type == "cuda":
            src = src_all[start:end].to(z.device, non_blocking=True).long()
            dst = dst_all[start:end].to(z.device, non_blocking=True).long()
            neg = neg_all[start:end].to(z.device, non_blocking=True).long()
            pos_src = z[src]
            pos_dst = z[dst]
            neg_pair_feature = pos_src.unsqueeze(1) * z[neg]
        else:
            src = src_all[start:end].cpu().long()
            dst = dst_all[start:end].cpu().long()
            neg = neg_all[start:end].cpu().long()
            pos_src = z[src].to(device, non_blocking=True)
            pos_dst = z[dst].to(device, non_blocking=True)
            neg_dst = z[neg.reshape(-1)].to(device, non_blocking=True).view(
                neg.size(0), neg.size(1), -1
            )
            neg_pair_feature = pos_src.unsqueeze(1) * neg_dst

        pos_score = predictor.score_pairs(pos_src, pos_dst)
        # Avoid materializing a second [batch*num_neg, dim] copy of every
        # source embedding. Linear layers accept the 3-D pair tensor directly.
        neg_score = predictor(neg_pair_feature).view(neg.size(0), neg.size(1))
        # 0901/RPTA protocol: pessimistic ties, i.e. an equal-scoring
        # negative is ranked ahead of the positive.
        ranks = 1.0 + (neg_score >= pos_score.view(-1, 1)).sum(dim=1).float()

        mrr_sum += float((1.0 / ranks).sum().item())
        hits1_sum += float((ranks <= 1).float().sum().item())
        hits3_sum += float((ranks <= 3).float().sum().item())
        hits10_sum += float((ranks <= 10).float().sum().item())

    denom = max(total, 1)
    return {
        "mrr": mrr_sum / denom,
        "hits@1": hits1_sum / denom,
        "hits@3": hits3_sum / denom,
        "hits@10": hits10_sum / denom,
    }


def _run_single_lp(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
    run_id: int,
    output_dir: str | Path | None,
) -> dict[str, float]:
    seed = int(cfg.seed) + run_id
    set_seed(seed)
    inference_mode = resolve_inference_mode(cfg)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info).to(device)
    if hasattr(model, "initialize_prototypes"):
        initialized = model.initialize_prototypes(data.x, device=device, seed=seed)
        if initialized:
            logger.info("Initialized modality prototypes with K-means from frozen node features")
    training_mode = _resolve_lp_training_mode(cfg)
    proj_dim = int(cfg.task.decoder.get("proj_dim", 0) or 0)
    projection = nn.Linear(model.out_dim, proj_dim).to(device) if proj_dim > 0 else None
    predictor_in_dim = proj_dim if projection is not None else model.out_dim
    predictor = LinkPredictor(
        in_dim=predictor_in_dim,
        hidden_dim=int(cfg.task.decoder.hidden_dim),
        num_layers=int(cfg.task.decoder.num_layers),
        dropout=float(cfg.task.decoder.dropout),
    ).to(device)
    optimizer_params = list(model.parameters()) + list(predictor.parameters())
    if projection is not None:
        optimizer_params += list(projection.parameters())
    optimizer = build_optimizer(optimizer_params, cfg, model=model)
    criterion = torch.nn.BCEWithLogitsLoss()
    uses_graph = _uses_graph_encoder(cfg)
    # LP is always sampled under the unified protocol. MLP has no message
    # graph and therefore uses ordinary edge-label mini-batches.
    loader_name = "LinkNeighborLoader" if uses_graph else "EdgeLabelDataLoader"
    x_all = data.x.to(device) if not uses_graph else None
    undirected_filter = bool(data.edge_split.metadata.get("undirected", cfg.dataset.get("make_undirected", True)))
    forbidden_keys = _build_forbidden_edge_keys(data.edge_split, data.num_nodes, undirected=undirected_filter)
    logger.info(
        "[Run %d/%d] seed=%d | model+decoder params=%d",
        run_id + 1,
        int(cfg.num_runs),
        seed,
        count_parameters(model) + count_parameters(predictor) + (count_parameters(projection) if projection is not None else 0),
    )
    train_pos_per_epoch = cfg.task.get("train_pos_per_epoch")
    if train_pos_per_epoch is not None:
        train_pos_per_epoch = int(train_pos_per_epoch)
    logger.info("Loader: %s", loader_name)
    logger.info("Protocol: %s", str(cfg.task.get("protocol_version", "unified_sampled_lp_v1")))
    logger.info("LP training mode: %s", training_mode)
    if bool(getattr(model, "requires_full_graph_training", False)):
        logger.info(
            "Unified sampled LP overrides model.full_graph_training; "
            "this run is a sampled adaptation"
        )
    if uses_graph:
        logger.info("Train neighbor sampling: %s", _resolve_lp_num_neighbors(cfg))
    logger.info("Inference mode: %s", inference_mode)
    logger.info("Train negative sampling: global filtered | num_neg=%d", int(cfg.task.num_train_neg))
    if train_pos_per_epoch is not None:
        logger.info("Train pos per epoch: %d", train_pos_per_epoch)
    logger.info("Training...")

    # Reuse immutable graph storage across epochs. Only supervision labels and
    # their deterministic loaders are rebuilt each epoch.
    pyg_data = Data(x=data.x, edge_index=data.edge_index) if uses_graph else None
    edge_mask_backend = str(cfg.task.get("positive_edge_mask_backend", "global_eid")).strip().lower()
    if edge_mask_backend not in {"global_eid", "local_keys"}:
        raise ValueError(
            "task.positive_edge_mask_backend must be 'global_eid' or 'local_keys'; "
            f"got {edge_mask_backend!r}"
        )
    message_lookup = (
        _build_message_edge_lookup(data.edge_index, data.num_nodes)
        if uses_graph and edge_mask_backend == "global_eid"
        else None
    )
    if uses_graph:
        logger.info("Positive message-edge masking: %s", edge_mask_backend)

    best_val = -1.0
    best_test: dict[str, float] = {}
    best_model_state = None
    best_predictor_state = None
    best_projection_state = None
    patience_value = cfg.task.get("patience")
    patience_total = None if patience_value is None else int(patience_value)
    patience_left = patience_total
    early_stop_min_epoch = int(cfg.task.get("early_stop_min_epoch", 1))
    early_stop_min_delta = float(cfg.task.get("early_stop_min_delta", 0.0))
    grad_clip = cfg.task.get("grad_clip")
    aux_weight = float(cfg.task.loss.aux_weight)
    max_train_batches = cfg.task.get("max_train_batches")
    inference_batch_size = int(cfg.task.inference_batch_size)
    eval_preload_node_emb = bool(cfg.task.get("eval_preload_node_emb", False))
    evaluate_test = bool(cfg.task.get("evaluate_test", True))
    logger.info("LP eval preload node embeddings: %s", eval_preload_node_emb)
    train_pos_edge_index_all = edge_dict_to_index(data.edge_split.train).cpu()

    for epoch in range(1, int(cfg.task.epochs) + 1):
        model.train()
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        predictor.train()
        if projection is not None:
            projection.train()
        total_loss = 0.0
        total_examples = 0
        positive_message_edges_removed = 0
        aux_sums: dict[str, float] = {}
        aux_counts: dict[str, float] = {}
        neg_seed = seed + int(cfg.task.get("negative_seed_offset", 10_000)) + epoch
        shuffle_seed = seed + int(cfg.task.get("shuffle_seed_offset", 20_000)) + epoch
        neighbor_seed = seed + int(cfg.task.get("neighbor_seed_offset", 30_000)) + epoch
        neg_generator = torch.Generator().manual_seed(neg_seed)
        batch_generator = torch.Generator().manual_seed(shuffle_seed)
        edge_label_index, edge_label = _build_epoch_train_labels(
            train_pos_edge_index_all,
            data.num_nodes,
            int(cfg.task.num_train_neg),
            forbidden_keys,
            neg_generator,
            train_pos_per_epoch=train_pos_per_epoch,
        )
        if uses_graph:
            loader = _build_link_loader(
                cfg,
                pyg_data,
                edge_label_index,
                edge_label,
                batch_generator,
                neighbor_seed,
            )
        else:
            loader = _build_edge_loader(cfg, edge_label_index, edge_label, batch_generator)

        for step, batch in enumerate(loader):
            if max_train_batches is not None and step >= int(max_train_batches):
                break
            optimizer.zero_grad(set_to_none=True)
            if uses_graph:
                if message_lookup is not None:
                    positive_message_edges_removed += _exclude_positive_label_edges_by_global_eid(
                        batch,
                        message_lookup,
                    )
                else:
                    original_num_edges = int(batch.edge_index.size(1))
                    batch.edge_index = _exclude_positive_label_edges_from_message_graph(
                        batch.edge_index,
                        batch.edge_label_index,
                        batch.edge_label,
                        num_nodes=int(batch.x.size(0)),
                    )
                    positive_message_edges_removed += original_num_edges - int(batch.edge_index.size(1))
                batch = batch.to(device)
                if hasattr(model, "_batch_n_id"):
                    model._batch_n_id = batch.n_id
                z, _, _, aux_loss, aux_info = model(batch.x, batch.edge_index)
                if projection is not None:
                    z = projection(z)
                src, dst = batch.edge_label_index
                logits = predictor.score_pairs(z[src], z[dst])
                labels = batch.edge_label.float()
            else:
                edges, labels = batch
                edges = edges.to(device)
                labels = labels.to(device)
                z_all, _, _, aux_loss, aux_info = model(x_all[edges.reshape(-1)], None)
                if projection is not None:
                    z_all = projection(z_all)
                z_pairs = z_all.view(edges.size(0), 2, -1)
                logits = predictor.score_pairs(z_pairs[:, 0], z_pairs[:, 1])
            loss = criterion(logits, labels) + aux_weight * aux_loss
            loss.backward()
            if grad_clip is not None:
                clip_params = list(model.parameters()) + list(predictor.parameters())
                if projection is not None:
                    clip_params += list(projection.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=float(grad_clip))
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_examples += int(labels.numel())
            update_aux_info_stats(aux_sums, aux_counts, aux_info, weight=float(labels.numel()))
            if hasattr(model, "_batch_n_id"):
                model._batch_n_id = None

        train_loss = total_loss / max(total_examples, 1)
        aux_stats = summarize_aux_info_stats(aux_sums, aux_counts)
        scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        if epoch % int(cfg.task.eval_every) != 0:
            logger.info(
                "Epoch %05d | Train Loss %.4f | Positive Message Edges Removed %d",
                epoch,
                train_loss,
                positive_message_edges_removed,
            )
            if aux_stats:
                logger.info("Aux %s", format_aux_info_stats(aux_stats))
            continue

        z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
        if projection is not None:
            projection.eval()
            with torch.no_grad():
                z = projection(z.to(device)).detach().cpu()
        z_eval = _prepare_eval_embeddings(z, device, eval_preload_node_emb, logger)
        val_metrics = _evaluate_split(z_eval, predictor, data.edge_split.valid, device, int(cfg.task.eval_edge_batch_size))
        logger.info(
            "Epoch %05d | Train Loss %.4f | Positive Message Edges Removed %d",
            epoch,
            train_loss,
            positive_message_edges_removed,
        )
        if aux_stats:
            logger.info("Aux %s", format_aux_info_stats(aux_stats))
        logger.info(
            "Val MRR %.2f | Val H@1 %.2f | Val H@3 %.2f | Val H@10 %.2f",
            format_pct(val_metrics["mrr"]),
            format_pct(val_metrics["hits@1"]),
            format_pct(val_metrics["hits@3"]),
            format_pct(val_metrics["hits@10"]),
        )

        stop_early = False
        if val_metrics["mrr"] > best_val + early_stop_min_delta:
            best_val = val_metrics["mrr"]
            best_test = {
                "val_mrr": val_metrics["mrr"],
            }
            best_model_state = clone_state_dict(model)
            best_predictor_state = clone_state_dict(predictor)
            best_projection_state = clone_state_dict(projection) if projection is not None else None
            patience_left = patience_total
        elif patience_left is not None and epoch >= early_stop_min_epoch:
            patience_left -= 1
            patience_used = int(patience_total) - patience_left
            logger.info("Patience %d/%d | Best Val MRR %.2f", patience_used, int(patience_total), format_pct(best_val))
            if patience_left <= 0:
                logger.info("Early stopping at epoch %03d", epoch)
                stop_early = True

        del z_eval
        del z
        torch.cuda.empty_cache()
        if stop_early:
            break

    if output_dir is not None:
        export_node_aux_stats(
            cfg=cfg,
            model=model,
            data=data,
            device=device,
            output_dir=output_dir,
            run_id=run_id,
            tag="final_epoch",
            task_name="lp",
            logger=logger,
        )
        export_edge_aux_stats(
            cfg=cfg,
            model=model,
            data=data,
            device=device,
            output_dir=output_dir,
            run_id=run_id,
            tag="final_epoch",
            task_name="lp",
            logger=logger,
        )
        export_node_prototype_stats(
            cfg=cfg,
            model=model,
            data=data,
            device=device,
            output_dir=output_dir,
            run_id=run_id,
            tag="final_epoch",
            task_name="lp",
            logger=logger,
        )

    if best_model_state is not None and best_predictor_state is not None:
        load_state_dict_cpu(model, best_model_state)
        load_state_dict_cpu(predictor, best_predictor_state)
        if best_projection_state is not None and projection is not None:
            load_state_dict_cpu(projection, best_projection_state)
        if output_dir is not None:
            export_node_aux_stats(
                cfg=cfg,
                model=model,
                data=data,
                device=device,
                output_dir=output_dir,
                run_id=run_id,
                tag="best_val",
                task_name="lp",
                logger=logger,
            )
            export_edge_aux_stats(
                cfg=cfg,
                model=model,
                data=data,
                device=device,
                output_dir=output_dir,
                run_id=run_id,
                tag="best_val",
                task_name="lp",
                logger=logger,
            )
            export_node_prototype_stats(
                cfg=cfg,
                model=model,
                data=data,
                device=device,
                output_dir=output_dir,
                run_id=run_id,
                tag="best_val",
                task_name="lp",
                logger=logger,
            )
        if evaluate_test:
            z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
            if projection is not None:
                projection.eval()
                with torch.no_grad():
                    z = projection(z.to(device)).detach().cpu()
            z_eval = _prepare_eval_embeddings(z, device, eval_preload_node_emb, logger)
            test_metrics = _evaluate_split(
                z_eval,
                predictor,
                data.edge_split.test,
                device,
                int(cfg.task.eval_edge_batch_size),
            )
            best_test["test_mrr"] = test_metrics["mrr"]
            best_test["test_hits@1"] = test_metrics["hits@1"]
            best_test["test_hits@3"] = test_metrics["hits@3"]
            best_test["test_hits@10"] = test_metrics["hits@10"]
            del z_eval
            del z
            best_test.update(
                _evaluate_modality_masks(
                    cfg,
                    model,
                    predictor,
                    projection,
                    data,
                    device,
                    uses_graph,
                    inference_mode,
                    inference_batch_size,
                    eval_preload_node_emb,
                    seed,
                    best_test,
                    logger,
                )
            )

    if not best_test:
        z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
        if projection is not None:
            projection.eval()
            with torch.no_grad():
                z = projection(z.to(device)).detach().cpu()
        z_eval = _prepare_eval_embeddings(z, device, eval_preload_node_emb, logger)
        val_metrics = _evaluate_split(z_eval, predictor, data.edge_split.valid, device, int(cfg.task.eval_edge_batch_size))
        best_test = {
            "val_mrr": val_metrics["mrr"],
        }
        if evaluate_test:
            test_metrics = _evaluate_split(z_eval, predictor, data.edge_split.test, device, int(cfg.task.eval_edge_batch_size))
            best_test.update(
                {
                    "test_mrr": test_metrics["mrr"],
                    "test_hits@1": test_metrics["hits@1"],
                    "test_hits@3": test_metrics["hits@3"],
                    "test_hits@10": test_metrics["hits@10"],
                }
            )
        del z_eval
        del z
        if evaluate_test:
            best_test.update(
                _evaluate_modality_masks(
                    cfg,
                    model,
                    predictor,
                    projection,
                    data,
                    device,
                    uses_graph,
                    inference_mode,
                    inference_batch_size,
                    eval_preload_node_emb,
                    seed,
                    best_test,
                    logger,
                )
            )

    if hasattr(model, "_batch_n_id"):
        model._batch_n_id = None

    save_ckpt_path = cfg.task.get("save_ckpt_path")
    if save_ckpt_path:
        path = Path(str(save_ckpt_path))
        if int(cfg.num_runs) > 1:
            path = path.with_name(f"{path.stem}_run{run_id + 1}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "task": "lp",
                "seed": seed,
                "model_state": clone_state_dict(model),
                "head_state": clone_state_dict(predictor),
                "proj_state": clone_state_dict(projection) if projection is not None else None,
                "data_info": data_info,
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)

    logger.info(
        "[Run %d] Best Val MRR %.2f",
        run_id + 1,
        format_pct(best_test["val_mrr"]),
    )
    if evaluate_test:
        logger.info(
            "[Run %d] Final Test MRR %.2f | H@1 %.2f | H@3 %.2f | H@10 %.2f",
            run_id + 1,
            format_pct(best_test["test_mrr"]),
            format_pct(best_test["test_hits@1"]),
            format_pct(best_test["test_hits@3"]),
            format_pct(best_test["test_hits@10"]),
        )
    return best_test


def run_lp(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
    output_dir: str | Path | None = None,
) -> dict[str, tuple[float, float]]:
    if data.edge_split is None:
        raise ValueError("LP data must contain edge_split")
    run_results = [
        _run_single_lp(cfg, data, device, logger, run_id, output_dir)
        for run_id in range(int(cfg.num_runs))
    ]
    output: dict[str, tuple[float, float]] = {}
    logger.info("============================================================")
    logger.info("Final Results over %d runs", int(cfg.num_runs))
    logger.info("============================================================")
    display_names = {
        "val_mrr": "Highest Valid MRR",
        "test_mrr": "Test MRR",
        "test_hits@1": "Test Hits@1",
        "test_hits@3": "Test Hits@3",
        "test_hits@10": "Test Hits@10",
    }
    keys = ["val_mrr"]
    if bool(cfg.task.get("evaluate_test", True)):
        keys.extend(["test_mrr", "test_hits@1", "test_hits@3", "test_hits@10"])
    for key in keys:
        values = [item[key] for item in run_results]
        mean, std = mean_std(values)
        output[key] = (mean, std)
        logger.info("%s: %.2f ± %.2f", display_names[key], format_pct(mean), format_pct(std))
    primary_keys = set(output)
    for key in sorted(set().union(*(item.keys() for item in run_results)) - primary_keys):
        values = [item[key] for item in run_results if key in item]
        if len(values) != len(run_results):
            continue
        mean, std = mean_std(values)
        output[key] = (mean, std)
        logger.info("%s: %.2f ± %.2f", key, format_pct(mean), format_pct(std))
    logger.info("============================================================")
    return output
