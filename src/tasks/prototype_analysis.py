from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import MAGData
from src.tasks.analysis import (
    _labels,
    _lp_split_degree,
    _lp_split_names,
    _nc_split_names,
)
from src.tasks.inference import infer_all_embeddings


def should_export_prototype_aux(cfg, model) -> bool:
    return bool(cfg.model.get("export_prototype_aux", False)) and hasattr(model, "prototypes")


@torch.no_grad()
def _prototype_aux_stats(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor | None,
    device: torch.device,
    topk: int,
) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    x = x.to(device)
    edge_index = model._edge_index_or_empty(edge_index, device)

    x_t, x_v = model._split_features(x)
    h_t = model.text_proj(x_t)
    h_v = model.visual_proj(x_v)

    mean_t = model._neighbor_mean(h_t, edge_index)
    mean_v = model._neighbor_mean(h_v, edge_index)
    r_t = model._reliability_gate(h_t, mean_t, model.text_reliability)
    r_v = model._reliability_gate(h_v, mean_v, model.visual_reliability)
    h_t_rel = r_t * h_t
    h_v_rel = r_v * h_v

    s_t = model._edge_cosine_mean(h_t, edge_index)
    s_v = model._edge_cosine_mean(h_v, edge_index)
    degree = model._degree(edge_index, int(x.size(0)), h_t.dtype)
    log_degree = torch.log1p(degree).unsqueeze(-1)

    if hasattr(model, "_reliability_base"):
        h0 = model._reliability_base(h_t, h_v, h_t_rel, h_v_rel, r_t, r_v)
    else:
        lambda_denom = (r_t + r_v).clamp_min(model.eps)
        h0 = (r_t / lambda_denom) * h_t_rel + (r_v / lambda_denom) * h_v_rel
    gamma = model._frequency_gate(torch.cat([h0, s_t, s_v, log_degree], dim=-1))

    router_input = torch.cat(
        [h_t_rel, h_v_rel, torch.abs(h_t_rel - h_v_rel), h_t_rel * h_v_rel, s_t, s_v, log_degree],
        dim=-1,
    )
    p = model._path_weights(model.router(router_input), model._active_path_mask(x.device, h_t.dtype))

    scale = math.sqrt(float(model.hidden_dim))
    proto_logits = torch.matmul(h0, model.prototypes.t()) / scale
    attn = torch.softmax(proto_logits, dim=-1)
    k = max(1, min(int(topk), int(attn.size(1))))
    top_prob, top_idx = torch.topk(attn, k=k, dim=-1)
    entropy = -(attn.clamp_min(model.eps) * torch.log(attn.clamp_min(model.eps))).sum(dim=-1)
    entropy_norm = entropy / math.log(max(int(attn.size(1)), 2))

    stats = {
        "degree": degree.detach().cpu(),
        "s_text": s_t.squeeze(-1).detach().cpu(),
        "s_visual": s_v.squeeze(-1).detach().cpu(),
        "r_text": r_t.squeeze(-1).detach().cpu(),
        "r_visual": r_v.squeeze(-1).detach().cpu(),
        "p_self": p[:, 0].detach().cpu(),
        "p_struct": p[:, 1].detach().cpu(),
        "p_proto": p[:, 2].detach().cpu(),
        "gamma": gamma.squeeze(-1).detach().cpu(),
        "top_proto_idx": top_idx.detach().cpu(),
        "top_proto_prob": top_prob.detach().cpu(),
        "prototype_entropy": entropy.detach().cpu(),
        "prototype_entropy_norm": entropy_norm.detach().cpu(),
    }
    if was_training:
        model.train()
    return stats


@torch.no_grad()
def _classification_stats(
    *,
    cfg,
    model,
    classifier,
    data: MAGData,
    device: torch.device,
    uses_graph: bool,
    inference_batch_size: int,
    inference_mode: str,
) -> dict[str, torch.Tensor]:
    if classifier is None or data.y is None:
        return {}
    z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
    logits_parts: list[torch.Tensor] = []
    classifier.eval()
    for start in range(0, int(z.size(0)), inference_batch_size):
        end = min(start + inference_batch_size, int(z.size(0)))
        logits_parts.append(classifier(z[start:end].to(device)).detach().cpu())
    logits = torch.cat(logits_parts, dim=0)
    probs = torch.softmax(logits, dim=-1)
    pred_confidence, pred = probs.max(dim=-1)
    labels = data.y.cpu().long()
    true_prob = probs[torch.arange(labels.numel()), labels]
    correct = (pred == labels).long()
    return {
        "pred": pred.cpu(),
        "pred_confidence": pred_confidence.cpu(),
        "true_prob": true_prob.cpu(),
        "correct": correct.cpu(),
    }


@torch.no_grad()
def export_node_prototype_stats(
    *,
    cfg,
    model,
    data: MAGData,
    device: torch.device,
    output_dir: str | Path,
    run_id: int,
    tag: str,
    task_name: str,
    logger: logging.Logger,
    classifier=None,
    uses_graph: bool | None = None,
    inference_batch_size: int | None = None,
    inference_mode: str | None = None,
) -> Path | None:
    if not should_export_prototype_aux(cfg, model):
        return None

    export_dir = Path(output_dir) / "prototype_aux"
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"run_{run_id + 1:02d}_{tag}_prototype_aux.csv"

    topk = int(cfg.model.get("prototype_export_topk", 3))
    stats = _prototype_aux_stats(model, data.x, data.edge_index, device=device, topk=topk)
    cls_stats: dict[str, torch.Tensor] = {}
    if task_name == "nc" and classifier is not None:
        cls_stats = _classification_stats(
            cfg=cfg,
            model=model,
            classifier=classifier,
            data=data,
            device=device,
            uses_graph=bool(uses_graph),
            inference_batch_size=int(inference_batch_size or cfg.task.inference_batch_size),
            inference_mode=str(inference_mode or cfg.task.get("inference_mode", "full")),
        )

    num_nodes = int(data.num_nodes)
    labels = _labels(data)
    if task_name == "lp" and data.edge_split is not None:
        split_names = _lp_split_names(data.edge_split, num_nodes)
        node_degree, train_edge_degree = _lp_split_degree(data.edge_split, num_nodes)
    else:
        split_names = _nc_split_names(data)
        node_degree = None
        train_edge_degree = None

    columns = ["node_id", "label", "split", "degree"]
    if task_name == "lp":
        columns.extend(["node_degree", "train_edge_degree"])
    columns.extend(
        [
            "s_text",
            "s_visual",
            "r_text",
            "r_visual",
            "p_self",
            "p_struct",
            "p_proto",
            "gamma",
            "top1_proto",
            "top1_prob",
            "top2_proto",
            "top2_prob",
            "top3_proto",
            "top3_prob",
            "prototype_entropy",
            "prototype_entropy_norm",
        ]
    )
    if cls_stats:
        columns.extend(["pred", "pred_confidence", "true_prob", "correct"])

    top_idx = stats["top_proto_idx"]
    top_prob = stats["top_proto_prob"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for node_id in range(num_nodes):
            row: list[int | float | str] = [
                node_id,
                labels[node_id],
                split_names[node_id],
                int(round(float(stats["degree"][node_id].item()))),
            ]
            if task_name == "lp":
                row.extend(
                    [
                        int(node_degree[node_id].item()) if node_degree is not None else "",
                        int(train_edge_degree[node_id].item()) if train_edge_degree is not None else "",
                    ]
                )
            row.extend(
                [
                    float(stats["s_text"][node_id].item()),
                    float(stats["s_visual"][node_id].item()),
                    float(stats["r_text"][node_id].item()),
                    float(stats["r_visual"][node_id].item()),
                    float(stats["p_self"][node_id].item()),
                    float(stats["p_struct"][node_id].item()),
                    float(stats["p_proto"][node_id].item()),
                    float(stats["gamma"][node_id].item()),
                ]
            )
            for rank in range(3):
                if rank < top_idx.size(1):
                    row.extend([int(top_idx[node_id, rank].item()), float(top_prob[node_id, rank].item())])
                else:
                    row.extend(["", ""])
            row.extend(
                [
                    float(stats["prototype_entropy"][node_id].item()),
                    float(stats["prototype_entropy_norm"][node_id].item()),
                ]
            )
            if cls_stats:
                row.extend(
                    [
                        int(cls_stats["pred"][node_id].item()),
                        float(cls_stats["pred_confidence"][node_id].item()),
                        float(cls_stats["true_prob"][node_id].item()),
                        int(cls_stats["correct"][node_id].item()),
                    ]
                )
            writer.writerow(row)

    logger.info("Saved node-level MAP-MAG prototype stats [%s]: %s", tag, path)
    return path
