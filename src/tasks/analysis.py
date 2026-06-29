from __future__ import annotations

import csv
import logging
from pathlib import Path

import torch

from src.data import EdgeSplit, MAGData
from src.data.graph_utils import edge_dict_to_index


def _edge_incident_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.long)
    endpoints = torch.cat([edge_index[0].cpu().long(), edge_index[1].cpu().long()], dim=0)
    return torch.bincount(endpoints, minlength=num_nodes).long()


def _lp_split_degree(edge_split: EdgeSplit, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    train_edge_index = edge_dict_to_index(edge_split.train)
    all_edge_index = torch.cat(
        [
            train_edge_index,
            edge_dict_to_index(edge_split.valid),
            edge_dict_to_index(edge_split.test),
        ],
        dim=1,
    )
    return _edge_incident_degree(all_edge_index, num_nodes), _edge_incident_degree(train_edge_index, num_nodes)


def _nc_split_names(data: MAGData) -> list[str]:
    split = ["none"] * int(data.num_nodes)
    if data.train_idx is not None:
        for node_id in data.train_idx.cpu().tolist():
            split[int(node_id)] = "train"
    if data.val_idx is not None:
        for node_id in data.val_idx.cpu().tolist():
            split[int(node_id)] = "val"
    if data.test_idx is not None:
        for node_id in data.test_idx.cpu().tolist():
            split[int(node_id)] = "test"
    return split


def _mark_lp_nodes(flags: torch.Tensor, edge_dict: dict[str, torch.Tensor], column: int) -> None:
    edge_index = edge_dict_to_index(edge_dict)
    if edge_index.numel() == 0:
        return
    nodes = torch.unique(torch.cat([edge_index[0].cpu().long(), edge_index[1].cpu().long()], dim=0))
    flags[nodes, column] = True


def _lp_split_names(edge_split: EdgeSplit, num_nodes: int) -> list[str]:
    flags = torch.zeros((num_nodes, 3), dtype=torch.bool)
    _mark_lp_nodes(flags, edge_split.train, 0)
    _mark_lp_nodes(flags, edge_split.valid, 1)
    _mark_lp_nodes(flags, edge_split.test, 2)

    names = ("train", "val", "test")
    split: list[str] = []
    for node_flags in flags.tolist():
        parts = [name for name, is_set in zip(names, node_flags, strict=True) if is_set]
        split.append("+".join(parts) if parts else "none")
    return split


def _labels(data: MAGData) -> list[int | str]:
    if data.y is None:
        return [""] * int(data.num_nodes)
    return [int(value) for value in data.y.cpu().tolist()]


def should_export_node_aux(cfg, model) -> bool:
    return bool(cfg.model.get("export_node_aux", False)) and hasattr(model, "node_aux_stats")


@torch.no_grad()
def export_node_aux_stats(
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
) -> Path | None:
    if not should_export_node_aux(cfg, model):
        return None

    export_dir = Path(output_dir) / "node_aux"
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"run_{run_id + 1:02d}_{tag}_node_aux.csv"

    stats = model.node_aux_stats(data.x, data.edge_index, device=device)
    num_nodes = int(data.num_nodes)
    labels = _labels(data)
    if task_name == "lp" and data.edge_split is not None:
        split_names = _lp_split_names(data.edge_split, num_nodes)
        node_degree, train_edge_degree = _lp_split_degree(data.edge_split, num_nodes)
    else:
        split_names = _nc_split_names(data)
        node_degree = None
        train_edge_degree = None

    columns = [
        "node_id",
        "label",
        "split",
        "degree",
    ]
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
            "text_visual_cosine",
        ]
    )

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
                    float(stats["text_visual_cosine"][node_id].item()),
                ]
            )
            writer.writerow(row)

    logger.info("Saved node-level MAP-MAG aux stats [%s]: %s", tag, path)
    return path
