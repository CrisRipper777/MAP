from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from .graph_utils import edge_dict_to_index, ensure_edge_index, preprocess_edge_index
from .splits import load_or_create_lp_split, load_or_create_nc_split
from .types import EdgeSplit, MAGData


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path) -> Path:
    path = Path(str(path_like)).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists() or candidate.parent.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def _load_numpy_feature(path: str | Path, dtype: str = "float32") -> torch.Tensor:
    arr = np.load(resolve_path(path), allow_pickle=False)
    if dtype == "float32":
        arr = arr.astype(np.float32, copy=False)
    return torch.from_numpy(np.asarray(arr)).contiguous()


def _load_torch_tensor(path: str | Path, dtype: str = "float32") -> torch.Tensor:
    value = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    tensor = torch.as_tensor(value)
    if dtype == "float32":
        tensor = tensor.float()
    return tensor.contiguous()


def _load_dgl_graph(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    import dgl

    graphs, _ = dgl.load_graphs(str(resolve_path(path)))
    if not graphs:
        raise ValueError(f"No DGL graph found in {path}")
    graph = graphs[0]
    src, dst = graph.edges()
    edge_index = torch.stack([src.long(), dst.long()], dim=0).contiguous()
    labels = graph.ndata.get("label")
    if labels is None:
        raise KeyError(f"DGL graph {path} does not contain ndata['label']")
    return edge_index, labels.long().contiguous(), int(graph.num_nodes())


def _normalize_split_dict(raw: dict[str, Any]) -> EdgeSplit:
    def pack(section: dict[str, Any]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for key, value in section.items():
            result[key] = torch.as_tensor(value, dtype=torch.long).contiguous()
        return result

    return EdgeSplit(
        train=pack(raw["train"]),
        valid=pack(raw["valid"]),
        test=pack(raw["test"]),
        metadata=dict(raw.get("metadata", {})),
    )


def _split_modalities_from_joint(
    x: torch.Tensor,
    text_dim: int | None,
    visual_dim: int | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not visual_dim or not text_dim:
        return None, None
    if x.size(1) < visual_dim + text_dim:
        raise ValueError(
            f"joint feature dim {x.size(1)} is smaller than visual_dim+text_dim={visual_dim + text_dim}"
        )
    x_t = x[:, :text_dim].contiguous()
    x_i = x[:, text_dim : text_dim + visual_dim].contiguous()
    return x_i, x_t


def _as_index_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.long).contiguous()


def _apply_feature_mode(data: MAGData, cfg: DictConfig) -> MAGData:
    mode = str(cfg.dataset.get("feature_mode", "both")).strip().lower()
    aliases = {
        "full": "both",
        "multi": "both",
        "multimodal": "both",
        "all": "both",
        "text_only": "text",
        "visual_only": "visual",
        "image": "visual",
        "image_only": "visual",
    }
    mode = aliases.get(mode, mode)
    if mode == "both":
        data.info = {**data.info, "feature_mode": "both"}
        return data
    if data.x_t is None or data.x_i is None:
        raise ValueError(
            f"dataset.feature_mode={mode!r} requires split text/visual features, "
            f"but dataset {data.name} did not expose both modalities"
        )

    if mode == "text":
        data.x = data.x_t.contiguous()
        data.x_i = None
    elif mode == "visual":
        data.x = data.x_i.contiguous()
        data.x_t = None
    elif mode == "zero_text":
        x_t = torch.zeros_like(data.x_t)
        x_i = data.x_i.contiguous()
        data.x_t = x_t
        data.x_i = x_i
        data.x = torch.cat([x_t, x_i], dim=1).contiguous()
    elif mode == "zero_visual":
        x_t = data.x_t.contiguous()
        x_i = torch.zeros_like(data.x_i)
        data.x_t = x_t
        data.x_i = x_i
        data.x = torch.cat([x_t, x_i], dim=1).contiguous()
    else:
        raise ValueError(
            "dataset.feature_mode must be one of both/text/visual/zero_text/zero_visual, "
            f"got {mode!r}"
        )
    data.info = {**data.info, "feature_mode": mode}
    return data


def _load_magb(cfg: DictConfig, task_name: str, seed: int) -> MAGData:
    ds = cfg.dataset
    edge_index_raw, labels, num_nodes = _load_dgl_graph(ds.graph_path)
    image_feat = _load_numpy_feature(ds.image_feat_path, ds.get("feature_dtype", "float32"))
    text_feat = _load_numpy_feature(ds.text_feat_path, ds.get("feature_dtype", "float32"))
    if image_feat.size(0) != num_nodes or text_feat.size(0) != num_nodes:
        raise ValueError(
            f"{ds.name}: feature/node mismatch: image={tuple(image_feat.shape)}, "
            f"text={tuple(text_feat.shape)}, num_nodes={num_nodes}"
        )
    x = torch.cat([text_feat, image_feat], dim=1).contiguous()

    if task_name == "nc":
        split = load_or_create_nc_split(
            resolve_path(ds.nc_split_path),
            labels,
            float(ds.get("nc_train_ratio", 0.6)),
            float(ds.get("nc_val_ratio", 0.2)),
            seed,
            bool(ds.get("auto_generate_splits", True)),
        )
        edge_index = preprocess_edge_index(
            edge_index_raw,
            num_nodes,
            make_undirected=bool(ds.get("make_undirected", True)),
            with_self_loops=bool(ds.get("add_self_loops", True)),
        )
        return MAGData(
            name=ds.name,
            source=ds.source,
            task=task_name,
            x=x,
            x_i=image_feat,
            x_t=text_feat,
            edge_index=edge_index,
            y=labels,
            train_idx=_as_index_tensor(split["train_idx"]),
            val_idx=_as_index_tensor(split["val_idx"]),
            test_idx=_as_index_tensor(split["test_idx"]),
            num_nodes=num_nodes,
            num_classes=int(ds.num_classes),
            info={"nc_split_path": str(resolve_path(ds.nc_split_path))},
        )

    if task_name == "lp":
        split_raw = load_or_create_lp_split(
            resolve_path(ds.lp_split_path),
            edge_index_raw,
            num_nodes,
            float(ds.get("lp_val_ratio", 0.2)),
            float(ds.get("lp_test_ratio", 0.2)),
            int(ds.get("lp_num_neg", 150)),
            seed,
            bool(ds.get("lp_split_undirected", True)),
            bool(ds.get("auto_generate_splits", True)),
        )
        edge_split = _normalize_split_dict(split_raw)
        train_edge_index = edge_dict_to_index(edge_split.train)
        train_edge_index = preprocess_edge_index(
            train_edge_index,
            num_nodes,
            make_undirected=bool(ds.get("make_undirected", True)),
            with_self_loops=False,
        )
        return MAGData(
            name=ds.name,
            source=ds.source,
            task=task_name,
            x=x,
            x_i=image_feat,
            x_t=text_feat,
            edge_index=train_edge_index,
            y=labels,
            edge_split=edge_split,
            num_nodes=num_nodes,
            num_classes=int(ds.num_classes),
            info={"lp_split_path": str(resolve_path(ds.lp_split_path))},
        )

    raise ValueError(f"Unsupported MAGB task: {task_name}")


def _load_mmgraph(cfg: DictConfig, task_name: str) -> MAGData:
    ds = cfg.dataset
    x = _load_torch_tensor(ds.joint_feat_path, ds.get("feature_dtype", "float32"))
    num_nodes = int(x.size(0))
    x_i, x_t = _split_modalities_from_joint(
        x,
        int(ds.text_dim) if "text_dim" in ds else None,
        int(ds.visual_dim) if "visual_dim" in ds else None,
    )

    if task_name == "nc":
        edge_pairs = torch.as_tensor(
            torch.load(resolve_path(ds.edge_path), map_location="cpu", weights_only=False), dtype=torch.long
        )
        labels = torch.as_tensor(
            torch.load(resolve_path(ds.label_path), map_location="cpu", weights_only=False), dtype=torch.long
        )
        split = torch.load(resolve_path(ds.node_split_path), map_location="cpu", weights_only=False)
        edge_index = preprocess_edge_index(
            ensure_edge_index(edge_pairs),
            num_nodes,
            make_undirected=bool(ds.get("make_undirected", True)),
            with_self_loops=bool(ds.get("add_self_loops", True)),
        )
        return MAGData(
            name=ds.name,
            source=ds.source,
            task=task_name,
            x=x,
            x_i=x_i,
            x_t=x_t,
            edge_index=edge_index,
            y=labels,
            train_idx=_as_index_tensor(split["train_idx"]),
            val_idx=_as_index_tensor(split["val_idx"]),
            test_idx=_as_index_tensor(split["test_idx"]),
            num_nodes=num_nodes,
            num_classes=int(ds.num_classes),
            info={"node_split_path": str(resolve_path(ds.node_split_path))},
        )

    if task_name == "lp":
        split_raw = torch.load(resolve_path(ds.edge_split_path), map_location="cpu", weights_only=False)
        edge_split = _normalize_split_dict(split_raw)
        train_edge_index = edge_dict_to_index(edge_split.train)
        train_edge_index = preprocess_edge_index(
            train_edge_index,
            num_nodes,
            make_undirected=bool(ds.get("make_undirected", True)),
            with_self_loops=bool(ds.get("add_self_loops", False)),
        )
        return MAGData(
            name=ds.name,
            source=ds.source,
            task=task_name,
            x=x,
            x_i=x_i,
            x_t=x_t,
            edge_index=train_edge_index,
            edge_split=edge_split,
            num_nodes=num_nodes,
            num_classes=None,
            info={"edge_split_path": str(resolve_path(ds.edge_split_path))},
        )

    raise ValueError(f"Unsupported MM-Graph task: {task_name}")


def load_mag_data(cfg: DictConfig, task_name: str, seed: int) -> MAGData:
    tasks = list(cfg.dataset.get("tasks", []))
    if task_name not in tasks:
        raise ValueError(f"Dataset {cfg.dataset.name} supports tasks {tasks}, but got task={task_name}")
    source = str(cfg.dataset.source).lower()
    if source == "magb":
        return _apply_feature_mode(_load_magb(cfg, task_name, seed), cfg)
    if source == "mmgraph":
        return _apply_feature_mode(_load_mmgraph(cfg, task_name), cfg)
    raise ValueError(f"Unknown dataset source: {cfg.dataset.source}")
