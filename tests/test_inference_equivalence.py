from __future__ import annotations

import logging

import torch
import torch.nn as nn
import pytest

from src.data import EdgeSplit, MAGData
from src.models import dip, gcn, map_mag, map_mag_v1, mlp, mmgcn, sage
from src.models.factory import build_model
from src.tasks.analysis import export_node_aux_stats
from src.tasks.inference import infer_all_embeddings, resolve_inference_mode


class _CfgNode(dict):
    def __getattr__(self, name: str):
        return self[name]


def _cfg() -> _CfgNode:
    return _CfgNode(
        model=_CfgNode(
            hidden_dim=5,
            num_layers=2,
            dropout=0.0,
            activation="relu",
            norm="batchnorm",
            aggr="mean",
        )
    )


def _small_graph() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5, 1.6],
            [1.7, 1.8, 1.9, 2.0],
            [2.1, 2.2, 2.3, 2.4],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0],
        ],
        dtype=torch.long,
    )
    return x, edge_index


def _build_model(model_cls):
    torch.manual_seed(123)
    model = model_cls(_cfg(), {"input_dim": 4})
    model.eval()
    return model


def _compare_full_and_inference(model, x: torch.Tensor, edge_index: torch.Tensor | None):
    graph_edge_index = edge_index if edge_index is not None else torch.empty((2, 0), dtype=torch.long)
    data = MAGData(
        name="small",
        source="test",
        task="test",
        x=x,
        edge_index=graph_edge_index,
        num_nodes=int(x.size(0)),
    )
    with torch.no_grad():
        model.eval()
        full = infer_all_embeddings(
            model,
            data,
            device=torch.device("cpu"),
            uses_graph=edge_index is not None,
            batch_size=2,
            inference_mode="full",
        )
        model.eval()
        inferred = infer_all_embeddings(
            model,
            data,
            device=torch.device("cpu"),
            uses_graph=edge_index is not None,
            batch_size=2,
            inference_mode="layerwise",
        )
    return full, inferred, float((full - inferred).abs().max().item())


def test_mlp_inference_matches_full_batch_forward() -> None:
    x, _ = _small_graph()
    model = _build_model(mlp.Model)

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, None)

    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4


def test_sage_layerwise_inference_matches_full_batch_forward() -> None:
    x, edge_index = _small_graph()
    model = _build_model(sage.Model)

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, edge_index)

    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4


def test_gcn_layerwise_inference_matches_full_batch_forward() -> None:
    x, edge_index = _small_graph()
    model = _build_model(gcn.Model)

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, edge_index)

    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4


def test_mmgcn_layerwise_inference_matches_full_batch_forward() -> None:
    _, edge_index = _small_graph()
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    torch.manual_seed(123)
    model = mmgcn.Model(_cfg(), {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6})
    model.eval()

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, edge_index)

    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4


def test_dip_layerwise_inference_matches_full_batch_forward() -> None:
    _, edge_index = _small_graph()
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    cfg = _cfg()
    cfg.model["d_model"] = 4
    cfg.model["q_dim"] = 4
    cfg.model["n_q"] = 2
    cfg.model["mp_hops"] = 2
    cfg.model["n_pnode_v"] = 3
    cfg.model["n_pnode_t"] = 2
    cfg.model["dropout"] = 0.0
    cfg.model["norm"] = True
    cfg.model["fusion_type"] = "path_integral"
    cfg.model["embedding_dim"] = None

    torch.manual_seed(123)
    model = dip.Model(cfg, {"input_dim": 10, "text_dim": 4, "visual_dim": 6})
    model.eval()

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, edge_index)

    assert full.shape == (6, 8)
    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4


def test_map_mag_can_be_built_by_factory() -> None:
    cfg = _cfg()
    cfg.model["name"] = "map_mag"
    cfg.model["hidden_dim"] = 5
    cfg.model["num_prototypes"] = 4
    cfg.model["num_hops"] = 1

    model = build_model(cfg, {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6})

    assert isinstance(model, map_mag.MAPMAG)
    assert model.out_dim == 5


def test_map_mag_v1_can_be_built_by_factory() -> None:
    cfg = _cfg()
    cfg.model["name"] = "map_mag_v1"
    cfg.model["hidden_dim"] = 5
    cfg.model["num_prototypes"] = 4
    cfg.model["num_hops"] = 1

    model = build_model(cfg, {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6})

    assert isinstance(model, map_mag_v1.MAPMAG)
    assert model.out_dim == 5


def test_map_mag_v1_node_aux_stats_contains_required_values() -> None:
    _, edge_index = _small_graph()
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    cfg = _cfg()
    cfg.model["hidden_dim"] = 5
    cfg.model["num_prototypes"] = 4
    cfg.model["num_hops"] = 2

    torch.manual_seed(123)
    model = map_mag_v1.Model(cfg, {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6})
    stats = model.node_aux_stats(x, edge_index, device=torch.device("cpu"))

    required = {
        "degree",
        "s_text",
        "s_visual",
        "r_text",
        "r_visual",
        "p_self",
        "p_struct",
        "p_proto",
        "gamma",
        "text_visual_cosine",
    }
    assert required <= set(stats)
    assert all(stats[key].shape == (6,) for key in required)
    assert torch.isfinite(torch.stack([stats[key].float() for key in required])).all()
    assert torch.allclose(stats["p_self"] + stats["p_struct"] + stats["p_proto"], torch.ones(6), atol=1e-6)


def test_map_mag_v1_reliability_ablation_sets_gates_to_one() -> None:
    _, edge_index = _small_graph()
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    cfg = _cfg()
    cfg.model["hidden_dim"] = 5
    cfg.model["num_prototypes"] = 4
    cfg.model["use_reliability"] = False

    torch.manual_seed(123)
    model = map_mag_v1.Model(cfg, {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6})
    stats = model.node_aux_stats(x, edge_index, device=torch.device("cpu"))

    assert torch.allclose(stats["r_text"], torch.ones(6), atol=1e-6)
    assert torch.allclose(stats["r_visual"], torch.ones(6), atol=1e-6)


def test_map_mag_v1_node_aux_export_writes_nc_and_lp_csv(tmp_path) -> None:
    _, edge_index = _small_graph()
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    cfg = _CfgNode(
        task=_CfgNode(),
        model=_CfgNode(
            hidden_dim=5,
            num_layers=2,
            dropout=0.0,
            norm="batchnorm",
            num_prototypes=4,
            export_node_aux=True,
        ),
    )
    torch.manual_seed(123)
    model = map_mag_v1.Model(cfg, {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6})
    logger = logging.getLogger("test_map_mag_v1_node_aux_export")

    nc_data = MAGData(
        name="small",
        source="test",
        task="nc",
        x=x,
        edge_index=edge_index,
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
        train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([2, 3]),
        test_idx=torch.tensor([4, 5]),
        num_nodes=6,
        num_classes=2,
    )
    nc_path = export_node_aux_stats(
        cfg=cfg,
        model=model,
        data=nc_data,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        run_id=0,
        tag="best_val",
        task_name="nc",
        logger=logger,
    )

    assert nc_path is not None
    nc_lines = nc_path.read_text(encoding="utf-8").splitlines()
    assert nc_lines[0].split(",") == [
        "node_id",
        "label",
        "split",
        "degree",
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
    assert len(nc_lines) == 7

    edge_split = EdgeSplit(
        train={"source_node": torch.tensor([0, 1]), "target_node": torch.tensor([1, 2])},
        valid={
            "source_node": torch.tensor([2]),
            "target_node": torch.tensor([3]),
            "target_node_neg": torch.tensor([[0, 4]]),
        },
        test={
            "source_node": torch.tensor([4]),
            "target_node": torch.tensor([5]),
            "target_node_neg": torch.tensor([[1, 3]]),
        },
    )
    lp_data = MAGData(
        name="small",
        source="test",
        task="lp",
        x=x,
        edge_index=edge_index,
        edge_split=edge_split,
        num_nodes=6,
    )
    lp_path = export_node_aux_stats(
        cfg=cfg,
        model=model,
        data=lp_data,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        run_id=0,
        tag="final_epoch",
        task_name="lp",
        logger=logger,
    )

    assert lp_path is not None
    lp_header = lp_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert lp_header[:6] == ["node_id", "label", "split", "degree", "node_degree", "train_edge_degree"]


def test_map_mag_layerwise_inference_matches_full_batch_forward() -> None:
    _, edge_index = _small_graph()
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    cfg = _cfg()
    cfg.model["hidden_dim"] = 5
    cfg.model["num_prototypes"] = 4
    cfg.model["num_hops"] = 2
    cfg.model["lambda_proto"] = 0.01
    cfg.model["lambda_gate"] = 0.001

    torch.manual_seed(123)
    model = map_mag.Model(cfg, {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6})
    model.eval()

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, edge_index)
    z, _, _, aux_loss, aux_info = model(x, edge_index)

    assert full.shape == (6, 5)
    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4
    assert z.shape == full.shape
    assert aux_loss.dim() == 0
    assert torch.isfinite(z).all()
    assert torch.isfinite(aux_loss)
    assert {"mean_r_text", "mean_r_visual", "mean_p_self", "mean_p_struct", "mean_p_proto"} <= set(aux_info)


def test_map_mag_warmup_and_fixed_gamma_controls() -> None:
    _, edge_index = _small_graph()
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    cfg = _cfg()
    cfg.model["hidden_dim"] = 5
    cfg.model["num_prototypes"] = 4
    cfg.model["num_hops"] = 2
    cfg.model["reliability_min"] = 0.2
    cfg.model["gate_warmup_epochs"] = 2
    cfg.model["gamma_warmup_epochs"] = 2

    torch.manual_seed(123)
    model = map_mag.Model(cfg, {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6})
    model.eval()
    model.set_epoch(1)

    with torch.no_grad():
        z, _, _, aux_loss, aux_info = model(x, edge_index)

    assert z.shape == (6, 5)
    assert aux_loss.device == z.device
    assert aux_info["mean_r_text"] >= 0.2
    assert aux_info["mean_r_visual"] >= 0.2
    assert torch.allclose(aux_info["mean_p_self"], torch.tensor(1.0 / 3.0), atol=1e-6)
    assert torch.allclose(aux_info["mean_p_struct"], torch.tensor(1.0 / 3.0), atol=1e-6)
    assert torch.allclose(aux_info["mean_p_proto"], torch.tensor(1.0 / 3.0), atol=1e-6)
    assert torch.allclose(aux_info["mean_gamma"], torch.tensor(0.5), atol=1e-6)

    cfg_fixed = _cfg()
    cfg_fixed.model["hidden_dim"] = 5
    cfg_fixed.model["num_prototypes"] = 4
    cfg_fixed.model["fixed_gamma"] = 0.4
    cfg_fixed.model["gamma_warmup_epochs"] = 2

    torch.manual_seed(123)
    fixed_model = map_mag.Model(cfg_fixed, {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6})
    fixed_model.eval()
    fixed_model.set_epoch(1)

    with torch.no_grad():
        _, _, _, _, fixed_aux_info = fixed_model(x, edge_index)

    assert torch.allclose(fixed_aux_info["mean_gamma"], torch.tensor(0.4), atol=1e-6)


def test_mmgcn_uses_global_batch_node_ids_for_id_embeddings() -> None:
    torch.manual_seed(123)
    model = mmgcn.Model(_cfg(), {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6})
    n_id = torch.tensor([3, 7, 1, 4], dtype=torch.long)

    assert model._batch_n_id is None
    model._batch_n_id = n_id

    assert torch.equal(model._get_id_emb(n_id.numel()), model.id_embedding[n_id])


def test_mmgcn_id_embedding_is_not_registered_parameter() -> None:
    torch.manual_seed(123)
    model = mmgcn.Model(_cfg(), {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6})

    assert isinstance(model.id_embedding, torch.Tensor)
    assert not isinstance(model.id_embedding, nn.Parameter)
    assert model.id_embedding.requires_grad
    assert "id_embedding" not in dict(model.named_parameters())


def test_mmgcn_honors_dropout_config() -> None:
    cfg = _cfg()
    cfg.model["dropout"] = 0.37

    model = mmgcn.Model(cfg, {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6})

    assert model.v_branch.dropout.p == 0.37
    assert model.t_branch.dropout.p == 0.37


def test_mmgcn_honors_norm_config() -> None:
    cfg = _cfg()
    cfg.model["norm"] = "batchnorm"
    model = mmgcn.Model(cfg, {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6})

    assert isinstance(model.v_branch.norms[0], nn.BatchNorm1d)
    assert isinstance(model.t_branch.norms[0], nn.BatchNorm1d)

    cfg.model["norm"] = "none"
    model = mmgcn.Model(cfg, {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6})

    assert isinstance(model.v_branch.norms[0], nn.Identity)
    assert isinstance(model.t_branch.norms[0], nn.Identity)


def test_inference_mode_validation_accepts_supported_modes() -> None:
    cfg = _CfgNode(task=_CfgNode(inference_mode="full"))
    assert resolve_inference_mode(cfg) == "full"

    cfg.task["inference_mode"] = "layerwise"
    assert resolve_inference_mode(cfg) == "layerwise"


def test_inference_mode_validation_rejects_invalid_mode() -> None:
    cfg = _CfgNode(task=_CfgNode(inference_mode="sampled"))

    with pytest.raises(ValueError, match="task.inference_mode"):
        resolve_inference_mode(cfg)
