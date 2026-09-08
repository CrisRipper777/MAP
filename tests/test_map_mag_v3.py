from __future__ import annotations

import logging

import pytest
import torch

from src.data import EdgeSplit, MAGData
from src.models import map_mag_v3
from src.models.factory import build_model
from src.tasks.analysis import export_edge_aux_stats, export_node_aux_stats
from src.tasks.inference import infer_all_embeddings


class _CfgNode(dict):
    def __getattr__(self, name: str):
        return self[name]


def _cfg(**overrides) -> _CfgNode:
    model = _CfgNode(
        name="map_mag_v3",
        hidden_dim=8,
        dropout=0.0,
        norm="layernorm",
        num_hops=2,
        num_layers=2,
        restart=0.15,
        diffusion_add_self_loops=True,
        full_graph_training=True,
        use_reliability=True,
        reliability_min=0.1,
        reliability_mode="single",
        use_modality_specific_edges=True,
        edge_weight_mode="separate_cos",
        edge_weight_min=0.1,
        edge_weight_temperature=2.0,
        edge_weight_learnable_init=1.0,
        propagation_mode="modality_specific",
        use_conflict_channel=False,
        conflict_min=0.0,
        conflict_scale=1.0,
        conflict_eta_max=0.2,
        modality_fusion_mode="learned_router",
        modality_router_temperature=2.0,
        modality_router_residual_scale=1.0,
        modality_router_residual_zero_init=True,
        lambda_modality_balance=0.0,
        use_self_residual=False,
        self_residual_max=0.1,
        use_modality_prototypes=False,
        num_text_prototypes=4,
        num_visual_prototypes=5,
        num_common_prototypes=3,
        prototype_residual_max=0.1,
        prototype_temperature=1.0,
        lambda_proto_diversity=0.0,
        prototype_init="random",
        lambda_edge_reg=0.0,
        export_aux_stats=False,
        export_node_aux=False,
        export_edge_aux=False,
        export_node_predictions=False,
    )
    model.update(overrides)
    return _CfgNode(model=model, task=_CfgNode())


def _graph() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0],
        ],
        dtype=torch.long,
    )
    return x, edge_index


def _data(x: torch.Tensor, edge_index: torch.Tensor) -> MAGData:
    return MAGData(
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


def _lp_data(x: torch.Tensor, edge_index: torch.Tensor) -> MAGData:
    return MAGData(
        name="small-lp",
        source="test",
        task="lp",
        x=x,
        edge_index=edge_index,
        edge_split=EdgeSplit(
            train={
                "source_node": torch.tensor([0, 1, 2, 3, 4, 0]),
                "target_node": torch.tensor([1, 2, 3, 4, 5, 5]),
            },
            valid={
                "source_node": torch.tensor([0]),
                "target_node": torch.tensor([2]),
                "target_node_neg": torch.tensor([[3, 4]]),
            },
            test={
                "source_node": torch.tensor([1]),
                "target_node": torch.tensor([4]),
                "target_node_neg": torch.tensor([[0, 3]]),
            },
        ),
        num_nodes=6,
    )


def _build(cfg: _CfgNode | None = None) -> map_mag_v3.MAPMAGV3:
    torch.manual_seed(123)
    return map_mag_v3.Model(
        cfg or _cfg(),
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )


def _forward(cfg: _CfgNode, edge_index: torch.Tensor | None = None):
    x, default_edge_index = _graph()
    if edge_index is None:
        edge_index = default_edge_index
    model = _build(cfg)
    model.eval()
    with torch.no_grad():
        z, _, _, aux_loss, aux_info = model(x, edge_index)
    assert z.shape == (6, 8)
    assert aux_loss.dim() == 0
    assert torch.isfinite(z).all()
    assert torch.isfinite(aux_loss)
    for value in aux_info.values():
        if torch.is_tensor(value):
            assert not value.requires_grad
            assert torch.isfinite(value).all()
    return model, z, aux_loss, aux_info


def test_map_mag_v3_factory_and_forward_contract() -> None:
    cfg = _cfg()
    model = build_model(
        cfg,
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )
    x, edge_index = _graph()
    z, none_a, none_b, aux_loss, aux_info = model(x, edge_index)

    assert isinstance(model, map_mag_v3.MAPMAGV3)
    assert model.out_dim == 8
    assert z.shape == (6, 8)
    assert none_a is None and none_b is None
    assert aux_loss.dim() == 0
    assert {
        "mean_r_text",
        "mean_r_visual",
        "mean_lambda_text",
        "mean_lambda_visual",
        "mean_edge_weight_text",
        "std_edge_weight_text",
        "mean_edge_weight_visual",
        "std_edge_weight_visual",
        "mean_cos_text",
        "mean_cos_visual",
        "mean_alpha_text",
        "mean_alpha_visual",
        "mean_degree",
    } <= set(aux_info)


@pytest.mark.parametrize("propagation_mode", ["shared_fused", "modality_specific"])
def test_map_mag_v3_propagation_modes_run(propagation_mode: str) -> None:
    _, _, _, aux_info = _forward(_cfg(propagation_mode=propagation_mode))

    assert ("mean_shared_edge_weight" in aux_info) == (propagation_mode == "shared_fused")


@pytest.mark.parametrize(
    "edge_weight_mode",
    [
        "separate_cos",
        "reliability_scaled",
        "learnable_scalar",
        "shared_avg_cos",
        "raw_uniform",
    ],
)
def test_map_mag_v3_edge_weight_modes_run(edge_weight_mode: str) -> None:
    _, _, _, aux_info = _forward(_cfg(edge_weight_mode=edge_weight_mode))

    assert 0.1 <= float(aux_info["mean_edge_weight_text"]) <= 1.0
    assert 0.1 <= float(aux_info["mean_edge_weight_visual"]) <= 1.0
    if edge_weight_mode == "raw_uniform":
        assert torch.allclose(aux_info["mean_edge_weight_text"], torch.tensor(1.0))
        assert torch.allclose(aux_info["std_edge_weight_text"], torch.tensor(0.0))


@pytest.mark.parametrize(
    "modality_fusion_mode",
    ["reliability", "reliability_residual", "learned_router", "uniform"],
)
def test_map_mag_v3_modality_fusion_modes_run(modality_fusion_mode: str) -> None:
    x, edge_index = _graph()
    model = _build(_cfg(modality_fusion_mode=modality_fusion_mode))
    stats = model.node_aux_stats(x, edge_index, device=torch.device("cpu"))

    assert torch.allclose(stats["alpha_text"] + stats["alpha_visual"], torch.ones(6), atol=1e-6)
    if modality_fusion_mode == "uniform":
        assert torch.allclose(stats["alpha_text"], torch.full((6,), 0.5), atol=1e-6)
    if modality_fusion_mode == "reliability":
        assert torch.allclose(stats["alpha_text"], stats["lambda_text"], atol=1e-6)
        assert torch.allclose(stats["alpha_visual"], stats["lambda_visual"], atol=1e-6)
    if modality_fusion_mode == "reliability_residual":
        assert torch.allclose(stats["alpha_text"], stats["lambda_text"], atol=1e-6)
        assert torch.allclose(stats["alpha_visual"], stats["lambda_visual"], atol=1e-6)


def test_map_mag_v3_reliability_residual_router_is_bounded_and_trainable() -> None:
    x, edge_index = _graph()
    model = _build(
        _cfg(
            modality_fusion_mode="reliability_residual",
            modality_router_residual_scale=0.5,
            modality_router_residual_zero_init=False,
        )
    )
    model.train()
    z, _, _, aux_loss, _ = model(x, edge_index)
    (z.square().mean() + aux_loss).backward()
    stats = model.node_aux_stats(x, edge_index, device=torch.device("cpu"))

    assert torch.allclose(
        stats["alpha_text"] + stats["alpha_visual"],
        torch.ones(6),
        atol=1e-6,
    )
    assert torch.isfinite(stats["alpha_text"]).all()
    assert model.modality_router[-1].weight.grad is not None


def test_map_mag_v3_zero_initialized_residual_router_learns_a_correction() -> None:
    x, edge_index = _graph()
    model = _build(
        _cfg(
            modality_fusion_mode="reliability_residual",
            modality_router_residual_scale=0.5,
            modality_router_residual_zero_init=True,
        )
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.node_aux_stats(x, edge_index, device=torch.device("cpu"))
    output_layer = model.modality_router[-1]
    before_weight = output_layer.weight.detach().clone()

    model.train()
    z, _, _, aux_loss, _ = model(x, edge_index)
    (z.square().mean() + aux_loss).backward()
    assert output_layer.weight.grad is not None
    assert float(output_layer.weight.grad.abs().sum()) > 0.0
    optimizer.step()

    assert torch.allclose(before["alpha_text"], before["lambda_text"], atol=1e-6)
    assert not torch.equal(output_layer.weight.detach(), before_weight)


@pytest.mark.parametrize("reliability_mode", ["single", "double", "none"])
def test_map_mag_v3_reliability_modes_run(reliability_mode: str) -> None:
    x, edge_index = _graph()
    model = _build(_cfg(reliability_mode=reliability_mode))
    stats = model.node_aux_stats(x, edge_index, device=torch.device("cpu"))

    assert torch.allclose(stats["lambda_text"] + stats["lambda_visual"], torch.ones(6), atol=1e-6)
    if reliability_mode == "none":
        assert torch.allclose(stats["r_text"], torch.ones(6), atol=1e-6)
        assert torch.allclose(stats["r_visual"], torch.ones(6), atol=1e-6)


@pytest.mark.parametrize("use_conflict_channel", [False, True])
def test_map_mag_v3_conflict_channel_switch_runs(use_conflict_channel: bool) -> None:
    _, _, _, aux_info = _forward(_cfg(use_conflict_channel=use_conflict_channel))

    assert ("mean_eta_conflict_text" in aux_info) == use_conflict_channel
    assert ("mean_conflict_weight_visual" in aux_info) == use_conflict_channel


@pytest.mark.parametrize("use_self_residual", [False, True])
def test_map_mag_v3_self_residual_switch_runs(use_self_residual: bool) -> None:
    _, _, _, aux_info = _forward(_cfg(use_self_residual=use_self_residual))

    assert ("mean_beta_self" in aux_info) == use_self_residual


@pytest.mark.parametrize("use_modality_prototypes", [False, True])
def test_map_mag_v3_modality_prototype_switch_runs(use_modality_prototypes: bool) -> None:
    model, _, _, aux_info = _forward(
        _cfg(
            use_modality_prototypes=use_modality_prototypes,
            lambda_proto_diversity=0.1 if use_modality_prototypes else 0.0,
        )
    )

    assert hasattr(model, "text_prototypes") == use_modality_prototypes
    assert ("mean_proto_residual" in aux_info) == use_modality_prototypes


def test_map_mag_v3_kmeans_prototype_initialization_is_one_time_and_finite() -> None:
    x, edge_index = _graph()
    cfg = _cfg(
        use_modality_prototypes=True,
        prototype_init="kmeans",
        num_text_prototypes=3,
        num_visual_prototypes=3,
        num_common_prototypes=3,
        prototype_init_max_samples=6,
        prototype_init_iters=4,
        prototype_init_batch_size=2,
    )
    model = _build(cfg)
    initial_text = model.text_prototypes.detach().clone()

    with pytest.raises(RuntimeError, match="initialize_prototypes"):
        model(x, edge_index)

    model.train()
    assert model.initialize_prototypes(x, device=torch.device("cpu"), seed=17)
    assert model.training
    assert bool(model._prototypes_initialized.item())
    assert not torch.allclose(initial_text, model.text_prototypes)
    assert torch.isfinite(model.text_prototypes).all()
    assert torch.isfinite(model.visual_prototypes).all()
    assert torch.isfinite(model.common_prototypes).all()
    assert not model.initialize_prototypes(x, device=torch.device("cpu"), seed=99)

    z, _, _, aux_loss, _ = model(x, edge_index)
    assert z.shape == (6, 8)
    assert torch.isfinite(aux_loss)


def test_map_mag_v3_optional_losses_support_backward() -> None:
    x, edge_index = _graph()
    model = _build(
        _cfg(
            use_conflict_channel=True,
            use_self_residual=True,
            use_modality_prototypes=True,
            lambda_modality_balance=0.1,
            lambda_proto_diversity=0.1,
            lambda_edge_reg=0.1,
        )
    )
    model.train()
    z, _, _, aux_loss, _ = model(x, edge_index)
    (z.square().mean() + aux_loss).backward()

    assert torch.isfinite(aux_loss)
    assert aux_loss.item() >= 0.0
    assert model.text_prototypes.grad is not None
    assert model.modality_router[0].weight.grad is not None
    assert model.text_proj.net[0].weight.grad is not None


def test_map_mag_v3_empty_edge_index_is_finite() -> None:
    empty_edge_index = torch.empty((2, 0), dtype=torch.long)
    _, _, _, aux_info = _forward(
        _cfg(use_conflict_channel=True, use_self_residual=True),
        edge_index=empty_edge_index,
    )

    assert torch.allclose(aux_info["mean_edge_weight_text"], torch.tensor(0.0))
    assert torch.allclose(aux_info["mean_cos_visual"], torch.tensor(0.0))
    assert torch.allclose(aux_info["mean_conflict_weight_text"], torch.tensor(0.0))


def test_map_mag_v3_full_and_layerwise_inference_match() -> None:
    x, edge_index = _graph()
    model = _build()
    model.eval()
    data = _data(x, edge_index)

    full = infer_all_embeddings(
        model,
        data,
        device=torch.device("cpu"),
        uses_graph=True,
        batch_size=2,
        inference_mode="full",
    )
    layerwise = infer_all_embeddings(
        model,
        data,
        device=torch.device("cpu"),
        uses_graph=True,
        batch_size=2,
        inference_mode="layerwise",
    )

    assert full.shape == (6, 8)
    assert torch.allclose(full, layerwise, atol=1e-6)


def test_map_mag_v3_node_aux_export_writes_dual_router_columns(tmp_path) -> None:
    x, edge_index = _graph()
    cfg = _cfg(export_node_aux=True, use_conflict_channel=True, use_self_residual=True)
    model = _build(cfg)
    path = export_node_aux_stats(
        cfg=cfg,
        model=model,
        data=_data(x, edge_index),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        run_id=0,
        tag="best_val",
        task_name="nc",
        logger=logging.getLogger("test_map_mag_v3_node_aux"),
    )

    assert path is not None
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {
        "lambda_text",
        "lambda_visual",
        "alpha_text",
        "alpha_visual",
        "beta_self",
        "eta_conflict_text",
        "eta_conflict_visual",
    } <= set(header)


def test_map_mag_v3_edge_aux_export_writes_modality_weights(tmp_path) -> None:
    x, edge_index = _graph()
    cfg = _cfg(export_edge_aux=True)
    model = _build(cfg)
    path = export_edge_aux_stats(
        cfg=cfg,
        model=model,
        data=_data(x, edge_index),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        run_id=0,
        tag="best_val",
        task_name="nc",
        logger=logging.getLogger("test_map_mag_v3_edge_aux"),
    )

    assert path is not None
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert header == [
        "src",
        "dst",
        "label_src",
        "label_dst",
        "same_label",
        "cos_text",
        "cos_visual",
        "edge_weight",
        "edge_weight_text",
        "edge_weight_visual",
        "edge_weight_shared",
        "src_degree",
        "dst_degree",
    ]
    assert len(lines) == edge_index.size(1) + 1


def test_map_mag_v3_lp_edge_aux_export_marks_train_message_edges(tmp_path) -> None:
    x, edge_index = _graph()
    cfg = _cfg(export_edge_aux=True)
    model = _build(cfg)
    path = export_edge_aux_stats(
        cfg=cfg,
        model=model,
        data=_lp_data(x, edge_index),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        run_id=0,
        tag="best_val",
        task_name="lp",
        logger=logging.getLogger("test_map_mag_v3_lp_edge_aux"),
    )

    assert path is not None
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert header == [
        "src",
        "dst",
        "positive_split",
        "label_src",
        "label_dst",
        "same_label",
        "cos_text",
        "cos_visual",
        "edge_weight",
        "edge_weight_text",
        "edge_weight_visual",
        "edge_weight_shared",
        "src_degree",
        "dst_degree",
    ]
    split_index = header.index("positive_split")
    assert {line.split(",")[split_index] for line in lines[1:]} == {"train"}
