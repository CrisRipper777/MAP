from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.map_mag_v2 import MAPMAGV2
from src.models.ored_components import RestartDiffusion, normalize_restart_adjacency
from src.models.ored_mag import Model


TEXT_DIM = 7
VISUAL_DIM = 11
INPUT_DIM = TEXT_DIM + VISUAL_DIM
N = 20


def make_model(variant: str = "p0", num_hops: int = 2) -> Model:
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "ored_mag",
                "variant": variant,
                "hidden_dim": 256,
                "factor_dim": 128,
                "dropout": 0.0,
                "activation": "gelu",
                "norm": "layernorm",
                "lambda_common": 0.02,
                "lambda_orth": 0.01,
                "lambda_recon": 0.3,
                "orth_fallback_batch": 16,
                "num_hops": num_hops,
                "restart": 0.15,
                "diffusion_add_self_loops": True,
            }
        }
    )
    return Model(
        cfg,
        {
            "input_dim": INPUT_DIM,
            "text_dim": TEXT_DIM,
            "visual_dim": VISUAL_DIM,
            "num_nodes": N,
            "num_classes": 3,
        },
    )


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    x = torch.randn(N, INPUT_DIM)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    return x, edge_index


def random_edges(num_edges: int = 17) -> torch.Tensor:
    return torch.randint(0, N, (2, num_edges), dtype=torch.long)


def test_shapes_and_variant_guard() -> None:
    model = make_model().eval()
    x, edge_index = make_inputs()
    z, aux_a, aux_b, aux_loss, aux_info = model(x, edge_index)
    assert z.shape == (N, 256)
    assert model.out_dim == 256
    assert aux_a is None and aux_b is None
    assert aux_loss.item() == 0.0
    assert aux_info == {}

    cfg = OmegaConf.create({"model": {"variant": "not_a_variant", "hidden_dim": 256, "factor_dim": 128, "dropout": 0.0}})
    try:
        Model(cfg, {"input_dim": INPUT_DIM, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM})
    except ValueError as exc:
        assert "ored_mag variant" in str(exc)
    else:
        raise AssertionError("non-P0 variant must be rejected")


def test_common_factor_is_exact_average() -> None:
    model = make_model().eval()
    x, _ = make_inputs()
    factors, _ = model._encode(x)
    assert torch.equal(factors["c"], (factors["c_t"] + factors["c_v"]) * 0.5)


def test_common_encoder_is_shared_and_private_encoders_are_separate() -> None:
    factorizer = make_model().factorizer
    calls: list[int] = []
    hook = factorizer.common_encoder.register_forward_hook(lambda *_: calls.append(1))
    x, _ = make_inputs()
    factorizer.eval()(x[:, :TEXT_DIM], x[:, TEXT_DIM:])
    hook.remove()
    assert len(calls) == 2
    assert factorizer.private_text_encoder is not factorizer.private_visual_encoder
    assert factorizer.common_encoder is not factorizer.private_text_encoder
    assert factorizer.common_encoder[0].weight.data_ptr() != factorizer.private_text_encoder[0].weight.data_ptr()


def test_topology_invariance_including_factors() -> None:
    model = make_model().eval()
    x, original = make_inputs()
    variants = [
        original,
        torch.empty((2, 0), dtype=torch.long),
        original[:, torch.randperm(original.size(1))],
        random_edges(),
    ]
    reference = model(x, variants[0])[0]
    for edge_index in variants[1:]:
        assert torch.equal(reference, model(x, edge_index)[0])
        ref_factors = model.encode_factors(x, variants[0])
        new_factors = model.encode_factors(x, edge_index)
        for key in ref_factors:
            assert torch.equal(ref_factors[key], new_factors[key]), key


def test_training_aux_is_finite_and_backpropagates_through_factorizer() -> None:
    model = make_model().train()
    x, edge_index = make_inputs()
    _, _, _, aux_loss, aux_info = model(x, edge_index)
    assert aux_loss.requires_grad
    assert torch.isfinite(aux_loss)
    assert {"p0_common_loss", "p0_orth_loss", "p0_recon_loss"}.issubset(aux_info)
    assert all(torch.isfinite(value).item() for value in aux_info.values())
    aux_loss.backward()
    common_grad = model.factorizer.common_encoder[0].weight.grad
    private_grad = model.factorizer.private_text_encoder[0].weight.grad
    assert common_grad is not None and torch.isfinite(common_grad).all() and common_grad.abs().sum() > 0
    assert private_grad is not None and torch.isfinite(private_grad).all() and private_grad.abs().sum() > 0


def test_eval_aux_is_exact_zero_and_empty() -> None:
    model = make_model().eval()
    x, edge_index = make_inputs()
    _, _, _, aux_loss, aux_info = model(x, edge_index)
    assert aux_loss.item() == 0.0
    assert aux_info == {}


def test_chunked_inference_matches_full_eval() -> None:
    model = make_model().eval()
    x, edge_index = make_inputs()
    full = model(x, edge_index)[0].cpu()
    chunked = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=5)
    assert chunked.device.type == "cpu"
    torch.testing.assert_close(full, chunked, rtol=1e-5, atol=1e-6)


def test_static_source_contains_no_propagation_implementation() -> None:
    root = Path(__file__).parents[1] / "src" / "models"
    source = "\n".join((root / name).read_text() for name in ("ored_components.py", "ored_mag.py")).lower()
    forbidden = ("semantic_edge", "cross_factor", "edge_attention")
    assert not any(token in source for token in forbidden)


def test_state_dict_manifest_matches_source_architecture_contract() -> None:
    model = make_model()
    names = tuple(model.state_dict())
    required = (
        "factorizer.text_projector.net.0.weight",
        "factorizer.visual_projector.net.0.weight",
        "factorizer.common_encoder.0.weight",
        "factorizer.private_text_encoder.0.weight",
        "factorizer.private_visual_encoder.0.weight",
        "recon_text_head.net.0.weight",
        "recon_visual_head.net.0.weight",
        "fusion.0.weight",
    )
    for name in required:
        assert name in names
    assert len(names) == len(set(names))


def test_o2_parameter_parity() -> None:
    counts = {variant: sum(p.numel() for p in make_model(variant).parameters()) for variant in (
        "p0_refine", "joint_rd", "ownership_rd"
    )}
    assert len(set(counts.values())) == 1, counts


def make_map_v2_for_diffusion() -> MAPMAGV2:
    cfg = OmegaConf.create(
        {
            "model": {
                "hidden_dim": 16,
                "num_hops": 2,
                "restart": 0.15,
                "diffusion_add_self_loops": True,
                "use_reliability": False,
                "use_semantic_edge_weight": False,
                "norm": "layernorm",
            }
        }
    )
    return MAPMAGV2(
        cfg,
        {"input_dim": INPUT_DIM, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM},
    )


def test_restart_diffusion_matches_map_v2_diffuse() -> None:
    torch.manual_seed(7)
    h = torch.randn(N, 16)
    edge_index = random_edges(31)
    reference = make_map_v2_for_diffusion()._diffuse(h, edge_index)
    candidate = RestartDiffusion(2, 0.15, True)(h, edge_index)
    torch.testing.assert_close(candidate, reference, rtol=1e-6, atol=1e-6)


def test_restart_diffusion_zero_hops_is_identity() -> None:
    h = torch.randn(N, 16)
    edge_index = random_edges()
    result = RestartDiffusion(0, 0.15, True)(h, edge_index)
    assert result is h
    torch.testing.assert_close(result, h)


def test_restart_diffusion_empty_graph_with_self_loops_is_identity() -> None:
    h = torch.randn(N, 16)
    empty = torch.empty((2, 0), dtype=torch.long)
    result = RestartDiffusion(2, 0.15, True)(h, empty)
    torch.testing.assert_close(result, h, rtol=1e-6, atol=1e-6)


def test_o2_empty_graph_matches_p0_refine() -> None:
    x, _ = make_inputs()
    empty = torch.empty((2, 0), dtype=torch.long)
    reference_model = make_model("p0_refine").eval()
    reference = reference_model(x, empty)[0]
    for variant in ("joint_rd", "ownership_rd"):
        candidate_model = make_model(variant).eval()
        candidate_model.load_state_dict(reference_model.state_dict())
        candidate = candidate_model(x, empty)[0]
        torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-6)


def test_o2_graph_sensitivity_is_nonzero() -> None:
    x, edge_index = make_inputs()
    empty = torch.empty((2, 0), dtype=torch.long)
    for variant in ("joint_rd", "ownership_rd"):
        model = make_model(variant).eval()
        z_empty = model(x, empty)[0]
        z_graph = model(x, edge_index)[0]
        assert (z_graph - z_empty).abs().max().item() > 1e-7, variant


def test_ownership_diffusion_has_no_cross_factor_transport() -> None:
    torch.manual_seed(8)
    h = torch.randn(N, 3, 8)
    edge_index = random_edges(25)
    norm_ei, norm_ew = normalize_restart_adjacency(edge_index, N, h.dtype, True)
    diffusion = RestartDiffusion(2, 0.15, True)
    baseline = diffusion(h, norm_edge_index=norm_ei, norm_edge_weight=norm_ew)
    perturbed = h.clone()
    perturbed[:, 1] += 3.0 * torch.randn(N, 8)
    changed = diffusion(perturbed, norm_edge_index=norm_ei, norm_edge_weight=norm_ew)
    torch.testing.assert_close(changed[:, 0], baseline[:, 0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(changed[:, 2], baseline[:, 2], rtol=1e-6, atol=1e-6)
    assert (changed[:, 1] - baseline[:, 1]).abs().max().item() > 1e-6


def test_restart_diffusion_has_no_learnable_parameters() -> None:
    assert sum(p.numel() for p in RestartDiffusion().parameters()) == 0


def test_all_o2_variants_forward_aux_backward_are_finite() -> None:
    x, edge_index = make_inputs()
    for variant in ("p0", "p0_refine", "joint_rd", "ownership_rd"):
        model = make_model(variant).train()
        z, _, _, aux_loss, aux_info = model(x, edge_index)
        total = z.square().mean() + aux_loss
        assert torch.isfinite(z).all()
        assert torch.isfinite(aux_loss)
        assert all(torch.isfinite(value).all() for value in aux_info.values())
        total.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert grads and all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_graph_variant_inference_matches_eval_forward() -> None:
    x, edge_index = make_inputs()
    for variant in ("joint_rd", "ownership_rd"):
        model = make_model(variant).eval()
        full = model(x, edge_index)[0].cpu()
        inferred = model.inference(x, edge_index, device=torch.device("cpu"))
        torch.testing.assert_close(full, inferred, rtol=1e-5, atol=1e-6)


def test_encode_ored_states_contains_variant_states() -> None:
    x, edge_index = make_inputs()
    joint = make_model("joint_rd").eval().encode_ored_states(x, edge_index)
    ownership = make_model("ownership_rd").eval().encode_ored_states(x, edge_index)
    assert {"C0", "Pt0", "Pv0", "joint0", "joint2", "z"}.issubset(joint)
    assert {"C0", "Pt0", "Pv0", "C2", "Pt2", "Pv2", "u", "z"}.issubset(ownership)
