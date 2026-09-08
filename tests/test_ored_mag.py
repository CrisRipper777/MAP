from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.ored_mag import Model


TEXT_DIM = 7
VISUAL_DIM = 11
INPUT_DIM = TEXT_DIM + VISUAL_DIM
N = 20


def make_model() -> Model:
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "ored_mag",
                "variant": "p0",
                "hidden_dim": 256,
                "factor_dim": 128,
                "dropout": 0.0,
                "activation": "gelu",
                "norm": "layernorm",
                "lambda_common": 0.02,
                "lambda_orth": 0.01,
                "lambda_recon": 0.3,
                "orth_fallback_batch": 16,
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

    cfg = OmegaConf.create({"model": {"variant": "not_p0", "hidden_dim": 256, "factor_dim": 128, "dropout": 0.0}})
    try:
        Model(cfg, {"input_dim": INPUT_DIM, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM})
    except ValueError as exc:
        assert "variant=p0" in str(exc)
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
    forbidden = ("gcn_norm", "messagepassing", "scatter", "index_add", "neighbor")
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
