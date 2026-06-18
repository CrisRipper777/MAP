# MAP-MAG Implementation Report

## 1. Files Added / Modified

- Added `src/models/map_mag.py`
  - Implements `MAPMAG` and exposes `Model = MAPMAG` for the existing dynamic model factory.
- Added `configs/model/map_mag.yaml`
  - Adds the default Hydra model config and ablation switches.
- Modified `tests/test_inference_equivalence.py`
  - Adds MAP-MAG factory construction and full/layerwise inference equivalence checks.

No data loading, split generation, negative sampling, NC trainer, or LP trainer logic was changed.

## 2. MAP-MAG Modules

MAP-MAG is implemented as an encoder that returns `(z, None, None, aux_loss, aux_info)`.

- Modality projection
  - Splits joint features using `data_info.text_dim` and `data_info.visual_dim`.
  - Projects text and visual features to `hidden_dim` with `Linear -> Norm -> ReLU -> Dropout`.
- Modality reliability estimation
  - Computes sparse neighbor means with `scatter`.
  - Learns scalar text and visual reliability gates from `[h, mean_h, h - mean_h, h * mean_h]`.
- Node-level pathway router
  - Computes text/visual edge cosine consistency using sparse edge aggregation.
  - Learns node-level weights over self, structure, and prototype paths.
- Self modality path
  - Encodes `[h_t_rel || h_v_rel]`.
- Structure-aware diffusion path
  - Builds reliability-weighted `h0`.
  - Applies GCN-style normalized sparse diffusion with restart.
  - Mixes low-frequency and high-frequency residual features through a scalar frequency gate.
- Semantic prototype path
  - Uses learnable prototypes `P` with `O(NK)` attention.
  - Adds prototype diversity regularization.
- Pathway fusion
  - Fuses enabled paths with learned router weights or fixed average weights.
  - Stabilizes output with residual MLP plus `LayerNorm`.

All graph operations use sparse `edge_index` scatter/GCN normalization. No dense adjacency or `N x N` attention is constructed.

## 3. Configuration Parameters

Main defaults in `configs/model/map_mag.yaml`:

| Parameter | Default | Meaning |
|---|---:|---|
| `hidden_dim` | 256 | Encoder output dimension |
| `dropout` | 0.2 | Dropout in projection/router/path MLPs |
| `norm` | layernorm | Normalization used inside MLPs |
| `num_hops` | 2 | Structure diffusion hops |
| `restart` | 0.15 | Restart weight for diffusion |
| `num_prototypes` | 32 | Number of semantic prototypes |
| `lambda_proto` | 0.01 | Prototype diversity loss weight |
| `lambda_gate` | 0.001 | Router balance loss weight |
| `full_graph_training` | false | Uses existing sampled training by default |

Ablation switches:

| Variant | Config override |
|---|---|
| w/o prototype path | `model.use_prototype_path=false` |
| w/o structure path | `model.use_structure_path=false` |
| w/o preference router | `model.use_preference_router=false` |
| low-pass only | `model.structure_low_pass_only=true` |
| self only | `model.self_only=true` |

## 4. Completed Sanity Checks

Commands run from `idea/MAG_baseline`:

```bash
conda run -n yhf_env python -m py_compile src/models/map_mag.py
conda run -n yhf_env python -m pytest tests/test_inference_equivalence.py -q
conda run -n yhf_env python -m pytest tests -q
```

Results:

| Check | Result |
|---|---|
| `map_mag.py` syntax compile | Passed |
| Inference equivalence tests | 13 passed |
| Full test suite | 19 passed |
| MAP-MAG factory import/build | Passed |
| MAP-MAG output shape / scalar aux loss / finite output | Passed |
| Full vs layerwise inference on toy graph | Max diff `< 1e-4` |

The initial bare `pytest` command failed because the default user Python environment was missing `exceptiongroup`; using the project environment through `conda run -n yhf_env python -m pytest` passed.

## 5. Preliminary Smoke Results

These are tiny wiring checks, not meaningful benchmark results.

| Dataset / Task | Overrides | Result |
|---|---|---|
| Movies / NC | `num_runs=1`, `epochs=1`, `max_train_batches=1`, `batch_size=128`, `num_neighbors=2`, `hidden_dim=16`, `num_prototypes=4`, CPU | Val Acc 32.93, Test Acc 32.92, Test Macro-F1 2.48 |
| sports-copurchase / LP | `num_runs=1`, `epochs=1`, `max_train_batches=1`, `train_pos_per_epoch=128`, `batch_size=128`, `num_neighbors=2`, `hidden_dim=8`, `num_prototypes=4`, CPU | Val MRR 4.89, Test MRR 4.65, Test H@10 9.18 |

## 6. Known Issues

- MAP-MAG supports `task.inference_mode=layerwise` through an exact full-graph fallback in `Model.inference`, similar to the DiP fallback pattern. It verifies equivalence, but it is not yet a memory-saving layerwise implementation.
- By default `full_graph_training=false`, so NC/LP training uses the existing neighbor-sampled trainer. This keeps the framework scalable, but the structure/prototype routing statistics are computed on sampled subgraphs during training.
- The smoke results use very small dimensions and only one training batch, so they should not be interpreted as model quality.

## 7. Next Steps

- Run controlled Movies-NC and sports-copurchase-LP experiments with the default `hidden_dim=256`.
- Run the five ablations listed above with the same seeds and report mean/std over multiple runs.
- If large-graph inference becomes a bottleneck, implement a true chunked/layerwise MAP-MAG inference path that materializes projected features on CPU and performs sparse propagation in bounded chunks.
