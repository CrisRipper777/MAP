# ORED-MAG Stage ORED-0: Reference Lock and Migration Audit

## 1. Starting SHA and branch/status

- Repository: `CrisRipper777/MAP/MAP`
- Branch: `ORED`
- Starting SHA: `1c01faec943dfca08e467d517d58276f7629457b`
- Starting status: clean; `origin/ORED` pointed to the same SHA.
- No reset or history rewrite was used.
- `docs/pard_mag_deep_research_proposal.md` was not modified.

## 2. Tests

`conda run --no-capture-output -n yhf_env python -m pytest tests/ -q`:

```text
97 passed, 1 warning in 3.55s
```

The warning is the PyG distributed deprecation warning. The bare `pytest` launcher
was incorrectly bound to a Python 3.8 user-site script and failed before collection;
the environment-native pytest command above passed. No framework regression was found.

## 3. Protocol lock

All formal runs use `task=nc`, `unified_full_graph_nc_v1`, `training_mode=full_graph`,
`evaluate_test=false`, Movies, seed 42, `num_runs=1`, validation-accuracy checkpoint
selection, and the existing split/masks/optimizer/early-stopping/gradient-clipping/
metric implementation. MAP-v2 defaults remain `hidden_dim=256`, `num_hops=2`,
`restart=0.15`, `structure_mode=lowpass`, `use_self_residual=false`, and
`use_prototype_path=false`.

The host has two RTX 3090 GPUs. The default agent sandbox did not expose
`/dev/nvidia*`, so formal runs were executed in the approved host environment on
`cuda:0`; no research protocol or model hyperparameter changed.

## 4. Reference A — MAP-v2 core

- Best Val Acc: **57.32%**
- Best-checkpoint Val Macro-F1: **49.73%**
- Best epoch: **84**; early-stop epoch: **114**
- Parameters: **1,226,652**
- Config/output: `outputs/ored/o0/map_v2_core_movies_seed42/`

## 5. Reference B — LOWPASS-UNIFORM

Overrides: `model.use_reliability=false`, `model.use_semantic_edge_weight=false`.

- Best Val Acc: **58.07%**
- Best-checkpoint Val Macro-F1: **49.86%**
- Best epoch: **71**; early-stop epoch: **101**
- Parameters: **1,226,652**
- Config/output: `outputs/ored/o0/lowpass_uniform_movies_seed42/`

## 6. Reference C — SEMANTIC-UNIFORM

Overrides: `model.use_reliability=false`, `model.use_semantic_edge_weight=true`,
`model.edge_weight_mode=avg_cos`.

- Best Val Acc: **57.68%**
- Best-checkpoint Val Macro-F1: **50.08%**
- Best epoch: **84**; early-stop epoch: **114**
- Parameters: **1,226,652**
- Config/output: `outputs/ored/o0/semantic_uniform_movies_seed42/`

The three resolved configs, logs and `results.json` files are preserved in those
directories. No test evaluation was executed.

## 7. Historical reference comparison

`docs/nc_benchmark_results.md` has historical `map_mag_v2` Movies Val Acc
`57.1086 ± 0.3532%` over seeds 42/43/44, but no exact historical seed-42 value.
The v2 diagnosis also reports controlled variants as three-seed aggregates. Therefore
the ORED-0 single-seed values are **NOT DIRECTLY COMPARABLE** to those historical
references. They are consistent in scale; no historical Val value was invented.

## 8. Semantic Ownership migration source

- Repository: `CrisRipper777/exp`
- Branch: `oft-mag`
- Audited SHA: `9f2ed253763c4bd2b7e14e534ae8c392c89c5551`
- Files: `src/models/biaxis_p0.py`, `src/models/biaxis_components.py`

P0 is topology-free and factorizes `c_t=E_C(h_t)`, `c_v=E_C(h_v)`,
`c=(c_t+c_v)/2`, `p_t=E_t(h_t)`, and `p_v=E_v(h_v)`. Its auxiliary losses are
common, orthogonality and reconstruction. No source code was copied into MAP.

## 9. Migration compatibility audit

| Audit item | Conclusion |
|---|---|
| Reuse verbatim | P0 factor/reconstruction mathematics, `Model(cfg,data_info)`, forward tuple, inference return contract, and `[x_t\|x_v]` split. |
| Imports/paths | Rename `.biaxis_components` to `.ored_components`; expose the model as `ored_mag`; relative `src.models` imports otherwise remain valid. |
| Current common API | MAP and `exp/oft-mag` `get_activation`/`make_norm` APIs match, including BatchNorm and LayerNorm. P0 defaults to LayerNorm. |
| `data_info` | MAP NC supplies `input_dim`, `text_dim`, and `visual_dim`, which are all P0 needs. |
| NC aux loss | Current trainer adds the scalar P0 `aux_loss` to CE and backpropagates it correctly. |
| P0 `aux_info` | Current MAP `AUX_INFO_KEYS` omits `p0_*`, so diagnostics are ignored by logging; optimization and checkpointing are unaffected. |
| Forward contract | Exact match: `(z, None, None, aux_loss, aux_info)`. |
| Inference contract | Exact match; P0 ignores `edge_index` by design and chunked inference is valid. Synthetic smoke confirmed edge-invariant outputs. |
| Full-graph training | Runner-zero-modification is valid for locked NC: P0 ignores topology, so full-graph execution is mathematically harmless. Its config `full_graph_training:false` is not itself consulted by MAP NC. |
| Runner change | None required for ORED-1 NC. Forcing sampled mode would be a protocol change and is not needed. |
| text/visual dims | Loader order is `[x_t\|x_i]`; P0 validates positive dimensions and input width. |

## 10. Files planned for ORED-1

Not created in ORED-0:

- `src/models/ored_components.py`
- `src/models/ored_mag.py`
- `configs/model/ored_mag.yaml`
- `tests/test_ored_mag.py`

## 11. Framework files that MUST remain untouched

`src/tasks/nc.py`, `src/main.py`, `src/tasks/inference.py`, `src/models/common.py`,
`src/models/factory.py`, `configs/task/nc.yaml`, dataset configs, split files and
split-generation logic, `docs/unified_training_evaluation_protocol.md`, and
`docs/pard_mag_deep_research_proposal.md`.

## 12. ORED Evidence Ledger summary

The ledger records shallow restart low-pass diffusion as a locked source, a small
semantic-edge signal, validated P0 ownership factors, and O1/O1.5 ownership states as
possible graph states. K4/cross-routing/operator-bank/prototype/high-pass/complex-router
defaults are closed. Joint-RD, ownership-conditioned Composition/Exposure,
Exposure × Composition, and same-node cross-factor conditioning remain open.

## 13. Anomalies and warnings

- A same-seed GPU rerun was needed to clean an interrupted CPU log. The earlier GPU
  attempt reached 57.56% at epoch 63; the clean final run reached 57.32% at epoch 84.
  This small non-bitwise difference is consistent with CUDA atomic scatter reduction;
  it is disclosed rather than claimed deterministic. The final lock uses the clean run.
- The interrupted CPU directory was moved recoverably to
  `/tmp/ored_o0_map_v2_core_cpu_interrupted_20260908` and is not summarized.

## 14. Exact commands

Git audit: `git status`; `git branch --show-current`; `git rev-parse HEAD`;
`git log --oneline -5`.

Tests: `conda run --no-capture-output -n yhf_env python -m pytest tests/ -q`.

Formal commands:

```bash
conda run --no-capture-output -n yhf_env python -m src.main dataset=Movies task=nc model=map_mag_v2 num_runs=1 seed=42 device=cuda:0 task.evaluate_test=false hydra.run.dir=outputs/ored/o0/map_v2_core_movies_seed42
conda run --no-capture-output -n yhf_env python -m src.main dataset=Movies task=nc model=map_mag_v2 num_runs=1 seed=42 device=cuda:0 task.evaluate_test=false model.use_reliability=false model.use_semantic_edge_weight=false hydra.run.dir=outputs/ored/o0/lowpass_uniform_movies_seed42
conda run --no-capture-output -n yhf_env python -m src.main dataset=Movies task=nc model=map_mag_v2 num_runs=1 seed=42 device=cuda:0 task.evaluate_test=false model.use_reliability=false model.use_semantic_edge_weight=true model.edge_weight_mode=avg_cos hydra.run.dir=outputs/ored/o0/semantic_uniform_movies_seed42
```

## 15. Explicit scope statement

```text
NO TEST
NO ORED MODEL IMPLEMENTED
NO TUNING
```

No Test metrics were read or used. No `ored_mag.py`, Semantic Ownership factorizer,
Joint-RD, Ownership-RD, evidence scorer, Composition, Exposure or FULL ORED code was
added.

## 16. GO / HOLD

**GO_TO_ORED_1.** Tests pass, all three formal reference runs completed normally,
resolved configs match their intended variants, test evaluation is disabled, and P0
migration preserves the locked NC runner/protocol. The CUDA non-bitwise caveat is
recorded for tolerance-based future reference checks and does not require a runner or
protocol change.
