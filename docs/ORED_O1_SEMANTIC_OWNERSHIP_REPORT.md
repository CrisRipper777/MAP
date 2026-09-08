# ORED-MAG Stage ORED-1: Semantic Ownership Migration and P0 Reproduction

## 1. Objective

Migrate the verified `exp/oft-mag` P0 factorization `(C, P_t, P_v)` into MAP as
`ored_mag`, without adding propagation, routing, evidence scoring, Composition,
Exposure, or a new training objective.

## 2. Decision

**HOLD_IMPLEMENTATION.** The implementation, unit tests, topology invariance, and
factor diagnostics pass. The required source-to-MAP migration parity gate does not:
Macro-F1 differs by `2.38pp`, above the allowed `0.50pp`.

## 3. Starting state

- Repository: `CrisRipper777/MAP/MAP`
- Branch: `ORED`
- ORED-0 tag/SHA: `ored-o0-reference` / `4c7321ba956c2fd0c4701eedcbe6c1d358d289d4`
- Source reference: `CrisRipper777/exp`, branch `oft-mag`
- Source SHA: `9f2ed253763c4bd2b7e14e534ae8c392c89c5551`
- No reset, rebase, or history rewrite was used.

## 4. Source truth

The source implementation is `exp/src/models/biaxis_components.py` and
`exp/src/models/biaxis_p0.py`. The migrated state-dict keys and tensor shapes match
the source architecture, with model parameter count `1,037,568`.

## 5. Added implementation

- `src/models/ored_components.py`
- `src/models/ored_mag.py`
- `configs/model/ored_mag.yaml`
- `tests/test_ored_mag.py`
- `scripts/ored_o1_diagnostics.py`

The existing task, runner, inference, factory, common-model, dataset, split, and
protocol files were not changed.

## 6. Modality split

The model validates positive text and visual dimensions and consumes
`x = [x_t | x_v]` using the MAP loader's existing order.

## 7. Factorization

The implementation uses separate modality projectors, one shared common encoder,
independent private encoders, and the exact consensus
`c = (c_t + c_v) / 2`. The fusion path is exactly
`Linear(3 * factor_dim, hidden_dim) -> norm -> activation -> Dropout`.

## 8. Auxiliary objective

The only auxiliary terms are the source P0 terms:

- common cosine alignment;
- cross-covariance orthogonality, with the source tiny-batch cosine fallback;
- two reconstruction heads from `[c, p]` to the projected modality embeddings.

MAP weights are `lambda_common=.02`, `lambda_orth=.01`, and `lambda_recon=.3`,
with `orth_fallback_batch=16`.

## 9. Framework interface

`forward` returns `(z, None, None, aux_loss, aux_info)`. Evaluation returns exact
zero auxiliary loss and an empty info dictionary. No framework file or protocol
override was introduced.

## 10. Inference and diagnostics API

`inference` returns CPU embeddings through chunked evaluation. `encode_factors`
returns CPU `c`, `c_t`, `c_v`, `p_t`, `p_v`, and `z_local`. Both APIs depend only on
features; the edge argument is accepted only for the existing interface.

## 11. Test result

The full suite passed:

```text
106 passed, 1 warning in 3.48s
```

The warning is the existing PyG distributed deprecation warning. The ORED-specific
suite passed `9/9`.

## 12. Test coverage

The ORED tests cover output shapes, P0 variant rejection, exact consensus averaging,
shared/independent encoder structure, topology invariance, finite train auxiliary
loss and backward gradients, evaluation zero auxiliary loss, chunked inference,
static absence of propagation operators, and state-dict architecture parity.

## 13. Formal run

The requested command was run on `cuda:0` with `Movies`, `seed=42`,
`num_runs=1`, `task.evaluate_test=false`, and the existing MAP unified full-graph NC
protocol. The checkpoint is
`outputs/ored/o1/ored_p0_movies_seed42_best.pt`.

## 14. ORED P0 result

- Best epoch: `71`
- Early stopping epoch: `101`
- Validation Accuracy: **51.829636%** (`51.83%`)
- Validation Macro-F1: **40.001156%** (`40.00%`)
- Model parameters: `1,037,568`
- NC head parameters: `5,140`
- Total parameters: `1,042,708`

No test evaluation or test metric was performed.

## 15. Historical comparison

Against the locked historical P0 reference (`51.71%` Val Acc, `37.62%` Val
Macro-F1), ORED differs by `+0.12pp` Acc and `+2.38pp` Macro-F1. Acc is within the
`0.30pp` allowance; Macro-F1 is outside the `1.00pp` allowance.

## 16. Current-source rerun

The unmodified source checkout was rerun val-only at the audited SHA. It reproduced
`51.71%` Val Acc and `37.62%` Val Macro-F1 at epoch `57`, with early stopping at
epoch `87`. Relative to that rerun, ORED differs by `+0.12pp` Acc and `+2.38pp`
Macro-F1, exceeding the required migration parity limits of `0.20pp` and `0.50pp`.

The runs use the existing configurations of their respective repositories; no
source edit, MAP protocol edit, or tuning was used. The first unreconciled training
difference is consistent with existing trainer configuration differences, including
MAP's configured gradient clipping and early-stopping settings. This is recorded as
a parity issue, not silently treated as migration success.

## 17. Factor diagnostics

`experiments/ored/o1/factor_diagnostics.json` reports finite values and nonzero norms
for every factor. Mean norms are `4.895 (c)`, `5.146 (c_t)`, `5.538 (c_v)`,
`2.844 (p_t)`, and `4.564 (p_v)`. Effective ranks are `4.33`, `4.06`, `9.72`,
`12.33`, and `28.72`, respectively; per-dimension standard deviations are nonzero.
Reconstruction MSE is `0.01064` for text and `0.05143` for visual. No NaN/Inf,
near-zero norm, all-constant factor, or effective-rank-one collapse was detected.

## 18. Topology and artifact audit

Original, empty, shuffled, and random edge inputs all produced maximum absolute
output delta `0.0`, including factor outputs. The saved artifacts are:

- `experiments/ored/o1/ored_p0_summary.json`
- `experiments/ored/o1/ored_p0_summary.csv`
- `experiments/ored/o1/factor_diagnostics.json`
- `experiments/ored/o1/state_dict_manifest.json`
- `experiments/ored/o1/source_rerun_comparison.json`

The ORED-0 ledger was not upgraded to `LOCKED IN ORED IMPLEMENTATION` because the
parity gate is HOLD.

## 19. Scope and next gate

```text
NO TEST
NO GRAPH PROPAGATION
NO JOINT-RD
NO OWNERSHIP-RD
NO COMPOSITION
NO EXPOSURE
NO TUNING
```

`GO_TO_ORED_2` is not granted. It requires an unreconciled source/current result to
be resolved while preserving the runner, core framework, split, and protocol. The
open items remain Joint-RD vs Ownership-RD, Composition, Exposure, and
Exposure×Composition; no ownership-conditioned graph evidence is claimed here.
