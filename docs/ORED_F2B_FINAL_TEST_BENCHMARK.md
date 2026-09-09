# ORED-F2B Final Test Benchmark

## Frozen protocol

- Architecture: ORED-MAG, variant `f1_dual_direct`.
- Frozen architecture: Semantic Ownership Decomposition; Dual-Granularity Relational Diffusion; Ownership-Projected Residual Collaboration.
- Architecture-freeze commit/tag: `5a53d1d` / `ored-f2a-architecture-freeze`.
- Checkpoint-only evaluation; validation-accuracy-selected checkpoints from F1/F2A.
- ARCHITECTURE FROZEN BEFORE TEST; NO TEST-DRIVEN TUNING; NO TEST-DRIVEN MODEL SELECTION; NO ENSEMBLE; NO SEED FILTERING.
- Exact full-graph inference on the fixed test split; no optimizer step was executed.

## Checkpoint manifest

The manifest contains 15 expected entries and 0 missing entries. No missing checkpoint was retrained.
See `experiments/ored/f2b/checkpoint_manifest.json` for paths, source configs, validation metadata, and SHA256 values.

## Per-dataset results

All values are percentages; `±` is population SD over seeds 42/43/44.

| Dataset | Val Acc | Test Acc | Test Macro-F1 | Test Acc − Val Acc |
|---|---:|---:|---:|---:|
| Movies | 57.51±0.38 | 55.57±0.25 | 49.80±1.41 | -1.94±0.63 pp |
| Toys | 80.23±0.21 | 79.89±0.32 | 77.17±0.31 | -0.34±0.53 pp |
| Grocery | 82.96±0.32 | 82.06±0.53 | 73.45±0.95 | -0.90±0.69 pp |
| ele-fashion | 87.19±0.22 | 87.08±0.22 | 74.74±0.64 | -0.11±0.01 pp |
| Reddit-S | 95.83±0.12 | 95.83±0.23 | 91.72±0.39 | -0.00±0.12 pp |

## Five-dataset macro mean

- ORED-MAG Test Accuracy: **80.0854%**.
- ORED-MAG Test Macro-F1: **73.3775%**.

| Reference | ORED minus Test Acc (pp) | ORED minus Test Macro-F1 (pp) |
|---|---:|---:|
| MLP | 3.1224 | 4.8678 |
| GCN | 1.7438 | 2.3218 |
| GraphSAGE | 0.7894 | 1.6175 |
| MMGCN | 0.2776 | 1.2734 |
| DGF | 1.1526 | 5.8945 |
| LGMRec | 0.2344 | 0.5180 |
| DiP | -0.3952 | -0.5111 |
| MAP | -0.0505 | 0.0771 |
| MAP-v2 | -0.4237 | -0.6624 |
| MAP-v3 | -0.2406 | -0.5962 |

## Unified NC benchmark comparison

The reference values are read from `docs/nc_benchmark_results.md`; no baseline was rerun.

| Model | Mean Test Accuracy | Mean Test Macro-F1 |
|---|---:|---:|
| MLP | 76.9630 | 68.5097 |
| GCN | 78.3416 | 71.0557 |
| GraphSAGE | 79.2960 | 71.7600 |
| MMGCN | 79.8078 | 72.1041 |
| DGF | 78.9328 | 67.4830 |
| LGMRec | 79.8510 | 72.8595 |
| DiP | 80.4806 | 73.8886 |
| MAP | 80.1359 | 73.3004 |
| MAP-v2 | 80.5091 | 74.0399 |
| MAP-v3 | 80.3260 | 73.9737 |
| ORED-MAG | 80.0854 | 73.3775 |

## Performance tier

**TIER_B**: mean Test Accuracy=80.0854 and mean Test Macro-F1=73.3775.

## Stability and anomalies

- All 15 expected frozen checkpoints were present and evaluated.
- Every checkpoint passed task, seed, data-info, strict model-state/head-state, variant, and frozen hyperparameter checks.
- The reported validation values are the source checkpoint-selection values; this evaluation computed metrics only on `test_idx`.
- No checkpoint was selected using Test metrics, no seed was filtered, and no ensemble was formed.

## Next paper-validation steps

Freeze these Test numbers as final evidence, preserve the manifest and source checkpoint hashes, and proceed to paper-level error analysis and reproducibility packaging. Do not reopen OCB, Composition, or Exposure based on this Test result.

## Evidence boundary

Final Dual ORED architecture = FROZEN BEFORE TEST. Test benchmark = FINAL PERFORMANCE EVIDENCE.
