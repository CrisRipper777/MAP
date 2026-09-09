# ORED-3A Composition Report

## 1. Scientific question

Does each semantic ownership state need a different connected-neighbor composition, and is ownership-specific composition better than one generic learned composition? This stage studies relational composition only. It does not study Exposure.

## 2. Starting SHA/tag

- Starting SHA: `a441a01faccb54d95c4bda29b25f1923e1f1adbe`
- Starting tag: `ored-o2-strong-parent`
- Final experiment tag: `ored-o3a-composition` (created after the formal artifacts are complete).

## 3. Matched variants

comp_uniform, comp_shared, comp_ownership all use the ORED-2 `ownership_rd` strong parent: `C→C`, `Pt→Pt`, `Pv→Pv`, `num_hops=2`, `restart=0.15`, `add_self_loops=true`, the same factorizer/fusion/output modules, and the same auxiliary loss. The only forward difference is conversion of scorer output into original-edge weights. `comp_uniform` still constructs and evaluates the scorer but ignores its output.

## 4. Scorer formula

For original edge `src=j, dst=i`, `H0=stack([C0,Pt0,Pv0])`:

```text
q_i^b = W_q H_i^b + E_own[b]
k_j^b = W_k H_j^b
g_ji^b = GELU(q_i^b + k_j^b)
s_ji^b = aᵀ g_ji^b
w_ji^b = 1 + 0.5*tanh(s_ji^b / 1.0)
```

`W_q/W_k: 128→64`, `E_own: 3×64`, and one shared `a:64→1`. The score output layer is zero-initialized, so all initial weights are exactly one. Added self-loops retain default weight one and are not scored. Scores are computed once from H0 and reused across both hops.

## 5. Strong-parent initialization and parameter parity

| Dataset | Model params | Head params | Total params | Parity |
|---|---|---|---|---|
| Movies | 1186817 | 5140 | 1191957 | True |
| Toys | 1186817 | 4626 | 1191443 | True |
| Grocery | 1186817 | 5140 | 1191957 | True |

Construction-order matched: `True`. Zero-score parent parity and all scorer sharing/isolation checks are in the test suite.

## 6. Tests

`tests/test_ored_mag.py`: 30 passed, including ORED-2 regression, exact parameter parity, zero-score parent parity, bounds, shared identity, ownership separation, no cross-factor payload transport, shared scorer parameters, scorer gradients, uniform isolation, finite forward/backward, and inference parity.

## 7. Movies seed42 smoke

The smoke is recorded under `outputs/ored/o3a/smoke/`. It is Val-only (`task.evaluate_test=false`) and is not used for tuning. {
  "available": true,
  "dataset": "Movies",
  "seed": 42,
  "task.evaluate_test": false,
  "no_tuning": true,
  "runs": {
    "comp_uniform": {
      "val_acc_pct": 56.83863162994385,
      "val_macro_f1_pct": 47.171976436423215,
      "score_abs_mean": 0.0,
      "score_min": 0.0,
      "score_max": 0.0,
      "factor_weights": {
        "C": {
          "mean": 1.0,
          "std": 0.0,
          "cv": 0.0,
          "min": 1.0,
          "max": 1.0,
          "fraction_lt_0_75": 0.0,
          "fraction_gt_1_25": 0.0,
          "fraction_near_lower": 0.0,
          "fraction_near_upper": 0.0
        },
        "Pt": {
          "mean": 1.0,
          "std": 0.0,
          "cv": 0.0,
          "min": 1.0,
          "max": 1.0,
          "fraction_lt_0_75": 0.0,
          "fraction_gt_1_25": 0.0,
          "fraction_near_lower": 0.0,
          "fraction_near_upper": 0.0
        },
        "Pv": {
          "mean": 1.0,
          "std": 0.0,
          "cv": 0.0,
          "min": 1.0,
          "max": 1.0,
          "fraction_lt_0_75": 0.0,
          "fraction_gt_1_25": 0.0,
          "fraction_near_lower": 0.0,
          "fraction_near_upper": 0.0
        }
      },
      "finite": true,
      "saturation_warning": false,
      "gradient_health": "covered_by_tests_test_o3a_scorer_gradients_are_finite_and_nonzero",
      "checkpoint": "outputs/ored/o3a/smoke_checkpoints/movies_seed42_comp_uniform.pt"
    },
    "comp_shared": {
      "val_acc_pct": 56.65866732597351,
      "val_macro_f1_pct": 47.66606266449937,
      "score_abs_mean": 1.2371622323989868,
      "score_min": -4.228809356689453,
      "score_max": 0.5527599453926086,
      "factor_weights": {
        "C": {
          "mean": 0.5959753394126892,
          "std": 0.05221132934093475,
          "cv": 0.08760652645860652,
          "min": 0.5034410953521729,
          "max": 0.8704978227615356,
          "fraction_lt_0_75": 0.9914055795325929,
          "fraction_gt_1_25": 0.0,
          "fraction_near_lower": 0.0013059539060459448,
          "fraction_near_upper": 0.0
        },
        "Pt": {
          "mean": 0.5959753394126892,
          "std": 0.05221132934093475,
          "cv": 0.08760652645860652,
          "min": 0.5034410953521729,
          "max": 0.8704978227615356,
          "fraction_lt_0_75": 0.9914055795325929,
          "fraction_gt_1_25": 0.0,
          "fraction_near_lower": 0.0013059539060459448,
          "fraction_near_upper": 0.0
        },
        "Pv": {
          "mean": 0.5959753394126892,
          "std": 0.05221132934093475,
          "cv": 0.08760652645860652,
          "min": 0.5034410953521729,
          "max": 0.8704978227615356,
          "fraction_lt_0_75": 0.9914055795325929,
          "fraction_gt_1_25": 0.0,
          "fraction_near_lower": 0.0013059539060459448,
          "fraction_near_upper": 0.0
        }
      },
      "finite": true,
      "saturation_warning": false,
      "gradient_health": "covered_by_tests_test_o3a_scorer_gradients_are_finite_and_nonzero",
      "checkpoint": "outputs/ored/o3a/smoke_checkpoints/movies_seed42_comp_shared.pt"
    },
    "comp_ownership": {
      "val_acc_pct": 56.80863857269287,
      "val_macro_f1_pct": 48.37165523400634,
      "score_abs_mean": 0.3244475722312927,
      "score_min": -1.2591747045516968,
      "score_max": 0.5926315188407898,
      "factor_weights": {
        "C": {
          "mean": 0.7202141284942627,
          "std": 0.05587117373943329,
          "cv": 0.07757578132525952,
          "min": 0.5745818018913269,
          "max": 0.9215862154960632,
          "fraction_lt_0_75": 0.6874354796582132,
          "fraction_gt_1_25": 0.0,
          "fraction_near_lower": 0.0,
          "fraction_near_upper": 0.0
        },
        "Pt": {
          "mean": 0.9774934649467468,
          "std": 0.06757329404354095,
          "cv": 0.06912915171992715,
          "min": 0.6970756649971008,
          "max": 1.1144345998764038,
          "fraction_lt_0_75": 0.001473862265394709,
          "fraction_gt_1_25": 0.0,
          "fraction_near_lower": 0.0,
          "fraction_near_upper": 0.0
        },
        "Pv": {
          "mean": 0.9430614709854126,
          "std": 0.11175894737243652,
          "cv": 0.1185065351632473,
          "min": 0.5905336141586304,
          "max": 1.265892744064331,
          "fraction_lt_0_75": 0.03962637280630838,
          "fraction_gt_1_25": 9.950124998445293e-05,
          "fraction_near_lower": 0.0,
          "fraction_near_upper": 0.0
        }
      },
      "finite": true,
      "saturation_warning": false,
      "gradient_health": "covered_by_tests_test_o3a_scorer_gradients_are_finite_and_nonzero",
      "checkpoint": "outputs/ored/o3a/smoke_checkpoints/movies_seed42_comp_ownership.pt"
    }
  }
}

## 8. Formal 3×3 results

All values are validation percentages, mean±population SD over seeds 42/43/44; no test labels or metrics were used.

| Dataset | Variant | Val Acc % | Val Macro-F1 % |
|---|---|---|---|
| Movies | comp_uniform | 57.33±0.35 | 49.42±1.73 |
| Movies | comp_shared | 56.94±0.48 | 49.76±1.20 |
| Movies | comp_ownership | 57.10±0.43 | 49.36±1.36 |
| Toys | comp_uniform | 80.00±0.12 | 77.16±0.51 |
| Toys | comp_shared | 80.12±0.09 | 76.98±0.33 |
| Toys | comp_ownership | 79.99±0.07 | 77.55±0.20 |
| Grocery | comp_uniform | 82.77±0.45 | 73.58±1.79 |
| Grocery | comp_shared | 82.77±0.47 | 74.05±2.11 |
| Grocery | comp_ownership | 82.84±0.35 | 74.22±1.36 |

## 9. Contrasts A/B/C

| Contrast | ΔAcc pp | ΔF1 pp | + Acc seeds/9 |
|---|---|---|---|
| A_shared_minus_uniform | -0.09±0.32 | 0.21±1.09 | 3 |
| B_ownership_minus_uniform | -0.06±0.19 | 0.33±1.04 | 4 |
| C_ownership_minus_shared | 0.03±0.21 | 0.12±0.64 | 5 |

- A: Shared − Uniform tests generic learned composition.
- B: Ownership − Uniform tests ownership-conditioned composition.
- C: Ownership − Shared is the core ownership-conditioning contrast.

## 10. Edge specialization diagnostics

Per-run factor values are mean/std/CV; full distributions, bound fractions, and saturation counts are in `experiments/ored/o3a/o3a_edge_diagnostics.json`.

| Run | C mean/std/CV | Pt mean/std/CV | Pv mean/std/CV |
|---|---|---|---|
| Movies/seed42/comp_uniform | 1.000/0.000/0.000 | 1.000/0.000/0.000 | 1.000/0.000/0.000 |
| Movies/seed42/comp_shared | 0.805/0.061/0.075 | 0.805/0.061/0.075 | 0.805/0.061/0.075 |
| Movies/seed42/comp_ownership | 0.605/0.044/0.072 | 0.937/0.099/0.105 | 0.888/0.131/0.148 |
| Movies/seed43/comp_uniform | 1.000/0.000/0.000 | 1.000/0.000/0.000 | 1.000/0.000/0.000 |
| Movies/seed43/comp_shared | 0.672/0.076/0.113 | 0.672/0.076/0.113 | 0.672/0.076/0.113 |
| Movies/seed43/comp_ownership | 1.010/0.048/0.048 | 0.944/0.216/0.229 | 0.694/0.131/0.189 |
| Movies/seed44/comp_uniform | 1.000/0.000/0.000 | 1.000/0.000/0.000 | 1.000/0.000/0.000 |
| Movies/seed44/comp_shared | 0.667/0.070/0.106 | 0.667/0.070/0.106 | 0.667/0.070/0.106 |
| Movies/seed44/comp_ownership | 0.657/0.034/0.052 | 0.990/0.119/0.120 | 0.840/0.094/0.112 |
| Toys/seed42/comp_uniform | 1.000/0.000/0.000 | 1.000/0.000/0.000 | 1.000/0.000/0.000 |
| Toys/seed42/comp_shared | 1.010/0.078/0.077 | 1.010/0.078/0.077 | 1.010/0.078/0.077 |
| Toys/seed42/comp_ownership | 1.118/0.069/0.062 | 0.970/0.050/0.052 | 1.203/0.099/0.082 |
| Toys/seed43/comp_uniform | 1.000/0.000/0.000 | 1.000/0.000/0.000 | 1.000/0.000/0.000 |
| Toys/seed43/comp_shared | 1.202/0.066/0.055 | 1.202/0.066/0.055 | 1.202/0.066/0.055 |
| Toys/seed43/comp_ownership | 1.068/0.102/0.095 | 0.874/0.071/0.081 | 1.139/0.129/0.113 |
| Toys/seed44/comp_uniform | 1.000/0.000/0.000 | 1.000/0.000/0.000 | 1.000/0.000/0.000 |
| Toys/seed44/comp_shared | 1.232/0.053/0.043 | 1.232/0.053/0.043 | 1.232/0.053/0.043 |
| Toys/seed44/comp_ownership | 1.235/0.051/0.042 | 0.977/0.082/0.084 | 1.299/0.102/0.079 |
| Grocery/seed42/comp_uniform | 1.000/0.000/0.000 | 1.000/0.000/0.000 | 1.000/0.000/0.000 |
| Grocery/seed42/comp_shared | 0.710/0.079/0.111 | 0.710/0.079/0.111 | 0.710/0.079/0.111 |
| Grocery/seed42/comp_ownership | 0.787/0.076/0.097 | 0.842/0.120/0.142 | 1.024/0.243/0.237 |
| Grocery/seed43/comp_uniform | 1.000/0.000/0.000 | 1.000/0.000/0.000 | 1.000/0.000/0.000 |
| Grocery/seed43/comp_shared | 0.899/0.143/0.159 | 0.899/0.143/0.159 | 0.899/0.143/0.159 |
| Grocery/seed43/comp_ownership | 1.004/0.048/0.047 | 0.903/0.113/0.125 | 1.216/0.168/0.139 |
| Grocery/seed44/comp_uniform | 1.000/0.000/0.000 | 1.000/0.000/0.000 | 1.000/0.000/0.000 |
| Grocery/seed44/comp_shared | 1.144/0.078/0.068 | 1.144/0.078/0.068 | 1.144/0.078/0.068 |
| Grocery/seed44/comp_ownership | 0.907/0.090/0.100 | 0.843/0.107/0.127 | 1.151/0.227/0.198 |

The declared peak scorer tensor is `[E,3,64]`; no `[E,3,3,d]` tensor is created. Runtime forward time and peak allocated GPU memory are recorded per run in the same JSON artifact.

## 11. Cross-factor weight correlations

Entries are Pearson/Spearman/MAD.

| Run | C-Pt P/S/MAD | C-Pv P/S/MAD | Pt-Pv P/S/MAD |
|---|---|---|---|
| Movies/seed42/comp_uniform | n/a/n/a/0.000 | n/a/n/a/0.000 | n/a/n/a/0.000 |
| Movies/seed42/comp_shared | 1.000/1.000/0.000 | 1.000/1.000/0.000 | 1.000/1.000/0.000 |
| Movies/seed42/comp_ownership | 0.415/0.429/0.332 | -0.158/-0.141/0.285 | 0.034/0.037/0.136 |
| Movies/seed43/comp_uniform | n/a/n/a/0.000 | n/a/n/a/0.000 | n/a/n/a/0.000 |
| Movies/seed43/comp_shared | 1.000/1.000/0.000 | 1.000/1.000/0.000 | 1.000/1.000/0.000 |
| Movies/seed43/comp_ownership | -0.001/0.014/0.197 | 0.111/0.081/0.320 | 0.077/0.076/0.291 |
| Movies/seed44/comp_uniform | n/a/n/a/0.000 | n/a/n/a/0.000 | n/a/n/a/0.000 |
| Movies/seed44/comp_shared | 1.000/1.000/0.000 | 1.000/1.000/0.000 | 1.000/1.000/0.000 |
| Movies/seed44/comp_ownership | 0.165/0.186/0.333 | 0.216/0.248/0.183 | 0.085/0.109/0.179 |
| Toys/seed42/comp_uniform | n/a/n/a/0.000 | n/a/n/a/0.000 | n/a/n/a/0.000 |
| Toys/seed42/comp_shared | 1.000/1.000/0.000 | 1.000/1.000/0.000 | 1.000/1.000/0.000 |
| Toys/seed42/comp_ownership | 0.551/0.535/0.148 | 0.040/0.077/0.120 | 0.045/0.040/0.235 |
| Toys/seed43/comp_uniform | n/a/n/a/0.000 | n/a/n/a/0.000 | n/a/n/a/0.000 |
| Toys/seed43/comp_shared | 1.000/1.000/0.000 | 1.000/1.000/0.000 | 1.000/1.000/0.000 |
| Toys/seed43/comp_ownership | 0.250/0.264/0.200 | 0.364/0.384/0.123 | 0.145/0.156/0.269 |
| Toys/seed44/comp_uniform | n/a/n/a/0.000 | n/a/n/a/0.000 | n/a/n/a/0.000 |
| Toys/seed44/comp_shared | 1.000/1.000/0.000 | 1.000/1.000/0.000 | 1.000/1.000/0.000 |
| Toys/seed44/comp_ownership | 0.000/-0.026/0.258 | 0.358/0.286/0.098 | -0.021/-0.026/0.322 |
| Grocery/seed42/comp_uniform | n/a/n/a/0.000 | n/a/n/a/0.000 | n/a/n/a/0.000 |
| Grocery/seed42/comp_shared | 1.000/1.000/0.000 | 1.000/1.000/0.000 | 1.000/1.000/0.000 |
| Grocery/seed42/comp_ownership | 0.008/-0.013/0.121 | -0.188/-0.193/0.298 | 0.011/0.061/0.270 |
| Grocery/seed43/comp_uniform | n/a/n/a/0.000 | n/a/n/a/0.000 | n/a/n/a/0.000 |
| Grocery/seed43/comp_shared | 1.000/1.000/0.000 | 1.000/1.000/0.000 | 1.000/1.000/0.000 |
| Grocery/seed43/comp_ownership | 0.358/0.339/0.124 | 0.269/0.277/0.233 | 0.106/0.131/0.326 |
| Grocery/seed44/comp_uniform | n/a/n/a/0.000 | n/a/n/a/0.000 | n/a/n/a/0.000 |
| Grocery/seed44/comp_shared | 1.000/1.000/0.000 | 1.000/1.000/0.000 | 1.000/1.000/0.000 |
| Grocery/seed44/comp_ownership | 0.077/0.091/0.119 | 0.227/0.234/0.295 | 0.284/0.261/0.334 |

## 12. Propagation/factor health

Entries are graph update ratio / cosine(H0,H2) / effective-rank change.

| Run | C update/cos/Δrank | Pt update/cos/Δrank | Pv update/cos/Δrank |
|---|---|---|---|
| Movies/seed42/comp_uniform | 0.3824/0.9518/-2.974 | 0.5430/0.8728/-1.333 | 0.5519/0.8747/-6.439 |
| Movies/seed42/comp_shared | 0.3721/0.9535/-2.798 | 0.5268/0.8786/-0.995 | 0.5269/0.8843/-4.849 |
| Movies/seed42/comp_ownership | 0.3564/0.9562/-2.877 | 0.5292/0.8801/-2.283 | 0.5588/0.8712/-7.315 |
| Movies/seed43/comp_uniform | 0.3843/0.9505/-2.887 | 0.5427/0.8730/-2.057 | 0.5629/0.8693/-7.416 |
| Movies/seed43/comp_shared | 0.3717/0.9509/-2.999 | 0.5210/0.8800/-1.396 | 0.5210/0.8854/-5.502 |
| Movies/seed43/comp_ownership | 0.4085/0.9412/-3.854 | 0.5382/0.8777/-1.836 | 0.5427/0.8766/-7.433 |
| Movies/seed44/comp_uniform | 0.3909/0.9487/-3.245 | 0.5366/0.8775/-1.129 | 0.5485/0.8755/-6.372 |
| Movies/seed44/comp_shared | 0.3577/0.9572/-2.593 | 0.5062/0.8885/-1.377 | 0.5235/0.8845/-6.474 |
| Movies/seed44/comp_ownership | 0.3676/0.9531/-3.080 | 0.5264/0.8821/-1.595 | 0.5502/0.8733/-6.989 |
| Toys/seed42/comp_uniform | 0.3287/0.9654/-2.445 | 0.4188/0.9280/-2.878 | 0.4637/0.9083/-4.955 |
| Toys/seed42/comp_shared | 0.3216/0.9681/-2.128 | 0.4487/0.9164/-4.718 | 0.4947/0.8955/-7.137 |
| Toys/seed42/comp_ownership | 0.3224/0.9687/-2.037 | 0.4274/0.9251/-3.984 | 0.4904/0.8982/-6.533 |
| Toys/seed43/comp_uniform | 0.3325/0.9640/-2.505 | 0.4500/0.9155/-5.001 | 0.4930/0.8959/-7.952 |
| Toys/seed43/comp_shared | 0.3493/0.9596/-2.879 | 0.4453/0.9190/-3.970 | 0.4823/0.9013/-6.080 |
| Toys/seed43/comp_ownership | 0.3322/0.9647/-2.438 | 0.4361/0.9203/-4.384 | 0.4913/0.8970/-7.095 |
| Toys/seed44/comp_uniform | 0.3358/0.9630/-2.708 | 0.4310/0.9225/-4.378 | 0.4728/0.9045/-6.611 |
| Toys/seed44/comp_shared | 0.3387/0.9638/-2.534 | 0.4349/0.9229/-4.100 | 0.4757/0.9045/-6.197 |
| Toys/seed44/comp_ownership | 0.3360/0.9648/-2.439 | 0.4231/0.9263/-3.780 | 0.4782/0.9037/-6.378 |
| Grocery/seed42/comp_uniform | 0.4057/0.9500/-4.034 | 0.5097/0.9061/-5.475 | 0.5668/0.8807/-7.573 |
| Grocery/seed42/comp_shared | 0.3661/0.9603/-3.007 | 0.4825/0.9139/-5.527 | 0.5411/0.8884/-7.865 |
| Grocery/seed42/comp_ownership | 0.3885/0.9536/-3.784 | 0.4853/0.9142/-4.729 | 0.5630/0.8774/-6.740 |
| Grocery/seed43/comp_uniform | 0.4153/0.9463/-4.314 | 0.4938/0.9133/-4.890 | 0.5465/0.8870/-7.114 |
| Grocery/seed43/comp_shared | 0.4028/0.9496/-4.116 | 0.4965/0.9112/-6.106 | 0.5477/0.8853/-7.829 |
| Grocery/seed43/comp_ownership | 0.4146/0.9466/-4.243 | 0.4845/0.9164/-4.806 | 0.5544/0.8828/-7.206 |
| Grocery/seed44/comp_uniform | 0.4152/0.9462/-4.148 | 0.4879/0.9151/-4.179 | 0.5295/0.8954/-6.431 |
| Grocery/seed44/comp_shared | 0.4212/0.9451/-4.173 | 0.5010/0.9106/-4.308 | 0.5376/0.8924/-6.687 |
| Grocery/seed44/comp_ownership | 0.4068/0.9482/-4.042 | 0.4835/0.9158/-4.798 | 0.5519/0.8845/-7.234 |

All formal runs were finite; no dynamic per-hop routing was used.

## 13. Available label-edge diagnostics

Offline only: train-induced and val-induced edges mean both endpoints are in the respective split. For each factor, the JSON artifact reports same-label mean weight, different-label mean weight, and AUC(weight→same-label). Test labels were excluded.

## 14. Movies MAP gap

The fixed reference is MAP-v2 `lowpass_uniform` Movies seed42 Val Acc 58.0684 / Macro-F1 49.86. The ORED-3A seed42 values and signed gaps are: `{
  "comp_uniform": {
    "val_acc_pct": 57.04858899116516,
    "val_macro_f1_pct": 47.16815496918626,
    "gap_vs_map_lowpass_uniform_acc_pp": -1.0197994361871423,
    "gap_vs_map_lowpass_uniform_macro_f1_pp": -2.691845030813738
  },
  "comp_shared": {
    "val_acc_pct": 56.29873871803284,
    "val_macro_f1_pct": 49.61331018221462,
    "gap_vs_map_lowpass_uniform_acc_pp": -1.7696497093194665,
    "gap_vs_map_lowpass_uniform_macro_f1_pp": -0.24668981778538068
  },
  "comp_ownership": {
    "val_acc_pct": 56.59868121147156,
    "val_macro_f1_pct": 49.14879418791159,
    "gap_vs_map_lowpass_uniform_acc_pp": -1.4697072158807458,
    "gap_vs_map_lowpass_uniform_macro_f1_pp": -0.7112058120884086
  }
}`. This is secondary and not a Gate A/B criterion.

## 15. Gate A — Does Composition help?

Best of Shared and Ownership against Uniform: `NO_GO_COMPOSITION`. Candidate and best-of metrics are recorded in `o3a_summary.json`; the thresholds are the registered +0.30pp macro Acc, at least 2/3 positive dataset means, and nonnegative macro-F1 for GO.

## 16. Gate B — Is Ownership conditioning necessary?

Ownership − Shared: `NO_GO_OWNERSHIP_COMPOSITION`. The registered strong gate requires macro ΔAcc≥+0.30pp, at least 2/3 positive dataset means, at least 6/9 positive paired seed Acc, and nonnegative macro-F1.

## 17. What is supported

- The ORED-2 ownership-preserving restart-diffusion parent remains a valid, finite matched-control implementation under the fixed Val-only protocol.
- The scorer learns non-uniform edge weights in the learned variants, and ownership-specific weights can differ by factor. These are diagnostic observations, not performance evidence.
- The formal results provide a paired comparison record for future work; they do not support promoting composition into the default core.

## 18. What is not supported

- Generic learned composition is **NOT SUPPORTED** as a stable validation-accuracy core gain: Gate A is `NO_GO_COMPOSITION`. The best candidate has `-0.057 pp` macro ΔAcc; its positive F1 signal is secondary and does not satisfy the Acc promotion band.
- Ownership-conditioned composition is **NOT SUPPORTED** as a core mechanism: Gate B is `NO_GO_OWNERSHIP_COMPOSITION`.
- This stage does not support Output Refinement as an independently useful mechanism; the registered ORED-2 finding remains `p0_refine` = `-0.28 Acc / -0.58 Macro-F1` versus P0.
- This report does not support Exposure, cross-factor transport, dynamic per-hop routing, test-time tuning, or any test-set claim.

## 19. Next-step recommendation

Proceed to an ownership-conditioned composition core only if Gate B is STRONG_GO; otherwise keep the composition result as diagnostic and do not enter Exposure×Composition FULL.

## 20. Exact commands

```bash
# Movies seed42 Val-only smoke (two GPUs may be used as separate shards)
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_o3a.py --gpu 0 --shard 0 --shards 2 --smoke
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_o3a.py --gpu 1 --shard 1 --shards 2 --smoke

# Formal 3 datasets × 3 seeds × 3 variants, Val-only
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_o3a.py --gpu 0 --shard 0 --shards 2
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_o3a.py --gpu 1 --shard 1 --shards 2

# Aggregate results and offline diagnostics
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/summarize_ored_o3a.py --device cuda:0
```

## 21. Scope

`NO TEST` · `NO EXPOSURE` · `NO CROSS-FACTOR TRANSPORT` · `NO DYNAMIC PER-HOP ROUTING` · `NO TUNING`.
