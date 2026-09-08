# LP Benchmark Results

## 实验设置

- 数据集：`sports-copurchase`
- 模型：13 个实现模型
- Seed：`42`
- LP 协议：`unified_sampled_lp_v1`
- 训练方式：sampled link prediction
- Neighbor sampling：两跳 `[5, 5]`
- Training negative：每条正边 1 个 filtered negative
- LP projection dimension：`128`
- Inference：full-graph exact inference
- 表中数值：百分比（`results.json` 中的 `[0, 1]` 数值乘以 100）
- 本实验为单 seed，因此没有跨 seed 方差；表中记录 seed 42 的结果
- 原始结果目录：`outputs/full_benchmark/lp/sports-copurchase/`

本次共完成 `1 × 13 = 13` 个 LP benchmark jobs。结果由各模型目录下的
`seed42_runs1/results.json` 汇总而来。`map_mag_v3` 的 LP 运行使用
`map_mag_v3_lp` 配置 preset。

## 结果总览

| 指标 | 最优模型 | 结果 |
|---|---|---:|
| Validation MRR | `map_mag_v3` | **40.4279** |
| Test MRR | `map_mag_v3` | **37.3798** |
| Test Hits@1 | `map_mag_v3` | **21.2840** |
| Test Hits@3 | `map_mag_v3` | **43.0252** |
| Test Hits@10 | `dip` | **73.6295** |

## 详细结果

| Model | Val MRR | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|
| mlp | 27.0150 | 25.0620 | 11.8889 | 26.3224 | 55.3711 |
| gcn | 33.0181 | 30.8291 | 17.1331 | 33.3921 | 61.7833 |
| sage | 35.0372 | 32.5724 | 16.9807 | 36.5301 | 69.0375 |
| mmgcn | 38.1364 | 35.0268 | 18.7582 | 40.0663 | 72.7768 |
| mgat | 35.4767 | 32.5305 | 16.8685 | 36.4419 | 69.4064 |
| dip | 39.1335 | 35.7429 | 19.2500 | 41.2584 | **73.6295** |
| dgf | 37.6213 | 34.4381 | 18.6887 | 38.9651 | 70.9459 |
| dmgc | 16.6131 | 15.7282 | 5.6772 | 14.9253 | 37.2170 |
| lgmrec | 35.3194 | 32.7383 | 18.2584 | 36.2975 | 65.3703 |
| map_mag | 37.7181 | 35.0568 | 19.7231 | 39.7268 | 69.7351 |
| map_mag_v1 | 37.7321 | 34.8233 | 19.1564 | 39.7616 | 70.0612 |
| map_mag_v2 | 38.5795 | 35.7234 | 20.2015 | 40.6009 | 70.8123 |
| map_mag_v3 | **40.4279** | **37.3798** | **21.2840** | **43.0252** | 73.6241 |

## 结果路径

- [LP benchmark outputs](../outputs/full_benchmark/lp/sports-copurchase/)
- [Benchmark manifest](../outputs/full_benchmark/benchmark_manifest.json)
