# MAP/MAP 统一训练与评估协议

协议版本：

- NC：`unified_full_graph_nc_v1`
- LP：`unified_sampled_lp_v1`

本文档冻结当前主实验协议。任何改变 split、消息图、负采样、训练模式、验证频率、早停或排名规则的实验，都必须使用新的协议名称，不能与本协议结果直接混报。

## 1. 共同规则

- 正式运行使用训练 seed `42/43/44`，同一数据集的所有模型使用相同 seed 集合。
- 多个训练 seed 共用同一份 split；seed 只改变初始化、dropout 和训练采样随机性。
- MM-Graph 数据集原样使用官方 split，不重排 query，不替换官方 valid/test negatives。
- 自建 MAGB split 只生成一次并跨模型复用；生成时按无向 pair 去重并隔离 train/valid/test，所有负例相对已知正边过滤。
- 每次运行记录 resolved config、split 路径和 split SHA-256。
- 模型选择只能使用 validation 指标。恢复最佳 validation checkpoint 后，test 只计算一次。
- 当前暂不统一或重新搜索学习率。任务默认值和现有模型级 `lr/weight_decay` 覆盖继续保留，相关比较限制必须在结果中披露。

## 2. 节点分类（NC）

- 所有模型采用 transductive full-graph training。
- 每个 epoch 对完整图执行一次 encoder forward，只在 train nodes 上计算交叉熵。
- validation/test 节点的特征和结构可以参与传播，但其标签不能进入训练。
- 主选择指标为 validation accuracy，同时报告 Accuracy 和 Macro-F1。
- 最大 300 epochs，每 epoch 验证一次；最早在 epoch 30 后停止，30 次验证无显著提升则早停。
- 显著提升阈值为 `1e-4`，gradient clipping max norm 为 `1.0`。
- 最佳 checkpoint 恢复后只执行一次 test。

## 3. 链路预测（LP）

### 3.1 数据与消息图

- message graph 只由 train positives 构成，不包含 validation/test positives。
- 消息图按数据集配置对称化，但官方有向 validation/test query 及其逐行固定 negatives 保持不变。
- 训练负例相对 train/valid/test 的全部已知正 pair 过滤，并排除 self-loop。

### 3.2 训练

- 所有图编码器统一使用 PyG `LinkNeighborLoader` sampled training。
- 采样参数固定为两跳 `[5, 5]`、`subgraph_type=bidirectional`、batch size 2048。
- 每个 epoch 使用全部训练正边；每个正例配一个当 epoch 重新采样的 filtered negative。
- 每个 sampled batch 在前向前删除当前正监督边的两个消息方向，避免目标边泄漏。
- 删除后端固定为 `global_eid`：在全局 train-only message graph 上预建
  `directed edge code -> e_id` 查询表，通过 `batch.e_id` 和全局 boolean scratch
  在 CPU 上删除正边，再把 batch 传入 GPU。查询覆盖重复边，并保持剩余边顺序不变。
- `local_keys` 旧后端仅保留用于实现等价性审计，不用于正式实验。
- MLP 不使用图采样，但使用完全相同的当 epoch 正负监督集合和 batch shuffle 规则。
- 全局模型（包括 DiP、MAP-MAG-v1/v2/v3）在本协议下属于 sampled adaptation；模型配置中的 native full-graph 偏好不会改变 LP 训练模式。
- 负采样、batch shuffle、neighbor sampling 使用从 run seed 派生的独立随机流，默认 offset 分别为 10000、20000、30000。
- 最大 150 epochs，每 2 epochs 验证一次；最早在 epoch 20 后停止，10 次验证无显著提升则早停。
- 显著提升阈值为 MRR `1e-4`，gradient clipping max norm 为 `1.0`。

### 3.3 共享预测头

- 所有 encoder 输出先投影至 128 维。
- 使用相同的三层 Hadamard-product MLP scorer：hidden 256、dropout 0.02。
- 主任务损失为 1:1 balanced `BCEWithLogitsLoss`。
- 模型固有辅助损失可以保留，但必须只使用训练数据并在配置和结果中披露。

### 3.4 验证与测试

- encoder 进入 `eval()`，在完整 train-only message graph 上生成所有节点 embedding。
- 使用 split 中固定的 `target_node_neg`；不重新采样验证或测试 negatives。
- 候选边可以分块打分，分块大小只属于内存实现参数，不改变指标定义。
- 排名采用 pessimistic tie：`rank = 1 + count(negative_score >= positive_score)`。
- validation 主指标为 MRR，同时报告 Hits@1、Hits@3 和 Hits@10。
- 按 validation MRR 保存最佳 checkpoint；训练结束或早停后恢复该 checkpoint，只运行一次 test。
- full forward 与经过等价性测试的 exact layerwise/chunked inference 均可使用；随机 sampled inference 不属于正式协议。

## 4. 正式结果报告

- 报告三个 seed 的 mean、population standard deviation 和逐 seed 数值。
- 报告模型参数量、最佳 epoch、训练时间、峰值显存、协议版本和完整 resolved config。
- 不把 OpenMAG 原始协议、DiP 官方 native full-graph 协议、严格清洗 split 或开发期子采样结果与本协议主表混报。

## 5. 允许的开发期加速

以下设置只能用于 smoke test 或调试，结果不能进入主表：

- `train_pos_per_epoch` 只取部分训练正边；
- `max_train_batches` 限制每 epoch batch 数；
- 限制 validation query 数；
- 减小 fanout 或使用 sampled validation。

不改变数学协议的正式加速包括：多 worker/prefetch、复用静态 PyG graph storage、候选边分块、避免重复 source embedding、混合精度（通过等价性和数值稳定性验证后）以及 exact chunked inference。
