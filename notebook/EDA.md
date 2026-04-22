# OpenOneRec Dataset EDA Design Document

## 研究背景与动机

### 项目方向
本项目基于 OpenOneRec 开源框架，在 DPO（Direct Preference Optimization）数据构建方式上进行改进。OpenOneRec 原始论文（IPA 模块）通过训练一个独立的 Reward Model 来为 beam search 生成的候选 session 打分，进而构建 (chosen, rejected) preference pair，再用 DPO loss 优化模型。

我们的核心研究问题是：

> **在离线推荐场景下，能否用用户行为信号直接替代 Reward Model 来构建 DPO preference pair？不同粒度的行为信号组合对模型偏好对齐效果的影响是什么？**

### 为什么原作者没有这么做
OpenOneRec 的工业在线场景存在时序约束——用户行为（like、follow）发生在推荐之后，训练时无法提前获知，因此必须依赖 Reward Model 提前预测。而在我们的**离线学术场景**下，所有行为信号均已记录在数据集中，时序约束不存在，因此可以直接用行为信号构建 preference pair，实现真正无额外模型的偏好对齐。

### 创新点
用规则化的行为信号 Reward Function 替代 Reward Model，系统分析以下四组信号粒度对 DPO 效果的影响：

| 实验组 | 信号构成 |
|--------|---------|
| Baseline | OneRec 原始 Reward Model |
| 实验1 | 只用 longview（纯 implicit） |
| 实验2 | longview + like + follow（implicit + explicit 正向） |
| 实验3 | 实验2 + not_interested 负向惩罚 |
| 实验4 | 只用 explicit（like + follow + not_interested） |

---

## EDA 总体设计

EDA 分为两大部分，由三个独立脚本完成：

```
Part 1: 数据健康度检查（能不能用）
  └── Script 1: eda_data_health.py

Part 2: 研究动机验证（值不值得这么做）
  ├── Script 2: eda_behavior_signals.py
  └── Script 3: eda_dpo_feasibility.py
```

每个脚本运行结束后生成一份 `report.txt` 文件，总结所有发现。

---

## Script 1: eda_data_health.py

**目标**：验证数据基础质量，确认数据集可用于训练。

对应 PDF 中的"主表基础健康度"和"映射表 EDA"部分。

### 分析维度

**1. 主表基础信息**
- 总行数、字段列表、数据类型
- `split` 字段分布（split=0 的占比，确认训练集规模）

**2. 序列长度分布**
- `hist_video_pid`：用户历史视频序列长度
  - 均值、中位数、P25、P75、P95、P99
  - 长度分布直方图统计
  - 空值率
- `target_video_pid`：目标视频序列长度
  - 同上

**3. 字段空值率**
- 对所有关键字段统计空值/空列表比例
- 重点关注训练所需字段的缺失情况

**4. 映射表健康度**
- `video_ad_pid2sid.parquet`
  - 总 pid 数量
  - sid 长度是否恒定为 3
  - 是否一对一映射（pid 唯一性）
- `pid2caption.parquet`
  - caption 覆盖率（与主表 pid 的 join 比例）
  - caption 文本长度分布（均值、中位数、P95）
  - 空 caption 比例
  - 字段名核验（caption vs dense_caption）

**5. PID 覆盖率**
- 主表中出现的 pid，能在 pid2sid 中找到映射的比例
- 主表中出现的 pid，能在 pid2caption 中找到文本的比例

### 输出
- `eda_data_health_report.txt`：包含所有统计数值和关键发现

---

## Script 2: eda_behavior_signals.py

**目标**：分析各行为信号的分布特征，验证用行为信号替代 Reward Model 的可行性。

这是针对我们研究问题新增的核心分析。

### 分析维度

**1. 各信号覆盖率**

对以下 5 个信号字段逐一分析：
- `hist_video_longview`（完播，implicit）
- `hist_video_like`（点赞，explicit 正向）
- `hist_video_follow`（关注，explicit 强正向）
- `hist_video_forward`（转发，explicit 正向）
- `hist_video_not_interested`（不感兴趣，explicit 负向）

每个信号统计：
- 非空用户比例（覆盖率）
- 每用户平均信号数量
- 信号数量分布（均值、中位数、P95）

**2. 信号稀疏性对比**
- 五个信号的覆盖率排序对比（预期：longview > like > follow ≈ forward > not_interested）
- 可视化为汇总表格（输出到 report）

**3. 信号共现分析**
- 有 `like` 的用户中，同时有 `longview` 的比例
- 有 `follow` 的用户中，同时有 `longview` 的比例
- 有 `not_interested` 的用户中，同时有 `longview` 的比例
- 验证信号之间的语义一致性（关注的视频是否一定看完了）

**4. 信号语义验证**
- 对同一个用户，分析有 `like` 的视频和无 `like` 的视频在 `longview` 上的差异
- 验证 explicit 信号确实代表比 implicit 信号更强的偏好

### 输出
- `eda_behavior_signals_report.txt`：包含覆盖率统计、共现矩阵、语义一致性分析

---

## Script 3: eda_dpo_feasibility.py

**目标**：分析在不同信号粒度下，DPO preference pair 能构建多少，评估各实验组的数据规模。

### 分析维度

**1. 各实验组 Pair 构建可行性**

模拟四个实验组的 pair 构建逻辑，统计每组能构建 pair 的用户数量和 pair 总数：

```
实验1（纯 implicit）:
  chosen  = longview 数量最多的 session
  rejected = longview 数量最少的 session
  可构建条件：用户有足够的历史记录

实验2（implicit + explicit 正向）:
  reward = longview*1.0 + like*1.5 + follow*2.0
  chosen/rejected 按 reward 排序
  可构建条件：用户至少有1个 like 或 follow 信号

实验3（完整信号）:
  reward = 实验2 + not_interested*(-2.0)
  可构建条件：用户同时有正向和负向信号

实验4（纯 explicit）:
  reward = like*1.5 + follow*2.0 - not_interested*2.0
  可构建条件：用户有 explicit 信号
```

**2. 数据规模对比**
- 各实验组可构建的用户数量
- 各实验组可构建的 pair 总数
- 相比 split=0 全量的覆盖比例

**3. Pair 质量分析**
- chosen 和 rejected session 的 reward score 差距分布
- 差距过小的 pair 比例（可能是噪声 pair，需要过滤）
- 建议的过滤阈值

**4. 用户行为丰富度分层**
- 按历史序列长度将用户分层（短：<20，中：20-100，长：>100）
- 分析不同层的信号覆盖率差异
- 为后续实验的用户过滤策略提供依据

### 输出
- `eda_dpo_feasibility_report.txt`：包含各实验组数据规模、pair 质量分析、过滤建议

---

## EDA 整体叙事结构

三个脚本的发现共同支撑以下研究叙事：

```
Script 1 发现：
"数据质量良好，split=0 包含 X 条样本，
 序列长度中位数为 Y，pid2sid 覆盖率 Z%，数据可用于训练"
        ↓
Script 2 发现：
"longview 覆盖率最高（A%），like/follow 覆盖率中等（B%/C%），
 not_interested 最稀疏（D%）。信号之间语义一致，
 有 follow 的视频中 E% 同时有 longview，支持信号粒度假设"
        ↓
Script 3 发现：
"实验1可构建 F 个 pair，实验3因需要同时有正负信号仅能构建 G 个 pair，
 建议对 reward score 差距 < H 的 pair 进行过滤"
        ↓
结论：
"行为信号覆盖率和语义一致性支持我们的研究方向，
 各实验组数据规模足够，实验设计可行"
```

---

## 数据文件路径约定

```
raw_data/
├── onerec_bench_release.parquet   # 主训练母表
├── video_ad_pid2sid.parquet       # 视频/广告 pid → semantic ID
├── product_pid2sid.parquet        # 商品 pid → semantic ID
└── pid2caption.parquet            # pid → 文本描述

outputs/
├── eda_data_health_report.txt
├── eda_behavior_signals_report.txt
└── eda_dpo_feasibility_report.txt
```

---

## 注意事项

1. **字段名核验**：`pid2caption.parquet` 的字段名需在运行前确认是 `caption` 还是 `dense_caption`，脚本中加入自动检测逻辑
2. **只使用 split=0**：所有分析仅针对训练子集，避免 benchmark 数据泄漏
3. **最小可行集**：本次 EDA 聚焦 video 推荐任务（`hist_video_*` 和 `target_video_pid`），不涉及 ad 和 product 任务

---

## EDA 跑出来后的关键发现（2026-04-07 更新）

### Script 1 实测
- `split=0` = 162,074 行 = 全量（HF 这版根本没有 split≠0 的样本，`split` 字段冗余）
- `hist_video_pid` 长度 P50/P95 = 484/508，已被官方截断到 512，**K 不需要我们决定**
- `target_video_pid` 长度 P50 = 9，max = 10
- 5,829 个用户 (3.60%) target/hist 全空，**有效用户 = 156,245**
- 6 个 hist/target pid 列在 pid2sid 映射表的覆盖率均为 **100.0000%**，无样本被静默丢弃
- 5 个 hist + 5 个 target behavior 列与对应 pid 列**长度严格对齐**（0 mismatch），DPO reward 计算的硬前提满足
- 映射表非严格 1-to-1：`video_ad_pid2sid` 有 2,658 个重复 pid (0.017%)，`product_pid2sid` 有 1 个 — 拼 sid 前必须 `drop_duplicates(subset='pid')`
- `reco_*` 三列 97.75% 空（rec_reason 任务专用），`inter_*` 三列 45.87% 空（interactive_rec 专用）— 我们 DPO 实验都不需要

### Script 2 实测
- 信号稀疏度阶梯（hist 端）：longview (96.40%) > like (92.86%) > forward (57.57%) > follow (39.37%) > **not_interested (6.53%)**
- Item 级正例率（hist 端，~7400 万 item-观测）：longview 26.65%, like 7.19%, forward 0.59%, follow 0.26%, **not_interested 0.07%**
- **语义一致性假设全部成立** ✅：P(longview | sig=1) / P(longview | sig=0) lift：
  - like: 1.36
  - **follow: 1.88**
  - forward: 1.87
  - **not_interested: 0.66**（确实是负信号）
- **重大发现**：target 端 not_interested **极度稀疏**（用户覆盖率 0.34%，item 正例率 0.06%），整个数据集只有 555 个用户、853 个 not_interested item

### Script 3 实测（pair 构造可行性）
- Pair 构造策略：m = L // 2（L=10 → m=5），eligibility 要求 target 长度 ≥ 8
- **134,210 用户**符合 eligibility（82.81%）
- 各 arm 在 gap > 1.5 阈值下的 pair 数：
  - **Arm 1** (longview only): 91,226
  - **Arm 2** (longview + like + follow + forward): **105,383** ← 主推
  - **Arm 3** (Arm 2 + not_interested penalty): 105,479
  - **Arm 4** (pure explicit): 29,021
- **重大发现 1**：Arm 2 vs Arm 3 的 chosen subset 一致率 = **99.83%**，Arm 3 几乎完全等价于 Arm 2。原因：not_interested 在 134K eligible 用户里只有 516 个 (0.38%) 触发 penalty。**Arm 3 ablation effect size 上限 ≈ 500 个样本**。
- **重大发现 2**：Arm 4 有 **59.47% zero-gap pair**（target 里完全没有 like/follow/forward 时整个公式输出 0），Arm 4 实际可用 ~29K，质量低于 Arm 1/2/3
- **结论**：原 EDA.md 4-arm 设计在数据上不完全成立。Arm 3 由"硬性需要正负组合"重定义为"Arm 2 + soft penalty"后可以训，但 vs Arm 2 几乎无差异

---

## 实验设计调整（基于 EDA 实测）

**放弃 4-arm 完整消融**，用单一配置直接构建主训练集：

- **Reward 公式**：Arm 2 = `1.0·longview + 1.5·like + 2.0·follow + 1.5·forward`
- **Pair 构造策略**：每个用户的 target (10 item) 内部 split top-5 / bot-5
  - chosen = score 最高的 5 个 item（按原 target 时间顺序重排）
  - rejected = score 最低的 5 个 item（按原 target 时间顺序重排）
- **过滤阈值**：`gap > 1.5`
- **样本量**：105,383 pair（每用户 1 pair），与 UltraFeedback / Zephyr-DPO 同量级
- **train/valid split**：按 uid hash 95/5

**为什么放弃消融**：
1. Arm 3 vs Arm 2 effect size 上限 ~500 样本，没有意义
2. Arm 4 zero-gap 60%，质量太低
3. Arm 1 阶梯化分布（gap 只能取 0/1/2/3/4/5），训练信号弱
4. Arm 2 是最干净、最丰富的配置，直接当主实验

**Arm 3 vs Arm 2 的 0.38% 差异是 paper 级发现**，可作为 negative result 写进讨论：
> 在工业级推荐数据上，显式负反馈信号（not_interested）稀疏到 0.38%，使得显式负惩罚在 DPO 偏好对齐中实质上无法生效。这从经验上证伪了"显式负样本是 RLHF 必需"的常见假设。

---

## DPO 数据集 Schema（build_dpo_dataset.py 输出）

```
data/dpo_dataset/
├── train.parquet     ~100K rows (95%)
├── valid.parquet     ~5K rows (5%)
└── meta.json         构造参数 + 统计信息
```

每行 schema：

```python
{
    "uid": int64,                                # 原始 user id（trace/debug）

    # ─── 上下文（encoder 输入，chosen/rejected 共享）───
    "history_pid":  list<int64>,                 # 长度 ≤512
    "history_sid":  list<list<int64>>,           # 长度 ≤512，每 item = 3-token sid

    # ─── DPO chosen ───
    "chosen_pid":   list<int64>,                 # 长度 5
    "chosen_sid":   list<list<int64>>,           # 5 × 3 = 15 tokens
    "chosen_score": float32,                     # session reward (sum of item scores)

    # ─── DPO rejected ───
    "rejected_pid":   list<int64>,
    "rejected_sid":   list<list<int64>>,
    "rejected_score": float32,

    # ─── 元数据 ───
    "gap":         float32,                      # chosen_score - rejected_score (>1.5)
    "target_len":  int16,                        # 8/9/10
    "m":           int8,                         # = target_len // 2
}
```

**为什么同时存 pid 和 sid**：sid 是模型实际吃的 token（训练用）；pid 用于 case study / 错误分析 / join 回 caption。多存 pid 文件大小只增加 ~10%。

**关键实现要点**：
1. chosen / rejected 内部**按时间顺序排列**（不是按 score 排）— DPO loss 对 token 序列顺序敏感，必须保持 OneRec 数据集原始的时间序
2. 拼 sid 前 `pid2sid_df.drop_duplicates(subset='pid')`
3. hist 和 target 由 dataset 时间分割保证 hist 全部早于 target，无需额外验证；但记录 hist/target pid overlap 率作为 sanity stat
4. 文件大小估计：未压缩 ~1.7 GB，parquet snappy ≈ 400-700 MB

### Known limitation：跨 item 时序因果被打破

top/bot-m 切法保留了 chosen 和 rejected **内部**的 per-item 时序，但**破坏了跨 item 的因果链**。例如 target = [t1..t10]，reward 选出 chosen = [t1, t6, t7, t8, t9]：在原始 log 里 t6..t9 的出现是因为用户先看了 t2..t5，推荐系统当场调整了后续推荐。重排之后 decoder 被要求最大化 `P(t6 | history, t1) · P(t7 | history, t1, t6) · ...`，这些条件概率**在真实世界里从未出现过**——chosen 序列整体是一个合成重排，不是 logged trajectory。

**为什么我们接受这个 limitation**：

1. **DPO 学的是偏好方向，不是绝对似然**。Anthropic HH-RLHF / UltraFeedback / OneRec IPA 的 chosen 都不是 base model verbatim 出现过的序列，DPO 经验上对 off-policy chosen 鲁棒
2. **强 base model prior 可以吸收噪声**：OneRec-1.7B 在 96M interactions 上预训练过，105K pair 的 DPO fine-tune 不太可能反转它已经学到的 token 共现 prior
3. **替代方案更差**：顺序切（`target[:5]` vs `target[5:]`）保留时序但 expected gap ≈ 0，没有偏好信号；滑动窗口切法 pair 数从 105K 降到 30-50K，且窗口级别仍有 off-policy 问题
4. **真正干净的做法是 on-policy DPO**（base model beam-search 出 candidate，用 reward 打分当 chosen/rejected）—— 但这丢掉了"用真实行为信号"的卖点，且推理成本巨大。作为 future work

**我们做的 mitigation**：
- chosen/rejected 内部按原 target 时间重排，至少保留 intra-sequence local order
- 训练后做 sanity check：在 OneRec-1.7B base 下计算 `logπ_ref(chosen)` 平均值 vs `logπ_ref(rejected)` 平均值。如果 chosen 显著低，说明 off-policy shift 严重；如果相近，说明 prior 起作用了

**Paper framing**：在 Limitations 段明确披露——我们用 distributional purity 换 behavioral grounding，logged user signal 是一个真实、可解释的监督源，代价是合成的 preference 序列。

---

## 训练阶段计划

**Base model**：[`OpenOneRec/OneRec-1.7B`](https://huggingface.co/OpenOneRec/OneRec-1.7B)
- 4.29 GB，公开非 gated
- **已经做完 SFT** 的 Standard 版（基于 Qwen3-1.7B + Itemic-Text Alignment + Co-Pretraining + Multi-task SFT）
- 与开源 `onerec_bench_release` 数据完全对齐
- 不下载 Pro 版（含 Kuaishou 内部数据，distribution 不一致，作为 baseline 不干净）
- 不下载 pretrain-only 版（需要自己做 SFT，工作量大且无必要）

**完整 pipeline**：

```
1. EDA (DONE: Scripts 1/2/3)
2. build_dpo_dataset.py            ← 下一步
   → data/dpo_dataset/{train,valid}.parquet
3. 下载 OneRec-1.7B (4.3 GB)
4. DPO 训练
   - trl DPOTrainer 或 OpenOneRec 仓库 RL 脚本
   - 单卡 A100 40GB / A6000 48GB 足够
   - 105K pair × 3 epoch ≈ 4-12 小时
5. Evaluation
   - benchmark_data/video/video_test.parquet (38,781 samples)
   - 主指标：Recall@10, Pass@32, Pass@1
   - Baseline: 未 DPO 的 OneRec-1.7B 直接 inference
6. (Optional) 论文里加 Arm 3 vs Arm 2 的 negative ablation 一段
```

