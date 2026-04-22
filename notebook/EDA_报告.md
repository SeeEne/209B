# OpenOneRec 数据集 EDA 结果汇报

> **项目**：CS 1090a — 用户行为信号驱动的 DPO 偏好对齐（基于 OpenOneRec）
> **数据**：`OpenOneRec/OpenOneRec-RecIF` 最小可行子集
> **完成日期**：2026-04-07
> **EDA 脚本**：[`notebook/eda_data_health.py`](eda_data_health.py)、[`notebook/eda_behavior_signals.py`](eda_behavior_signals.py)、[`notebook/eda_dpo_feasibility.py`](eda_dpo_feasibility.py)
> **原始报告**：[`outputs/eda_data_health_report.txt`](outputs/eda_data_health_report.txt)、[`outputs/eda_behavior_signals_report.txt`](outputs/eda_behavior_signals_report.txt)、[`outputs/eda_dpo_feasibility_report.txt`](outputs/eda_dpo_feasibility_report.txt)

---

## TL;DR（一分钟看完）

1. **数据非常干净**：162,074 行 = 全部训练池，PID 映射 100% 命中，behavior 标签和 pid 长度严格对齐，无脏数据
2. **行为信号语义假设全部成立**：explicit 正向信号（like/follow/forward）能显著提升 longview 概率，not_interested 是真负信号
3. **重大发现**：target 端 not_interested 极度稀疏（0.06% item 正例率），原计划的 4-arm 消融实验不可行
4. **决定**：放弃 4-arm 消融，用单一 Arm 2 配置 + `gap > 1.5` 过滤构建训练集
5. **数据规模**：**105,383** 个 (chosen, rejected) pair，与 Zephyr-DPO / UltraFeedback 同量级
6. **下一步**：写 `build_dpo_dataset.py` → 下载 `OpenOneRec/OneRec-1.7B`（已 SFT 的 4.3 GB checkpoint） → trl DPOTrainer 训练 → benchmark_data/video/ 评测

---

## 1. 数据集全貌

### 1.1 核心理解

OpenOneRec 提供的不是原始 impression log，而是**已经按用户聚合好的 wide table**。每一行 = 一个用户的完整行为切片，已经预先切好 history / target：

```
hist_video_pid    = [v1, v2, ..., v484]   ← 上下文，喂 encoder
target_video_pid  = [v485, ..., v494]     ← 待预测，10 个 item
```

模型任务：**给定 484 个历史视频，autoregressive 生成接下来 5 个推荐**（target=10 是 dataset 给的，我们 split 成 chosen 5 个 + rejected 5 个）。

### 1.2 数据规模

| 指标 | 数值 |
|---|---|
| 总行数 | 162,074 |
| 唯一用户 | 162,074（每行一个用户） |
| 列数 | 25 |
| `split=0` 占比 | **100%**（HF 这版根本没有 split≠0 的样本） |
| 有效用户（hist/target 非空） | **156,245**（96.40%） |
| 空 video 用户 | 5,829（3.60%） |

### 1.3 25 列分类

| 类别 | 列 | 我们用吗 |
|---|---|---|
| **主键** | `uid`, `split` | ✅ |
| **视频域**（核心） | `hist_video_pid` + 5 个 `hist_video_<behavior>` + `target_video_pid` + 5 个 `target_video_<behavior>` | ✅✅✅ |
| **广告域** | `hist_ad_pid`, `target_ad_pid` | ❌ |
| **商品域** | `hist_goods_pid`, `target_goods_pid` | ❌ |
| **辅助 video** | `hist_longview_video_list` | ❌ |
| **interactive 任务** | `inter_keyword_to_items`, `inter_user_profile_with_pid`, `inter_user_profile_with_sid` | ❌（45.87% 空） |
| **rec_reason 任务** | `reco_gsu_caption`, `reco_target_caption`, `reco_cot` | ❌（97.75% 空） |

**核心理解**：5 个 `hist_video_<behavior>` 列是**和 `hist_video_pid` 长度一致的并行数组**，而不是聚合标量。例如 `hist_video_pid = [v1, v2, v3]`、`hist_video_like = [0, 1, 0]` 表示用户看了 v1/v2/v3，只对 v2 点了赞。

### 1.4 三张映射表

| 文件 | 行数 | 用途 |
|---|---|---|
| `video_ad_pid2sid.parquet` | 15,885,203 | 视频/广告 pid → sid（3-token 元组） |
| `product_pid2sid.parquet` | 2,066,115 | 商品 pid → sid |
| `pid2caption.parquet` | 未下载 | item 文本（item_understand 任务用） |

每个 sid = 3 个 int64 token，类似 (大类, 子类, 实例)。模型不直接吃 pid，吃 sid。

---

## 2. Script 1 — 数据健康度

### 2.1 序列长度分布（关键）

| 列 | P50 | P95 | P99 | max | 空率 |
|---|---|---|---|---|---|
| `hist_video_pid` | 484 | 508 | 511 | **512** | 3.60% |
| `target_video_pid` | 9 | 10 | 10 | **10** | 3.60% |

**关键事实**：
- `hist_video_pid` 已经被 OneRec 官方截断到 512，**K 不需要我们决定**
- `target_video_pid` 长度 P99 = 10，几乎所有用户都是满 10 个 item
- 21% 用户 target 长度 < 10（最小到 1），我们后续要求 `target_len >= 8` 时会丢掉一部分

### 2.2 PID 覆盖率（关键）

任何在主表里但不在映射表里的 pid 都会被 SFT 脚本静默丢弃。检查结果：

| 列 | pid 总数 | 在映射表中 | 覆盖率 |
|---|---|---|---|
| `hist_video_pid` | 74,276,520 | 74,276,520 | **100.0000%** |
| `target_video_pid` | 1,394,622 | 1,394,622 | **100.0000%** |
| `hist_ad_pid` | 3,613,364 | 3,613,364 | **100.0000%** |
| `target_ad_pid` | 660,359 | 660,359 | **100.0000%** |
| `hist_goods_pid` | 15,352,929 | 15,352,929 | **100.0000%** |
| `target_goods_pid` | 778,098 | 778,098 | **100.0000%** |

**所有 pid 列 100% 命中映射表**——零样本会被丢弃。✅

### 2.3 Behavior 标签 / pid 长度对齐（DPO 硬前提）

5 个 `hist_video_<behavior>` 列必须和 `hist_video_pid` 长度一致，否则 reward 计算会错位。检查结果：

| 列对 | mismatch 行数 |
|---|---|
| `hist_video_pid` ↔ 5 个 hist_video_behavior | **0** |
| `target_video_pid` ↔ 5 个 target_video_behavior | **0** |

**完全对齐，DPO reward 计算的硬前提满足**。✅

### 2.4 字段空值率（决定哪些任务能做）

| 字段 | 空率 | 含义 |
|---|---|---|
| `reco_*`（3 列） | **97.75%** | rec_reason 任务专用，我们不用 |
| `inter_*`（3 列） | 45.87% | interactive_rec 任务专用，我们不用 |
| `hist_ad_pid` | 31.60% | ad 域，~30% 用户没广告行为 |
| `hist_goods_pid` | 30.15% | product 域 |
| `hist_video_*`（6 列） | 3.60% | 全部和 `hist_video_pid` 同步空 |
| `target_video_*`（6 列） | 3.60% | 同上 |

### 2.5 映射表健康度

| 表 | 行数 | 唯一 pid 数 | 严格 1-to-1 | sid 长度恒为 3 |
|---|---|---|---|---|
| `video_ad_pid2sid` | 15,885,203 | 15,882,545 | ❌（2,658 个重复，0.017%） | ✅ |
| `product_pid2sid` | 2,066,115 | 2,066,114 | ❌（1 个重复） | ✅ |

⚠️ **构建数据集时必须 `df.drop_duplicates(subset='pid')` 否则 `set_index('pid')` 会报错**。

### 2.6 一句话总结

**数据质量极佳**：156K 有效用户，PID 100% 覆盖，behavior 与 pid 长度 0 mismatch，可以直接进入下一步分析。唯一的小坑是映射表有 0.017% 的重复 pid，构建时 drop 一下即可。

---

## 3. Script 2 — 行为信号分析

### 3.1 信号稀疏度阶梯

#### Hist 端（每用户约 484 个 item，全数据 ~7400 万 item-观测）

| Signal | 用户覆盖率 | item 正例率 | 性质 |
|---|---|---|---|
| longview | **96.40%** | **26.65%** | implicit，密集 |
| like | 92.86% | 7.19% | explicit 正向，半密集 |
| forward | 57.57% | 0.59% | explicit 正向，稀疏 |
| follow | 39.37% | 0.26% | explicit 强正向，稀疏 |
| not_interested | **6.53%** | **0.07%** | explicit 负向，**极稀疏** |

**直觉解读**：用户看 100 个视频，平均完播 27 个，点赞 7 个，关注 0.26 个，标"不感兴趣"0.07 个。信号稀疏度阶梯非常陡峭。

#### Target 端（每用户约 9 个 item，决定 DPO 实验可行性）

| Signal | 用户覆盖率 | item 正例率 |
|---|---|---|
| longview | 79.05% | 29.01% |
| like | 37.46% | 9.38% |
| forward | 4.65% | 0.69% |
| follow | 1.92% | 0.26% |
| **not_interested** | **0.34%** | **0.06%** |

⚠️ **整个数据集 156K 用户里，target 端含 not_interested 的用户只有 555 个 / item 853 个**。这是后面 4-arm 实验设计崩塌的直接原因。

### 3.2 语义一致性验证（核心假设）

我们假设：explicit 正向信号 → 视频更可能被完播；not_interested → 视频更不可能被完播。验证方法：在所有 item 上计算 lift = `P(longview | sig=1) / P(longview | sig=0)`。

#### Hist 端结果

| 条件 | P(longview\|sig=1) | P(longview\|sig=0) | **Lift** |
|---|---|---|---|
| like=1 | 35.25% | 25.98% | **1.36** ✅ |
| **follow=1** | **50.00%** | 26.59% | **1.88** ✅ |
| forward=1 | 49.57% | 26.51% | **1.87** ✅ |
| **not_interested=1** | **17.53%** | 26.66% | **0.66** ✅（负信号成立） |

#### Target 端结果（基本一致）

| 条件 | Lift |
|---|---|
| like | 1.16 |
| follow | **1.89** |
| forward | 1.50 |
| not_interested | **0.53** |

**结论**：**研究假设全部成立**。
- explicit 正向信号确实代表比 implicit 更强的偏好（lift 1.36–1.88）
- follow 是最强正信号（lift 1.88–1.89）
- not_interested 是真负信号（lift 0.53–0.66），不是噪声
- **数据级别支持"用行为信号替代 Reward Model"的核心假设**

### 3.3 一句话总结

行为信号语义非常清晰，5 种信号有明显的强弱阶梯，研究方向有数据基础支持。**但 not_interested 在 target 端只有 0.06% 的 item 正例率，这是一个严重的数据约束**。

---

## 4. Script 3 — DPO Pair 可行性

### 4.1 Pair 构造算法

```
对每个用户:
  1. 取 target_video_pid 里的 10 个 item
  2. 用 reward 公式给每个 item 算 score
  3. 按 score 排序：top-5 = chosen, bot-5 = rejected
  4. 计算 gap = sum(chosen scores) - sum(rejected scores)
  5. 如果 gap > 阈值，保留这条 pair
```

**关键数学事实**：top-5 by per-item score = max sum 5-subset。这等价于在所有 C(10,5)=252 种切法里枚举 gap 最大的那个，但只需 O(L log L)。

**Eligibility**：要求 target 长度 ≥ 8（即 m ≥ 4），共 134,210 个用户符合条件（占有效用户的 86%）。

### 4.2 4 个实验组的 reward 公式

| Arm | 公式 |
|---|---|
| **Arm 1** | `1.0 · longview` |
| **Arm 2** | `1.0 · longview + 1.5 · like + 2.0 · follow + 1.5 · forward` |
| **Arm 3** | Arm 2 + `−2.0 · not_interested` |
| **Arm 4** | `1.5 · like + 2.0 · follow + 1.5 · forward − 2.0 · not_interested` |

### 4.3 各 Arm 的 pair 数（不同 gap 阈值）

| Arm | gap > 0.5 | gap > 1.5 | gap > 2 | gap > 3 |
|---|---|---|---|---|
| Arm 1 longview | 115,487 | 91,226 | 64,097 | 35,978 |
| **Arm 2** | 123,041 | **105,383** ⭐ | 89,055 | 65,260 |
| Arm 3 | 123,073 | 105,479 | 89,191 | 65,424 |
| Arm 4 | 54,393 | 29,021 | 28,027 | 15,977 |

### 4.4 🚨 重大发现 1：Arm 3 实际等于 Arm 2

**Cross-arm chosen subset agreement**（同一用户在不同 arm 下选出的 chosen 5-item 是不是同一个集合）：

```
                  Arm 1     Arm 2     Arm 3    Arm 4
Arm 1            100%      75.39%    75.27%   22.45%
Arm 2            75.39%    100%      99.83%   33.04%
Arm 3            75.27%    99.83%    100%     33.13%
Arm 4            22.45%    33.04%    33.13%   100%
```

**Arm 2 vs Arm 3 一致率 = 99.83%** —— 几乎完全相同。

**原因**：134,210 个 eligible 用户里，target 含 not_interested 的只有 516 个 (0.38%)，penalty 几乎从不触发。Arm 3 vs Arm 2 的训练效果差异**上限就是这 ~500 个样本**。

### 4.5 🚨 重大发现 2：Arm 4 有 60% zero-gap pair

| Arm | zero-gap 用户数 | 占比 |
|---|---|---|
| Arm 1 | 18,723 | 13.95% |
| Arm 2 | 11,142 | 8.30% |
| Arm 3 | 11,110 | 8.28% |
| **Arm 4** | **79,808** | **59.47%** |

**原因**：Arm 4 不用 longview，target 里完全没有 like/follow/forward 的用户的整个 chosen/rejected score 都是 0，pair 退化。Arm 4 实际可用 ~29K 用户，质量远低于其他 arm。

### 4.6 🚨 重大发现 3：Arm 1 阶梯化分布

```
gap 直方图（Arm 1）:
  [0, 0.5):    13.95%   ← zero-gap
  [1, 1.5):    18.08%   ← 全是 gap=1
  [2, 3):      20.21%   ← 全是 gap=2
  [3, 5):      41.06%   ← 大部分 gap=3 或 4
  [5, 10):      6.70%   ← gap=5
```

**原因**：Arm 1 只用 longview（0/1 二值），m=5 时 gap 只能取整数 0/1/2/3/4/5。max=5 是因为 chosen 5 个全 longview、rejected 0 个 longview。gap=1 的 pair 训练信号弱。

### 4.7 一句话总结

**Arm 2 是唯一既丰富又干净的配置**：105,383 个 gap > 1.5 的 pair，没有 zero-gap 灾难，没有阶梯化噪声，没有和其他 arm 几乎等价的退化。**4-arm 消融实验在 OpenOneRec 数据上不可行**。

---

## 5. 实验设计调整

### 5.1 决策：放弃 4-arm 消融，使用单一 Arm 2 主配置

**理由**：

| Arm | 问题 | 决定 |
|---|---|---|
| Arm 1 | 阶梯化 gap 分布，gap=1 弱信号 | 放弃 |
| **Arm 2** | **数量丰富，gap 分布健康** | **采用** ⭐ |
| Arm 3 | 99.83% 等于 Arm 2，effect size 最多 500 样本 | 放弃 |
| Arm 4 | 59% zero-gap，可用样本仅 29K | 放弃 |

### 5.2 最终配置

```python
REWARD_WEIGHTS = {
    "longview": 1.0,
    "like":     1.5,
    "follow":   2.0,
    "forward":  1.5,
    "not_interested": 0.0,
}
GAP_THRESHOLD = 1.5
MIN_TARGET_LEN = 8         # m = L // 2 >= 4
PAIR_PER_USER = 1          # top-m vs bot-m, disjoint
```

**预期数据规模**：
- Eligible 用户：134,210
- Pair 数（gap > 1.5）：**105,383**
- Train / Valid 切分（按 uid hash）：~100K / ~5K

**对比文献**：
- Zephyr-DPO: 62K
- UltraFeedback: 64K
- Anthropic HH-RLHF: 170K
- **我们的 105K 在主流 DPO 训练数据规模区间内** ✅

### 5.3 Arm 3 作为 paper 级 negative result

虽然 Arm 3 不进主实验，但**这个发现本身有论文价值**：

> "在工业级推荐数据上，显式负反馈信号（not_interested）稀疏到 0.38%，使得显式负惩罚在 DPO 偏好对齐中实质上无法生效。这从经验上证伪了'显式负样本是 RLHF 必需'的常见假设，也部分解释了为什么 OneRec 论文需要训练独立的 Reward Model 而非直接使用行为信号。"

可以放在 paper 的 Discussion 或 Limitations 段，作为一段独立的"负面发现"。

---

## 6. 已知 Limitation：跨 item 时序因果被打破

### 6.1 问题陈述

我们的 top-5 / bot-5 切法保留了 chosen 和 rejected **内部**的 per-item 时序，但**破坏了跨 item 的因果链**。

举例：原 target = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]，假设 reward 选出：
- chosen = [t1, t6, t7, t8, t9]
- rejected = [t2, t3, t4, t5, t10]

在原始 log 里，t6..t9 的出现是因为推荐系统观察到用户先看了 t2..t5 的反馈才推出 t6..t9 的。重排后，autoregressive decoder 被要求最大化：

```
P(t6 | history, t1) · P(t7 | history, t1, t6) · ...
```

——这些条件概率**在真实世界里从未存在过**。chosen 序列整体是一个合成重排，不是 logged trajectory。

### 6.2 为什么我们接受这个 limitation

1. **DPO 学的是偏好方向，不是绝对似然**。Anthropic HH-RLHF / UltraFeedback / OneRec IPA 模块的 chosen 都不是 base model verbatim 出现过的序列。文献和实践都证明 DPO 对 off-policy chosen 鲁棒，只要相对 gap 信息量够。

2. **强 base model prior 吸收噪声**：OneRec-1.7B 在 96M interactions 上预训练过。105K pair 的 DPO fine-tune 不太可能反转这种 token 共现 prior——我们是在一个良好分布内做偏好微调，不是从零构建分布。

3. **替代方案更差**：
   - **顺序切**（target[:5] vs target[5:]）：保留时序但 expected gap ≈ 0，没有偏好信号，退化成 SFT
   - **滑动 5-window**：可训练 pair 数从 105K 降到 30-50K，且窗口级别仍有 off-policy 问题
   - **跨用户切**：用另一种分布问题换掉这个分布问题，不划算

4. **真正干净的做法是 on-policy DPO**：base model beam-search 出 candidate session → 用 reward 打分 → 高分当 chosen 低分当 rejected。这就是 OneRec IPA 的做法。但代价是丢掉"用真实行为信号"的卖点（candidate 不是真实日志），且推理成本极大。**留给 future work**。

### 6.3 我们做的 mitigation

- **chosen / rejected 内部按原 target 时间重排**（不按 score 排），至少保留 intra-sequence 局部时序
- **训练后做 sanity check**：在 OneRec-1.7B base 下计算 `logπ_ref(chosen)` 平均值 vs `logπ_ref(rejected)` 平均值。如果 chosen 显著低于 rejected，说明 off-policy shift 严重；如果接近或更高，说明 prior 起作用了

### 6.4 Paper 写作建议

在 Limitations 段明确披露：

> "Our pair construction selects top/bottom-m items by reward within the same target sequence and reorders by original timestamp. This preserves per-item temporal order within chosen and rejected but breaks the cross-item causal chain present in the original log. We trade distributional purity for behavioral grounding: logged user signals provide a real, interpretable supervision source at the cost of constructing synthetic preference sequences. Quantifying this trade-off via on-policy comparison is left to future work."

---

## 7. 训练阶段计划

### 7.1 Base model 选择

**`OpenOneRec/OneRec-1.7B`**（HuggingFace）

| 项 | 数值 |
|---|---|
| 大小 | 4.29 GB |
| 架构 | Qwen3-1.7B + Itemic-Text Alignment + Co-Pretraining + Multi-task SFT |
| 状态 | **已经做完 SFT** ✅ |
| 访问 | 公开（非 gated）✅ |
| 与我们的数据 | 完全对齐（用同一份 open-source data 训练） |

**不用以下版本**：

| 版本 | 不用的原因 |
|---|---|
| `OneRec-1.7B-pro` | 含 Kuaishou 内部数据，distribution 不一致，baseline 不干净 |
| `OneRec-1.7B-pretrain` | 仅预训练，需要自己做 SFT，工作量大且无收益 |
| `OneRec-8B-*` | 需要多卡，对学术消融过度 |

### 7.2 完整 Pipeline

```
1. EDA Scripts 1/2/3                                    [DONE ✅]
2. build_dpo_dataset.py                                 [下一步]
   → data/dpo_dataset/{train,valid}.parquet (~105K pair)
   → meta.json (构造参数 + 统计)
3. huggingface-cli download OpenOneRec/OneRec-1.7B
4. DPO 训练
   - trl DPOTrainer (首选) 或 OpenOneRec 仓库 RL 脚本
   - 单卡 A100 40GB / A6000 48GB
   - ~105K pair × 3 epoch ≈ 4-12 小时
5. Evaluation
   - benchmark_data/video/video_test.parquet (38,781 samples)
   - 主指标：Recall@10, Pass@32, Pass@1
   - Baseline: 未 DPO 的 OneRec-1.7B 直接 inference
6. (Optional) Paper 写作时加入 Arm 3 vs Arm 2 negative ablation 段
```

### 7.3 计算资源

| 资源 | 需求 |
|---|---|
| 磁盘 | 4.3 GB 模型 + ~700 MB 数据集 (parquet snappy) + checkpoint 空间 |
| GPU | 单卡 A100 40GB / A6000 48GB（DPO 需 policy + reference model 同时载入） |
| 显存 | 1.7B × 2 ≈ 8 GB params + activations + optimizer，40GB 卡 batch 4-8 舒适 |
| 时长 | 4-12 小时 / 训练 run |

---

## 8. DPO 数据集 Schema（待生成）

```
data/dpo_dataset/
├── train.parquet     ~100K rows (95%)
├── valid.parquet     ~5K rows (5%)
└── meta.json         构造参数 + 统计信息
```

### 每行 schema

```python
{
    "uid": int64,                                # 原始 user id（trace/debug）

    # ─── 上下文（encoder 输入，chosen/rejected 共享）───
    "history_pid":  list<int64>,                 # 长度 ≤ 512
    "history_sid":  list<list<int64>>,           # 长度 ≤ 512，每 item = 3-token sid

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

**为什么同时存 pid 和 sid**：sid 是模型实际吃的 token（训练用），pid 用于 case study / 错误分析 / join 回 caption。多存 pid 文件大小只增加 ~10%，换来调试便利。

**实现要点**：
1. chosen / rejected 内部**按时间顺序排列**（不按 score 排），保留 intra-sequence 局部时序
2. 拼 sid 前 `pid2sid_df.drop_duplicates(subset='pid')`
3. 文件大小估计：未压缩 ~1.7 GB，parquet snappy ≈ 400-700 MB

---

## 9. 团队讨论 Q&A

### Q: 为什么不用论文里的 4-arm 消融？
A: 数据约束。Arm 3 vs Arm 2 在数据上 99.83% 等价（not_interested 太稀疏），Arm 4 60% pair 是 zero-gap noise。强行做消融的成本远大于科学价值。Arm 3 的"失败"本身作为 negative result 写进 paper 即可。

### Q: 一个用户只能产一个 pair，是不是太少？
A: 105K pair 已经和 Zephyr-DPO / UltraFeedback 同量级，DPO 训练绰绰有余。OneRec base model 的 prior 已经很强，DPO 是微调而非从零训练。

### Q: chosen 序列在真实世界从未出现过，会有问题吗？
A: 见 §6（Known Limitation）。简短回答：DPO 学偏好方向不学序列真实性，文献支持这种做法，但我们会在 paper 里明确披露。训练后会做 logπ_ref sanity check。

### Q: 为什么不自己做预训练或 SFT？
A: 预训练成本天文数字（96M interactions × 数百 GPU × 数周），SFT 也需要大量工程。OneRec-1.7B 已经把这两步都做完了，直接接 DPO 就行——这是 maximum-leverage / minimum-risk 的路线。

### Q: 为什么是 1.7B 不是 8B？
A: 1.7B 单卡能装下 policy + reference model，8B 需要多卡。学术消融实验 1.7B 完全够用。

### Q: 评测用什么？
A: `benchmark_data/video/video_test.parquet`（38,781 个测试样本，OneRec 官方 benchmark）。指标：Recall@10、Pass@32、Pass@1。Baseline 是未 DPO 的 vanilla OneRec-1.7B。

---

## 10. 当前进度与下一步

```
[✅] Script 1 — eda_data_health.py
[✅] Script 2 — eda_behavior_signals.py
[✅] Script 3 — eda_dpo_feasibility.py
[✅] EDA 报告（本文件）
[ ] build_dpo_dataset.py            ← 下一步
[ ] 下载 OpenOneRec/OneRec-1.7B
[ ] DPO 训练 (trl DPOTrainer)
[ ] Evaluation on benchmark_data/video/
[ ] Paper writing
```

**最近 action item**：写 `build_dpo_dataset.py`，用 Arm 2 + gap > 1.5 配置生成 ~105K 条训练 pair。

---

## 附录 A：相关文档链接

- [`notebook/EDA.md`](EDA.md) — 原始 EDA 设计文档（含中文版调整后的 limitation）
- [`oneRec/eda_findings.md`](../oneRec/eda_findings.md) — 英文版 EDA findings（paper writing 用）
- [`oneRec/training_plan.md`](../oneRec/training_plan.md) — 英文版 training plan
- [`CLAUDE.md`](../CLAUDE.md) — 项目根目录的 Claude Code 上下文文档
- [`data_construction.md`](../data_construction.md) — 旧的 MIND pipeline 设计（**已废弃**，仅保留作为 context）

## 附录 B：原始 EDA 报告路径

- [`outputs/eda_data_health_report.txt`](outputs/eda_data_health_report.txt)
- [`outputs/eda_behavior_signals_report.txt`](outputs/eda_behavior_signals_report.txt)
- [`outputs/eda_dpo_feasibility_report.txt`](outputs/eda_dpo_feasibility_report.txt)