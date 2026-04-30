# EDA Findings and Pair-Construction Decision

> **Update (2026-04-30) — Method pivoted from DPO to contrastive learning.**
> The findings below are still load-bearing (signal sparsity, lift validation,
> per-user reward gap), but the loss they were collected to support (DPO over
> 5-item chosen / 5-item rejected sequences) was abandoned. The "Known
> limitation: cross-item temporal causality is broken" section at the bottom
> is precisely what motivated the pivot. The current method uses a pairwise
> contrastive loss on **single items** (`prediction_length=1` in
> [../build_contrastive_dataset.py](../build_contrastive_dataset.py)),
> sidestepping the synthetic-sequence problem entirely. See
> [../notebook/eda_ms2.ipynb](../notebook/eda_ms2.ipynb) §6 for the structural
> argument and [training_plan.md](training_plan.md) for the resulting pipeline.

EDA on `data/OpenOneRec/onerec_bench_release.parquet` (162,074 rows = full split=0 = entire training pool — `split` field is effectively a no-op in the HF release) completed 2026-04-07.

## Key empirical facts

- **Effective users**: 156,245 (3.60% have empty hist/target and are unusable)
- **hist_video_pid length**: P50/P95 = 484/508, max 512 — already truncated by OneRec, K is fixed
- **target_video_pid length**: median 9, max 10 — this is the entire space for chosen/rejected
- **PID coverage in pid2sid maps**: 100.0000% on all 6 hist/target columns
- **Behavior/pid length alignment**: 0 mismatches across all 5 hist + 5 target behavior columns (hard prerequisite for DPO reward)
- **video_ad_pid2sid has 2,658 duplicate pids** (0.017%); product_pid2sid has 1. Build script must `drop_duplicates(subset='pid')` before joining.

## Behavior signal sparsity (target side, decisive for DPO)

| signal | user coverage | item-level positive rate |
|---|---|---|
| longview | 79.05% | 29.01% |
| like | 37.46% | 9.38% |
| forward | 4.65% | 0.69% |
| follow | 1.92% | 0.26% |
| not_interested | **0.34%** | **0.06%** |

Semantic-consistency hypothesis (Script 2 §4) **fully supported**: lift of P(longview|sig=1) / P(longview|sig=0) is 1.36 (like), 1.88 (follow), 1.87 (forward), 0.66 (not_interested). Explicit positive signals genuinely indicate stronger preference; not_interested genuinely suppresses longview.

## Why the 4-arm ablation was abandoned

Script 3 used m = L//2, eligibility target_len ≥ 8 (134,210 users), top-m vs bottom-m by per-item reward.

Pair counts at gap > 1.5:
- Arm 1 (longview only): 91,226
- Arm 2 (longview + like + follow + forward): **105,383**
- Arm 3 (Arm 2 + −2.0·not_interested): 105,479
- Arm 4 (pure explicit): 29,021

**Arm 2 vs Arm 3 chosen-subset agreement = 99.83%** — only 516 of 134,210 eligible users (0.38%) have any not_interested in their target, so the penalty fires almost never. Arm 3's effective ablation surface is ~500 samples. **Not worth running as a separate arm.**

**Arm 4 has 59.47% zero-gap pairs** (target with zero like/follow/forward → entire formula = 0). Quality much lower than Arm 1/2/3.

**Arm 1's gap distribution is staircased** (only takes integer values 0/1/2/3/4/5 because longview is 0/1) — gap=1 pairs carry weak signal.

**Decision**: skip the ablation entirely. Use Arm 2 as the single main config. Treat the Arm 2 vs Arm 3 99.83% agreement as a paper-worthy negative finding ("explicit negative feedback is too sparse in industrial data to drive DPO alignment").

## Final pair-construction recipe

```
For each user with len(target_video_pid) >= 8:
  m = len(target_video_pid) // 2
  per-item reward = 1.0*longview + 1.5*like + 2.0*follow + 1.5*forward
  chosen_items   = top-m item indices by reward (re-sorted by original target time)
  rejected_items = bottom-m item indices by reward (re-sorted by original target time)
  gap = sum(chosen rewards) - sum(rejected rewards)
  Keep iff gap > 1.5
```

Result: ~105,383 pairs, one per user. Same scale as Zephyr-DPO / UltraFeedback.

**Implementation notes**:
- chosen/rejected lists must be reordered by original target time before being written to disk — DPO loss is sequence-order sensitive.
- Default to Arm 2 + gap>1.5 + top/bot-m construction. The 4-arm ablation is abandoned unless explicitly revived.
- The Arm 3 finding remains valuable as a negative result for the paper's discussion section.

## Known limitation: cross-item temporal causality is broken

Picking top-m / bot-m by per-item reward and reordering by timestamp preserves the **per-item temporal order within** chosen and rejected, but it **breaks the cross-item causal chain** in the original target sequence.

Concretely: suppose target = [t1..t10] and reward selects chosen = [t1, t6, t7, t8, t9]. In the original log, t6..t9 appeared *because* the user had just interacted with t2..t5 — the recommender adapted on the fly. After re-ordering, the chosen sequence is `[t1, t6, t7, t8, t9]`, and the autoregressive decoder is asked to maximize

```
P(t6 | history, t1) · P(t7 | history, t1, t6) · ...
```

— conditional probabilities that **never existed in the real world**. The chosen sequence as a whole is a synthetic re-arrangement, not a logged trajectory.

**Why we accept this limitation:**

1. **DPO learns preference direction, not absolute likelihood.** Anthropic HH-RLHF, UltraFeedback, and OneRec's own IPA module all use chosen/rejected drawn from distributions the base model never produced verbatim. Empirically DPO is robust to off-policy chosen as long as the relative gap is informative.
2. **Strong base-model prior absorbs the noise.** OneRec-1.7B was pretrained on 96M interactions. A 105K-pair DPO fine-tune is unlikely to invert the token-level co-occurrence prior the base model already encodes; we are nudging preferences within a well-shaped distribution, not building one from scratch.
3. **The alternative is worse.** Sequential split (`target[:5]` vs `target[5:]`) preserves causality but has an expected reward gap of ~0 — there is no preference signal to learn from. Sliding-window splits inside the target produce far fewer pairs (~30-50K vs 105K) and still suffer from the same off-policy issue at the window level.
4. **The honest fix is on-policy DPO** (beam-search candidates from the SFT model, score with our reward, treat as chosen/rejected). That removes the temporal-causality issue entirely but discards the "real logged behavior" selling point and adds substantial inference cost. We treat it as future work.

**Mitigation we DO apply:**
- Reorder chosen/rejected by original timestamp so that *intra*-sequence local order is preserved (avoids feeding the decoder a reward-sorted permutation, which would compound the problem).
- After training, run a sanity check: average `logπ_ref(chosen)` vs `logπ_ref(rejected)` under the OneRec-1.7B base model. If chosen has dramatically lower log-prob, the off-policy shift is severe enough to worry about. If they're comparable, the prior is doing its job.

**Paper framing:** explicitly disclose this in the Limitations section. The phrasing should emphasize that we trade distributional purity for behavioral grounding — we use logged user signals (a real, interpretable supervision source) at the cost of constructing synthetic preference sequences.