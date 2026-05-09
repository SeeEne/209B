# OpenOneRec — Behavior-Signal Alignment without a Reward Model

CS 209B course project (Harvard, Spring 2026) on Kuaishou's
[OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) generative
recommender. We replace OneRec's separately-trained Reward Model with
**direct user behavior signals** (longview / like / follow / forward) as the
supervision source for offline preference alignment, and ablate seven
training recipes head-to-head.

---

## TL;DR

We tried seven losses on the same data + same eval. Five contrastive variants
either **mode-collapsed**, **crashed absolute chosen recall below baseline**,
or **converged to a degenerate basin where both chosen and rejected log-probs
plummet together**. The two single-objective arms that don't have a
contrastive component (SFT-only and ORPO) are the only ones that lift chosen
recall above baseline without a failure mode — and they tie within noise.

| Arm | recall_chosen | recall_rejected | Δrecall | pass_chosen | pass_rejected | Δpass |
|---|---:|---:|---:|---:|---:|---:|
| OneRec baseline | 0.0120 | 0.0246 | **−0.0126** | 0.0344 | 0.0640 | **−0.0296** |
| **SFT-50k** | **0.0143** | 0.0146 | −0.0003 | **0.0411** | 0.0393 | +0.0018 |
| ORPO 50k (λ=0.1) | 0.0141 | 0.0143 | −0.0002 | 0.0407 | 0.0385 | +0.0021 |
| DPO + SFT-anchor 50k | 0.0098 | **0.0041** | **+0.0057** | 0.0282 | **0.0121** | **+0.0161** |

n = 7940 (full v1 valid). SFT-50k is the headline winner: highest absolute
chosen recall, Δ flipped from baseline, no failure mode. SFT vs ORPO chosen is
within noise (paired SE ≈ 0.0013).

---

## Research question

> In offline recommendation, can user behavior signals directly replace the
> Reward Model when constructing preference pairs for alignment training?

OneRec's online IPA module needs a Reward Model because user behaviors only
arrive *after* recommendation. We work **offline** on logged data, where
behavior signals are already in the parquet — so we can construct
chosen/rejected pairs directly and skip the Reward Model entirely.

---

## What we tried — and why most things failed

Seven arms, ordered by the chronology of the project. Detailed
trajectories are in [`archive/EXPERIMENTS.md`](archive/EXPERIMENTS.md).

### 1. Length=1 pure contrastive — **mode collapse**

`L = softplus(−(c − r)/τ)` on single chosen/rejected items, τ=0.1, 5k pairs.

Looked healthy at training time (`eval_pref_acc` 0.522 → 0.595) but on the
benchmark **Recall@32 = 0.0000** vs baseline 0.0231. The model produced the
same ~10 "popular" SIDs for every user, regardless of history. Inspection
showed first-position logits crashed (top-15 magnitude ~2.25 vs baseline ~28)
and were identical across users.

**Root cause**: pure contrastive admits a degenerate solution — crash both
chosen and rejected log-probs below "popular" tokens, while keeping
`chosen > rejected` by ε. The loss goes down, but the generation distribution
collapses onto a few high-prior tokens.

### 2. Length=1 contrastive + KL — **distribution healed, Recall@K still 0**

Added token-level KL(trained ‖ ref) on the SID positions. With kl_weight=0.5,
the trained model's logit magnitudes recovered (~26+), top-15 tokens
differentiated across prompts. Distribution looked **fine**.

But on the OneRec benchmark, **Recall@32 was still 0**.

**Root cause** (pivotal insight): the OneRec benchmark scores against
*next-shown* items. Most next-shown items are **non-engaged** (passive
scrolls, autoplay continuations). Our training signal was *engaged* > *not*.
A model that learned what we wanted *should* underperform on benchmark
Recall@K — they measure different things. This is when we built the
engagement-aware Δrecall metric on a held-out subset of our own
chosen/rejected data.

### 3. Length=3 contrastive (no KL) — **first directional reversal, but absolute numbers low**

Predict 3 items as a sequence; score = mean log P over 9 SID tokens. KL
dropped because Step 2 showed it didn't fix anything once we had the right
metric. 30k full run: `Δrecall = +0.0030`, `Δpass = +0.0080` vs baseline
`−0.0070 / −0.020`. **First clean sign reversal.**

But absolute `recall_chosen` = 0.0047, **half** of baseline's 0.0093. The
model was discriminating direction correctly while suppressing chosen items
overall. Foreshadowed the failure mode in Step 5.

### 4. Group-normalized contrastive (5k) — **matched 30k with 1/6 data**

Per-prompt loss-variance reduction: each user's G=3 chosen/rejected pairs
share a `group_id`, per-group std rescales the per-pair loss. Same
contrastive loss as Step 3, only addition is `/ std_g`. **Matched length=3
30k results with 1/6 the data and 0.72× wall-clock.** Variance reduction
worked.

> We call this "GRPO" in code for brevity but it is not GRPO — no rollouts,
> no advantage subtraction, just per-group `1/σ` loss scaling. In writing:
> **"group-normalized contrastive loss"**.

### 5. Group-normalized contrastive (50k) — **chosen recall crashed; run killed**

Step 4 looked promising at 5k — scaled to 50k for the headline number. At
step 2500 (1/5 epoch):

| | recall_chosen | recall_rejected | Δrecall |
|---|---:|---:|---:|
| baseline | 0.0093 | 0.0163 | −0.0070 |
| 5k smoke (1 epoch) | 0.0033 | 0.0003 | +0.0030 |
| **50k @step 2500** | **0.0043** | **0.0027** | +0.0017 |

Chosen recall **decreasing from baseline** (0.0093 → 0.0043). The +Δ was
purely "rejected drops faster than chosen". Run killed.

**Root cause**: pure contrastive `softplus(−(c − r)/τ)` only constrains
*relative* ranking. Two basins minimize it:
1. Push chosen up + push rejected down (intended)
2. Push **both** down, just push rejected harder (cheaper, picks this)

Group normalization scales loss *magnitude* per group but not its
*direction*, so the cheap basin is still reachable. The fix has to come from
adding an **explicit penalty on lowering chosen** — i.e. an SFT anchor.

### 6. Joint DPO + SFT-anchor + group norm — **calibration bug, then ineffective**

DPO term provides implicit KL via reference baseline; SFT term anchors
chosen absolutely:
```
margin     = β·[(c_θ − c_ref) − (r_θ − r_ref)]
L_dpo_grpo = (softplus(−margin).view(B, G) / std_g).mean()
L_sft      = −c_θ.mean()
L_total    = L_dpo_grpo + sft_weight · L_sft
```

First 5k smoke produced `eval_pref_acc = 0.546` (random). Tracing showed
`mean(1/std_g) ≈ 16` — the group normalization was amplifying L_dpo by 16×
while sft_weight=0.1 stayed at raw scale. **SFT was ~1–3% of total loss**,
not the intended 10%.

After fixing (`--sft_scale_mode match_dpo`, `sft_weight=1.0`),
`recall_chosen = 0.0093` at n=1000 — identical to baseline. The earlier
"0.0113 at n=5000" was sampling noise (n=1000 SE ≈ 0.003, swing inside 1σ).

**Root cause**: even with calibration fixed, joint DPO+SFT at 5k didn't beat
baseline. SFT's anchor was strong enough to prevent the Step 5 collapse but
not strong enough to lift chosen meaningfully — and with both terms competing
for gradient direction, neither got 100% of the signal.

### 7. Sequential SFT → DPO (5k) — **second collapse**

Standard post-training recipe: SFT first, then DPO with SFT model as ref.
Stage 1 SFT-only 5k lifted chosen 0.0093 → 0.0097 (+4%, within noise — same
data-scale ceiling as Step 6). Stage 2 DPO from this barely-moved SFT:

| ckpt | chosen_score | rejected_score |
|---|---:|---:|
| 1 (step 125) | −7.51 | −8.43 |
| 5 (step 625) | **−13.44** | **−17.68** |

Chosen log P crashed **8.6 nats** below the SFT starting point. recall_chosen
halved (0.0097 → 0.0043), recall_rejected dropped 17×.

**Root cause**: with margin ≈ 0 at step 0 (trained = ref = barely-moved SFT),
DPO sees the same two-basin problem as Step 5. `best_metric=chosen_score`
selected the least-collapsed ckpt 1 but couldn't prevent the collapse — it's
ex-post selection, not ex-ante constraint.

**Lesson**: Stage 2 DPO is gated on Stage 1 producing a *real* lift, not a
marginal one. SFT at 5k can't.

### 8. SFT-only 50k — **the ceiling broke (winner)**

5k SFT only moves chosen 0.05 nats. Was that an optimization failure or a
data-scale limit? Two sweeps cleared it.

**lr sweep** (lr ∈ {5e-5, 2e-4, 5e-4}, 5k): all converge to chosen ≈ −4.84.
**lora_r sweep** (r=16 → 32): +0.002 nats, pure noise.

Both negatives → 5k ceiling is **data**, not optimization. Ran 50k SFT with
no other change.

| ckpt | step | eval_chosen_score |
|---|---|---:|
| base | 0 | ≈ −4.92 |
| 5 | 6250 | **−4.795** |

**0.125 nats vs base, 0.046 nats below the 5k ceiling.** Engagement-aware
n=5000: `recall_chosen = 0.0140` (+50% vs baseline 0.0093, t≈4.7),
`Δrecall = +0.0008`. **First arm to lift chosen recall absolutely above
baseline AND flip Δrecall sign.** Wall clock 17 h on RTX 6000 Pro.

### 9. ORPO 50k — **tied with SFT**

`L = −log P_θ(c) + λ·−log σ(log_odds_θ(c) − log_odds_θ(r))`, λ=0.1
(paper default). Single-stage, no ref model, no group norm.

Final n=7940: `recall_chosen = 0.0141`, `recall_rejected = 0.0143`, Δ ≈ 0
(see TL;DR table). **Tied with SFT-50k within noise** despite a fundamentally
different objective. See "Why ORPO ≈ SFT" below — this is the project's most
generalizable finding.

### 10. SFT-50k → DPO + SFT-anchor 50k — **chosen crashed below baseline**

Stage 2 of the sequential recipe, but with `sft_weight=0.15` (raw, ~19%
gradient contribution post-warmup) added to prevent the Step 7 collapse.
Trained from SFT-50k (chosen=0.0140 starting point) so DPO had real headroom.

Final n=7940: `recall_chosen = 0.0098` — **below baseline's 0.0120**.
`recall_rejected` dropped 83% (0.0246 → 0.0041), giving the largest Δ of any
arm (+0.0057), but at the cost of crashing chosen. The 0.15 anchor weight
prevented total collapse (Step 7 went 0.0097 → 0.0043) but wasn't enough to
keep chosen above baseline.

**Root cause** (different from Step 7 — this is *post-warmup* contrastive
collapse, not start-of-training collapse): once DPO momentum builds, the
gradient direction that minimizes the contrastive loss most efficiently is
"push rejected hard down, let chosen drift down with it." The implicit KL via
ref baseline is a *direction* constraint, not an *absolute* constraint on
chosen — it allows chosen to drift below ref as long as rejected drifts
faster. SFT-anchor at 0.15 raw is the only force pulling chosen up, and it
loses.

**Could it be salvaged?** Probably yes by raising sft_weight to 0.5–1.0
(raw). Not pursued in this project's timeline; the SFT-50k baseline already
delivers the project's claim.

---

## Final results (n=7940 full v1 valid)

The four arms that survived through the final eval pass:

| Arm | recall_chosen | recall_rejected | Δrecall | pass_chosen | pass_rejected | Δpass |
|---|---:|---:|---:|---:|---:|---:|
| OneRec baseline | 0.0120 | 0.0246 | −0.0126 | 0.0344 | 0.0640 | −0.0296 |
| **SFT-50k** | **0.0143** | 0.0146 | −0.0003 | **0.0411** | 0.0393 | +0.0018 |
| ORPO 50k | 0.0141 | 0.0143 | −0.0002 | 0.0407 | 0.0385 | +0.0021 |
| DPO + SFT-anchor 50k | 0.0098 | 0.0041 | +0.0057 | 0.0282 | 0.0121 | +0.0161 |

Same loss configurations, same data (`contrastive_dataset_v1_grpo`, 50k
groups), same eval (`evaluate_engaged.py`, num_beams=32, top-K=96). Per-arm
JSON in [`evaluation_results/`](evaluation_results/).

---

## Why DPO + SFT-anchor crashed chosen recall

Mechanically: when DPO and SFT-anchor are combined post-warmup, the
contrastive term dominates the gradient direction (~81% in our setup), and
the cheapest way to satisfy it is to **lower P(rejected)** rather than
**raise P(chosen)** — because chosen and rejected share the same output
space (see next section). The SFT-anchor at sft_weight=0.15 (raw) provides a
restoring force on chosen, but it's overpowered.

Two collapse modes, one principle:
- **Step 7** (sequential 5k, no anchor): start-of-training collapse. Margin =
  0 at step 0, both basins are equally close, model picks the cheap one.
- **Step 10** (DPO+anchor 50k, weak anchor): post-warmup collapse. Margin is
  positive but the contrastive gradient still finds it cheaper to drag both
  down asymmetrically.

The "implicit KL via ref baseline" that DPO papers cite is a *direction*
constraint (chosen must move differently than rejected), not an *absolute*
floor on chosen. Without an explicit anchor strong enough to dominate the
contrastive pull, chosen drifts.

---

## Why ORPO ≈ SFT — a finding that generalizes

ORPO and SFT have very different objectives, yet their chosen and rejected
recall both end up tied within noise. Three structural reasons stack up:

**1. Small vocabulary makes zero-sum redistribution local.** At each SID-token
generation step, the model effectively picks from a codebook of **K = 8,192**
entries (vs ~50k–150k for standard LLMs — 6–18× smaller; we confirm K=8192
empirically from the data in the notebook). When SFT pushes P(chosen_token)
up by Δ, the remaining K−1 entries proportionally lose Δ. Smaller K → less
dilution → more concentrated zero-sum.

**2. Chosen and rejected share user context.** Both come from the same
session. They live in the same high-prob region of the codebook. So when SFT
pushes chosen up, **rejected is the nearest neighbor that gets drained**.
Numerical evidence: SFT (with no rejected supervision at all) drops
recall_rejected from baseline 0.0246 → 0.0146 *as a side effect*. ORPO's
explicit OR term tries to suppress rejected, but SFT already finished it via
zero-sum.

**3. LoRA r=16 + 50k pairs hits a capacity ceiling.** Both arms saturate
per-token chosen log P at ≈ −4.795 nats. With 5k data the ceiling was
−4.840; lr ∈ {5e-5, 2e-4, 5e-4} and r ∈ {16, 32} sweeps all hit the same
number → the bottleneck is data, not optimization. The OR term's residual
contrastive signal doesn't have free LoRA capacity to walk to a different
fixed point.

**The deeper point** — this is where the finding generalizes:

> Whenever chosen and rejected share a narrow categorical output space,
> contrastive alignment objectives' marginal gains over plain SFT are
> absorbed into SFT's implicit zero-sum.

This applies to most generative recommenders: small token vocabularies,
chosen/rejected from the same user context, low-rank fine-tunes. For these
tasks, **you don't need a contrastive objective if your output space is
narrow and chosen/rejected come from the same context**. Plain SFT on the
chosen branch suffices.

---

## What's the right metric

OneRec's official `Recall@K` rewards "what gets shown next," which on logged
data is mostly non-engaged items (passive scrolling). Baseline: rejected
recall = 0.0246 is **2× chosen recall** = 0.0120 — the model is *more*
likely to surface non-engaged items than engaged ones. Standard Recall@K
rewards exactly this bias.

We report **engagement-aware Δrecall** (chosen recall minus rejected recall)
on a held-out subset of `contrastive_dataset_v1/valid.parquet` as the primary
metric. Standard Recall@K on `video_test.parquet` is reported in
[`evaluation_results/`](evaluation_results/) for arms that have it but
treated as secondary — our training deliberately deprioritizes "next-shown
non-engaged" items, so a successful arm is *expected to score lower* on
official Recall@K.

---

## Quick start

```bash
# 1. Install (Python 3.10–3.12, CUDA 12)
pip install "torch==2.4.*" --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# 2. Build pair data (one-time, ~30 min)
python build_contrastive_dataset.py --length 3
python build_contrastive_dataset_GRPO.py --G 3

# 3. Train one arm (50k variants, ~17–38 h depending on arm)
bash scripts/run_sft_50k.sh                       # SFT-only (winner)
bash scripts/run_orpo_50k.sh                      # ORPO
bash scripts/run_dpo_anchor_from_sft_50k.sh       # DPO + SFT-anchor

# 4. Evaluate (engagement-aware Δrecall, ~2h on full v1 valid)
bash scripts/run_final_eval_baseline_sft.sh       # baseline + SFT
bash scripts/run_final_eval_orpo.sh               # ORPO
bash scripts/run_final_eval_dpo_anchor.sh         # DPO + anchor
```

Full hyperparameters: see [CLAUDE.md](CLAUDE.md) § Commands.

---

## Repository layout

```
project/
├── README.md                   ⭐ this file (results)
├── CLAUDE.md                      dev guide / conventions
│
├── train/
│   ├── train_sft_only.py          SFT-only arm (winner)
│   ├── train_orpo.py              ORPO arm
│   ├── train_dpo_from_sft.py      Sequential DPO Stage 2
│   ├── train_contrastive_dpo_g_normalize.py   DPO + SFT-anchor (group-normed)
│   ├── evaluate_engaged.py     ⭐ engagement-aware Δrecall (primary)
│   └── evaluate_origin.py         OneRec official Recall@K (secondary)
│
├── scripts/                       run_*.sh launchers
├── diagnose/                      base-vs-trained inspection tools
├── archive/
│   ├── EXPERIMENTS.md          ⭐ 13-step research journey, all dead ends
│   ├── train_contrastive.py        length=1 (Step 1, mode collapse)
│   ├── train_contrastive_length3.py  length=3 (Step 3)
│   └── train_contrastive_grpo.py     group-norm contrastive (Steps 4–5)
│
├── evaluation_results/            final per-arm CSVs + summary JSONs
├── data/                          (not in git)
└── runs/                          (not in git)
```

---

## Documentation map

| Doc | When to read |
|---|---|
| [README.md](README.md) | What did this project find? |
| [CLAUDE.md](CLAUDE.md) | How is the repo organized? Hyperparameters? |
| [archive/EXPERIMENTS.md](archive/EXPERIMENTS.md) | Full step-by-step journey of what didn't work, with logs |
| [train/README.md](train/README.md) | How do I train? |
| [diagnose/README.md](diagnose/README.md) | How do I inspect a trained model? |

---

## Course

CS 209B *Statistical Learning for Decision Making*, Harvard SEAS, Spring 2026.

## Acknowledgments

- Kuaishou OpenOneRec team for the model and dataset release
- Hong et al. (2024) for ORPO
- DeepSeek-V3 / R1 papers for the GRPO normalization inspiration
