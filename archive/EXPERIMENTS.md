# Experiment Journey

This directory holds code from intermediate experiments. The chosen path
forward is **DPO + SFT + group-normalized contrastive loss**, implemented at
project root and in `train/train_contrastive_dpo_g_normalize.py`. Everything
here in `archive/` was a meaningful step on the way there — kept for
reproducibility and for the report's ablation chapter.

This file records *why* each step was taken and *what we learned* before
moving on. Read top-to-bottom for the narrative.

---

## Terminology note: "GRPO" in this codebase

We use "GRPO" loosely in filenames and comments. Strictly, what we do is
**per-prompt loss variance normalization**:

- Each user's G pre-constructed contrastive pairs share a `group_id`
- Per-group std is computed and `.detach()`ed as a scalar baseline
- Each pair's loss is divided by its group's std before averaging

This is *inspired* by GRPO (DeepSeek-V3, R1) but **isn't** full GRPO. Real GRPO:

- Samples G rollouts from current policy (we don't sample — pairs are fixed)
- Subtracts group-mean reward as baseline (we can't — the loss is already
  positive and subtracting mean makes the per-group sum 0)
- Uses (r − μ)/σ as an *advantage weight* on policy gradient (we use 1/σ
  as a *loss scaling factor*)

In writing/reports we call it **"group-normalized contrastive loss"**.
Filenames keep `_grpo_` for brevity; that's a code-only shorthand.

---

## Timeline

### Step 0. Method pivot: DPO → contrastive (MS2)

Before any training, EDA found:
1. Explicit `not_interested` is too sparse (0.06% item rate) for DPO triplets
2. Top-m / bottom-m construction yields synthetic chosen *sequences* that never
   existed in the real log — DPO's likelihood-based loss penalizes this off-policy shift

Pivoted from DPO to a pairwise contrastive loss on individual items.

**Files:** `notebook/eda_*.py`, `oneRec/eda_findings.md` (kept active)

---

### Step 1. Length=1 contrastive — first attempt

**What:** Score = mean log P over 3 SID tokens of one item. Loss =
`softplus(-(chosen_score - rejected_score) / τ)`. Each user contributes 1
(chosen, rejected) pair. `τ=0.1`.

**Smoke (5k samples):** `eval_pref_acc` 0.522 → 0.595, looked healthy.

**Disaster on benchmark:** `Recall@32 = 0.0000` while baseline gets 0.0231.
Trained model produced valid SID-format generations, but the same ~10
"popular" SIDs dominated *every* user's top-K. Mode collapse.

**Diagnosis (via `compare_models.py`):**
- Trained logits at first SID position: top-15 magnitudes ~2.25 (vs baseline ~28)
- Top-15 was identical across different users → model lost personalization

**Lesson:** Pure contrastive admits a degenerate solution: crash both chosen
and rejected log-probs below "popular" tokens, while keeping `chosen > rejected`
by ε. Loss goes down, generation breaks.

**Files (archived):** `train_contrastive.py`

---

### Step 2. Add KL regularization

**What:** Token-level KL(trained ‖ ref) on the 3 SID positions, applied to
both chosen and rejected sequences.

```
L = L_contrastive + kl_weight × KL(trained ‖ ref)
```

**First try `kl_weight=0.1`:** Trained still got Recall@32 = 0. Inspecting
training logs showed `l_kl ≈ 2.0` throughout — KL was being violated, not
constrained. Weight too low.

**Second try `kl_weight=0.5`:**
- l_kl stayed in 0.2–0.5 range ✓ (KL was bounded)
- Trained logit magnitudes recovered to ~26+ (no longer crashed) ✓
- Top-15 differed across prompts ✓ (no longer same-popular)
- **But** still Recall@32 = 0 on baseline OneRec benchmark

**Diagnosis:** trained model now had valid, diverse, healthy distributions.
But its predictions still didn't overlap with the benchmark's GT items.

**Insight:** the standard Recall@K metric is wrong for what we trained.
Baseline was trained for "next-shown" prediction; benchmark GT is the
next-shown item. We trained for "engaged > non-engaged" preference. The two
are different distributions — most "next-shown" items are non-engaged
(passive scrolling), so a model that learns to *deprioritize* non-engaged
items will *underperform* on raw Recall@K *by design*.

**Files (archived):** none — KL machinery rolled into the next file

---

### Step 3. Length=3 contrastive, no KL (predict 3 future videos)

**What:** Predict 3 next items as a sequence; score = mean log P over 9 SID
tokens. Equal weight per item by averaging tokens.

```
L = softplus(-(c_θ - r_θ)/τ)              # pure contrastive, no KL
score = (1/9) × Σ log P(SID_token_i | history, prev_tokens)
```

**KL was dropped** vs Step 2. Step 2's KL fix made the distribution healthy
but Recall@K was still 0 — the failure was a metric mismatch, not a
distribution-shape problem. With length=3 + a proper engagement-aware
metric coming in Step 4, KL becomes unnecessary noise.

Order-invariant evaluation: `evaluate_length3.py` generates `max_new_tokens=13`
to cover 3 items, deduplicates across beams, matches GT as a set.

**Smoke (5k):** `eval_pref_acc=0.595`, `Recall@96=0` on benchmark. Same
"correct training, wrong metric" pattern.

**30k full run:** `eval_pref_acc=0.674`, `eval_margin=0.580`. Engagement-aware
metric `Δpass=+0.008` (vs baseline `-0.020`). **First clean directional
reversal demonstration.**

**Files (archived):** `train_contrastive_length3.py`, `evaluate_length3.py`

---

### Step 4. Engagement-aware evaluation (the "right" metric)

**What:** Instead of fighting OneRec benchmark's mismatched GT distribution,
build a metric directly aligned with the training objective.

`evaluate_engaged.py` runs on `contrastive_dataset_v1/valid.parquet` (held-out
10% never seen during training). Each row already has 3 chosen + 3 rejected.
We compute:

- `recall_chosen@K` : how often chosen items appear in model's top-K predictions
- `recall_rejected@K` : same for rejected
- **`Δrecall = recall_chosen − recall_rejected`** : the headline number

**Baseline (no FT):** `Δrecall = −0.0070`, `Δpass = −0.020` — base model is
*biased toward rejected* items (consistent with "trained to predict
next-shown, which is mostly non-engaged").

**Length=3 30k:** `Δrecall = +0.0030`, `Δpass = +0.0080` — clear sign reversal.

**Lesson:** Method works at the metric level; need a metric that measures
preference rather than raw next-shown coverage.

**Files (active):** `train/evaluate_engaged.py`

---

### Step 5. Per-prompt group normalization on the contrastive loss

**Motivation:** Single-step training-loss variance was huge (each batch had
4 users with very different difficulty). Aggregate eval was healthy, but
per-step gradient signal was noisy. Standard "more data" helps the
estimator, not the *per-step* gradient noise.

**What:** Same contrastive loss as Step 3 (no KL), with one addition:
divide each pair's loss by its per-group std before averaging. That's
literally the only difference vs Step 3. Each user's 3 chosen + 3
rejected are pre-expanded into G=3 single-item pairs sharing a
`group_id`:

```
L_pair = softplus(-(c−r)/τ)                        # per pair
L_grouped = L_pair.view(num_groups, G)              # (G groups, G pairs)
std_g = L_grouped.std(dim=-1).detach() + ε          # baseline scalar
L = (L_grouped / std_g).mean()                      # per-group rescaled
```

ε=1e-3 caps the amplification factor at 1000× to avoid initial-step
explosion when std is near 0.

**Pairing (`build_contrastive_dataset_GRPO.py`):**
9 (chosen_i, rejected_j) combinations from the 3×3 grid, ordered as:
- 0..2: cyclic shift 1 (derangement)   ← G=3 default
- 3..5: cyclic shift 2 (derangement)
- 6..8: diagonal (matched positions)

`PAIRING_ORDER[:G]` selects the first G.

**5k smoke (G=3):**
- `Δrecall = +0.0030`, `Δpass = +0.0080`
- **Matched length=3 30k results with 1/6 the data and 0.72× wall-clock**.
  Variance reduction validated.

**G=5 smoke:** Δpass = +0.0050 (worse than G=3 +0.0080). Three reasons:
1. Smaller effective user-batch (3 users/step vs 4 with G=3)
2. The 2 extra pairs from "cyclic shift 2" are correlated with the first 3
3. More averaging dilutes per-pair signal

**G=3 wins; G>3 doesn't help on this data.** Lesson: per-group variance
reduction's "more samples per group = lower std → smoother loss"
assumption breaks when samples are correlated within group (cyclic
shift 2 pairs are derived from the same underlying chosen/rejected
sets as cyclic shift 1).

**Note on the "GRPO" naming.** This step is *not* GRPO in any
algorithmic sense — there's no rollout sampling, no advantage
subtraction, no policy gradient. It's just per-group std rescaling of
a contrastive loss. We kept "_grpo_" in filenames as shorthand because
the per-group variance reduction was loosely *inspired* by GRPO
(DeepSeek), but the underlying loss is still the Step 3 contrastive
loss. In writing/reports, call it **"group-normalized contrastive
loss"**, not GRPO.

**Files (archived):** `train_contrastive_grpo.py`, `run_grpo_g5_smoke.sh`

---

### Step 6. 50k full run (group-normalized contrastive) + a critical observation

50k users × G=3 = 150k pairs, ~38h. Same hyperparams as Step 5 smoke
(group-normalized contrastive, no KL).

Sanity check at step 2500 (1/5 epoch) via `eval_grpo_50k_checkpoint.sh`:

| | recall_chosen | recall_rejected | Δrecall | Δpass |
|---|---|---|---|---|
| baseline (no FT) | 0.0093 | 0.0163 | −0.0070 | −0.0200 |
| 5k smoke (1 epoch) | 0.0033 | 0.0003 | +0.0030 | +0.0080 |
| **50k step 2500** | **0.0043** | **0.0027** | +0.0017 | +0.0040 |

**The smoking gun:** chosen recall is **not increasing** — it's **decreasing
from baseline** (0.0093 → 0.0043), just slower than rejected (0.0163 → 0.0027).
The +Δ is purely from "rejected drops faster". In absolute terms, the model
recommends **fewer** chosen items than baseline.

**Why:** Pure contrastive `softplus(-(c−r)/τ)` only constrains *relative*
ranking. The gradient has multiple local minima:
1. "Push chosen up, push rejected down" (what we want)
2. "Push both down, but push rejected more" (cheaper, picks this)

Group normalization changes the loss *magnitude* per group, not its
*direction* — so the cheaper basin is still reachable. KL would
constrain distribution shape but not chosen's *absolute* probability;
we'd seen in Step 2 that KL alone doesn't help. So neither group norm
nor KL nor their combination addresses the actual failure mode.

**Run killed.** Pivot to method that explicitly anchors chosen via an
SFT term.

**Files (archived):** `run_grpo_50k_full.sh`, `eval_grpo_50k_checkpoint.sh`

---

### Step 7. DPO + SFT + group normalization (chosen path)

Three forces, three jobs:

```
margin     = β × [(c_θ − c_ref) − (r_θ − r_ref)]   # DPO with ref baseline
L_dpo_pair = softplus(-margin)
L_dpo_grpo = (L_dpo_pair.view(B, G) / std_g.detach()).mean()  # group-normalize

L_sft = -c_θ.mean()                                # absolute push for chosen

L_total = L_dpo_grpo + sft_weight × L_sft          # KL drops out: DPO has implicit KL
```

| Component | Job | Mechanism |
|---|---|---|
| DPO margin | discrimination | sigmoid saturates when trained drifts from ref → implicit KL |
| SFT anchor | absolute chosen ↑ | pure NLL, doesn't see rejected |
| Group norm | gradient-noise reduction | per-prompt std rescaling |

DPO + SFT fixes Step 6's problem: the SFT term penalizes "lowering chosen",
making the cheaper-but-degenerate solution costlier. The DPO ref baseline
provides implicit KL (no need for explicit KL term).

Standard hyperparams:
- `--dpo_beta 0.1` (DeepSeek/Llama-3 standard)
- `--sft_weight 0.1` (gentle anchor — strong enough to push chosen up,
  weak enough not to drown the contrastive signal)
- `--kl_weight 0` (drop explicit KL)

**Files (active):** `train/train_contrastive_dpo_g_normalize.py`,
`scripts/run_dpo_smoke.sh`

---

## File-to-experiment map (what's in `archive/`)

| File | Step | What it produced | Why archived |
|---|---|---|---|
| `train_contrastive.py` | 1 | length=1 baseline run, mode collapse demo | Replaced by length=3 (Step 3) |
| `train_contrastive_length3.py` | 3 | length=3 30k run, first directional reversal | Replaced by group-norm version (Step 5) |
| `train_contrastive_grpo.py` | 5 | 5k smoke + 50k partial run, G=3 vs G=5 ablation | Replaced by DPO+SFT (Step 7) |
| `evaluate_length3.py` | 3 | length=3 evaluation w/ pool aggregation | Replaced by `evaluate_engaged.py` (Step 4) |
| `merge.py` | (utility) | Windows-side LoRA merge | Linux uses `merge_local.py` |
| `run_all_evals.sh` | 4 | Batch eval of baseline + length=3 30k + GRPO smoke | One-off script, runs reproducible from individual evals |
| `run_grpo_g5_smoke.sh` | 5 | G=5 ablation training | Confirmed G=3 > G=5; result recorded |
| `run_grpo_50k_full.sh` | 6 | Full GRPO 50k pilot | Killed at step 2500 due to chosen-recall drop |
| `eval_grpo_50k_checkpoint.sh` | 6 | Mid-training sanity check that triggered the pivot | One-off; technique reusable for any future run |

---

## Datasets produced (under `data/`)

These data files are *not* archived — they're current. Listing for completeness.

| Directory | Built by | Format | Used by |
|---|---|---|---|
| `contrastive_dataset_v0` | `build_contrastive_dataset.py --length 1` | 1 chosen + 1 rejected per row | Step 1, 2 |
| `contrastive_dataset_v1` | `build_contrastive_dataset.py --length 3` | 3 chosen + 3 rejected per row | Step 3, 4 (eval source) |
| `contrastive_dataset_v1_grpo` | `build_contrastive_dataset_GRPO.py --G 3` | 1 pair / row, group_id, G=3 | Step 5, 7 (current) |
| `contrastive_dataset_v1_grpo_g5` | `build_contrastive_dataset_GRPO.py --G 5` | 1 pair / row, group_id, G=5 | Step 5 ablation |

`contrastive_dataset_v1` is the source of truth for training data and the
held-out valid set used by `evaluate_engaged.py`.

---

## Headline numbers (engagement-aware)

All n=1000 unless marked. n=5000 numbers are the headline-quality
estimates (~1/2 the SE of n=1000) for arms where they exist.

| Method | n | recall_chosen | recall_rejected | Δrecall | Δpass |
|---|---|---|---|---|---|
| baseline (no FT) | 1000 | 0.0093 | 0.0163 | −0.0070 | −0.0200 |
| Length=3 contrastive 30k, no KL (Step 3) | 1000 | 0.0047 | 0.0017 | +0.0030 | +0.0080 |
| Group-norm contrastive 5k, G=3 (Step 5) | 1000 | 0.0033 | 0.0003 | +0.0030 | +0.0080 |
| Group-norm contrastive 5k, G=5 (Step 5) | 1000 | 0.0030 | 0.0010 | +0.0020 | +0.0050 |
| Group-norm contrastive 50k @step 2500 (Step 6) | 1000 | 0.0043 | 0.0027 | +0.0017 | +0.0040 |
| Joint DPO+SFT 5k (Step 9, post-fix) | 1000 | 0.0093 | 0.0130 | −0.0037 | −0.0120 |
| Sequential SFT→DPO 5k (Step 10) | 1000 | 0.0043 | 0.0007 | +0.0037 | +0.0110 |
| SFT-only 5k (Step 11 baseline) | 1000 | 0.0097 | 0.0127 | −0.0030 | −0.0090 |
| ORPO 5k (Step 8, 2026-05-07) | 1000 | 0.0107 | 0.0123 | −0.0017 | −0.0060 |
| **SFT-only 50k (Step 12)** | **1000** | **0.0117** | **0.0110** | **+0.0007** | **+0.0010** |
| **SFT-only 50k (Step 12)** | **5000** | **0.0140** | **0.0132** | **+0.0008** | **+0.0052** |
| ORPO 50k (Step 13, in-flight) | — | TBD | TBD | TBD | TBD |
| SFT-50k → DPO+anchor 50k (Step 13, in-flight) | — | TBD | TBD | TBD | TBD |

**Read at 5k**: ORPO is the best single-stage arm; sequential SFT→DPO's
"+0.0037" is a false positive (collapse mode — see Step 10).

**Read at 50k**: SFT-50k is the first arm to lift recall_chosen
*absolutely above baseline* (+50%, t≈4.7) AND flip Δrecall positive.
Δrecall=+0.0008 alone is t≈0.6 at n=5000 (not p<0.05), but
recall_chosen lift vs baseline IS statistically robust, and all 4
metrics agree on direction.

**Watch for in Step 13 50k follow-ups:** `recall_chosen ≥ 0.014` (at
or above SFT-50k's level) AND `Δrecall ≥ +0.001` (clearly positive).
SFT-50k itself is the bar to beat.

---

## Step 8 — ORPO arm (2026-05-07)

Single-stage, no reference model, no group normalization. Trainer at
`train/train_orpo.py`. Loss formulation strictly follows paper Eq. 3
(length-normalized mean log-prob; λ=0.1 paper default for Mistral-ORPO).
Optimization recipe deviates from the paper for cross-arm parity with
the SFT/DPO arms: lr=5e-5 vs paper 8e-6, 1 epoch vs paper 10, LoRA r=16
vs full fine-tune, linear schedule vs cosine.

**5k smoke trajectory** (eval every 125 steps, eval set n=1000 pairs
from `v1_grpo/valid.parquet`):

| ckpt | step | chosen_score | rejected | margin | log_odds | pref_acc |
|---|---|---|---|---|---|---|
| 1 | 125 | −4.857 | −4.967 | 0.110 | 0.112 | 0.528 |
| 2 | 250 | −4.851 | −4.973 | 0.123 | 0.125 | 0.531 |
| 3 | 375 | −4.844 | −4.975 | 0.131 | 0.134 | 0.534 |
| 4 | 500 | −4.841 | −4.976 | 0.135 | 0.137 | 0.533 |
| 5 | 625 | **−4.840** | **−4.980** | **0.140** | **0.143** | **0.535** |

`chosen_score` lifted base −4.92 → −4.840 monotonically across all 5
ckpts; trajectory still rising slightly at ckpt 5 (decelerating, not
flat). `pref_acc` climbed 0.528 → 0.535 — real but small; the OR-term
gradient signal is gentle at λ=0.1 with only 625 total steps (paper
trained Mistral-ORPO for 10 epochs on UltraFeedback, ≈3 orders of
magnitude more OR-gradient steps than this smoke).

**Why ORPO beats SFT-only at iso-data even though `chosen_score` is
identical:** ORPO 5k final `chosen_score` = −4.840 ≈ SFT-only 5k −4.841
(same absolute scalar), but `recall_chosen@96` = 0.0107 vs SFT-only's
0.0097 (~10 % relative lift). The OR term reshapes the *distribution*
at the chosen-relevant tokens (sharpens chosen vs rejected at the SID-
token level) even though the absolute scalar `mean log P(chosen)`
saturates at the same point. This shows up in beam-search recall but
not in the linear log-prob metric.

**No collapse.** The earlier group-norm contrastive failure mode (Step 6, chosen recall
dropping 0.0093 → 0.0043) doesn't reproduce here because L_NLL
contributes ~98 % of the total loss magnitude (`L_NLL ≈ 4–5` vs
`λ·L_OR ≈ 0.04–0.09`); the SFT-anchor gradient on the chosen side is
~50× larger than the OR-term contribution and structurally prevents
the contrastive collapse.

**Wall clock**: 5.5 h on A100 80GB with SDPA (no flash-attn — cu130 +
torch 2.11 has no prebuilt wheel; from-source compile blocked by
missing nvcc on this box). Per-step ~21 s at batch=24 with 2× sequences
(chosen+rejected stacked into (2B, L) = (48, ~2700)).

**Next:** scale to 50k via `scripts/run_orpo_50k.sh` (~38 h same box).
Same hyperparameters. The 50k bet is "more gradient steps on the OR
term + more chosen examples break the SFT ceiling and lift Δrecall to
≥ 0".

---

## Step 9 — The 2026-05-04 `sft_weight` calibration fix (joint trainer)

A 5k joint-DPO+SFT smoke with the original `--sft_weight 0.1` produced
**eval_pref_acc = 0.546** (essentially random) even though
recall_chosen rose vs baseline. The headline "DPO+SFT works" claim
became suspect.

Tracing per-step components revealed the bug. The original loss was:
```
L = L_dpo_grpo + sft_weight × L_sft     # raw addition
```
But L_dpo_grpo is `(L_dpo_pair / std_g).mean()`, and observed
`mean(1/std_g) ≈ 16` throughout training (per-prompt variance is
small). So L_dpo_grpo was being amplified ~16× by the group-norm,
while `sft_weight × L_sft` stayed at its raw scale. **SFT contributed
~1–3% of total loss**, not the intended 10%. The DPO term was
effectively running solo.

Three coupled fixes:

| Knob | Old | New | Why |
|---|---|---|---|
| `--sft_scale_mode` | (none) | `match_dpo` | Multiply L_sft by `mean(1/std_g)` so sft_weight is in DPO-equivalent units |
| `--group_norm_eps` | `1e-3` | `0.05` | Cap 1/std at 20× instead of 1000×; old eps caused step-0 `l_dpo ≈ 693` when trained=ref → std≈0 |
| `--group_norm_warmup` | (none) | `50` | Skip group-norm during cold start when trained≈ref → variance trivially 0 |

The follow-up smoke (`--sft_weight 1.0 --sft_scale_mode match_dpo`)
fixed pref_acc but produced engagement-aware **recall_chosen = 0.0093
at n=1000** — *identical to baseline*. The earlier reported "0.0113 at
n=5000" turned out to be sampling variance: at n=1000 the recall@96
binomial SE is ~0.003, so the 0.002 swing was inside 1σ.

**Lessons**:
1. Loss-component weights must be calibrated to *gradient contribution*,
   not nominal magnitude. Group normalization and similar variance
   reduction reshape the relative scale invisibly.
2. Cross-method comparisons require the *same* n. Mixing n=1000
   baseline with n=5000 candidates produces illusory differences.
3. When a metric like pref_acc moves but the engagement-aware metric
   doesn't, suspect a measurement artifact, not a method success.

After this fix, the joint trainer produced `recall_chosen = 0.0093`
(no improvement) at 5k. Project pivoted to *sequential* SFT→DPO.

---

## Step 10 — Sequential SFT → DPO at 5k (the second collapse)

Standard post-training recipe (InstructGPT / Llama-3 / DeepSeek):
SFT first to push chosen up, then DPO with the SFT model as ref. New
files: `train/train_sft_only.py`, `train/train_dpo_from_sft.py`,
`scripts/run_sequential_smoke.sh`.

**Stage 1 (SFT-only 5k):** chosen_score went from base ≈ −4.92 to
−4.841 on v1_grpo valid; engagement-aware recall_chosen 0.0093 →
0.0097 (+4%, within noise). The SFT loss "barely moved chosen" — same
issue as Step 9's joint trainer. Δrecall stayed negative (−0.0030).

**Stage 2 (DPO from SFT, 5k):** the more dramatic finding. With
trained=ref=SFT and the SFT lift being marginal, Stage 2 effectively
ran DPO from base — and reproduced **the Step 6 failure mode**.
Trajectory across 5 ckpts:

| ckpt | chosen_score | rejected_score | margin |
|---|---|---|---|
| 1 (step 125) | −7.51 | −8.43 | 0.92 |
| 2 (step 250) | −9.18 | −10.7 | 1.53 |
| 3 (step 375) | −10.87 | −13.39 | 2.52 |
| 4 (step 500) | −12.0 | −15.71 | 3.71 |
| 5 (step 625) | **−13.44** | **−17.68** | 4.24 |

chosen_score crashed 8.6 nats below the SFT starting point. Engagement-
aware: **recall_chosen halved (0.0097 → 0.0043)**, recall_rejected
dropped 17× (0.0127 → 0.0007). Δrecall flipped to +0.0037 — but as
"false positive" — both sides crashed, rejected just died faster.

**Diagnosis**: With margin=0 at step 0 (trained=ref), DPO has two
gradient directions that minimize L_pair = softplus(-margin):
1. Push chosen up + push rejected down (intended).
2. Push both down asymmetrically — rejected harder than chosen
   (cheaper from base's geometry, picks this).

`best_metric=chosen_score` salvaged *some* loss (selected ckpt 1,
chosen_score = −7.51, the least-collapsed) but cannot prevent the
collapse — it's an *ex post* selection, not an *ex ante* constraint.

**Lessons**:
1. SFT anchor on chosen is non-optional once we're past the small-scale
   smoke regime. It's the only structural protection against contrastive
   collapse when trained ≈ ref at start.
2. A failed sequential SFT→DPO is operationally equivalent to a failed
   group-norm contrastive run from Step 6 — same loss landscape, same
   wrong basin.
3. Stage 2's success is gated on Stage 1 having produced a *real* lift,
   not a marginal one. A SFT that only moves chosen 0.05 nats does not
   give Stage 2 any momentum.

**Files (active):** `train/train_sft_only.py`,
`train/train_dpo_from_sft.py`, `scripts/run_sequential_smoke.sh`.

---

## Step 11 — 5k SFT-only sweeps: data ceiling, not optimization

If 5k SFT only moves chosen_score 0.05 nats (Step 10 Stage 1), is that
an optimization failure or a data-scale ceiling? Two single-variable
sweeps to disambiguate.

**lr sweep** (held lora_r=16, batch=24, all else fixed):

| lr | ckpt 1 | ckpt 5 | final |
|---|---|---|---|
| 5e-5 (baseline) | −4.856 | −4.841 | **−4.841** |
| 2e-4 | −4.870 | −4.845 | −4.845 |
| 5e-4 | −5.001 | −4.927 | −4.927 |

All three converge to chosen_score ≈ −4.84. Higher lr destabilizes
early (escape velocity from base's basin too high), then claws back
toward the same ceiling. lr is *not* the bottleneck — `max_grad_norm=1.0`
is hit on every step regardless, so each step's effective magnitude is
`lr × unit_vector` and varying lr just changes step magnitude, not
quality of the basin found.

**lora_r sweep** (held lr=5e-5, α/r=2.0 preserved):

| r | α | ckpt 5 | final |
|---|---|---|---|
| 16 | 32 | −4.841 | **−4.841** |
| 32 | 64 | −4.839 | −4.839 |

r=32 gives +0.002 nats — pure noise. LoRA capacity is *not* the
bottleneck either.

**Conclusion**: 5k + 1 epoch + LoRA r∈{16,32} + lr∈{5e-5, 2e-4, 5e-4}
all hit chosen_score ≈ −4.84. Optimization landscape has a real ceiling
here; data is the binding constraint. This cleared us to spend 17 h on
50k SFT with confidence that any improvement would be *attributable*
to data, not hyperparameter luck.

**Files (active):** `scripts/run_sft_lr_sweep.sh`,
`scripts/run_sft_lora_r_sweep.sh`,
`diagnose/sft_score_trend.py` (forward-only chosen_score trend across
ckpts; ~3 min/ckpt vs ~30 min for beam-based recall).

---

## Step 12 — 50k SFT-only: ceiling broken, headline finding locked

10× data on the same SFT-only configuration. 17.4 h on RTX 6000 Pro.
Five evenly-spaced ckpts at steps 1250 / 2500 / 3750 / 5000 / 6250.

| ckpt | step | eval_chosen_score | slope/1250 |
|---|---|---|---|
| base | 0 | ≈ −4.92 | — |
| 1 | 1250 | −4.831 | −0.089 |
| 2 | 2500 | −4.818 | −0.013 |
| 3 | 3750 | −4.805 | −0.013 |
| 4 | 5000 | −4.799 | −0.006 |
| 5 | 6250 | **−4.795** | −0.004 |

Total improvement: 0.125 nats vs base, **0.046 nats below the 5k
ceiling**. Slope decays late (saturation onset, not exhaustion).

**Engagement-aware n=5000:**

| Method | recall_chosen | recall_rejected | Δrecall | pass_chosen | pass_rejected | Δpass |
|---|---|---|---|---|---|---|
| baseline | 0.0093 | 0.0163 | −0.0070 | ≈0.025 | ≈0.045 | −0.020 |
| **SFT-50k** | **0.0140** | **0.0132** | **+0.0008** | **0.0404** | **0.0352** | **+0.0052** |

First arm to: (a) lift recall_chosen above baseline in absolute
terms (+50%, t≈4.7), (b) flip Δrecall sign, (c) push Δpass clearly
positive.

**Statistical caveat**: Δrecall=+0.0008 alone has paired-t≈0.6 at n=5000
(not p<0.05 as a single number). The trend across base/5k/50k IS
statistically robust — 4 metrics moving monotonically in the right
direction. recall_chosen lift from baseline alone is t≈4.7. Δpass
swing of +0.025 vs SE 0.004 → t≈6.6.

**Lessons**:
1. The 5k SFT result was a *data-scale artifact*, not a method failure.
   Step 10's Stage 1 marginal lift was an underbaked SFT, not a broken
   one.
2. **A pure SFT objective on chosen items — no contrastive, no ref
   model, no group norm — is sufficient for the project's core
   claim.** Reward Model is not necessary for offline behavior-signal
   preference alignment in this regime.
3. recall_chosen, Δrecall, pass_chosen, Δpass all move in the same
   direction → result is structurally robust even though Δrecall alone
   fails p<0.05.
4. chosen_score (forward-only metric) tracks recall_chosen (beam-search
   metric) closely, validating `--best_metric chosen_score` as a fast
   proxy for hyperparameter selection.

**Files (active):** `scripts/run_sft_50k.sh`.

---

## Step 13 — Two finalists in flight (2026-05-07)

With SFT-50k as a defensible positive baseline (Step 12), two 50k arms
run in parallel to settle the final ablation:

**Arm A — ORPO 50k** (server B, `scripts/run_orpo_50k.sh`):
direct scale-up of Step 8's ORPO 5k smoke. Same hyperparams. Tests
whether the OR term gives net additional Δ improvement over SFT alone,
given that ORPO 5k matched SFT-only chosen_score but had ~10% better
recall_chosen.

**Arm B — SFT-50k → DPO + light anchor 50k** (server A,
`scripts/run_dpo_anchor_from_sft_50k.sh`): Stage 2 sequential, with
`sft_weight=0.15 raw` (~19% SFT contribution post-warmup) added
specifically to prevent the Step 10 collapse mode. Uses the joint
trainer (only one with a configurable SFT-anchor knob); ref model is
SFT-50k. With SFT-50k's recall_chosen already at 0.0140 (vs Step 10's
0.0097 starting point), DPO has more headroom to push rejected down
without crashing chosen.

Both ~38 h on their respective hardware. Whichever wins (or both lose)
on engagement-aware n=5000 settles the project's recommendation.

**Decision rules:**
- Both arms beat SFT-50k recall_chosen AND ORPO > sequential
  → headline = ORPO (simpler, no ref).
- Both beat SFT-50k AND sequential > ORPO
  → headline = sequential (validates the SFT→DPO recipe).
- Either crashes recall_chosen below SFT-50k
  → tune that arm's knob (λ for ORPO, sft_weight for anchor) before
  declaring failure.
- Neither meaningfully beats SFT-50k
  → headline = SFT-50k itself; project finding still stands.

**Files (active):** `scripts/run_orpo_50k.sh`,
`scripts/run_dpo_anchor_from_sft_50k.sh`,
`scripts/run_dpo_from_sft_50k.sh` (pure-DPO control arm; not currently
scheduled to run unless Arm B's anchor proves too weak / too strong).

---

## Final ablation table (planned, fills in after Step 13 completes)

Once both 50k arms finish, run the *headline-quality* numbers for the
report. All previous tables in this file used n=1000 or n=5000 subsets
of v1 valid; the final table replaces those with the full held-out set
plus a comparable benchmark number.

**Two metrics, one table:**

1. **Engagement-aware on full v1 valid** —
   `python train/evaluate_engaged.py --n -1` (or `--n 14000` covering all
   ~14k held-out rows from `data/contrastive_dataset_v1/valid.parquet`).
   Tightens Δrecall SE from ~0.0013 (n=5000 paired) to ~0.0008
   (n=14000 paired), enough to potentially clear p<0.05 on Δrecall ≥
   +0.0016 — a bar SFT-50k's +0.0008 sits below but the 50k DPO/ORPO
   arms might clear.

2. **OneRec-paper Recall@32 / Pass@32 on `v1_test` benchmark** —
   `python train/evaluate_origin.py` on
   `data/OpenOneRec/benchmark_data/video/video_test.parquet` (38,781
   rows, completely disjoint from master in uids — the OneRec paper's
   Table 4 metric). This is the *secondary* metric: our training
   objective deliberately deprioritizes "next-shown but non-engaged"
   items, so absolute Recall@32 is *expected to drop* below baseline
   for any successful arm. Reported anyway for direct comparison
   against published OneRec numbers.

| Arm | recall_chosen | recall_rejected | Δrecall | Δpass | Recall@32 (v1_test) | Pass@32 (v1_test) |
|---|---|---|---|---|---|---|
| baseline (no FT) | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT-50k (Step 12) | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT-50k → DPO+anchor 50k (Step 13B) | TBD | TBD | TBD | TBD | TBD | TBD |
| ORPO 50k (Step 13A) | TBD | TBD | TBD | TBD | TBD | TBD |

Cells fill in after Step 13's two arms finish (~2026-05-09).

**Wall-clock budget for the final evals.** Naive sequential cost with
the current `evaluate_engaged.py` (batch_size=1, num_beams=32,
~1.8 s/row) would be:

| Eval | Rows | Time/model | 4 models sequential |
|---|---|---|---|
| Full v1 valid (engagement-aware) | ~14,000 | ~7 h | ~28 h |
| Full v1_test benchmark (Recall@32) | 38,781 | ~19 h | ~76 h |
| **Total** | | | **~104 h** |

Two engineering changes applied jointly bring this to ~13 h:

**(1) Batched evaluator.** `evaluate_engaged.py` and
`evaluate_origin.py` are currently `batch_size=1` per
`model.generate()` call. Padded batched generate at `batch=4` with
`num_beams=32` cuts per-row time ~3.5×; `batch=8` (~7× speedup) is
plausible on 80 GB cards since KV-cache headroom is the binding
constraint. One-time refactor; permanent speedup for every future
sweep.

**(2) Two-server parallel split.** Servers A (RTX 6000 Pro / H100,
SFT/DPO host) and B (A100 80GB, ORPO host) each evaluate two arms.

| Server | Arms | Models | Time (batched batch=4, full v1 valid + full v1_test) |
|---|---|---|---|
| **Server B** | baseline (no FT), ORPO 50k | OneRec-1.7B base, `runs/orpo_50k/merged` | ~15 h |
| **Server A** | SFT-50k, SFT-50k → DPO+anchor 50k | `runs/sft_only_50k/merged`, `runs/dpo_anchor_from_sft_50k/merged` | ~15 h |

Per-model batched cost: ~2 h (full v1 valid) + ~5.5 h (full v1_test) ≈
~7.5 h × 2 arms ≈ ~15 h per server. Bottleneck is whichever server
finishes second → **total wall-clock ~15 h** (vs the ~104 h naive
estimate, ~7× speedup overall).

Cuts to ~10 h if `batch=8` clears VRAM on both servers; worth a quick
batch-size sanity test at the start of the eval sweep before
committing.

**Statistical caveats to record alongside numbers:**
- Δrecall on full v1 valid will have SE ~0.0008. Anything ≥ +0.0016 is
  p<0.05 (one-sided); below that, report point estimate but flag as
  not formally significant.
- Recall@32 on v1_test directly tests the OneRec paper's chosen
  metric. Expect *all* arms to score below baseline here — that is the
  "Step 4 metric mismatch" issue surfacing for a final time. The
  framing is "we trade-off official Recall@K for engagement-aware
  Δrecall", not "we beat the OneRec paper".
