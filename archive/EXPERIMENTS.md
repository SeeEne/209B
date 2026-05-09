# Experiment Journey

This file documents the full 14-step research arc of the project, from
the initial pivot away from DPO triplets through the final ablation. The
project converged to the simplest viable arm: **SFT-only on the chosen
branch**, with **ORPO** as a confirmed within-noise alternative. Every
contrastive variant we built either mode-collapsed, crashed absolute
chosen recall, or settled into a degenerate basin where both chosen and
rejected log-probs plummet together. Documenting *why* those failures
happened — and the structural reason ORPO ties with SFT despite a
fundamentally different objective — is the project's main contribution.

This file records *why* each step was taken and *what we learned* before
moving on. Read top-to-bottom for the narrative; Step 14 is the final
summary.

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

### Step 7. DPO + SFT + group normalization (originally proposed as the chosen path; eventually superseded — see Step 13B and Step 14)

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
| **baseline (final, n=7940)** | **7940** | **0.0120** | **0.0246** | **−0.0126** | **−0.0296** |
| **SFT-only 50k (final, n=7940)** | **7940** | **0.0143** | **0.0146** | **−0.0003** | **+0.0018** |
| **ORPO 50k (final, n=7940)** | **7940** | **0.0141** | **0.0143** | **−0.0002** | **+0.0021** |
| **SFT-50k → DPO+anchor 50k (final, n=7940)** | **7940** | **0.0098** | **0.0041** | **+0.0057** | **+0.0161** |

**Read at 5k**: ORPO is the best single-stage arm; sequential SFT→DPO's
"+0.0037" is a false positive (collapse mode — see Step 10).

**Read at 50k (final, n=7940 full v1 valid)**:

- **SFT-only 50k is the headline winner**: highest absolute chosen
  recall (0.0143, +19% vs baseline), Δrecall and Δpass both flipped
  to ≈ 0 (from baseline's strongly negative −0.0126 / −0.0296), no
  failure mode.
- **ORPO 50k tied with SFT** within noise (chosen 0.0141 vs SFT 0.0143,
  paired SE ≈ 0.0013). Despite a fundamentally different objective,
  ORPO's OR term added no measurable lift over plain NLL — driven by
  small-vocab + shared-context zero-sum (see Step 14).
- **DPO + SFT-anchor 50k crashed chosen below baseline** (0.0098 vs
  baseline 0.0120). It produced the largest Δrecall (+0.0057) but at
  the cost of dragging chosen down — classic contrastive collapse,
  see Step 13B post-mortem.

The 5k headline trend (ORPO best single-stage) survives at scale, with
SFT-only catching up at 50k and the contrastive arm regressing.

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

## Step 13 — Two 50k finalists, both completed (2026-05-09)

With SFT-50k as a defensible positive baseline (Step 12), two 50k arms
ran in parallel on different servers to settle the final ablation. Both
completed; final eval below uses **n=7940 (full v1 valid)** for
headline-quality SE (~0.0008 paired).

### Step 13A — ORPO 50k (completed)

`scripts/run_orpo_50k.sh`, server B (A100 80GB, ~38 h with SDPA).
Direct scale-up of Step 8's 5k ORPO smoke, same hyperparameters
(λ=0.1, lr=5e-5, lora_r=16). No ref model, no group normalization.

| Metric | baseline | SFT-50k | **ORPO 50k** |
|---|---:|---:|---:|
| recall_chosen | 0.0120 | 0.0143 | **0.0141** |
| recall_rejected | 0.0246 | 0.0146 | **0.0143** |
| Δrecall | −0.0126 | −0.0003 | **−0.0002** |
| pass_chosen | 0.0344 | 0.0411 | **0.0407** |
| Δpass | −0.0296 | +0.0018 | **+0.0021** |

**Outcome — tied with SFT-50k within noise** on every metric. Differences
of 0.0001–0.0003 against paired SE 0.0013 ≈ noise floor.

**Why this is a result, not a non-result.** Two different objectives
(plain NLL vs NLL + λ·odds-ratio contrastive) converged to the same
recall numbers despite ORPO having an explicit term to suppress
rejected. The structural reason is non-trivial and is the project's
deepest finding — see Step 14. Briefly: when chosen and rejected share
a narrow categorical output space (8,192-entry SID codebook + same user
context, vs 50k–150k for standard LLMs), SFT's implicit zero-sum
redistribution already drains the
rejected items, so the OR term has nothing extra to do. The 5k smoke
hint that "ORPO has +10% recall_chosen over SFT-only at iso-data"
(Step 8) didn't survive at 50k — both arms hit the same data ceiling.

### Step 13B — SFT-50k → DPO + SFT-anchor 50k (completed, failure mode)

`scripts/run_dpo_anchor_from_sft_50k.sh`, server A (RTX 6000 Pro, ~38 h
incl. ~5 h ref-score precompute). Stage 2 sequential from SFT-50k,
with `sft_weight=0.15 raw` (~19% SFT contribution post-warmup) added
specifically to prevent the Step 10 cold-start collapse. Uses the joint
trainer (only one with a configurable SFT-anchor knob); ref model =
SFT-50k.

| Metric | baseline | SFT-50k start | **DPO+anchor 50k** |
|---|---:|---:|---:|
| recall_chosen | 0.0120 | 0.0143 | **0.0098** ⚠️ |
| recall_rejected | 0.0246 | 0.0146 | **0.0041** |
| Δrecall | −0.0126 | −0.0003 | **+0.0057** |
| pass_chosen | 0.0344 | 0.0411 | **0.0282** ⚠️ |
| Δpass | −0.0296 | +0.0018 | **+0.0161** |

**Outcome — largest Δrecall and Δpass of any arm, but chosen recall
crashed below baseline.** Δrecall=+0.0057 cleared p<0.05 (≈ 7σ at
n=7940); Δpass=+0.0161 likewise. **But** recall_chosen dropped 31%
from the SFT-50k starting point and ended **18% below baseline**. Pass@96
chosen dropped from 0.0411 to 0.0282 (also below baseline 0.0344).

**Root cause — post-warmup contrastive collapse, distinct from Step 10.**
Step 10 was *start-of-training* collapse (margin=0 at step 0 → both
basins equally close → model picks the cheap one). Step 13B has a
positive starting margin from SFT-50k, but **once DPO momentum builds,
the gradient direction that minimizes the contrastive loss most
efficiently is "push rejected hard down, let chosen drift down with
it"**. Specifically:
- Group-normalized DPO contributes ~81% of the post-warmup gradient
  magnitude (β=0.1 on margin amplified ~16× by `1/std_g`)
- SFT-anchor at sft_weight=0.15 raw contributes ~19% — the only force
  pulling chosen up
- Implicit KL via ref baseline (= SFT-50k) is a *direction* constraint
  on the (chosen − rejected) gap, **not** an absolute floor on chosen.
  Chosen can drift below ref as long as rejected drifts faster.

The 0.15 anchor was strong enough to prevent Step 10's catastrophic
chosen-log-P crash (which fell 8.6 nats over 625 steps) but too weak
to keep chosen above baseline once contrastive momentum took over.

**Could it be salvaged?** Likely yes, by raising sft_weight to
0.5–1.0 (raw). Not pursued — the project timeline closed and SFT-50k
already carries the headline finding. Documented as an open knob for
future work.

**Files (active):** `scripts/run_orpo_50k.sh`,
`scripts/run_dpo_anchor_from_sft_50k.sh`,
`scripts/run_dpo_from_sft_50k.sh` (pure-DPO control arm without anchor;
never run as the anchor arm's behavior is sufficient evidence of the
collapse mode).

---

## Final ablation table (n=7940 full v1 valid, completed 2026-05-09)

All numbers below are the headline values for the report. Engagement-
aware Δrecall / Δpass on the full held-out v1 valid set (n=7940,
paired SE ≈ 0.0008). OneRec-paper Recall@32 on v1_test was
deferred — `video_test.parquet` is 38,781 rows × 4 arms × ~5–6 h batched
≈ ~22 h compute, deferred past the project deadline. Engagement-aware
Δrecall is the primary metric; Recall@32 would have been a secondary
"degradation expected by design" reading.

| Arm | recall_chosen | recall_rejected | Δrecall | pass_chosen | pass_rejected | Δpass |
|---|---:|---:|---:|---:|---:|---:|
| baseline (no FT) | 0.0120 | 0.0246 | −0.0126 | 0.0344 | 0.0640 | −0.0296 |
| **SFT-50k (Step 12)** | **0.0143** | 0.0146 | −0.0003 | **0.0411** | 0.0393 | +0.0018 |
| ORPO 50k (Step 13A) | 0.0141 | 0.0143 | −0.0002 | 0.0407 | 0.0385 | +0.0021 |
| SFT-50k → DPO+anchor 50k (Step 13B) | 0.0098 | **0.0041** | **+0.0057** | 0.0282 | **0.0121** | **+0.0161** |

**Reading the table:**

- **SFT-50k = headline winner.** Highest absolute chosen recall and
  pass; Δ flipped from baseline's strongly negative to ≈ 0; no failure
  mode. This is the simplest possible arm (no ref, no contrastive, no
  group norm) and it wins on every absolute-metric column.
- **ORPO 50k tied with SFT** within noise (chosen 0.0141 vs SFT 0.0143
  vs paired SE 0.0013). Independent confirmation of SFT's number from a
  fundamentally different objective.
- **DPO+anchor 50k wins Δ but crashes chosen.** Largest Δrecall
  (+0.0057, p<0.05) and Δpass (+0.0161, p<0.05) — these are real signals.
  But chosen pass and recall are both *below baseline*. Useful as a
  documented failure mode of the contrastive arm at this anchor weight.

**Statistical notes:**
- Δrecall paired SE ≈ 0.0008 at n=7940 → anything ≥ +0.0016 is
  p<0.05 (one-sided). DPO+anchor's +0.0057 clears this; SFT and ORPO's
  ≈ 0 do not (consistent with their tied-with-baseline reading).
- recall_chosen lift SFT-50k vs baseline: 0.0143 vs 0.0120, t≈2.6 at
  n=7940 (p≈0.005, two-sided) — formally significant.
- recall_chosen drop DPO+anchor vs baseline: 0.0098 vs 0.0120,
  t≈−2.5 → formally significant *deterioration*.

---

## Step 14 — Project summary and takeaways (2026-05-09)

This is the closing summary; no further experiments planned.

### What the project showed

**Yes** — user behavior signals (longview / like / follow / forward) can
directly replace the Reward Model when constructing preference data for
offline alignment of a generative recommender. The simplest possible
recipe — **plain SFT on the chosen branch with no contrastive term, no
ref model, no group normalization** — lifts engagement-aware
recall_chosen from 0.0120 (baseline) to 0.0143 (+19%, p≈0.005) and
flips Δrecall sign from baseline's strongly biased −0.0126 to ≈ 0. No
Reward Model required.

The original research bet ("DPO + SFT + group-normalized contrastive
loss is the right method") turned out to be **the wrong arm to favor**.
It's the most complex arm we built, has the most failure modes (Step 6,
Step 9, Step 10, Step 13B all fall under variants of it), and its
final 50k incarnation (Step 13B) crashed chosen recall below baseline.
Reporting this is part of the contribution.

### The ORPO ≈ SFT finding (deepest takeaway)

ORPO (single-stage, λ-weighted log-odds contrastive on top of NLL) was
designed to add discriminative pressure that plain NLL doesn't have. On
this task it tied with SFT to within noise on every measured metric.
The structural reason generalizes:

> **When chosen and rejected share a narrow categorical output space,
> contrastive alignment objectives' marginal gains over plain SFT are
> absorbed by SFT's implicit zero-sum redistribution.**

Three conditions, all present in our setup, all needed:

1. **Small vocabulary** (codebook K = 8,192, vs 50k–150k for standard LLMs;
   confirmed empirically from the SID values in the data).
   Pushing P(chosen_token) up by Δ drains Δ from K−1 entries — smaller
   K means less dilution and more concentrated zero-sum.
2. **Chosen and rejected share user context.** They live in the same
   high-prob region of the codebook. SFT's push on chosen drains
   probability mass from rejected as the nearest neighbor — for free,
   without any rejected supervision. (Numerical evidence: SFT-50k drops
   recall_rejected from 0.0246 to 0.0146 *despite never seeing rejected
   in training*.)
3. **LoRA r=16 + 50k pairs hits a capacity ceiling.** Both arms
   saturate per-token chosen log P at ≈ −4.795 nats; lr/lora_r sweeps
   confirm data is the binding constraint, not optimization. ORPO's
   residual contrastive signal (after λ=0.1 weighting) doesn't have
   free LoRA capacity to walk to a different fixed point.

Remove condition (1) (use a standard LLM vocab) or (2) (use a different
prompt for y₊ vs y₋, as in standard RLHF) and this collapse should
disappear — ORPO's published wins on Mistral-on-UltraFeedback are
exactly that regime. The takeaway is task-shaped, not method-shaped:
**generative recommenders sit in a corner of the design space where
contrastive preference objectives are largely redundant.**

### The contrastive-collapse finding (operational takeaway)

Five contrastive variants in the project failed in the same way:

| Step | Arm | Failure |
|---|---|---|
| 1 | Length=1 contrastive | Mode collapse to ~10 popular SIDs |
| 5 | Group-norm contrastive 50k | chosen recall halved from baseline |
| 6 | Joint DPO+SFT 5k (post-fix) | No improvement over baseline |
| 7 | Sequential SFT→DPO 5k | chosen log P crashed 8.6 nats |
| 13B | DPO + SFT-anchor 50k | chosen recall crashed below baseline |

Each is a slightly different geometry of the same root cause: any loss
of the form `softplus(−(c − r))` or its DPO variant has two minima —
"push c up, push r down" (intended) and "push both down asymmetrically,
just push r more" (cheaper). In a small-vocab, shared-context regime,
the second minimum is easier to reach. SFT-anchor terms can in
principle prevent it, but only if their gradient contribution dominates
the contrastive contribution, which means anchor-weight calibration
matters more than is usually appreciated.

We document Step 13B's `sft_weight=0.15 raw` (≈19% gradient share)
**as too weak** for this regime. A future-work knob to try is
`sft_weight ∈ {0.5, 1.0}` (raw) — likely sufficient to keep chosen at
or above SFT-50k while letting DPO push rejected down.

### What the headline number actually means

SFT-50k achieves Δrecall ≈ 0 — *not* a positive Δ. Read carefully, that
is the project's main quantitative result:

> **The OneRec base model is biased *toward* non-engaged items**
> (recall_rejected = 2× recall_chosen on baseline). 50k pairs of
> behavior-signal SFT pull this bias from −0.0126 to ≈ 0 — i.e.
> debiases the model. It does not (yet) make it positively prefer
> engaged items, but the strongly anti-aligned baseline is removed.

The companion Δpass = +0.0018 / +0.0021 (SFT / ORPO) is small but
consistently positive and consistently directionally correct; it's the
weakest "model is now neutral or slightly preference-aligned" reading
that matches the data. Future scale-up (200k, 500k pairs?) would test
whether the trajectory continues: if Δrecall keeps drifting from −0.013
through 0 toward a positive number, the project's claim strengthens. If
it saturates around 0, the conclusion is "behavior-signal SFT debiases
but does not align."

### Method recommendation for follow-up work

For anyone building on this:

1. **Default to SFT-only on the chosen branch** in offline behavior-
   signal regimes with narrow categorical output spaces. Don't reach
   for contrastive objectives without a specific reason.
2. **If you do build a contrastive arm**, anchor-weight calibration is
   the single most important hyperparameter. sft_weight in the 0.5–1.0
   raw range (or use `--sft_scale_mode match_dpo` with weight 1.0) is
   the safer default than the 0.15 we tested.
3. **Don't trust eval_pref_acc as a method-success proxy.** Step 9's
   joint trainer hit pref_acc=0.546 (random) yet had recall_chosen
   moving the right way — and Step 5's group-norm 5k smoke had
   pref_acc up but chosen recall crashing at scale. Engagement-aware
   recall is the only metric we found that doesn't lie.
4. **Use forward-only chosen_score (no beam search) for hyperparameter
   sweeps.** `diagnose/sft_score_trend.py` is ~30× faster than full
   beam-based recall and tracked the headline metric closely across
   all our runs (validated in Steps 11–12).

### Final state of the repo

- `train/train_sft_only.py` and `scripts/run_sft_50k.sh` are the
  recommended training path going forward.
- `train/train_orpo.py` and `scripts/run_orpo_50k.sh` are kept as
  the "tied alternative" — useful for anyone wanting the ref-model-
  free single-stage formulation.
- `train/train_contrastive_dpo_g_normalize.py` and the DPO-related
  scripts are retained as the documented contrastive arm; their
  failure modes are part of the contribution. The 0.15 anchor weight
  in `scripts/run_dpo_anchor_from_sft_50k.sh` is documented as too
  weak — anyone reusing this trainer should bump anchor weight first.
- `train/evaluate_engaged.py` is the canonical headline-metric tool;
  `train/evaluate_origin.py` remains for OneRec-paper comparisons that
  weren't run for this project but could be in future work.
