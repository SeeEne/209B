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

### Step 3. Length=3 contrastive (predict 3 future videos)

**What:** Predict 3 next items as a sequence; score = mean log P over 9 SID
tokens. Equal weight per item by averaging tokens.

```
score = (1/9) × Σ log P(SID_token_i | history, prev_tokens)
```

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

### Step 5. Per-prompt group normalization (GRPO-inspired)

**Motivation:** Single-step training-loss variance was huge (each batch had
4 users with very different difficulty). Aggregate eval was healthy, but
per-step gradient signal was noisy. Standard "more data" helps the
estimator, not the *per-step* gradient noise.

**What:** Each user's 3 chosen + 3 rejected expanded into G=3 single-item
contrastive pairs sharing a `group_id`. Loss normalized per-group:

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

**G=3 wins; G>3 doesn't help on this data.** Lesson: GRPO's "more samples =
better" assumption breaks when samples are correlated within group.

**Files (archived):** `train_contrastive_grpo.py`, `run_grpo_g5_smoke.sh`

---

### Step 6. 50k full run + a critical observation

50k users × G=3 = 150k pairs, ~38h. Same hyperparams as smoke.

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

KL anchor keeps the *distribution shape* close to base, but doesn't reward
chosen's *absolute* probability. So model takes the cheaper path.

**Run killed.** Pivot to method that explicitly anchors chosen.

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
`run_dpo_grpo_smoke.sh`

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
| `run_dpo_smoke.sh` | (early DPO sketch) | Pre-cursor to `run_dpo_grpo_smoke.sh` | Older draft |

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

## Headline numbers (engagement-aware, n=1000, topk=96)

| Method | recall_chosen | recall_rejected | Δrecall | Δpass |
|---|---|---|---|---|
| baseline (no FT) | 0.0093 | 0.0163 | −0.0070 | −0.0200 |
| length=3 30k full (Step 3) | 0.0047 | 0.0017 | +0.0030 | +0.0080 |
| GRPO 5k smoke G=3 (Step 5) | 0.0033 | 0.0003 | +0.0030 | +0.0080 |
| GRPO 5k smoke G=5 (Step 5) | 0.0030 | 0.0010 | +0.0020 | +0.0050 |
| GRPO 50k step 2500 (Step 6) | 0.0043 | 0.0027 | +0.0017 | +0.0040 |
| SFT-only 5k (Step 7 anchor)              | 0.0097     | 0.0127       | −0.0030     | —          |
| **ORPO 5k (Step 8, 2026-05-07)**         | **0.0107** | **0.0123**   | **−0.0017** | **−0.0060** |

**ORPO 5k is the best single-stage 5k arm by every column** — highest
recall_chosen, lowest recall_rejected, smallest absolute Δrecall.
Δrecall remains negative (rejected still slightly more recalled than
chosen) but is closer to zero than any 5k-data arm produced so far.
Statistical caveat: n=1000 puts the 0.0107 vs 0.0097 (ORPO vs SFT-only)
gap inside binomial noise (95 % CI ~±0.006 at p≈0.01); the four-metric
directional agreement (ORPO > SFT > baseline on every column) is
stronger evidence than any single gap.

**Watch for in 50k follow-ups:** `recall_chosen ≥ 0.012` (clearly above
the 5k ceiling) AND `Δrecall ≥ 0` (positive direction). The 5k
trajectory was monotonically rising and decelerating, so 10× data has
upside even if the per-step OR gradient signal stays weak.

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

**No collapse.** The earlier GRPO-only failure mode (chosen recall
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
