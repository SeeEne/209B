# OpenOneRec — Behavior-Signal Alignment without a Reward Model

CS 209B course project (Harvard, Spring 2026) building on Kuaishou's
[OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) generative
recommender. We replace OneRec's separately-trained Reward Model with
**direct user behavior signals** as the supervision source for preference
alignment, training with a **DPO + SFT + group-normalized contrastive loss**
on a single GPU.

---

## Research question

> In offline recommendation, can user behavior signals (longview / like /
> follow / forward) directly replace the Reward Model when constructing
> preference pairs for alignment training?

OneRec's online IPA module needs a Reward Model because user behaviors only
arrive *after* recommendation. We work **offline** on logged data, where
behavior signals are already in the parquet — so we can construct
positive/negative pairs directly and skip the Reward Model entirely.

---

## Method

For each user we have 3 chosen items (positive engagement: longview / like /
follow / forward) and 3 rejected items (no signal or `not_interested`). Each
user contributes G=3 single-item (chosen, rejected) pairs sharing a `group_id`,
paired by cyclic shift to break position alignment.

**Loss = DPO + SFT anchor + per-prompt group normalization:**

```
margin_p   = β × [(c_θ − c_ref) − (r_θ − r_ref)]          # DPO with ref baseline
L_dpo_p    = softplus(−margin_p)                           # per pair
L_dpo_grpo = (L_dpo_pair.view(num_groups, G) / std_g).mean()   # group-normalize
L_sft      = −c_θ.mean()                                   # absolute push for chosen
L_total    = L_dpo_grpo + sft_weight × L_sft
```

Each component does one job:
- **DPO** — discriminate chosen ≻ rejected, with implicit KL via reference baseline
- **SFT anchor** — keep absolute P(chosen) up; without it, contrastive can be minimized by *lowering* both chosen and rejected
- **Group normalization** — per-prompt std rescaling, reduces gradient noise from heterogeneous per-prompt difficulty

Standard hyperparameters: `--dpo_beta 0.1 --sft_weight 0.1 --kl_weight 0`.
LoRA on Qwen-3 attention + MLP layers (r=16, α=32).

> **Terminology note**: in code we call it "GRPO normalization" for brevity, but it's not full GRPO (no rollouts, no advantage subtraction — just per-group `1/σ` loss scaling). In writing we call it **"group-normalized contrastive loss"**.

---

## Key finding: standard Recall@K is the wrong metric

OneRec base model is biased toward *non-engaged* items in its top-K predictions
(it was trained for "next-shown" prediction, and most next-shown items are
passively scrolled rather than engaged). Standard Recall@K rewards exactly
this bias:

| | recall_chosen | recall_rejected | **Δ (C − R)** |
|---|---|---|---|
| OneRec baseline (no FT) | 0.0093 | 0.0163 | **−0.0070** ❌ biased toward rejected |
| Length=3 contrastive (30k) | 0.0047 | 0.0017 | **+0.0030** ✓ direction reversed |
| Length=3 + group norm (5k smoke) | 0.0033 | 0.0003 | **+0.0030** ✓ matches 30k with 1/6 data |
| **DPO + SFT + G-norm** | *running* | | |

The headline number is **Δ** (recall on engaged minus recall on non-engaged) on
the held-out contrastive-pair valid set. A baseline trained for "what gets
shown next" gives **Δ < 0**; our method flips the sign.

The OneRec paper's standard `Recall@K` measures next-shown coverage and is
*expected to drop* after our training — that's the price of teaching the model
to discriminate engagement. We report both metrics for context but treat the
engagement-aware Δ as the headline.

---

## Quick start

### Setup
```bash
python -m venv venv && source venv/bin/activate
pip install torch transformers peft accelerate pandas pyarrow tqdm
```

Download `OneRec-1.7B` to `model/` and the OpenOneRec dataset to `data/`. See
[CLAUDE.md](CLAUDE.md) § Auto-resolution for HF auth and template handling.

### Build data (one-time)
```bash
python build_contrastive_dataset.py --length 3        # → data/contrastive_dataset_v1/
python build_contrastive_dataset_GRPO.py --G 3        # → data/contrastive_dataset_v1_grpo/
```

### Train + evaluate (smoke, ~5h on RTX 6000 Pro)
```bash
bash run_dpo_smoke.sh
```

This runs the full pipeline end-to-end: train 5000 groups → merge LoRA →
evaluate engagement-aware metric. Logs to `runs/dpo_grpo_smoke.log`.

### Full 50k training
Edit `run_dpo_smoke.sh` (or call `train/train_contrastive_dpo_g_normalize.py`
directly) with `--max_train_groups 50000 --eval_steps 2500 --save_steps 2500`.
Wall clock ~38h on a single RTX 6000 Pro.

---

## Repository layout

```
project/
├── README.md                                  # this file (project overview)
├── CLAUDE.md                                  # internal dev guide (Claude Code agents)
│
├── build_contrastive_dataset.py               # master table → length-3 pairs (v1)
├── build_contrastive_dataset_GRPO.py          # v1 → per-pair format with group_id
├── merge_local.py                             # merge LoRA adapter into base for inference
├── run_dpo_smoke.sh                           # end-to-end smoke pipeline
│
├── train/                                     # active training + evaluation
│   ├── README.md
│   ├── train_contrastive_dpo_g_normalize.py   # ⭐ main trainer
│   ├── evaluate_engaged.py                    # ⭐ engagement-aware Δrecall metric
│   ├── evaluate_origin.py                     # OneRec official Recall@K (secondary)
│   ├── dataset.py                             # tokenization + prompt helpers
│   └── utils.py                               # template resolver
│
├── diagnose/                                  # diagnostic tools (see diagnose/README.md)
│   ├── compare_models.py                      # base vs trained side-by-side outputs
│   └── debug_recall.py                        # baseline-reproduction sanity check
│
├── archive/                                   # superseded experiments + journey log
│   ├── EXPERIMENTS.md                         # ⭐ 7-step research journey
│   └── ...                                    # old trainers, evaluators, scripts
│
├── notebook/                                  # MS2 EDA + MS3 narrative
├── oneRec/                                    # OneRec template + planning docs
├── data/                                      # not in git
└── runs/                                      # experiment outputs (not in git)
```

---

## Documentation map

| Doc | Audience | Read when |
|---|---|---|
| [README.md](README.md) (this file) | new readers, course graders | "What is this project?" |
| [CLAUDE.md](CLAUDE.md) | dev / agent | "How is the repo organized? What are the conventions?" |
| [train/README.md](train/README.md) | dev | "How do I train? What hyperparameters?" |
| [diagnose/README.md](diagnose/README.md) | dev | "How do I inspect a trained model?" |
| [archive/EXPERIMENTS.md](archive/EXPERIMENTS.md) | reviewer / future-self | "Why this method? What didn't work?" |
| [oneRec/eda_findings.md](oneRec/eda_findings.md) | dev / reviewer | "What did EDA reveal? Why pivot from DPO originally?" |

---

## What's *not* in the active path

Lots of code lives in `archive/` because it represents intermediate
experiments that taught us something but are no longer the production path.
The full chronology — what we tried, what broke, why we moved on — is in
[`archive/EXPERIMENTS.md`](archive/EXPERIMENTS.md).

Highlights:
- **Step 1**: pure length=1 contrastive → mode collapse (everyone got the same
  ~10 popular SIDs)
- **Step 2**: + KL regularization → recovered distribution shape but standard
  Recall@K still 0
- **Step 3**: realized standard Recall@K measures the *wrong thing* given our
  training signal → built engagement-aware metric
- **Step 5**: GRPO-style group normalization → matched length=3 30k results
  with 1/6 the data (variance reduction confirmed)
- **Step 6**: but pure GRPO causes chosen recall to **drop in absolute terms**
  → motivated SFT anchor → current method

---

## Course

CS 209B *Statistical Learning for Decision Making*, Harvard SEAS, Spring 2026.
Course project replacing the IPA Reward Model with offline behavior signals.

## Acknowledgments

- Kuaishou OpenOneRec team for the model and dataset release
- Course staff for project guidance
- DeepSeek-V3 / R1 papers for the GRPO normalization inspiration
