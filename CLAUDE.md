# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project (current direction)

CS 209B course project based on the **OpenOneRec** open-source dataset/framework (Kuaishou, 2025). The project replaces OneRec's separately-trained Reward Model with **direct user behavior signals** as the supervision source for preference alignment.

**Core research question:**
> In offline recommendation, can user behavior signals (longview / like / follow / forward) directly replace the Reward Model when constructing preference pairs for alignment training?

**Why this is novel.** OneRec's IPA module trains a separate Reward Model to score beam-search candidates because in their **online** setting user behaviors only arrive *after* recommendation. We work **offline** on logged data — every behavior signal is already in the parquet — so we can construct positive/negative pairs directly from behavior.

## Current method: DPO + SFT anchor + group-normalized loss

Per-pair DPO loss (reference model's scores baked in as a baseline → implicit KL):
```
margin_p = β × [(c_θ − c_ref) − (r_θ − r_ref)]
L_dpo_p  = softplus(−margin_p)
```

GRPO-inspired group normalization. Each user contributes G=3 (chosen, rejected) pairs sharing a `group_id`; per-group std rescales the per-pair loss:
```
L_dpo_grpo = (L_dpo_pair.view(num_groups, G) / std_g.detach()).mean()
```

SFT anchor on chosen — keeps absolute P(chosen) up so the contrastive part can't be minimized by *lowering* both sides:
```
L_sft   = −c_θ.mean()
L_total = L_dpo_grpo + sft_weight × L_sft
```

Standard hyperparameters: `--dpo_beta 0.1`, `--sft_weight 0.1`, `--kl_weight 0` (DPO has implicit KL).

**Why this stack** — see [`archive/EXPERIMENTS.md`](archive/EXPERIMENTS.md) for the 7-step research journey. Short version:
1. Pure contrastive (length=1) → mode collapse
2. + KL → distribution recovers, but Recall@K still 0
3. **Insight: Recall@K measures next-shown; we trained for engaged. Mismatched metric.**
4. Build engagement-aware Δrecall metric on held-out contrastive valid
5. Length=3 contrastive: clean directional reversal (Δpass +0.0080 vs baseline −0.020)
6. Group normalization: matches length=3 30k Δ with 1/6 the data
7. **Pure GRPO crashes chosen recall in absolute terms** → add SFT anchor + DPO formulation → current path.

## Terminology note

We call our group normalization "GRPO" in filenames (`_grpo_`), but **it isn't strictly GRPO**:
- We don't sample rollouts (pairs are pre-constructed from data)
- We don't subtract group mean as advantage (we'd get 0)
- We use `1/std` as a *loss scaling factor*, not as advantage weight on policy gradient

In writing/reports, call it **"group-normalized contrastive loss"** or **"GRPO-inspired"**. Filenames keep `_grpo_` for brevity.

## Data

Source dataset: `OpenOneRec/OpenOneRec-RecIF` (HF, gated).

```
data/OpenOneRec/
├── onerec_bench_release.parquet      # master training table, 162,074 rows
├── video_ad_pid2sid.parquet          # 15.9M video/ad pid → semantic ID
├── product_pid2sid.parquet           # 2.1M product pid → semantic ID
└── benchmark_data/
    ├── video/video_test.parquet      # 38,781 rows  (held-out, official benchmark)
    └── ...
```

Derived contrastive datasets:

```
data/contrastive_dataset_v1/          # ⭐ length=3 source — 3 chosen + 3 rejected per row
│   ├── train.parquet  ~125k rows     # also feeds evaluate_engaged.py
│   ├── valid.parquet  ~14k rows
│   └── meta.json
data/contrastive_dataset_v1_grpo/     # ⭐ G=3 per-pair format derived from v1
│   ├── train.parquet  ~214k rows  (= 71k groups × 3 pairs)
│   ├── valid.parquet
│   └── meta.json
data/contrastive_dataset_v1_grpo_g5/  # G=5 ablation (smoke only)
data/contrastive_dataset_v0/          # length=1 (legacy, length=1 era)
```

**v1_grpo** is what the current trainer reads. **v1** is the source for `evaluate_engaged.py` (the engagement-aware metric needs the 3+3 chosen/rejected per user).

**Schema notes:**
- All rows have `split=0` in HF release; `split==0` filter is a no-op.
- `target_video_*` behavior columns are `list<int64>` aligned 1:1 to `target_video_pid`.
- `sid` is a `list<int64>` of length 3.
- Benchmark `video_test.parquet` is **completely disjoint** from master in uids — engagement labels not recoverable for benchmark items via uid-join.

## Repo layout

```
project/
├── CLAUDE.md                                  # this file
├── build_contrastive_dataset.py               # master → v1 (length=3) data
├── build_contrastive_dataset_GRPO.py          # v1 → v1_grpo (per-pair, group_id, configurable G)
├── merge_local.py                             # merge LoRA adapter into base for inference
├── run_dpo_smoke.sh                           # main run script (5k smoke pipeline)
│
├── train/
│   ├── README.md
│   ├── dataset.py                             # SYSTEM_PROMPT + helpers (used by trainer)
│   ├── utils.py                               # resolve_template
│   ├── train_contrastive_dpo_g_normalize.py   # ⭐ MAIN trainer (DPO + SFT + G-norm)
│   ├── evaluate_engaged.py                    # ⭐ MAIN evaluator (engagement-aware Δrecall/Δpass)
│   └── evaluate_origin.py                     # OneRec official Recall@K (baseline reproduction)
│
├── diagnose/                                  # diagnostic tools
│   ├── README.md
│   ├── compare_models.py                      # base vs trained logits + beam outputs
│   └── debug_recall.py                        # quick OneRec baseline-reproduction sanity check
│
├── archive/                                   # superseded experiments + journey log
│   ├── EXPERIMENTS.md                         # ⭐ 7-step research journey, file map, headline numbers
│   ├── train_contrastive.py                   # length=1 (Step 1)
│   ├── train_contrastive_length3.py           # length=3 (Step 3)
│   ├── train_contrastive_grpo.py              # GRPO contrastive without DPO (Step 5)
│   ├── evaluate_length3.py                    # length=3 evaluation
│   ├── debug_pipeline.py                      # CPU pipeline test (length=1 era)
│   ├── merge.py                               # Windows merge variant
│   ├── run_all_evals.sh
│   ├── run_grpo_g5_smoke.sh                   # G=5 ablation
│   ├── run_grpo_50k_full.sh                   # GRPO-only 50k (killed)
│   └── eval_grpo_50k_checkpoint.sh            # mid-training sanity check
│
├── notebook/                                  # MS2 EDA + MS3 narrative
├── oneRec/                                    # template + planning docs
├── data/                                      # not in git
└── runs/                                      # experiment outputs (not in git)
```

## Current status (2026-05-02)

(See `archive/EXPERIMENTS.md` for full timeline.)

**Method finalized**: DPO + SFT anchor + group normalization. Code at `train/train_contrastive_dpo_g_normalize.py`.

**Validated by smoke runs (5k samples each):**
- length=3 30k full: Δpass=+0.0080, eval_pref_acc=0.674
- GRPO 5k smoke (G=3): matched length=3 30k Δ with 1/6 data
- G=5 ablation: G=5 worse than G=3 — variance reduction saturates beyond G=3 due to within-group correlation
- GRPO 50k partial (step 2500): chosen recall **drops** from baseline 0.0093 → 0.0043 → motivated SFT anchor + DPO

**Next steps:**
1. DPO + SFT smoke (5k, ~5h on RTX 6000 Pro): `bash run_dpo_smoke.sh`
2. If smoke shows chosen recall maintained or rising vs baseline → scale to 50k full
3. Final ablation table for the report: baseline vs length=3 30k vs GRPO 5k vs DPO+SFT 50k

## Dependencies

Python 3.10+. Key packages: `torch`, `transformers`, `peft`, `accelerate`, `pandas`, `pyarrow`, `tqdm`, `huggingface_hub`.

## Commands

### Build data (only re-run if you need to rebuild)

```bash
python build_contrastive_dataset.py --length 3        # → data/contrastive_dataset_v1/
python build_contrastive_dataset_GRPO.py --G 3        # v1 → data/contrastive_dataset_v1_grpo/
```

### Train (DPO + SFT + group-normalized loss)

Smoke pipeline (5000 groups, ~5h on RTX 6000 Pro):

```bash
bash run_dpo_smoke.sh
```

Full hyperparameter form (override anything):

```bash
python train/train_contrastive_dpo_g_normalize.py \
    --model_path model/OneRec-1.7B \
    --template model/qwen3_soft_switch.jinja2 \
    --train_parquet data/contrastive_dataset_v1_grpo/train.parquet \
    --valid_parquet data/contrastive_dataset_v1_grpo/valid.parquet \
    --output_dir runs/dpo_grpo_50k \
    --G 3 \
    --max_train_groups 50000 --max_eval_groups 2000 \
    --eval_steps 2500 --save_steps 2500 --logging_steps 50 \
    --per_device_batch_size 12 --grad_accum 1 \
    --lr 5e-5 --dpo_beta 0.1 --sft_weight 0.1 --kl_weight 0 \
    --merge_and_save
```

### Evaluate

**Engagement-aware (primary metric)** — runs on `contrastive_dataset_v1/valid.parquet`:

```bash
python train/evaluate_engaged.py \
    --model_path runs/dpo_grpo_50k/merged \
    --valid_parquet data/contrastive_dataset_v1/valid.parquet \
    --template model/qwen3_soft_switch.jinja2 \
    --n 5000 --num_beams 32 --topk 96 \
    --output_csv runs/eval_engaged_50k.csv
```

**OneRec official Recall@K (secondary)** — comparable to paper Table 4:

```bash
python train/evaluate_origin.py \
    --model_path runs/dpo_grpo_50k/merged \
    --benchmark data/OpenOneRec/benchmark_data/video/video_test.parquet \
    --template model/qwen3_soft_switch.jinja2 \
    --n 100 --output_csv runs/eval_origin_50k.csv
```

### Merge LoRA adapter (if not done with --merge_and_save)

```bash
python merge_local.py \
    --base model/OneRec-1.7B \
    --adapter runs/<run_name>/adapter \
    --out runs/<run_name>/merged
```

### Diagnostic tools

```bash
# Side-by-side base vs trained model outputs on benchmark prompts
python diagnose/compare_models.py \
    --trained runs/<run_name>/merged --num_beams 8 --max_new_tokens 13

# Verify generation pipeline reproduces OneRec Table 4 baseline
python diagnose/debug_recall.py --n 100
```

## Auto-resolution

`--model_path` defaults to `OpenOneRec/OneRec-1.7B` (HF Hub) — but on offline servers always pass an explicit local path (`model/OneRec-1.7B`). Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to disable HF retry on offline boxes (the smoke `.sh` script does this).

`--template` is resolved by [train/utils.py](train/utils.py) `resolve_template()` in this order: explicit CLI flag → `<project_root>/oneRec/qwen3_soft_switch.jinja2` → `~/.cache/onerec_template/qwen3_soft_switch.jinja2` → download from upstream GitHub. Override the URL with `ONEREC_TEMPLATE_URL`.

HF auth: token at `~/.cache/huggingface/token`. The OneRec-1.7B model is ungated; OpenOneRec-RecIF *dataset* is gated.

## Conventions

- **All EDA filters to `split=0`.** Never touch `benchmark_data/` for training analysis — it's the held-out test set, completely disjoint from master.
- **Two evaluators, two purposes:**
  - `train/evaluate_engaged.py` (Δrecall / Δpass) — **headline metric**, aligned with what we trained for
  - `train/evaluate_origin.py` (Recall@K / Pass@K) — **secondary metric**, comparable to OneRec paper Table 4
- **Don't run training experiments from `archive/`.** Those scripts produced known results; if you want to revisit, reference `archive/EXPERIMENTS.md` first.
- **Group normalization terminology**: in code we call it `_grpo`, but it's *not* full GRPO. In writing, use **"group-normalized contrastive loss"** or **"GRPO-inspired"**.
- **SID matching** is at the **string level** (`<s_a_X><s_b_Y><s_c_Z>`), not at the PID level (PID→SID is many-to-one).
- **Three loss weights to keep in mind:**
  - `--dpo_beta 0.1` — standard DPO scaling (DeepSeek/Llama-3 use this)
  - `--sft_weight 0.1` — gentle anchor on chosen (raise if chosen recall drops)
  - `--kl_weight 0` — explicit KL is **off** by default; DPO has implicit KL via ref baseline
