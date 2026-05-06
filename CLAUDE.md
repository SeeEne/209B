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

Standard hyperparameters: `--dpo_beta 0.1`, `--sft_weight 1.0 --sft_scale_mode match_dpo` (the SFT term is rescaled by `mean(1/std_g)` so 1.0 is the *true* relative magnitude vs `L_dpo_grpo`; bare `--sft_weight 0.1` was effectively ~1–3%), `--kl_weight 0` (DPO has implicit KL), `--group_norm_eps 0.05` (caps `1/std` at 20× — was 1e-3 → step-0 `l_dpo=693` explosion), `--group_norm_warmup 50` (skip group-norm during cold start when trained≈ref), `--best_metric chosen_score` (correlates with `recall_chosen`; old `eval_pref_acc` was decoupled).

**Why this stack** — see [`archive/EXPERIMENTS.md`](archive/EXPERIMENTS.md) for the 7-step research journey. Short version:
1. Pure contrastive (length=1) → mode collapse
2. + KL → distribution recovers, but Recall@K still 0
3. **Insight: Recall@K measures next-shown; we trained for engaged. Mismatched metric.**
4. Build engagement-aware Δrecall metric on held-out contrastive valid
5. Length=3 contrastive: clean directional reversal (Δpass +0.0080 vs baseline −0.020)
6. Group normalization: matches length=3 30k Δ with 1/6 the data
7. **Pure GRPO crashes chosen recall in absolute terms** → add SFT anchor + DPO formulation → current path.

## ORPO arm (alternative single-stage formulation)

ORPO (Hong et al. 2024) folds preference learning into the SFT loss with **no reference model**, using a log-odds-ratio term:

```
log_odds_θ(y|x) = log P_θ(y|x) − log(1 − P_θ(y|x))
ratio           = log_odds_θ(c|x) − log_odds_θ(r|x)
L_ORPO          = −mean log P_θ(c|x)        # NLL term, same as our SFT
                  + λ × −mean log σ(ratio)  # OR term
```

Why it's worth running here: no ref model → no `_ref_cache/` precompute, ~50% less peak VRAM, single stage from base. If it matches or beats SFT-only at the same data scale, it's the simpler default for this offline behavior-signal regime. Trainer at [`train/train_orpo.py`](train/train_orpo.py); smoke at [`scripts/run_orpo_smoke.sh`](scripts/run_orpo_smoke.sh) (5k, ~1.5h on A100 80GB). λ default 0.1 (paper).

Implementation note: log_odds is computed with a numerically stable `log1mexp` (Mächler 2012) — log P close to 0 would otherwise produce −∞. On this task log P ≈ −4.84 nats, far from saturation, but the guard is essentially free.

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
├── pyproject.toml                             # ⭐ deps for new server setup (pip install -e .)
├── build_contrastive_dataset.py               # master → v1 (length=3) data
├── build_contrastive_dataset_GRPO.py          # v1 → v1_grpo (per-pair, group_id, configurable G)
├── merge_local.py                             # merge LoRA adapter into base for inference
├── scripts/                                   # all run_*.sh launchers (run from project root)
│   ├── run_dpo_smoke.sh                       # DPO+SFT joint smoke (5k, primary path)
│   ├── run_sft_50k.sh / run_sft_*_sweep.sh    # SFT-only ablation arm (lr/r sweeps + 50k full)
│   ├── run_sequential_smoke.sh                # sequential SFT → DPO ablation arm
│   └── run_orpo_smoke.sh                      # ⭐ ORPO ablation arm (single-stage, no ref)
│
├── train/
│   ├── README.md
│   ├── dataset.py                             # SYSTEM_PROMPT + helpers (used by all trainers)
│   ├── utils.py                               # resolve_template
│   ├── train_contrastive_dpo_g_normalize.py   # ⭐ MAIN trainer (DPO + SFT + G-norm)
│   ├── train_sft_only.py                      # Stage 1 / SFT-only ablation
│   ├── train_dpo_from_sft.py                  # Stage 2 / DPO from SFT init
│   ├── train_orpo.py                          # ⭐ single-stage ORPO (no ref model)
│   ├── evaluate_engaged.py                    # ⭐ MAIN evaluator (engagement-aware Δrecall/Δpass)
│   └── evaluate_origin.py                     # OneRec official Recall@K (baseline reproduction)
│
├── diagnose/                                  # diagnostic tools
│   ├── README.md
│   ├── compare_models.py                      # base vs trained logits + beam outputs
│   ├── checkpoint_recall_trend.py             # engagement-aware recall across checkpoints (named-adapter swap, no merging)
│   ├── sft_score_trend.py                     # fast forward-only chosen/rejected score trend (no beam)
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

## Current status (2026-05-06)

(See `archive/EXPERIMENTS.md` for full timeline.)

**Three ablation arms now in place**, sharing the same `data/contrastive_dataset_v1_grpo` 5k subset (deterministic via `subsample_seed`):

| Arm | Trainer | Loss | Ref model | Group norm |
|-----|---------|------|-----------|------------|
| DPO+SFT joint | `train_contrastive_dpo_g_normalize.py` | `L_dpo_grpo + λ·L_sft` | yes (cached) | yes |
| SFT → DPO sequential | `train_sft_only.py` → `train_dpo_from_sft.py` | stage-isolated | stage 2 only | stage 2 only |
| SFT-only | `train_sft_only.py` | `L_sft` (16× scale) | no | no |
| ORPO | `train_orpo.py` | `L_NLL + λ·L_OR` | no | no |

**Empirical findings (5k smoke):**
- length=3 30k full: Δpass=+0.0080, eval_pref_acc=0.674
- GRPO 5k (G=3): matched length=3 30k Δ with 1/6 data
- G=5 ablation: G=5 worse than G=3 — variance reduction saturates due to within-group correlation
- GRPO 50k partial: chosen recall drops 0.0093 → 0.0043 → motivated SFT anchor
- DPO+SFT joint 5k: Δ improves but eval_pref_acc only 0.546 (sft_scale_mode match_dpo overpowers DPO discriminative signal)
- SFT-only 5k sweep: lr ∈ {5e-5, 2e-4, 5e-4} all converge to chosen_score ≈ −4.84; LoRA r=16 → 32 changes chosen by 0.002 nats (noise) → **5k ceiling = data, not optimization**

**In-flight (2026-05-06):**
- 50k SFT (`scripts/run_sft_50k.sh`, ~17h on RTX 6000 Pro / H100) — testing whether 10× data breaks the −4.84 ceiling.
- 5k ORPO (`scripts/run_orpo_smoke.sh`, ~1.5h on A100 80GB) — separate server, single-stage from base, no ref model. λ=0.1 (paper default).

**Decision logic for next steps:**
1. SFT-50k breaks ceiling → run Stage 2 DPO from SFT-50k → final ablation table.
2. ORPO 5k beats SFT-only 5k → scale ORPO to 50k.
3. Both stay at ≈ −4.84 → reconsider OneRec base or eval methodology.

## Dependencies

Python 3.10–3.12. All runtime deps are declared in [`pyproject.toml`](pyproject.toml).

**Setup on a fresh server (e.g. A100 80GB, Ubuntu 22, CUDA 12):**

```bash
# 1) Install torch FIRST with the CUDA-matched wheel (NOT pinned in pyproject):
pip install "torch==2.4.*" --index-url https://download.pytorch.org/whl/cu121

# 2) Then install the rest (editable so train/, diagnose/ stay importable):
pip install -e .

# 3) Optional: flash-attn 2 for ~1.4× faster forward (Linux only).
#    Trainers fall back to sdpa automatically if absent — never a hard error.
pip install -e ".[flash]"

# 4) Sanity check the env:
python -c "
import torch, transformers, peft, accelerate, pandas, pyarrow
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)
print('transformers', transformers.__version__, 'peft', peft.__version__)
print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"
```

**transformers ≥ 4.44 is a hard requirement** — `Trainer.compute_loss` uses the `num_items_in_batch` kwarg added in 4.40 and stabilized in 4.44.

**Files to ship to a fresh server**: `model/OneRec-1.7B/`, `model/qwen3_soft_switch.jinja2`, `data/contrastive_dataset_v1_grpo/`, `data/contrastive_dataset_v1/`, plus the repo source. Do **not** ship `runs/` (outputs) or `data/contrastive_dataset_v1_grpo/_ref_cache/` (DPO-only cache; ORPO doesn't use it).

## Commands

### Build data (only re-run if you need to rebuild)

```bash
python build_contrastive_dataset.py --length 3        # → data/contrastive_dataset_v1/
python build_contrastive_dataset_GRPO.py --G 3        # v1 → data/contrastive_dataset_v1_grpo/
```

### Train (DPO + SFT + group-normalized loss)

Smoke pipeline (5000 groups, ~5h on RTX 6000 Pro):

```bash
bash scripts/run_dpo_smoke.sh
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
    --num_checkpoints 5 --logging_steps 50 \
    --per_device_batch_size 24 --grad_accum 1 \
    --lr 5e-5 \
    --dpo_beta 0.1 \
    --sft_weight 1.0 --sft_scale_mode match_dpo \
    --group_norm_eps 0.05 --group_norm_warmup 50 \
    --best_metric chosen_score \
    --kl_weight 0 \
    --merge_and_save
```

Notes on the post-2026-05-04 flags (see `scripts/run_dpo_smoke.sh` header for the full rationale):
- `--num_checkpoints 5` replaces manual `--eval_steps`/`--save_steps` — Trainer derives evenly-spaced ckpts.
- `--per_device_batch_size 24` works because ref scores are pre-computed (the ref model is freed after the precompute pass), freeing ~3.4 GB VRAM.
- `--sft_scale_mode match_dpo` rescales `L_sft` by `mean(1/std_g)` so `--sft_weight 1.0` is commensurate with `L_dpo_grpo`. Without this, the SFT contribution silently shrinks to ~1–3% of DPO.
- `--group_norm_warmup 50` skips group-norm for the first 50 steps. At cold start trained≈ref → `std_g ≈ 0` → loss explodes; the warmup makes those steps plain DPO.
- `--group_norm_eps 0.05` caps the `1/std` amplification factor at 20×. The old default `1e-3` (= 1000×) caused step-0 `l_dpo ≈ 693`.
- `--best_metric chosen_score` selects the best ckpt by absolute chosen log-prob (correlates with `recall_chosen`). Old default `eval_pref_acc` measured *separation* and was decoupled from the headline metric.

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

### Train (ORPO — single-stage, no ref model)

Smoke pipeline (5000 groups, ~1.5h on A100 80GB):

```bash
bash scripts/run_orpo_smoke.sh
```

Full hyperparameter form:

```bash
python train/train_orpo.py \
    --model_path model/OneRec-1.7B \
    --template model/qwen3_soft_switch.jinja2 \
    --train_parquet data/contrastive_dataset_v1_grpo/train.parquet \
    --valid_parquet data/contrastive_dataset_v1_grpo/valid.parquet \
    --output_dir runs/orpo_5k \
    --max_train_groups 5000 --max_eval_groups 1000 \
    --num_checkpoints 5 --logging_steps 25 \
    --per_device_batch_size 24 --grad_accum 1 \
    --lr 5e-5 \
    --lambda_or 0.1 \
    --nll_loss_scale 1.0 \
    --lora_r 16 --lora_alpha 32 \
    --best_metric chosen_score \
    --merge_and_save
```

Key knobs:
- `--lambda_or 0.1` — weight on the odds-ratio term. Paper range 0.1–1.0. Larger = more discriminative pressure; smaller = closer to pure SFT.
- `--nll_loss_scale 1.0` — paper formulation. Set to 16.0 *only* if you also want to match the SFT-only ablation's `match_dpo` scaling — but then you should rescale `lambda_or` to keep the OR/NLL ratio you want.
- No `--ref_model_path`, no `--group_norm_*`, no `--dpo_beta` — ORPO has none of those.

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

# Track recall_chosen / Δ across LoRA checkpoints (catches the GRPO-only failure
# mode where margin grows but chosen recall drops in absolute terms)
python diagnose/checkpoint_recall_trend.py \
    --base model/OneRec-1.7B \
    --adapters runs/<run_name>/checkpoint-600 runs/<run_name>/checkpoint-800 runs/<run_name>/checkpoint-1000 \
    --include_base \
    --template model/qwen3_soft_switch.jinja2 \
    --n 1000 --num_beams 32 --topk 96 \
    --output_csv runs/checkpoint_trend_<run_name>.csv
```

## Auto-resolution

`--model_path` defaults to `OpenOneRec/OneRec-1.7B` (HF Hub) — but on offline servers always pass an explicit local path (`model/OneRec-1.7B`). Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to disable HF retry on offline boxes (the smoke `.sh` script does this).

`--template` is resolved by [train/utils.py](train/utils.py) `resolve_template()` in this order: explicit CLI flag → `<project_root>/oneRec/qwen3_soft_switch.jinja2` → `~/.cache/onerec_template/qwen3_soft_switch.jinja2` → download from upstream GitHub. Override the URL with `ONEREC_TEMPLATE_URL`.

HF auth: token at `~/.cache/huggingface/token`. The OneRec-1.7B model is ungated; OpenOneRec-RecIF *dataset* is gated.

### Ref-score cache

Reference-model log-probs (`c_ref`, `r_ref`) are precomputed once and cached to `data/contrastive_dataset_v1_grpo/_ref_cache/`. The ref model is then freed before training starts.

- **First run**: ~30 min precompute (full pass over all groups under the configured `max_train_groups` / `max_hist`).
- **Subsequent runs**: ~1 sec load — cache hits whenever data and prompt construction are unchanged.
- **Stays hot when tuning**: `--sft_weight`, `--dpo_beta`, `--lr`, `--group_norm_*`, `--best_metric` (anything that doesn't change ref inputs).
- **Invalidates on**: `--max_train_groups` / `--max_eval_groups` (different row subset), `--max_hist` (different prompt length), `--model_path` (different ref weights), `--template` (different prompt format).
- **To force rebuild**: `rm -rf data/contrastive_dataset_v1_grpo/_ref_cache/`.

This cache is the reason hyperparameter sweeps are fast — keep it in mind when changing flags.

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
  - `--sft_weight 1.0 --sft_scale_mode match_dpo` — anchor on chosen, rescaled to match `L_dpo_grpo` magnitude. Raise if chosen recall drops; lower if Δ shrinks because SFT dominates. (Without `match_dpo`, you'd want ~10–30 to get the same effect — the old `0.1` was effectively ~1–3%.)
  - `--kl_weight 0` — explicit KL is **off** by default; DPO has implicit KL via ref baseline
- **ORPO is a parallel arm, not a replacement.** The DPO+SFT joint trainer remains the primary path; ORPO is run on a separate server to test whether dropping the ref model and merging stages costs anything in this offline behavior-signal regime. They are evaluated by the SAME engagement-aware metric, so results go side-by-side in the final ablation table.
