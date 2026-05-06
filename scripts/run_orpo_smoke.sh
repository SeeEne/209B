#!/usr/bin/env bash
#
# run_orpo_smoke.sh
#
# 5,000-group ORPO smoke. Single-stage from base, no ref model, no group
# normalization. Counterpart to:
#   - run_sft_smoke.sh / run_sft_50k.sh    (SFT-only ablation)
#   - run_dpo_smoke.sh                     (DPO+SFT joint baseline)
#   - run_sequential_smoke.sh              (SFT → DPO sequential)
#
# Why ORPO is in this ablation table:
#   ORPO collapses the SFT+preference pipeline into one stage with no ref
#   model. If it matches or beats SFT-only at the same data scale (5k), it's
#   the better default for this offline behavior-signal regime — fewer
#   moving parts, less VRAM (no ref model), no two-stage hyperparameter
#   product.
#
#   Configuration locked from the SFT 5k sweep (see run_sft_lr_sweep.sh and
#   run_sft_lora_r_sweep.sh): lr=5e-5 wins at 5k, r=16 / α=32 saturates
#   capacity. So we use the same SFT-arm hyperparameters and add ORPO's
#   one new knob:
#
#     lambda_or = 0.3
#       Paper's Table 4 default is 0.1 under the *sum*-log-p convention
#       (log P(y|x) = sum over tokens). This repo scores SIDs with the
#       *mean*-log-p convention for parity with all other trainers, which
#       scales the OR-term gradient by 1/3 vs the paper. 0.3 here ≈ paper's
#       0.1 in equivalent gradient strength.
#
# Hardware target: 1× A100 80GB (Ubuntu 22, CUDA 12). No ref model means
# peak VRAM is ~30–40 GB at batch=24, leaving plenty of headroom — we
# could push to 32 or 48 if needed (kept at 24 here for clean parity with
# SFT/DPO arms; same step count for direct comparison).
#
# Wall clock estimate at 5k:
#   - 5,000 groups × G=3 = 15,000 pairs
#   - batch=24 → 625 steps/epoch
#   - per-step ~9–10 s (no ref; same forward as SFT but on 2× sequences)
#   - train wall: ~1.5 h
#   - eval (n=1000, beam=32, ~6 min)
#   - total ~1.6 h
#
# Output:
#   runs/orpo_5k/
#     ├── checkpoint-125 / checkpoint-250 / ... / checkpoint-625
#     ├── adapter/    (LoRA at best ckpt by eval_chosen_score)
#     └── merged/     (full model = base + adapter)
#   runs/orpo_5k.log
#   runs/eval_engaged_orpo_5k_n1000.csv

set -e
set -o pipefail

# ---- Environment ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

# ---- Paths ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_GRPO="data/contrastive_dataset_v1_grpo"
DATA_V1="data/contrastive_dataset_v1"
TEMPLATE="model/qwen3_soft_switch.jinja2"
BASE_MODEL="model/OneRec-1.7B"

RUN="runs/orpo_5k"
TRAIN_LOG="${RUN}.log"
EVAL_DIR="runs/eval_logs"
EVAL_LOG="$EVAL_DIR/orpo_5k_n1000.log"
EVAL_CSV="runs/eval_engaged_orpo_5k_n1000.csv"

mkdir -p "$EVAL_DIR"

EVAL_N=1000
LAMBDA_OR=0.3

START=$(date +%s)

# ---- Sanity ----
if [ ! -f "$DATA_GRPO/train.parquet" ] || [ ! -f "$DATA_GRPO/valid.parquet" ]; then
    echo ">>> [error] G=3 dataset not found at $DATA_GRPO"; exit 1
fi
if [ ! -f "$DATA_V1/valid.parquet" ]; then
    echo ">>> [error] v1 valid not found at $DATA_V1/valid.parquet"; exit 1
fi
if [ ! -d "$BASE_MODEL" ]; then
    echo ">>> [error] base model not found at $BASE_MODEL"; exit 1
fi

# ============================================================
# STEP 1/2 — Train ORPO 5k (~1.5 h)
# ============================================================
if [ -d "$RUN/merged" ]; then
    echo ">>> [skip] $RUN/merged already exists; skipping training."
    echo "    rm -rf $RUN to retrain from scratch."
else
    echo ""
    echo "============================================================"
    echo ">>> STEP 1/2: ORPO train  (5,000 groups, ~1.5 h)"
    echo ">>> lr=5e-5, lora_r=16/alpha=32, batch=24, 1 epoch"
    echo ">>> lambda_or=$LAMBDA_OR  (≈ paper's 0.1 under sum-log-p; we use mean-log-p so ×3)"
    echo ">>> nll_loss_scale=1.0    (paper formulation; not match_dpo)"
    echo ">>> from base (single stage, no SFT init, no ref model)"
    echo ">>> out: $RUN"
    echo ">>> log: $TRAIN_LOG"
    echo "============================================================"

    if [ -d "$RUN" ]; then
        echo ">>> [warn] $RUN exists without merged/. HF Trainer will resume."
        echo "    rm -rf $RUN for a fresh start. Continuing in 5s..."
        sleep 5
    fi

    python train/train_orpo.py \
        --model_path "$BASE_MODEL" \
        --template "$TEMPLATE" \
        --train_parquet "$DATA_GRPO/train.parquet" \
        --valid_parquet "$DATA_GRPO/valid.parquet" \
        --output_dir "$RUN" \
        --max_train_groups 5000 --max_eval_groups 1000 \
        --num_checkpoints 5 --logging_steps 25 \
        --per_device_batch_size 24 --grad_accum 1 \
        --lr 5e-5 \
        --lambda_or $LAMBDA_OR \
        --nll_loss_scale 1.0 \
        --lora_r 16 --lora_alpha 32 \
        --merge_and_save \
        2>&1 | tee "$TRAIN_LOG"
fi

T_AFTER_TRAIN=$(date +%s)
TRAIN_MIN=$(( (T_AFTER_TRAIN - START) / 60 ))

# ============================================================
# STEP 2/2 — Engagement-aware eval (n=1000, ~6 min)
# ============================================================
if [ ! -d "$RUN/merged" ]; then
    echo ">>> [error] $RUN/merged missing — training likely failed"
    exit 1
fi

echo ""
echo "============================================================"
echo ">>> STEP 2/2: engagement-aware eval (n=$EVAL_N)"
echo ">>> model: $RUN/merged"
echo ">>> log:   $EVAL_LOG"
echo ">>> csv:   $EVAL_CSV"
echo "============================================================"

python train/evaluate_engaged.py \
    --model_path "$RUN/merged" \
    --valid_parquet "$DATA_V1/valid.parquet" \
    --template "$TEMPLATE" \
    --n $EVAL_N --num_beams 32 --topk 96 \
    --output_csv "$EVAL_CSV" \
    2>&1 | tee "$EVAL_LOG"

END=$(date +%s)
EVAL_MIN=$(( (END - T_AFTER_TRAIN) / 60 ))
TOTAL_MIN=$(( (END - START) / 60 ))

# ============================================================
# Summary
# ============================================================
echo ""
echo "############################################################"
echo "# ORPO 5K SMOKE COMPLETE  (train ${TRAIN_MIN} min + eval ${EVAL_MIN} min = ${TOTAL_MIN} min)"
echo "############################################################"

# Train-time eval_chosen_score trend across the 5 ORPO checkpoints
echo ""
echo "============================================================"
echo "Train-time eval_chosen_score trend (ORPO 5k, 5 ckpts + final)"
echo "  Reference points:"
echo "    base                 ≈ -4.92"
echo "    SFT-only 5k final    = -4.841"
echo "    SFT-only 5k chosen_recall@96 (n=1000) = 0.0097"
echo "============================================================"

# HF Trainer's per-step eval prints `'eval_chosen_score': -4.84` (single-quoted
# key, unquoted float). The final json.dumps prints `"eval_chosen_score": -4.84`
# (double-quoted key, unquoted float). The regex below accepts either quote
# style around the key, and either no quotes / single / double around the value.
if [ -f "$TRAIN_LOG" ]; then
    extract_metric() {
        local key="$1"
        sed -nE "s/.*[\"']${key}[\"']: [\"']?([-+0-9.eE]+)[\"']?.*/\1/p" "$TRAIN_LOG"
    }

    mapfile -t SCORES < <(extract_metric eval_chosen_score)
    mapfile -t PREF < <(extract_metric eval_pref_acc)
    mapfile -t MARGIN < <(extract_metric eval_margin)
    mapfile -t LOG_ODDS_M < <(extract_metric eval_log_odds_margin)
    LABELS=("ckpt 1 (step 125)" "ckpt 2 (step 250)" "ckpt 3 (step 375)" \
            "ckpt 4 (step 500)" "ckpt 5 (step 625)" "final (best ckpt)")
    for ((i=0; i<${#SCORES[@]} && i<6; i++)); do
        echo "    ${LABELS[$i]}: eval_chosen_score = ${SCORES[$i]}"
    done
    if [ ${#PREF[@]} -gt 0 ]; then
        echo ""
        echo "    OR-term diagnostics (linear margin = chosen-rejected score gap;"
        echo "    log_odds_margin = what L_OR optimizes — sigmoid input):"
        for ((i=0; i<${#PREF[@]} && i<6; i++)); do
            echo "    ${LABELS[$i]}: pref_acc=${PREF[$i]}  margin=${MARGIN[$i]}  log_odds=${LOG_ODDS_M[$i]:-n/a}"
        done
    fi
fi

# Headline engagement-aware metrics
echo ""
echo "============================================================"
echo "Engagement-aware Recall@K  (n=$EVAL_N, on v1 valid)"
echo "  Baseline (no FT)        : chosen=0.0093  rejected=0.0163  Δ=-0.0070"
echo "  SFT-only 5k (lr=5e-5)   : chosen=0.0097  rejected=0.0127  Δ=-0.0030"
echo "  → ORPO target: chosen ≥ 0.0097, AND ideally Δ closer to zero or positive."
echo "============================================================"
if [ -f "$EVAL_LOG" ]; then
    grep -E "^recall@96|^pass@96" "$EVAL_LOG" | tail -n 2 | sed 's/^/  /'
fi

echo ""
echo "============================================================"
echo "Decision tree for next step (after reading the numbers above)"
echo "============================================================"
echo "If chosen_score < -4.84  AND  chosen_recall > 0.0097:"
echo "  → ORPO beats SFT-only at 5k."
echo "  → Next: scale ORPO to 50k (run_orpo_50k.sh, ~12 h on A100)."
echo ""
echo "If chosen_score ≈ -4.84 (matches SFT-only ceiling):"
echo "  → ORPO no worse than SFT-only at 5k. The OR term didn't help here"
echo "    but it didn't hurt either; the data ceiling dominates."
echo "  → Next: still consider 50k ORPO if you want to test if ORPO scales"
echo "    BETTER than SFT-only with more data."
echo ""
echo "If chosen_score > -4.84 (worse than SFT-only):"
echo "  → OR term is harming chosen log-prob. Try lambda_or=0.1 or"
echo "    nll_loss_scale=2.0 to up-weight the SFT term."
echo ""
echo "For richer trend across the 5 ckpts (~15 min, no beam search):"
echo "  python diagnose/sft_score_trend.py \\"
echo "    --base $BASE_MODEL --include_base \\"
echo "    --adapters $RUN/checkpoint-125 $RUN/checkpoint-250 \\"
echo "               $RUN/checkpoint-375 $RUN/checkpoint-500 \\"
echo "               $RUN/checkpoint-625 \\"
echo "    --template $TEMPLATE --n 1000 --batch_size 16 \\"
echo "    --output_csv runs/orpo_5k_trend.csv"
