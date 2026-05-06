#!/usr/bin/env bash
#
# run_sft_50k.sh
#
# Full-scale SFT-only training: 50,000 groups (10× the smoke), confirmed
# data-bottlenecked configuration after the 2026-05-05/06 sweeps:
#
#   lr      = 5e-5   (lr=2e-4, 5e-4 sweeps both inferior at 5k)
#   lora_r  = 16     (r=32 sweep showed +0.002 nats only — capacity not used)
#   alpha   = 32     (α/r = 2.0)
#   epochs  = 1      (no benefit from data repeats expected)
#   batch   = 24, grad_accum = 1
#   sft_loss_scale = 16.0   (match joint trainer's match_dpo SFT term)
#
# Why 50k:
#   At 5k, all three lr values (5e-5, 2e-4, 5e-4) and both LoRA r values
#   (16, 32) converge to chosen_score ≈ -4.84 — a hard ceiling. r=16→32
#   moved chosen by 0.002 nats (noise). Conclusion: ceiling is the *data
#   scale*, not the optimization config. 50k = 10× data is the next test.
#
#   Expected best-case outcome: chosen_score breaks below -4.84 and pushes
#   recall_chosen on the v1 engagement valid above the 0.0093 baseline.
#
# Schedule details for 50k (vs 5k smoke):
#   total steps        : 6,250  (vs 625 at 5k)
#   warmup steps       : 312    (=0.05 × 6250, vs 31 at 5k)
#   ckpt interval      : 1,250  (=6250/5, vs 125 at 5k)
#   per-step time      : ~9.7s  (unchanged — same data dim/model size)
#   train wall clock   : ~17h
#   final engagement eval (n=1000, ~6 min)
#   total wall clock   : ~17h
#
# Output:
#   runs/sft_only_50k/
#     ├── checkpoint-1250 / checkpoint-2500 / ... / checkpoint-6250
#     ├── adapter/    (LoRA at best ckpt by eval_chosen_score)
#     └── merged/     (full model = base + adapter merged)
#   runs/sft_only_50k.log
#   runs/eval_engaged_sft_only_50k_n1000.csv

set -e
set -o pipefail

# ---- Environment ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

# ---- Paths ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATA_GRPO="data/contrastive_dataset_v1_grpo"
DATA_V1="data/contrastive_dataset_v1"
TEMPLATE="model/qwen3_soft_switch.jinja2"
BASE_MODEL="model/OneRec-1.7B"

RUN="runs/sft_only_50k"
TRAIN_LOG="${RUN}.log"
EVAL_DIR="runs/eval_logs"
EVAL_LOG="$EVAL_DIR/sft_only_50k_n1000.log"
EVAL_CSV="runs/eval_engaged_sft_only_50k_n1000.csv"

mkdir -p "$EVAL_DIR"

EVAL_N=1000

START=$(date +%s)

# ---- Sanity ----
if [ ! -f "$DATA_GRPO/train.parquet" ] || [ ! -f "$DATA_GRPO/valid.parquet" ]; then
    echo ">>> [error] G=3 dataset not found at $DATA_GRPO"; exit 1
fi
if [ ! -f "$DATA_V1/valid.parquet" ]; then
    echo ">>> [error] v1 valid not found at $DATA_V1/valid.parquet"; exit 1
fi

# ============================================================
# STEP 1/2 — Train SFT 50k (~17h)
# ============================================================
if [ -d "$RUN/merged" ]; then
    echo ">>> [skip] $RUN/merged already exists; skipping training."
    echo "    rm -rf $RUN to retrain from scratch."
else
    echo ""
    echo "============================================================"
    echo ">>> STEP 1/2: SFT-only train  (50,000 groups, ~17h)"
    echo ">>> lr=5e-5, lora_r=16/alpha=32, batch=24, 1 epoch"
    echo ">>> sft_loss_scale=16.0 (matches joint trainer's match_dpo term)"
    echo ">>> out: $RUN"
    echo ">>> log: $TRAIN_LOG"
    echo "============================================================"

    if [ -d "$RUN" ]; then
        echo ">>> [warn] $RUN exists without merged/. HF Trainer will resume."
        echo "    rm -rf $RUN for a fresh start. Continuing in 5s..."
        sleep 5
    fi

    python train/train_sft_only.py \
        --model_path "$BASE_MODEL" \
        --template "$TEMPLATE" \
        --train_parquet "$DATA_GRPO/train.parquet" \
        --valid_parquet "$DATA_GRPO/valid.parquet" \
        --output_dir "$RUN" \
        --max_train_groups 50000 --max_eval_groups 1000 \
        --num_checkpoints 5 --logging_steps 50 \
        --per_device_batch_size 24 --grad_accum 1 \
        --lr 5e-5 \
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
echo ">>> log: $EVAL_LOG"
echo ">>> csv: $EVAL_CSV"
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
echo "# SFT 50K COMPLETE  (train ${TRAIN_MIN} min + eval ${EVAL_MIN} min = ${TOTAL_MIN} min)"
echo "############################################################"

# eval_chosen_score trend across the 5 50k checkpoints
echo ""
echo "============================================================"
echo "Train-time eval_chosen_score trend (50k, 5 ckpts + final)"
echo "  Reference points (5k smoke, lr=5e-5):"
echo "    base     ≈ -4.92"
echo "    ckpt 1   = -4.856   (already 80% of total 5k gain)"
echo "    final    = -4.841   ← 5k ceiling we want to break"
echo "============================================================"

if [ -f "$TRAIN_LOG" ]; then
    mapfile -t SCORES < <(
        sed -nE "s/.*'eval_chosen_score': '([-+0-9.eE]+)'.*/\1/p" "$TRAIN_LOG"
    )
    LABELS=("ckpt 1 (step 1250)" "ckpt 2 (step 2500)" "ckpt 3 (step 3750)" \
            "ckpt 4 (step 5000)" "ckpt 5 (step 6250)" "final (best ckpt)")
    for ((i=0; i<${#SCORES[@]} && i<6; i++)); do
        echo "    ${LABELS[$i]}: eval_chosen_score = ${SCORES[$i]}"
    done
fi

# Headline engagement-aware metrics
echo ""
echo "============================================================"
echo "Engagement-aware Recall@K  (n=$EVAL_N, on v1 valid)"
echo "  Reference (5k SFT, lr=5e-5):"
echo "    chosen=0.0097  rejected=0.0127  Δ=-0.0030"
echo "  Baseline (no FT):"
echo "    chosen=0.0093  rejected=0.0163  Δ=-0.0070"
echo "============================================================"
if [ -f "$EVAL_LOG" ]; then
    grep -E "^recall@96|^pass@96" "$EVAL_LOG" | tail -n 2 | sed 's/^/  /'
fi

echo ""
echo "============================================================"
echo "Decision tree for next step (after reading the numbers above)"
echo "============================================================"
echo "If chosen_score breaks well below -4.84 AND chosen_recall > 0.0097:"
echo "  → Data scale was indeed the ceiling. Project finding confirmed."
echo "  → Next: run Stage 2 (DPO from this SFT-50k checkpoint), then"
echo "    we have a defensible final ablation table for the report."
echo ""
echo "If chosen_score breaks below -4.84 BUT chosen_recall stays at ~0.0097:"
echo "  → log-prob improved but generation still doesn't surface chosen items"
echo "  → suggests eval (Recall@96 with beam=32) is the bottleneck, not training"
echo ""
echo "If chosen_score stays at -4.84 (no break):"
echo "  → 50k didn't help either. Reconsider: prompt format, target domain,"
echo "    or the OneRec-1.7B base is genuinely overfit to next-shown."
echo ""
echo "For richer trend across the 5 50k ckpts (~1h):"
echo "  python diagnose/sft_score_trend.py \\"
echo "    --base $BASE_MODEL --include_base \\"
echo "    --adapters $RUN/checkpoint-1250 $RUN/checkpoint-2500 \\"
echo "               $RUN/checkpoint-3750 $RUN/checkpoint-5000 \\"
echo "               $RUN/checkpoint-6250 \\"
echo "    --template $TEMPLATE --n 1000 --batch_size 16 \\"
echo "    --output_csv runs/sft_50k_trend.csv"