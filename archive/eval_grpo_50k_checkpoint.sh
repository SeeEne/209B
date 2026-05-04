#!/usr/bin/env bash
#
# eval_grpo_50k_checkpoint.sh
#
# Merge a specific intermediate checkpoint of the GRPO 50k run into the
# base model and run engagement-aware eval on it.
#
# Use this to sanity-check progress mid-training without waiting the full 38h.
# Training MUST be stopped first (it owns the GPU), then resumed afterwards
# via ./run_grpo_50k_full.sh (HF Trainer auto-resumes from latest checkpoint).
#
# Usage:
#   ./eval_grpo_50k_checkpoint.sh             # auto-pick latest checkpoint
#   ./eval_grpo_50k_checkpoint.sh 2500        # eval checkpoint-2500 specifically
#
# Wall clock: ~1 min merge + ~15 min eval = ~16 min total downtime for training.

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

OUT_DIR="runs/contrastive_grpo_50k"
EVAL_LOG_DIR="runs/eval_logs"
mkdir -p "$EVAL_LOG_DIR"

# ---- Pick checkpoint ----
if [ -z "$1" ]; then
    CKPT_NAME=$(ls -1 "$OUT_DIR" 2>/dev/null | grep "^checkpoint-" | sort -V | tail -n 1)
    if [ -z "$CKPT_NAME" ]; then
        echo "[error] no checkpoint-* dir in $OUT_DIR/"
        echo "Available:"
        ls "$OUT_DIR" 2>/dev/null | sed 's/^/  /'
        exit 1
    fi
    STEP=${CKPT_NAME#checkpoint-}
    echo ">>> auto-picked latest: $CKPT_NAME"
else
    STEP=$1
    CKPT_NAME="checkpoint-$STEP"
fi

CKPT_DIR="$OUT_DIR/$CKPT_NAME"
MERGED_DIR="$OUT_DIR/merged_step$STEP"
EVAL_LOG="$EVAL_LOG_DIR/grpo_50k_step${STEP}.log"
EVAL_CSV="runs/eval_engaged_grpo_50k_step${STEP}.csv"

if [ ! -d "$CKPT_DIR" ]; then
    echo "[error] checkpoint not found: $CKPT_DIR"
    echo "Available checkpoints:"
    ls "$OUT_DIR" 2>/dev/null | grep "^checkpoint-" | sed 's/^/  /'
    exit 1
fi

# ---- Detect if training is still running (warn, don't block) ----
TRAIN_PID=$(pgrep -f "train_contrastive_grpo.py" || true)
if [ -n "$TRAIN_PID" ]; then
    echo "[warn] training is still running (PID $TRAIN_PID)."
    echo "       This eval needs the GPU exclusively — it will likely OOM."
    echo "       Stop training first:  kill -SIGINT $TRAIN_PID"
    echo "       Continuing in 5s anyway..."
    sleep 5
fi

START=$(date +%s)

# ---- Step 1: Merge ----
echo ""
echo "============================================================"
echo ">>> STEP 1/2: Merge checkpoint-$STEP → $MERGED_DIR"
echo "============================================================"

if [ -d "$MERGED_DIR" ]; then
    echo ">>> [skip] merged dir already exists"
else
    python merge_local.py \
        --base model/OneRec-1.7B \
        --adapter "$CKPT_DIR" \
        --out "$MERGED_DIR"
fi

# ---- Step 2: Engagement-aware eval ----
echo ""
echo "============================================================"
echo ">>> STEP 2/2: Engagement-aware eval (n=5000, ~15 min)"
echo ">>> log: $EVAL_LOG"
echo ">>> csv: $EVAL_CSV"
echo "============================================================"

python train/evaluate_engaged.py \
    --model_path "$MERGED_DIR" \
    --valid_parquet data/contrastive_dataset_v1/valid.parquet \
    --template model/qwen3_soft_switch.jinja2 \
    --n 5000 --num_beams 32 --topk 96 \
    --output_csv "$EVAL_CSV" \
    2>&1 | tee "$EVAL_LOG"

END=$(date +%s)
TOTAL_MIN=$(( (END - START) / 60 ))

# ---- Summary ----
echo ""
echo "############################################################"
echo "# CHECKPOINT-$STEP EVAL COMPLETE  (${TOTAL_MIN} min downtime)"
echo "############################################################"
echo ""
echo "--- Results for checkpoint-$STEP ---"
grep -E "^recall@96" "$EVAL_LOG" | tail -n 1 | sed 's/^/  /'
grep -E "^pass@96"   "$EVAL_LOG" | tail -n 1 | sed 's/^/  /'

echo ""
echo "--- Reference: trajectory we want to see ---"
echo "step      | recall_chosen   pass_chosen     Δrecall    Δpass"
echo "----------|----------------------------------------------------"
echo "baseline  |    0.0093          0.0260       -0.0070   -0.0200"
echo "5k smoke  |    0.0033          0.0090       +0.0030   +0.0080"
echo "step 2500 | this run, see above (epoch 0.20)"
echo "step 5000 | (epoch 0.40, predicted ~0.005 / ~0.013)"
echo "step 12500| FINAL, predicted Δrecall +0.008, Δpass +0.022"
echo ""
echo "Sanity check: recall_chosen should be > 0.0033 (smoke level)"
echo "              AND Δpass should be > +0.008 (smoke level)"
echo ""
echo "To resume training:  ./run_grpo_50k_full.sh"
echo "  HF Trainer will auto-resume from $CKPT_NAME"
