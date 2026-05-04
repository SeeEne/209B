#!/usr/bin/env bash
#
# run_grpo_g5_smoke.sh
#
# Train + evaluate the GRPO G=5 smoke. Assumes
# data/contrastive_dataset_v1_grpo_g5/ is already built.
#
# Steps:
#   1. Train GRPO at G=5, 5000 groups (= 25k pairs, ~6h)
#   2. Evaluate engagement-aware metric on the merged model (~15min)
#
# Total wall clock: ~6.5 hours.

set -e

# ---- Environment ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

# ---- Paths ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data/contrastive_dataset_v1_grpo_g5"
OUT_DIR="runs/contrastive_grpo_g5_smoke"
TRAIN_LOG="runs/grpo_g5_smoke.log"
EVAL_LOG_DIR="runs/eval_logs"
EVAL_LOG="$EVAL_LOG_DIR/grpo_g5_smoke.log"
EVAL_CSV="runs/eval_engaged_grpo_g5_smoke.csv"

mkdir -p "$EVAL_LOG_DIR"

# Sanity: dataset must exist
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/valid.parquet" ]; then
    echo ">>> [error] G=5 dataset not found at $DATA_DIR"
    echo "    Build it first: python build_contrastive_dataset_GRPO.py --G 5"
    exit 1
fi

START_TIME=$(date +%s)

# ============================================================
# STEP 1/2: Train GRPO G=5 smoke
# ============================================================
echo "============================================================"
echo ">>> STEP 1/2: Train GRPO G=5 smoke (5000 groups, ~6h)"
echo ">>> log: $TRAIN_LOG"
echo "============================================================"

if [ -d "$OUT_DIR" ]; then
    echo ">>> [warn] $OUT_DIR already exists."
    echo "    HF Trainer may try to resume. Rename or rm -rf if you want a fresh run."
    echo "    Continuing in 5s..."
    sleep 5
fi

python train/train_contrastive_grpo.py \
    --model_path model/OneRec-1.7B \
    --template model/qwen3_soft_switch.jinja2 \
    --train_parquet "$DATA_DIR/train.parquet" \
    --valid_parquet "$DATA_DIR/valid.parquet" \
    --output_dir "$OUT_DIR" \
    --G 5 \
    --max_train_groups 5000 --max_eval_groups 1000 \
    --eval_steps 200 --save_steps 200 --logging_steps 25 \
    --per_device_batch_size 15 --grad_accum 1 \
    --lr 5e-5 --temperature 0.5 --kl_weight 0.5 \
    --merge_and_save \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_DONE=$(date +%s)
TRAIN_MIN=$(( (TRAIN_DONE - START_TIME) / 60 ))

# ============================================================
# STEP 2/2: Evaluate engagement-aware metric
# ============================================================
echo ""
echo "============================================================"
echo ">>> STEP 2/2: Evaluate engagement-aware (n=5000, ~15min)"
echo ">>> log: $EVAL_LOG"
echo ">>> csv: $EVAL_CSV"
echo "============================================================"

if [ ! -d "$OUT_DIR/merged" ]; then
    echo ">>> [error] merged checkpoint not found at $OUT_DIR/merged"
    echo "    Did training crash before --merge_and_save? Check $TRAIN_LOG."
    exit 1
fi

python train/evaluate_engaged.py \
    --model_path "$OUT_DIR/merged" \
    --valid_parquet data/contrastive_dataset_v1/valid.parquet \
    --template model/qwen3_soft_switch.jinja2 \
    --n 5000 --num_beams 32 --topk 96 \
    --output_csv "$EVAL_CSV" \
    2>&1 | tee "$EVAL_LOG"

END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))
EVAL_MIN=$(( (END_TIME - TRAIN_DONE) / 60 ))

# ============================================================
# Summary
# ============================================================
echo ""
echo "############################################################"
echo "# G=5 SMOKE PIPELINE COMPLETE"
echo "############################################################"
echo "  train:    ${TRAIN_MIN} min"
echo "  evaluate: ${EVAL_MIN} min"
echo "  TOTAL:    ${TOTAL_MIN} min"
echo ""
echo "Artifacts:"
echo "  train log:    $TRAIN_LOG"
echo "  eval log:     $EVAL_LOG"
echo "  eval csv:     $EVAL_CSV"
echo "  merged ckpt:  $OUT_DIR/merged"
echo ""

echo "============================================================"
echo "G=5 smoke results (this run)"
echo "============================================================"
grep -E "^recall@96" "$EVAL_LOG" | tail -n 1 | sed 's/^/  /'
grep -E "^pass@96"   "$EVAL_LOG" | tail -n 1 | sed 's/^/  /'

echo ""
echo "============================================================"
echo "G=3 smoke baseline (previous run, for comparison)"
echo "============================================================"
echo "  recall@96       0.0033        0.0003       +0.0030"
echo "  pass@96         0.0090        0.0010       +0.0080"

echo ""
echo "Verdict guide:"
echo "  Δpass(G=5) > 0.0080 + headroom  → G=5 wins, consider 50k full at G=5"
echo "  Δpass(G=5) ≈ 0.0080             → G=3 is sufficient (cheaper compute)"
echo "  Δpass(G=5) < 0.0080             → larger G doesn't help here"
