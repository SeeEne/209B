#!/usr/bin/env bash
#
# run_dpo_grpo_smoke.sh
#
# Train + evaluate the DPO + SFT + GRPO normalization smoke (5000 groups).
#
# Goal: verify chosen recall ACTUALLY GOES UP (vs the pure-GRPO trajectory
# where chosen drops from baseline 0.0093 to ~0.0033).
#
# Loss = L_dpo_grpo + 0.1 × L_sft   (KL dropped, DPO has implicit KL)
#
# Hyperparameters:
#   --dpo_beta 0.1     standard DPO value
#   --sft_weight 0.1   gentle anchor on chosen — strong enough to push it up,
#                      weak enough not to drown the DPO signal
#   --kl_weight 0      DPO already has implicit KL via ref baseline in margin
#
# Total wall clock: ~5h training + ~15min eval ≈ 5.5h.

set -e

# ---- Environment ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

# ---- Paths ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data/contrastive_dataset_v1_grpo"      # G=3 dataset
OUT_DIR="runs/dpo_grpo_smoke"
TRAIN_LOG="runs/dpo_grpo_smoke.log"
EVAL_LOG_DIR="runs/eval_logs"
EVAL_LOG="$EVAL_LOG_DIR/dpo_grpo_smoke.log"
EVAL_CSV="runs/eval_engaged_dpo_grpo_smoke.csv"

mkdir -p "$EVAL_LOG_DIR"

if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/valid.parquet" ]; then
    echo ">>> [error] G=3 dataset not found at $DATA_DIR"
    echo "    Build it: python build_contrastive_dataset_GRPO.py --G 3"
    exit 1
fi

START_TIME=$(date +%s)

# ============================================================
# STEP 1/2: Train DPO + SFT + GRPO smoke
# ============================================================
echo "============================================================"
echo ">>> STEP 1/2: DPO + SFT + GRPO normalize smoke (5000 groups, ~5h)"
echo ">>> loss = L_dpo_grpo + 0.1 × L_sft  (kl_weight=0)"
echo ">>> log: $TRAIN_LOG"
echo "============================================================"

if [ -d "$OUT_DIR" ]; then
    echo ">>> [warn] $OUT_DIR exists. HF Trainer will resume."
    echo "    rm -rf $OUT_DIR for fresh run."
    echo "    Continuing in 5s..."
    sleep 5
fi

python train/train_contrastive_dpo_g_normalize.py \
    --model_path model/OneRec-1.7B \
    --template model/qwen3_soft_switch.jinja2 \
    --train_parquet "$DATA_DIR/train.parquet" \
    --valid_parquet "$DATA_DIR/valid.parquet" \
    --output_dir "$OUT_DIR" \
    --G 3 \
    --max_train_groups 5000 --max_eval_groups 1000 \
    --eval_steps 200 --save_steps 200 --logging_steps 25 \
    --per_device_batch_size 12 --grad_accum 1 \
    --lr 5e-5 \
    --dpo_beta 0.1 \
    --sft_weight 0.1 \
    --kl_weight 0 \
    --merge_and_save \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_DONE=$(date +%s)
TRAIN_MIN=$(( (TRAIN_DONE - START_TIME) / 60 ))

# ============================================================
# STEP 2/2: Engagement-aware eval
# ============================================================
echo ""
echo "============================================================"
echo ">>> STEP 2/2: Engagement-aware eval (n=5000, ~15min)"
echo ">>> log: $EVAL_LOG"
echo ">>> csv: $EVAL_CSV"
echo "============================================================"

if [ ! -d "$OUT_DIR/merged" ]; then
    echo ">>> [error] merged checkpoint not found at $OUT_DIR/merged"
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
echo "# DPO + SFT + GRPO SMOKE COMPLETE"
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
echo "DPO+SFT+GRPO smoke results (this run)"
echo "============================================================"
grep -E "^recall@96" "$EVAL_LOG" | tail -n 1 | sed 's/^/  /'
grep -E "^pass@96"   "$EVAL_LOG" | tail -n 1 | sed 's/^/  /'

echo ""
echo "============================================================"
echo "All-method comparison (engagement-aware, n=5000, topk=96)"
echo "============================================================"
echo "model                       chosen   rejected     Δrecall    Δpass"
echo "-----------------------------------------------------------------------"
echo "baseline (no FT)            0.0093    0.0163    -0.0070   -0.0200"
echo "v1_30k length=3 (30k full)  0.0047    0.0017    +0.0030   +0.0080"
echo "GRPO 5k smoke (G=3)         0.0033    0.0003    +0.0030   +0.0080"
echo "GRPO 5k smoke (G=5)         0.0030    0.0010    +0.0020   +0.0050"
echo "GRPO 50k step 2500          0.0043    0.0027    +0.0017   +0.0040"
echo "DPO+SFT+GRPO smoke (this)   see above"
echo ""
echo "Verdict guide for chosen recall (the key question):"
echo "  chosen ≥ baseline 0.0093    → SFT anchor works, chosen is RISING ✓"
echo "  chosen ∈ [0.005, 0.0093)    → partial improvement, can tune sft_weight up"
echo "  chosen < 0.005              → SFT not strong enough at 0.1; try 0.3"
echo ""
echo "Δ stays positive AND chosen ≥ baseline = the goal achieved."
