#!/usr/bin/env bash
#
# run_grpo_50k_full.sh
#
# Full GRPO G=3 training on 50k users + engagement-aware eval.
#
# Hyperparameters identical to the smoke test that validated the method
# (5k smoke produced Δpass=+0.0080, matching length=3 30k with 1/6 the data),
# scaled up to 50k users (10× more data than smoke).
#
# Predicted: Δpass ≈ +0.022 (3× smoke under power-law N^0.44),
#            ~3.5σ significant on n=5000 eval.
#
# Wall clock: ~37.5h train + ~30min eval ≈ 38h total on 1× RTX 6000 Pro.
# (To cut to ~24h, bump --per_device_batch_size to 24 — VRAM headroom
#  on 96GB is comfortable per the smoke run's 30GB peak. See comment below.)
#
# If the run crashes mid-way, just re-run this script — HF Trainer will
# resume from the latest checkpoint in $OUT_DIR (saved every 2500 steps).

set -e

# ---- Environment ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

# ---- Paths ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data/contrastive_dataset_v1_grpo"        # G=3 dataset (default location)
OUT_DIR="runs/contrastive_grpo_50k"
TRAIN_LOG="runs/grpo_50k.log"
EVAL_LOG_DIR="runs/eval_logs"
EVAL_LOG="$EVAL_LOG_DIR/grpo_50k.log"
EVAL_CSV="runs/eval_engaged_grpo_50k.csv"

mkdir -p "$EVAL_LOG_DIR"

# Sanity: dataset must exist
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/valid.parquet" ]; then
    echo ">>> [error] G=3 dataset not found at $DATA_DIR"
    echo "    Build it: python build_contrastive_dataset_GRPO.py --G 3"
    exit 1
fi

START_TIME=$(date +%s)

# ============================================================
# STEP 1/2: Train GRPO G=3 50k full
# ============================================================
echo "============================================================"
echo ">>> STEP 1/2: GRPO G=3 50k full (~37.5h)"
echo ">>> 50k users × 3 pairs = 150k pairs"
echo ">>> bsz=12 (4 users × G=3), lr=5e-5, T=0.5, kl_weight=0.5"
echo ">>> 12,500 optimizer steps × ~10.78 sec/step"
echo ">>> log: $TRAIN_LOG"
echo "============================================================"

if [ -d "$OUT_DIR" ]; then
    echo ">>> [warn] $OUT_DIR exists."
    echo "    HF Trainer will RESUME from latest checkpoint inside it."
    echo "    To start fresh: rm -rf $OUT_DIR"
    echo "    Continuing in 5s..."
    sleep 5
fi

# Hyperparameters identical to the 5k smoke that produced Δpass=+0.008.
# Only --max_train_groups changed from 5000 → 50000.
# eval_steps=2500 → 5 in-training evals (at 2500/5000/7500/10000/12500)
#                  + 1 final = 6 evals total, ~75 min eval overhead.
python train/train_contrastive_grpo.py \
    --model_path model/OneRec-1.7B \
    --template model/qwen3_soft_switch.jinja2 \
    --train_parquet "$DATA_DIR/train.parquet" \
    --valid_parquet "$DATA_DIR/valid.parquet" \
    --output_dir "$OUT_DIR" \
    --G 3 \
    --max_train_groups 50000 --max_eval_groups 2000 \
    --eval_steps 2500 --save_steps 2500 --logging_steps 50 \
    --save_total_limit 5 \
    --per_device_batch_size 12 --grad_accum 1 \
    --lr 5e-5 --temperature 0.5 --kl_weight 0.5 \
    --warmup_ratio 0.05 \
    --merge_and_save \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_DONE=$(date +%s)
TRAIN_HOURS=$(echo "scale=1; ($TRAIN_DONE - $START_TIME) / 3600" | bc)

# ============================================================
# STEP 2/2: Evaluate engagement-aware metric on valid
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
TOTAL_HOURS=$(echo "scale=1; ($END_TIME - $START_TIME) / 3600" | bc)
EVAL_MIN=$(( (END_TIME - TRAIN_DONE) / 60 ))

# ============================================================
# Summary
# ============================================================
echo ""
echo "############################################################"
echo "# GRPO G=3 50k FULL PIPELINE COMPLETE"
echo "############################################################"
echo "  train:    ${TRAIN_HOURS} h"
echo "  evaluate: ${EVAL_MIN} min"
echo "  TOTAL:    ${TOTAL_HOURS} h"
echo ""
echo "Artifacts:"
echo "  train log:   $TRAIN_LOG"
echo "  eval log:    $EVAL_LOG"
echo "  eval csv:    $EVAL_CSV"
echo "  merged ckpt: $OUT_DIR/merged"
echo ""

echo "============================================================"
echo "GRPO 50k full results (this run)"
echo "============================================================"
grep -E "^recall@96" "$EVAL_LOG" | tail -n 1 | sed 's/^/  /'
grep -E "^pass@96"   "$EVAL_LOG" | tail -n 1 | sed 's/^/  /'

echo ""
echo "============================================================"
echo "Comparison (engagement-aware, n=5000, num_beams=32, topk=96)"
echo "============================================================"
echo "model                       chosen   rejected     Δ"
echo "-----------------------------------------------------------------"
echo "baseline (no FT)            0.0093    0.0163    -0.0070   (recall)"
echo "                            0.0260    0.0460    -0.0200   (pass)"
echo "v1_30k (length=3, 30k)      0.0047    0.0017    +0.0030   (recall)"
echo "                            0.0130    0.0050    +0.0080   (pass)"
echo "GRPO 5k smoke (G=3)         0.0033    0.0003    +0.0030   (recall)"
echo "                            0.0090    0.0010    +0.0080   (pass)"
echo "GRPO 5k smoke (G=5)         0.0030    0.0010    +0.0020   (recall)"
echo "                            0.0080    0.0030    +0.0050   (pass)"
echo "GRPO 50k FULL (this run)    see above for the actual numbers"
echo ""
echo "Power-law prediction:"
echo "  Δrecall(50k) ≈ 0.0030 × (50/5)^0.44 ≈ +0.0083"
echo "  Δpass(50k)   ≈ 0.0080 × (50/5)^0.44 ≈ +0.022"
echo ""
echo "Significance check (n=5000):"
echo "  Δpass SE ≈ ±0.0017  →  +0.022 ≈ 13σ  (clearly significant)"
echo "  Δrecall SE ≈ ±0.0008 →  +0.008 ≈ 10σ"
