#!/usr/bin/env bash
#
# run_dpo_grpo_smoke.sh
#
# Train + evaluate the DPO + SFT + GRPO normalization smoke (5000 groups).
#
# Goal: verify chosen recall ACTUALLY GOES UP (vs the pure-GRPO trajectory
# where chosen drops from baseline 0.0093 to ~0.0033).
#
# Loss = L_dpo_grpo + sft_weight × (1/std_g).mean × L_sft   (kl off)
#
# Hyperparameters (post-2026-05-04 fixes — see train_contrastive_dpo_g_normalize.py
# DESIGN NOTES for what changed and why):
#   --dpo_beta 0.1            standard DPO value
#   --sft_weight 1.0          with --sft_scale_mode match_dpo, this is the TRUE
#                             relative magnitude (SFT ≈ DPO contribution).
#                             Old smoke used 0.1 which was effectively ~1-3%.
#   --sft_scale_mode match_dpo  scale L_sft by mean(1/std_g) so sft_weight is
#                               commensurate with L_dpo_grpo.
#   --group_norm_eps 0.05     cap 1/std at 20× (was 1e-3 = 1000× → step 0
#                             l_dpo=693 explosion).
#   --group_norm_warmup 50    skip group-norm for first 50 steps (cold start
#                             when trained=ref → std≈0).
#   --best_metric chosen_score  pick best ckpt by absolute chosen log-prob,
#                               which correlates with recall_chosen at eval.
#                               Old default eval_pref_acc was decoupled.
#   --kl_weight 0             DPO has implicit KL via ref baseline.
#
# Total wall clock:
#   First run:        ~30min ref precompute + ~2.5h train + ~15min ckpt evals
#                     + ~30min final eval (n=5000 engagement) ≈ 3.5h
#   Subsequent runs:  ref scores cache HIT (1 sec load) ≈ 3.0h
#                     Cache lives at data/contrastive_dataset_v1_grpo/_ref_cache/
#                     Tuning sft_weight / dpo_beta / lr / group_norm_* keeps
#                     cache hot; changing max_*_groups / max_hist invalidates.
# Old smoke (batch=12, live ref every step, 6×1000-group evals) was ~6.5h.

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
echo ">>> STEP 1/2: DPO + SFT + GRPO normalize smoke (5000 groups)"
echo ">>> loss = L_dpo_grpo + 1.0 × (1/std).mean × L_sft  (kl_weight=0)"
echo ">>> 5 evenly-spaced checkpoints; eval at each save (300 groups)"
echo ">>> per_device_batch_size=24, ref scores pre-computed (free ref_model)"
echo ">>> ref scores cached to disk → first run ~30min precompute, then 1sec"
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
    --max_train_groups 5000 --max_eval_groups 300 \
    --num_checkpoints 5 --logging_steps 25 \
    --per_device_batch_size 24 --grad_accum 1 \
    --lr 5e-5 \
    --dpo_beta 0.1 \
    --sft_weight 1.0 \
    --sft_scale_mode match_dpo \
    --group_norm_eps 0.05 \
    --group_norm_warmup 50 \
    --best_metric chosen_score \
    --kl_weight 0 \
    --merge_and_save \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_DONE=$(date +%s)
TRAIN_MIN=$(( (TRAIN_DONE - START_TIME) / 60 ))

# ============================================================
# STEP 2/2: Engagement-aware evaldaima
# ============================================================
echo ""
echo "============================================================"
echo ">>> STEP 2/2: Engagement-aware eval (n=5000, ~30min — beam search, "
echo "    standalone evaluate_engaged.py, not affected by pre-compute)"
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
echo "DPO+SFT 5k (broken design)  0.0047    0.0018    +0.0029   +0.0084  ← sft_weight 0.1 was effectively ~1-3% of loss"
echo "DPO+SFT+GRPO smoke (this)   see above"
echo ""
echo "Verdict guide for chosen recall (the key question this smoke answers):"
echo "  chosen ≥ baseline 0.0093    → SFT anchor works, chosen is RISING ✓"
echo "  chosen ∈ [0.005, 0.0093)    → partial improvement, bump sft_weight to 2.0 or 3.0"
echo "  chosen < 0.005              → match_dpo scaling not enough either; try sft_weight 5.0"
echo "                                or lower lr to 2e-5 (less drift from ref)"
echo ""
echo "Δ stays positive AND chosen ≥ baseline = the goal achieved."
