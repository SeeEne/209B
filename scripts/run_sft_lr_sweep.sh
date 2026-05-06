#!/usr/bin/env bash
#
# run_sft_lr_sweep.sh
#
# Single-dimension lr sweep for the SFT-only Stage 1 trainer.
#
# Motivation: in the 2026-05-05 sequential smoke, SFT-only with lr=5e-5
# moved chosen log P from base -4.92 → ckpt-125 -4.86 (+0.063 nats), but
# only +0.014 nats over the next 500 steps. The diagnosis is that grad_clip
# (max_grad_norm=1.0) was hit on every step (grad_norm 35-80 all training),
# making each step's effective magnitude = lr × unit_vector. With lr=5e-5
# the optimizer gets stuck in the first local minimum it finds.
#
# This sweep tests whether bumping lr lets SFT escape that local optimum:
#
#   reference  lr=5e-5  (existing run at runs/sft_only_5k)
#   sweep      lr=2e-4  (4×; standard LoRA SFT lower bound)
#   sweep      lr=5e-4  (10×; standard LoRA SFT upper-mid)
#
# All other hyperparams identical to the 2026-05-05 SFT smoke (so the
# only variable IS lr — clean ablation):
#   per_device_batch_size=24, grad_accum=1, warmup_ratio=0.05,
#   weight_decay=0.0, max_grad_norm=1.0, lora_r=16, alpha=32, dropout=0.05,
#   epochs=1, num_checkpoints=5, sft_loss_scale=16.0, seed=42.
#
# Usage (run from project root):
#     bash scripts/run_sft_lr_sweep.sh
#
# Total wall clock ~3h on RTX 6000 Pro / H100 (1.5h × 2 lr values).
# Each run skips if its output dir already exists (resume-friendly).

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
TEMPLATE="model/qwen3_soft_switch.jinja2"
BASE_MODEL="model/OneRec-1.7B"

# lr values to sweep. Edit this array to add/remove runs.
LRS=(2e-4 5e-4)

# Reference run (already trained, included in summary table only).
REF_RUN="runs/sft_only_5k"
REF_LR="5e-5"

START=$(date +%s)

# ---- Sanity ----
if [ ! -f "$DATA_GRPO/train.parquet" ] || [ ! -f "$DATA_GRPO/valid.parquet" ]; then
    echo ">>> [error] G=3 dataset not found at $DATA_GRPO"; exit 1
fi
if [ ! -d "$REF_RUN" ]; then
    echo ">>> [warn] reference run $REF_RUN not found — summary will skip lr=$REF_LR"
fi

# ============================================================
# Train each lr (skip if already done)
# ============================================================
RUN_DIRS=()
for LR in "${LRS[@]}"; do
    OUT="runs/sft_only_5k_lr${LR}"
    LOG="${OUT}.log"
    RUN_DIRS+=("$OUT")

    if [ -d "$OUT/adapter" ]; then
        echo ">>> [skip] $OUT already trained (adapter exists)"
        continue
    fi

    echo ""
    echo "============================================================"
    echo ">>> Training SFT-only with lr=$LR"
    echo ">>> out: $OUT"
    echo "============================================================"

    python train/train_sft_only.py \
        --model_path "$BASE_MODEL" \
        --template "$TEMPLATE" \
        --train_parquet "$DATA_GRPO/train.parquet" \
        --valid_parquet "$DATA_GRPO/valid.parquet" \
        --output_dir "$OUT" \
        --max_train_groups 5000 --max_eval_groups 1000 \
        --num_checkpoints 5 --logging_steps 25 \
        --per_device_batch_size 24 --grad_accum 1 \
        --lr "$LR" \
        --merge_and_save \
        2>&1 | tee "$LOG"
done

END=$(date +%s)
TOTAL_MIN=$(( (END - START) / 60 ))

# ============================================================
# Summary: extract eval_chosen_score per ckpt for each run
# ============================================================
echo ""
echo "############################################################"
echo "# SFT lr SWEEP COMPLETE  (total ${TOTAL_MIN} min)"
echo "############################################################"

# extract_trend <run_dir> <lr>
# Greps the eval_chosen_score values from the train log. Trainer emits
# 5 ckpt evals (single-line dict with single-quoted strings) plus 1 final
# eval after load_best_model_at_end → 6 lines total. Output labels them as
# ckpt 1-5 + 'final' (the best, post-reload).
extract_trend() {
    local RUN="$1"
    local LR="$2"
    local LOG="${RUN}.log"

    echo ">>> lr=$LR  ($RUN)"
    if [ ! -f "$LOG" ]; then
        echo "    [no log] $LOG"
        return
    fi

    # Extract just the float value after 'eval_chosen_score': '...'
    # using sed capture group (avoids the [-+0-9.eE]+ pitfall where 'e'
    # in 'eval_chosen_score' itself would be matched standalone).
    mapfile -t SCORES < <(
        sed -nE "s/.*'eval_chosen_score': '([-+0-9.eE]+)'.*/\1/p" "$LOG"
    )
    local N=${#SCORES[@]}
    if [ "$N" -eq 0 ]; then
        echo "    (no eval_chosen_score lines found in log)"
        return
    fi

    local LABELS=("ckpt 1 (step 125)" "ckpt 2 (step 250)" "ckpt 3 (step 375)" \
                  "ckpt 4 (step 500)" "ckpt 5 (step 625)" "final (best ckpt)")
    for ((i=0; i<N && i<6; i++)); do
        echo "    ${LABELS[$i]}: eval_chosen_score = ${SCORES[$i]}"
    done
}

# Reference run first.
if [ -d "$REF_RUN" ]; then
    extract_trend "$REF_RUN" "$REF_LR"
fi
# Then sweep runs.
for i in "${!LRS[@]}"; do
    extract_trend "${RUN_DIRS[$i]}" "${LRS[$i]}"
done

echo ""
echo "============================================================"
echo "How to read the trend"
echo "============================================================"
echo "Reference (lr=5e-5) v1_grpo eval_chosen_score curve:"
echo "  ckpt 1 (step 125): -4.856"
echo "  ckpt 5 (step 625): -4.841   ← gain saturated early (~step 125)"
echo ""
echo "If lr=2e-4 / 5e-4 gives ckpt 5 score < -4.84  →  lr broke the local"
echo "  optimum, sweep WORKED, scale up data next."
echo "If still ≈ -4.84  →  lr is not the bottleneck. Try LoRA r=32, more"
echo "  epochs, or scale data straight to 50k."
echo "If much worse (e.g. -5.5+)  →  lr too high, training diverged."
echo ""
echo "For a finer comparison on the v1 engagement valid (40 min):"
echo "  python diagnose/sft_score_trend.py \\"
echo "    --base $BASE_MODEL \\"
echo "    --adapters $REF_RUN/adapter \\"
for D in "${RUN_DIRS[@]}"; do
    echo "               $D/adapter \\"
done
echo "    --include_base --template $TEMPLATE \\"
echo "    --n 1000 --batch_size 16 \\"
echo "    --output_csv runs/sft_lr_sweep_trend.csv"
echo ""
echo "Note on duplicate adapter names: if the diagnose run reports adapter"
echo "names as adapter_0/1/2, the order matches the --adapters list above."