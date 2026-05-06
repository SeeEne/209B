#!/usr/bin/env bash
#
# run_sft_lora_r_sweep.sh
#
# Single-variable sweep: LoRA r=16 → r=32 on the SFT-only Stage 1 trainer.
#
# Why this exists: the lr sweep on 2026-05-05 (run_sft_lr_sweep.sh) showed
# that lr=5e-5, 2e-4, 5e-4 all converge to chosen_score ≈ -4.84 — a hard
# ceiling at 5k-data + LoRA r=16. The remaining question is whether that
# ceiling is the *data scale* or the *LoRA capacity* that's limiting.
#
# This sweep doubles LoRA capacity (r=16 → 32) while holding everything
# else fixed, including alpha/r ratio (alpha=32 → 64 keeps the effective
# update scale = α/r = 2.0). Trainable params: 17M → 35M.
#
# Three possible outcomes:
#   r=32 final < -4.841   →  LoRA capacity is the bottleneck.
#                            Next step: 50k data with r=32.
#   r=32 final ≈ -4.841   →  Capacity is NOT the issue → DATA is.
#                            Next step: 50k data with r=16 (cheaper).
#   r=32 final > -4.841   →  Overfit / divergence (unlikely at 5k 1 epoch).
#
# All non-LoRA hyperparams identical to runs/sft_only_5k:
#   lr=5e-5, batch=24, grad_accum=1, warmup_ratio=0.05, 1 epoch,
#   5000 train groups, 1000 valid groups, num_checkpoints=5,
#   sft_loss_scale=16.0, max_grad_norm=1.0, seed=42, dtype=bf16.
#
# Wall clock ~1.5h on RTX 6000 Pro / H100. r=32 doubles trainable params
# but compute is dominated by the frozen 1.7B backbone, so step time is
# essentially unchanged.

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
TEMPLATE="model/qwen3_soft_switch.jinja2"
BASE_MODEL="model/OneRec-1.7B"

REF_RUN="runs/sft_only_5k"     # r=16, lr=5e-5 baseline
REF_LABEL="lora_r=16 alpha=32  (reference, lr=5e-5)"

RUN="runs/sft_only_5k_r32"
LOG="${RUN}.log"
RUN_LABEL="lora_r=32 alpha=64  (sweep, lr=5e-5)"

START=$(date +%s)

# ---- Sanity ----
if [ ! -f "$DATA_GRPO/train.parquet" ] || [ ! -f "$DATA_GRPO/valid.parquet" ]; then
    echo ">>> [error] G=3 dataset not found at $DATA_GRPO"; exit 1
fi
if [ ! -d "$REF_RUN" ]; then
    echo ">>> [warn] reference $REF_RUN not found — summary will skip baseline"
fi

# ============================================================
# Train r=32
# ============================================================
if [ -d "$RUN/adapter" ]; then
    echo ">>> [skip] $RUN already trained (adapter exists). rm -rf to redo."
else
    echo ""
    echo "============================================================"
    echo ">>> Training SFT-only  $RUN_LABEL"
    echo ">>> alpha/r kept at 2.0 (= same effective update scale as r=16)"
    echo ">>> all other hyperparams match $REF_RUN"
    echo ">>> out: $RUN"
    echo "============================================================"

    python train/train_sft_only.py \
        --model_path "$BASE_MODEL" \
        --template "$TEMPLATE" \
        --train_parquet "$DATA_GRPO/train.parquet" \
        --valid_parquet "$DATA_GRPO/valid.parquet" \
        --output_dir "$RUN" \
        --max_train_groups 5000 --max_eval_groups 1000 \
        --num_checkpoints 5 --logging_steps 25 \
        --per_device_batch_size 24 --grad_accum 1 \
        --lr 5e-5 \
        --lora_r 32 --lora_alpha 64 \
        --merge_and_save \
        2>&1 | tee "$LOG"
fi

END=$(date +%s)
TOTAL_MIN=$(( (END - START) / 60 ))

# ============================================================
# Summary
# ============================================================
echo ""
echo "############################################################"
echo "# SFT LoRA r SWEEP COMPLETE  (total ${TOTAL_MIN} min)"
echo "############################################################"

extract_trend() {
    local RUN_DIR="$1"
    local LABEL="$2"
    local LOG_PATH="${RUN_DIR}.log"

    echo ">>> $LABEL"
    if [ ! -f "$LOG_PATH" ]; then
        echo "    [no log] $LOG_PATH"
        return
    fi

    mapfile -t SCORES < <(
        sed -nE "s/.*'eval_chosen_score': '([-+0-9.eE]+)'.*/\1/p" "$LOG_PATH"
    )
    local N=${#SCORES[@]}
    if [ "$N" -eq 0 ]; then
        echo "    (no eval_chosen_score lines found in $LOG_PATH)"
        return
    fi

    local LABELS=("ckpt 1 (step 125)" "ckpt 2 (step 250)" "ckpt 3 (step 375)" \
                  "ckpt 4 (step 500)" "ckpt 5 (step 625)" "final (best ckpt)")
    for ((i=0; i<N && i<6; i++)); do
        echo "    ${LABELS[$i]}: eval_chosen_score = ${SCORES[$i]}"
    done
}

if [ -d "$REF_RUN" ]; then
    extract_trend "$REF_RUN" "$REF_LABEL"
fi
extract_trend "$RUN" "$RUN_LABEL"

echo ""
echo "============================================================"
echo "Decision rule"
echo "============================================================"
echo "Compare r=32 final to r=16 reference (-4.841):"
echo "  r=32 final  <  -4.841  →  capacity bottleneck → next: 50k + r=32"
echo "  r=32 final  ≈  -4.841  →  data bottleneck    → next: 50k + r=16"
echo "  r=32 final  >  -4.841  →  unexpected divergence; investigate"
echo ""
echo "For an engagement-aware confirmation on v1 valid (~10 min):"
echo "  python diagnose/sft_score_trend.py \\"
echo "    --base $BASE_MODEL \\"
echo "    --adapters $REF_RUN/adapter $RUN/adapter \\"
echo "    --include_base --template $TEMPLATE \\"
echo "    --n 1000 --batch_size 16 \\"
echo "    --output_csv runs/sft_lora_r_sweep_trend.csv"