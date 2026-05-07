#!/usr/bin/env bash
#
# run_orpo_50k.sh
#
# Full ORPO run at 50,000 groups (10× the smoke). Same config as the
# 5k smoke, just more data + more steps. The 5k smoke established:
#   - chosen_score saturated at the SFT-only ceiling (-4.840 vs -4.841)
#     but kept rising monotonically through ckpt 5 (decelerating, not flat)
#   - engagement-aware recall_chosen = 0.0107 vs SFT-only 0.0097 vs base 0.0093
#   - Δrecall = -0.0017 (closer to zero than any prior 5k arm)
#   - pref_acc plateaued at 0.535 — OR-term signal weak at λ=0.1
#
# The 50k bet: more gradient steps on the OR term + more chosen examples
# break the SFT ceiling, lift pref_acc above 0.55, and push Δrecall to
# zero or positive.
#
# Hyperparameters: identical to the 5k smoke (paper Eq. 3 mean-log-p,
# λ=0.1 paper default, lr=5e-5, batch=24, 1 epoch, LoRA r=16/α=32). NO
# extra knobs to ablate here — that's a separate λ sweep if needed.
#
# Schedule details for 50k (vs 5k smoke):
#   total steps        : 6,250 (vs 625 at 5k)
#   warmup steps       : 312 (=0.05 × 6250)
#   ckpt interval      : 1,250 (=6250/5, vs 125 at 5k)
#   per-step time      : ~21 s (unchanged — same shape, GPU-bound on SDPA)
#   train wall clock   : ~36.5 h
#   in-training evals  : 5 × ~12 min (n=1000 pairs) = ~1 h
#   final engagement eval (n=1000, ~40 min)
#   total wall clock   : ~38 h
#
# If you'd rather drop to ~25k for a faster middle-ground (~19 h),
# override via env: MAX_TRAIN_GROUPS=25000 bash scripts/run_orpo_50k.sh
#
# Output:
#   runs/orpo_50k/
#     ├── checkpoint-1250 / 2500 / 3750 / 5000 / 6250
#     ├── adapter/    (LoRA at best ckpt by eval_chosen_score)
#     └── merged/     (full model = base + adapter)
#   runs/orpo_50k.log
#   runs/eval_engaged_orpo_50k_n1000.csv
#
# HF backup (always-on by default): every checkpoint streams to Hub as
# it's saved. ESSENTIAL for a 38h run on shared compute — a disconnect
# at hour 30 doesn't lose 30 hours of compute, only the most recent
# 250 steps (~1.5h) since the previous save.

set -e
set -o pipefail

# ---- Environment ----
HF_PUSH="${HF_PUSH:-1}"
HF_REPO_ID="${HF_REPO_ID:-SeeEne/onerec-209b-runs}"
HF_RUN_NAME="${HF_RUN_NAME:-orpo_50k}"
BASE_REPO="${BASE_REPO:-OpenOneRec/OneRec-1.7B}"

# Override knob: lower MAX_TRAIN_GROUPS for a faster middle-ground run
# without copying the script (e.g. 25000 → ~19h, 10000 → ~7.5h).
MAX_TRAIN_GROUPS="${MAX_TRAIN_GROUPS:-50000}"
LAMBDA_OR="${LAMBDA_OR:-0.1}"

# Don't set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE — both block Hub
# traffic and modern huggingface_hub treats them as the same flag.
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# ---- Paths ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_GRPO="data/contrastive_dataset_v1_grpo"
DATA_V1="data/contrastive_dataset_v1"
TEMPLATE="oneRec/qwen3_soft_switch.jinja2"
BASE_MODEL="model/OneRec-1.7B"

RUN="runs/${HF_RUN_NAME}"   # runs/orpo_50k by default
TRAIN_LOG="${RUN}.log"
EVAL_DIR="runs/eval_logs"
EVAL_LOG="$EVAL_DIR/${HF_RUN_NAME}_n1000.log"
EVAL_CSV="runs/eval_engaged_${HF_RUN_NAME}_n1000.csv"

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
# Auto-pull base model if missing (~3.5 GB, excludes assets/* preview PNGs).
if [ ! -d "$BASE_MODEL" ] || [ -z "$(ls -A "$BASE_MODEL" 2>/dev/null)" ]; then
    echo ""
    echo "============================================================"
    echo ">>> base model not at $BASE_MODEL — pulling $BASE_REPO from HF Hub"
    echo "============================================================"
    python train/hf_sync.py pull \
        --repo_id "$BASE_REPO" \
        --local_path "$BASE_MODEL" \
        --ignore_patterns "assets/*"
fi

# ============================================================
# STEP 1/2 — Train ORPO 50k (~36.5 h)
# ============================================================
if [ -d "$RUN/merged" ]; then
    echo ">>> [skip] $RUN/merged already exists; skipping training."
    echo "    rm -rf $RUN to retrain from scratch."
else
    echo ""
    echo "============================================================"
    echo ">>> STEP 1/2: ORPO train  ($MAX_TRAIN_GROUPS groups, ~36.5h at 50k)"
    echo ">>> lr=5e-5, lora_r=16/alpha=32, batch=24, 1 epoch"
    echo ">>> lambda_or=$LAMBDA_OR  (paper default; mean-log-p, no rescaling)"
    echo ">>> nll_loss_scale=1.0    (paper formulation)"
    echo ">>> from base (single stage, no SFT init, no ref model)"
    echo ">>> out: $RUN"
    echo ">>> log: $TRAIN_LOG"
    echo "============================================================"

    if [ -d "$RUN" ]; then
        echo ">>> [warn] $RUN exists without merged/. HF Trainer will resume."
        echo "    rm -rf $RUN for a fresh start. Continuing in 5s..."
        sleep 5
    fi

    HF_FLAGS=()
    if [ "$HF_PUSH" = "1" ]; then
        HF_FLAGS=(
            --hf_push
            --hf_repo_id "$HF_REPO_ID"
            --hf_run_name "$HF_RUN_NAME"
            --hf_log_path "$TRAIN_LOG"
        )
        echo ">>> HF backup: ON  → $HF_REPO_ID:$HF_RUN_NAME"
        echo ">>> (essential at this scale — disconnect at hr 30 only loses ~1.5h)"
    else
        echo ">>> HF backup: OFF  (NOT recommended at 50k — set HF_PUSH=1)"
    fi

    python train/train_orpo.py \
        --model_path "$BASE_MODEL" \
        --template "$TEMPLATE" \
        --train_parquet "$DATA_GRPO/train.parquet" \
        --valid_parquet "$DATA_GRPO/valid.parquet" \
        --output_dir "$RUN" \
        --max_train_groups "$MAX_TRAIN_GROUPS" --max_eval_groups 1000 \
        --num_checkpoints 5 --logging_steps 50 \
        --per_device_batch_size 24 --grad_accum 1 \
        --lr 5e-5 \
        --lambda_or $LAMBDA_OR \
        --nll_loss_scale 1.0 \
        --lora_r 16 --lora_alpha 32 \
        --merge_and_save \
        "${HF_FLAGS[@]}" \
        2>&1 | tee "$TRAIN_LOG"
fi

T_AFTER_TRAIN=$(date +%s)
TRAIN_MIN=$(( (T_AFTER_TRAIN - START) / 60 ))

# ============================================================
# STEP 2/2 — Engagement-aware eval (n=1000, ~40 min beam search)
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
# STEP 3 — push eval artifacts to HF Hub (training-side already pushed)
# ============================================================
if [ "$HF_PUSH" = "1" ]; then
    echo ""
    echo "============================================================"
    echo ">>> STEP 3/3: push eval artifacts to $HF_REPO_ID:$HF_RUN_NAME"
    echo "============================================================"
    EXTRA_FILES=()
    [ -f "$EVAL_LOG" ] && EXTRA_FILES+=("$EVAL_LOG")
    [ -f "$EVAL_CSV" ] && EXTRA_FILES+=("$EVAL_CSV")
    if [ ${#EXTRA_FILES[@]} -gt 0 ]; then
        python train/hf_sync.py push \
            --repo_id "$HF_REPO_ID" \
            --run_name "$HF_RUN_NAME" \
            --run_dir "$RUN" \
            --include \
            --extra_files "${EXTRA_FILES[@]}" \
            || echo "[hf-sync] eval push failed (training artifacts already on Hub)"
    fi
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "############################################################"
echo "# ORPO 50K COMPLETE  (train ${TRAIN_MIN} min + eval ${EVAL_MIN} min = ${TOTAL_MIN} min)"
echo "############################################################"

# eval_chosen_score trend across the 5 50k checkpoints.
echo ""
echo "============================================================"
echo "Train-time eval_chosen_score trend (50k, 5 ckpts + final)"
echo "  5k smoke reference points (lambda_or=0.1, lr=5e-5):"
echo "    base                  ≈ -4.92"
echo "    SFT-only 5k final     = -4.841"
echo "    ORPO 5k final         = -4.840"
echo "    ORPO 5k pref_acc      = 0.535  (the 5k ceiling we want to break)"
echo "============================================================"

if [ -f "$TRAIN_LOG" ]; then
    extract_metric() {
        local key="$1"
        sed -nE "s/.*[\"']${key}[\"']: [\"']?([-+0-9.eE]+)[\"']?.*/\1/p" "$TRAIN_LOG"
    }
    mapfile -t SCORES < <(extract_metric eval_chosen_score)
    mapfile -t PREF < <(extract_metric eval_pref_acc)
    mapfile -t MARGIN < <(extract_metric eval_margin)
    mapfile -t LOG_ODDS_M < <(extract_metric eval_log_odds_margin)

    # Compute step labels dynamically (50k → 1250/2500/3750/5000/6250).
    STEPS_TOTAL=$(( MAX_TRAIN_GROUPS * 3 / 24 ))   # G=3, batch=24
    STEP_INTERVAL=$(( STEPS_TOTAL / 5 ))
    LABELS=()
    for i in 1 2 3 4 5; do
        LABELS+=("ckpt $i (step $(( i * STEP_INTERVAL )))")
    done
    LABELS+=("final (best ckpt)")

    for ((i=0; i<${#SCORES[@]} && i<6; i++)); do
        echo "    ${LABELS[$i]}: eval_chosen_score = ${SCORES[$i]}"
    done
    if [ ${#PREF[@]} -gt 0 ]; then
        echo ""
        echo "    OR-term diagnostics:"
        for ((i=0; i<${#PREF[@]} && i<6; i++)); do
            echo "    ${LABELS[$i]}: pref_acc=${PREF[$i]}  margin=${MARGIN[$i]}  log_odds=${LOG_ODDS_M[$i]:-n/a}"
        done
    fi
fi

# Headline engagement-aware metrics.
echo ""
echo "============================================================"
echo "Engagement-aware Recall@K  (n=$EVAL_N, on v1 valid)"
echo "  Baseline (no FT)        : chosen=0.0093  rejected=0.0163  Δ=-0.0070"
echo "  SFT-only 5k (lr=5e-5)   : chosen=0.0097  rejected=0.0127  Δ=-0.0030"
echo "  ORPO 5k (lambda_or=0.1) : chosen=0.0107  rejected=0.0123  Δ=-0.0017"
echo "  → ORPO 50k target: chosen >= 0.012, AND ideally Δ ≥ 0 (positive)."
echo "============================================================"
if [ -f "$EVAL_LOG" ]; then
    grep -E "^recall@96|^pass@96" "$EVAL_LOG" | tail -n 2 | sed 's/^/  /'
fi

echo ""
echo "============================================================"
echo "Decision tree for the report's headline ablation row"
echo "============================================================"
echo "If chosen_score breaks below -4.84 AND chosen_recall > 0.012 AND Δrecall ≥ 0:"
echo "  → ORPO 50k is the best arm. Headline result for the report."
echo "  → Compare to SFT 50k once that's also done."
echo ""
echo "If chosen_score breaks below -4.84 BUT Δrecall stays negative:"
echo "  → OR term is helping chosen but not pushing rejected down enough."
echo "  → Try lambda_or=0.25 at 50k (~36h) for stronger discrimination."
echo ""
echo "If chosen_score stays at -4.84 (ORPO didn't break the SFT ceiling):"
echo "  → Data scale wasn't the bottleneck for chosen log-prob; both ORPO and"
echo "    SFT-only saturate the OneRec base around -4.84. Reconsider:"
echo "    prompt format, eval methodology, or whether the base is genuinely"
echo "    overfit to next-shown."
echo ""
echo "Trend across the 5 50k ckpts (~1.5h, no beam search) for paper figure:"
echo "  python diagnose/sft_score_trend.py \\"
echo "    --base $BASE_MODEL --include_base \\"
echo "    --adapters $RUN/checkpoint-1250 $RUN/checkpoint-2500 \\"
echo "               $RUN/checkpoint-3750 $RUN/checkpoint-5000 \\"
echo "               $RUN/checkpoint-6250 \\"
echo "    --template $TEMPLATE --n 1000 --batch_size 16 \\"
echo "    --output_csv runs/${HF_RUN_NAME}_trend.csv"
