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
#     lambda_or = 0.1   (paper default for Mistral-ORPO; Hong et al. 2024)
#       Paper Eq. 3 defines log P(y|x) as the LENGTH-NORMALIZED mean
#       per-token log-likelihood = (1/m) Σ log p_t. That's exactly the
#       convention every trainer in this repo uses, so 0.1 here is the
#       literal paper value, no scaling needed. The paper used larger λ
#       for smaller models (0.25 for Phi-2 2.7B, 0.2 for Llama-2 7B), so
#       at 1.7B 0.1–0.2 is the defensible range; start at 0.1.
#
#   Deviations from the paper's optimization recipe (kept for cross-arm
#   parity with our SFT/DPO arms, NOT for paper-faithfulness):
#     - lr = 5e-5 vs paper 8e-6  (parity)
#     - 1 epoch vs paper 10      (parity + wall clock)
#     - LoRA r=16 vs full FT     (single-GPU compute)
#     - linear schedule vs cosine (parity)
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
# HF Hub backup of every checkpoint as it's saved + final adapter/merged/log.
# Set HF_PUSH=0 to disable (e.g. for offline dev runs). Repo is created
# private on first push; layout mirrors runs/<HF_RUN_NAME>/ inside the repo.
HF_PUSH="${HF_PUSH:-1}"
HF_REPO_ID="${HF_REPO_ID:-SeeEne/onerec-209b-runs}"
HF_RUN_NAME="${HF_RUN_NAME:-orpo_5k}"

# Base model: pulled on first run if not already on disk. Override BASE_REPO
# if you want a different OneRec checkpoint.
BASE_REPO="${BASE_REPO:-OpenOneRec/OneRec-1.7B}"

# DO NOT set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE here. Both block Hub
# traffic, and modern huggingface_hub treats them as the same flag (see
# constants.py: HF_HUB_OFFLINE = is_true(HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE)).
# Setting either would break: (a) the auto-pull of the base model below,
# (b) the checkpoint-streaming callback during training, (c) the eval-
# artifact push at end-of-run. Local-path model loads via transformers
# don't phone home in practice, so the offline flag is unnecessary.
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export OMP_NUM_THREADS=8

# Force python stdout/stderr to flush line-by-line so `tee` shows training
# progress live in the terminal AND writes the same content to the .log
# file. Without this, python uses BLOCK buffering when stdout is a pipe,
# and the terminal sits idle for minutes between bursts.
export PYTHONUNBUFFERED=1

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
LAMBDA_OR=0.1

START=$(date +%s)

# ---- Sanity ----
if [ ! -f "$DATA_GRPO/train.parquet" ] || [ ! -f "$DATA_GRPO/valid.parquet" ]; then
    echo ">>> [error] G=3 dataset not found at $DATA_GRPO"; exit 1
fi
if [ ! -f "$DATA_V1/valid.parquet" ]; then
    echo ">>> [error] v1 valid not found at $DATA_V1/valid.parquet"; exit 1
fi
# Auto-pull base model if missing (fresh shared-compute box, ~3.5 GB).
# We exclude assets/* (preview PNGs) — not needed for training.
if [ ! -d "$BASE_MODEL" ] || [ -z "$(ls -A "$BASE_MODEL" 2>/dev/null)" ]; then
    echo ""
    echo "============================================================"
    echo ">>> base model not at $BASE_MODEL — pulling $BASE_REPO from HF Hub"
    echo ">>> (~3.5 GB; needs network. Retry on flaky links via --force.)"
    echo "============================================================"
    python train/hf_sync.py pull \
        --repo_id "$BASE_REPO" \
        --local_path "$BASE_MODEL" \
        --ignore_patterns "assets/*"
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
    echo ">>> lr=5e-5, lora_r=$LORA_R/alpha=$LORA_ALPHA, batch=$PER_DEVICE_BATCH_SIZE, 1 epoch"
    echo ">>> lambda_or=$LAMBDA_OR  (paper default; loss uses paper Eq. 3 mean-log-p, no rescaling)"
    echo ">>> nll_loss_scale=$NLL_LOSS_SCALE    (paper formulation; not match_dpo)"
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
    else
        echo ">>> HF backup: OFF (set HF_PUSH=1 to enable)"
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
        "${HF_FLAGS[@]}" \
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
# STEP 3 — push eval artifacts to HF Hub (training-side already pushed)
# ============================================================
# The training callback already streamed every checkpoint, and the post-
# training hook pushed adapter/, merged/, and the train log. The eval
# log + CSV are produced AFTER train_orpo.py exits, so push them now.
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
echo "  → OR term is harming chosen log-prob. Try lambda_or=0.05 (half of"
echo "    paper default) or nll_loss_scale=2.0 to up-weight the NLL term."
echo ""
echo "For richer trend across the 5 ckpts (~15 min, no beam search):"
echo "  python diagnose/sft_score_trend.py \\"
echo "    --base $BASE_MODEL --include_base \\"
echo "    --adapters $RUN/checkpoint-125 $RUN/checkpoint-250 \\"
echo "               $RUN/checkpoint-375 $RUN/checkpoint-500 \\"
echo "               $RUN/checkpoint-625 \\"
echo "    --template $TEMPLATE --n 1000 --batch_size 16 \\"
echo "    --output_csv runs/orpo_5k_trend.csv"
