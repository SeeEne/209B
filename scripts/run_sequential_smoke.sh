#!/usr/bin/env bash
#
# run_sequential_smoke.sh
#
# Sequential SFT → DPO smoke + ablation (5000 groups each stage).
#
# Motivation: in the joint DPO+SFT smoke (sft_weight=1.0 match_dpo, run on
# 2026-05-04), eval_pref_acc was only 0.546 (~random) because SFT was
# claiming ~87% of the gradient and swamping DPO's discriminative signal.
# Splitting into two stages — SFT first, then DPO from the SFT checkpoint —
# lets each stage get 100% of the gradient on a single objective. This is
# also the standard post-training recipe (InstructGPT / Llama-3 / DeepSeek).
#
# As a bonus, the Stage 1 checkpoint IS the SFT-only ablation arm — the
# missing entry from our 2026-05-04 ablation table.
#
# Pipeline (~5.5h total on RTX 6000 Pro, first run):
#   STEP 0 (~6 min)  : re-eval the joint DPO+SFT smoke at n=1000 so its
#                       numbers are directly comparable to the n=1000
#                       baseline / length=3 / GRPO entries in the table.
#   STEP 1 (~1.5h)   : Stage 1 SFT-only on base OneRec, 5k groups
#   STEP 2 (~6 min)  : engagement-aware eval of Stage 1 (n=1000)  ← SFT-only ablation
#   STEP 3 (~3.5h)   : Stage 2 DPO from Stage 1, 5k groups (ref = Stage 1 merged)
#                       breakdown: ~60 min ref precompute (cache COLD on first
#                       run because the SFT-merged ref is brand-new — its hash
#                       differs from the joint-smoke cache) + ~135 min train
#                       + ~6 min final eval.
#   STEP 4 (~6 min)  : engagement-aware eval of Stage 2 (n=1000)  ← final SFT→DPO
#
# All eval CSVs land in runs/. The reference-score cache for Stage 2 lives
# under data/contrastive_dataset_v1_grpo/_ref_cache/ — Stage 2's cache key
# includes ref_model_path, so a fresh SFT checkpoint always invalidates.
#
# Usage (run from project root):
#     bash scripts/run_sequential_smoke.sh
#
# Each step writes its own log under runs/eval_logs/ or runs/<run>/.

set -e
set -o pipefail   # Critical for overnight: without this, `python ... | tee` hides
                  # python failures (tee always succeeds) and downstream stages run
                  # on broken artifacts. Stage 2 starting on a half-trained SFT
                  # checkpoint would burn ~3.5h before failing the engagement eval.

# ---- Environment ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

# ---- Paths ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_GRPO="data/contrastive_dataset_v1_grpo"
DATA_V1="data/contrastive_dataset_v1"
TEMPLATE="model/qwen3_soft_switch.jinja2"
BASE_MODEL="model/OneRec-1.7B"

JOINT_RUN="runs/dpo_grpo_smoke"        # already trained 2026-05-04
SFT_RUN="runs/sft_only_5k"
DPO_RUN="runs/dpo_from_sft_5k"

EVAL_DIR="runs/eval_logs"
mkdir -p "$EVAL_DIR"

EVAL_N=1000  # match baseline / length=3 / GRPO ablation table entries

# ---- Sanity checks (fail FAST before a long sleep run) ----
if [ ! -f "$DATA_GRPO/train.parquet" ] || [ ! -f "$DATA_GRPO/valid.parquet" ]; then
    echo ">>> [error] G=3 dataset not found at $DATA_GRPO"
    echo "    build: python build_contrastive_dataset_GRPO.py --G 3"
    exit 1
fi
if [ ! -f "$DATA_V1/valid.parquet" ]; then
    echo ">>> [error] v1 valid (engagement eval source) not found at $DATA_V1/valid.parquet"
    exit 1
fi
if [ ! -d "$JOINT_RUN/merged" ]; then
    echo ">>> [warn] $JOINT_RUN/merged not found — STEP 0 (joint re-eval) will be skipped"
    SKIP_STEP0=1
fi

START=$(date +%s)

# ============================================================
# STEP 0 — re-eval joint DPO+SFT smoke at n=1000 (apples-to-apples vs baseline)
# ============================================================
if [ -z "$SKIP_STEP0" ]; then
    echo ""
    echo "============================================================"
    echo ">>> STEP 0/4: re-eval joint DPO+SFT (n=$EVAL_N)"
    echo ">>> model: $JOINT_RUN/merged"
    echo "============================================================"
    python train/evaluate_engaged.py \
        --model_path "$JOINT_RUN/merged" \
        --valid_parquet "$DATA_V1/valid.parquet" \
        --template "$TEMPLATE" \
        --n $EVAL_N --num_beams 32 --topk 96 \
        --output_csv "runs/eval_engaged_joint_n${EVAL_N}.csv" \
        2>&1 | tee "$EVAL_DIR/joint_reval_n${EVAL_N}.log"
fi

T_AFTER_STEP0=$(date +%s)

# ============================================================
# STEP 1 — Stage 1: SFT-only on base OneRec, 5k groups
# ============================================================
echo ""
echo "============================================================"
echo ">>> STEP 1/4: Stage 1 SFT-only train (5000 groups, ~1.5h)"
echo ">>> base : $BASE_MODEL"
echo ">>> out  : $SFT_RUN"
echo ">>> loss : -mean( log P(chosen | history) )   [no DPO, no ref]"
echo "============================================================"
if [ -d "$SFT_RUN" ]; then
    echo ">>> [warn] $SFT_RUN exists; HF Trainer will resume. rm -rf to start fresh."
    sleep 5
fi
python train/train_sft_only.py \
    --model_path "$BASE_MODEL" \
    --template "$TEMPLATE" \
    --train_parquet "$DATA_GRPO/train.parquet" \
    --valid_parquet "$DATA_GRPO/valid.parquet" \
    --output_dir "$SFT_RUN" \
    --max_train_groups 5000 --max_eval_groups 1000 \
    --num_checkpoints 5 --logging_steps 25 \
    --per_device_batch_size 24 --grad_accum 1 \
    --lr 5e-5 \
    --merge_and_save \
    2>&1 | tee "$SFT_RUN.log"

T_AFTER_SFT_TRAIN=$(date +%s)

# ============================================================
# STEP 2 — engagement-aware eval of Stage 1 (SFT-only ablation arm)
# ============================================================
echo ""
echo "============================================================"
echo ">>> STEP 2/4: engagement eval of Stage 1 (n=$EVAL_N)"
echo ">>> answers: 'does pure SFT push chosen ↑ AND rejected ↓,"
echo ">>>          or does it lift both?'"
echo "============================================================"
if [ ! -d "$SFT_RUN/merged" ]; then
    echo ">>> [error] $SFT_RUN/merged missing — Stage 1 train likely failed"
    exit 1
fi
python train/evaluate_engaged.py \
    --model_path "$SFT_RUN/merged" \
    --valid_parquet "$DATA_V1/valid.parquet" \
    --template "$TEMPLATE" \
    --n $EVAL_N --num_beams 32 --topk 96 \
    --output_csv "runs/eval_engaged_sft_only_n${EVAL_N}.csv" \
    2>&1 | tee "$EVAL_DIR/sft_only_n${EVAL_N}.log"

T_AFTER_SFT_EVAL=$(date +%s)

# ============================================================
# STEP 3 — Stage 2: DPO from SFT checkpoint, 5k groups
# ============================================================
echo ""
echo "============================================================"
echo ">>> STEP 3/4: Stage 2 DPO from SFT (5000 groups, ~3h)"
echo ">>> base = ref = $SFT_RUN/merged"
echo ">>> out  : $DPO_RUN"
echo ">>> loss : L_dpo_grpo only  (no SFT term, no KL)"
echo "============================================================"
if [ -d "$DPO_RUN" ]; then
    echo ">>> [warn] $DPO_RUN exists; HF Trainer will resume. rm -rf to start fresh."
    sleep 5
fi
python train/train_dpo_from_sft.py \
    --model_path "$SFT_RUN/merged" \
    --ref_model_path "$SFT_RUN/merged" \
    --template "$TEMPLATE" \
    --train_parquet "$DATA_GRPO/train.parquet" \
    --valid_parquet "$DATA_GRPO/valid.parquet" \
    --output_dir "$DPO_RUN" \
    --max_train_groups 5000 --max_eval_groups 1000 \
    --num_checkpoints 5 --logging_steps 25 \
    --per_device_batch_size 24 --grad_accum 1 \
    --lr 5e-5 \
    --dpo_beta 0.1 \
    --group_norm_eps 0.05 --group_norm_warmup 50 \
    --best_metric chosen_score \
    --merge_and_save \
    2>&1 | tee "$DPO_RUN.log"

T_AFTER_DPO_TRAIN=$(date +%s)

# ============================================================
# STEP 4 — engagement-aware eval of Stage 2 (final SFT → DPO)
# ============================================================
echo ""
echo "============================================================"
echo ">>> STEP 4/4: engagement eval of Stage 2 (n=$EVAL_N)"
echo ">>> the headline number for the sequential ablation"
echo "============================================================"
if [ ! -d "$DPO_RUN/merged" ]; then
    echo ">>> [error] $DPO_RUN/merged missing — Stage 2 train likely failed"
    exit 1
fi
python train/evaluate_engaged.py \
    --model_path "$DPO_RUN/merged" \
    --valid_parquet "$DATA_V1/valid.parquet" \
    --template "$TEMPLATE" \
    --n $EVAL_N --num_beams 32 --topk 96 \
    --output_csv "runs/eval_engaged_dpo_from_sft_n${EVAL_N}.csv" \
    2>&1 | tee "$EVAL_DIR/dpo_from_sft_n${EVAL_N}.log"

END=$(date +%s)

# ============================================================
# Summary
# ============================================================
fmt_min() { echo "$(( ($2 - $1) / 60 )) min"; }

echo ""
echo "############################################################"
echo "# SEQUENTIAL SMOKE COMPLETE"
echo "############################################################"
if [ -z "$SKIP_STEP0" ]; then
    echo "  STEP 0 joint re-eval     : $(fmt_min $START $T_AFTER_STEP0)"
fi
echo "  STEP 1 SFT train         : $(fmt_min $T_AFTER_STEP0 $T_AFTER_SFT_TRAIN)"
echo "  STEP 2 SFT eval          : $(fmt_min $T_AFTER_SFT_TRAIN $T_AFTER_SFT_EVAL)"
echo "  STEP 3 DPO-from-SFT train: $(fmt_min $T_AFTER_SFT_EVAL $T_AFTER_DPO_TRAIN)"
echo "  STEP 4 DPO eval          : $(fmt_min $T_AFTER_DPO_TRAIN $END)"
echo "  TOTAL                    : $(fmt_min $START $END)"
echo ""
echo "Headline CSVs:"
[ -z "$SKIP_STEP0" ] && echo "  joint  (re-eval n=$EVAL_N): runs/eval_engaged_joint_n${EVAL_N}.csv"
echo "  SFT-only        (n=$EVAL_N): runs/eval_engaged_sft_only_n${EVAL_N}.csv"
echo "  SFT → DPO       (n=$EVAL_N): runs/eval_engaged_dpo_from_sft_n${EVAL_N}.csv"
echo ""

# Quick side-by-side recall@96 / pass@96 grep across all four eval logs.
echo "============================================================"
echo "Engagement-aware metrics (grep recall@96 / pass@96)"
echo "============================================================"
for tag in joint_reval sft_only dpo_from_sft; do
    log="$EVAL_DIR/${tag}_n${EVAL_N}.log"
    if [ -f "$log" ]; then
        echo "--- $tag ---"
        grep -E "^recall@96|^pass@96" "$log" | tail -n 2 | sed 's/^/  /'
    fi
done