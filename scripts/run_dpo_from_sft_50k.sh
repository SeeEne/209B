#!/usr/bin/env bash
#
# run_dpo_from_sft_50k.sh
#
# Stage 2 of the sequential SFT → DPO ablation, scaled to 50k.
#
# Starting point: runs/sft_only_50k/merged (chosen_score = -4.795 at n=1000;
# Δrecall flipped to +0.0007). DPO Stage 2 task: keep that chosen-recall
# gain while pushing rejected_recall further down via the contrastive
# margin. Ref model = SFT-50k itself (standard SFT→DPO post-training).
#
# Loss (no SFT term, no KL):
#     margin = β × [(c_θ - c_ref) - (r_θ - r_ref)]
#     L_pair = softplus(-margin)
#     L_dpo  = (L_pair / std_g.detach()).mean()    (warmup steps: plain mean)
#
# Configuration (mirrors the SFT-50k locked config + DPO knobs from the
# 5k DPO smoke):
#     lr         = 5e-5              (parity with SFT-50k)
#     dpo_beta   = 0.1               (DeepSeek/Llama-3 standard)
#     batch      = 24, grad_accum=1  (parity with SFT-50k)
#     lora_r=16 / alpha=32           (parity with SFT-50k)
#     epochs     = 1                 (parity)
#     group_norm_eps     = 0.05      (cap 1/std at 20×)
#     group_norm_warmup  = 50        (cold-start trained=ref → std≈0)
#     best_metric        = chosen_score   ← critical: protects against the
#                                          GRPO-only failure mode where
#                                          chosen drops while rejected
#                                          drops faster (pref_acc rises
#                                          but absolute chosen recall
#                                          collapses, as observed in the
#                                          5k SFT→DPO smoke run).
#
# Wall clock estimate (RTX 6000 Pro / H100):
#     ref precompute (cold cache, ref=SFT-50k, 50k train + 1k valid):
#         ~5h    (3125 batches × 5.7s + 63 batches × 5.7s)
#     train (6250 steps × ~18.8s/step including 5 evals):
#         ~33h
#     final eval n=1000:
#         ~6 min
#     TOTAL: ~38–40h  (~1.6 days continuous)
#
# Why the cache is necessarily cold:
#     The ref-cache key hashes ref_model_path. Stage 2's ref is the freshly
#     produced runs/sft_only_50k/merged, never seen before → cache miss
#     guaranteed. Once computed, cache hits on any sweep over Stage 2 knobs
#     (β / lr / norm settings) keep this run-cost amortized.
#
# Output:
#     runs/dpo_from_sft_50k/
#         ├── checkpoint-1250 / ... / checkpoint-6250
#         ├── adapter/    (LoRA at best ckpt by eval_chosen_score)
#         └── merged/
#     runs/dpo_from_sft_50k.log
#     runs/eval_engaged_dpo_from_sft_50k_n1000.csv
#     data/contrastive_dataset_v1_grpo/_ref_cache/ref_train_n50000_<hash>.pt
#     data/contrastive_dataset_v1_grpo/_ref_cache/ref_valid_n1000_<hash>.pt

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
DATA_V1="data/contrastive_dataset_v1"
TEMPLATE="model/qwen3_soft_switch.jinja2"

SFT_RUN="runs/sft_only_50k"          # Stage 1 — must exist (its merged/)
RUN="runs/dpo_from_sft_50k"
TRAIN_LOG="${RUN}.log"
EVAL_DIR="runs/eval_logs"
EVAL_LOG="$EVAL_DIR/dpo_from_sft_50k_n1000.log"
EVAL_CSV="runs/eval_engaged_dpo_from_sft_50k_n1000.csv"

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
if [ ! -d "$SFT_RUN/merged" ]; then
    echo ">>> [error] Stage 1 SFT-50k checkpoint missing at $SFT_RUN/merged"
    echo "    Run scripts/run_sft_50k.sh first."
    exit 1
fi

# ============================================================
# STEP 1/2 — Train DPO from SFT-50k (~38h)
# ============================================================
if [ -d "$RUN/merged" ]; then
    echo ">>> [skip] $RUN/merged already exists; skipping training."
    echo "    rm -rf $RUN to retrain from scratch."
else
    echo ""
    echo "============================================================"
    echo ">>> STEP 1/2: DPO-from-SFT train  (50,000 groups, ~33h + ~5h ref precompute)"
    echo ">>> base = ref = $SFT_RUN/merged   (standard SFT→DPO setup)"
    echo ">>> lr=5e-5, dpo_beta=0.1, lora_r=16/alpha=32, batch=24, 1 epoch"
    echo ">>> group_norm_eps=0.05, group_norm_warmup=50  (cold-start guard)"
    echo ">>> best_metric=chosen_score  (protects against chosen-collapse failure)"
    echo ">>> out: $RUN"
    echo ">>> log: $TRAIN_LOG"
    echo "============================================================"

    if [ -d "$RUN" ]; then
        echo ">>> [warn] $RUN exists without merged/. HF Trainer will resume."
        echo "    rm -rf $RUN for a fresh start. Continuing in 5s..."
        sleep 5
    fi

    python train/train_dpo_from_sft.py \
        --model_path "$SFT_RUN/merged" \
        --ref_model_path "$SFT_RUN/merged" \
        --template "$TEMPLATE" \
        --train_parquet "$DATA_GRPO/train.parquet" \
        --valid_parquet "$DATA_GRPO/valid.parquet" \
        --output_dir "$RUN" \
        --max_train_groups 50000 --max_eval_groups 1000 \
        --num_checkpoints 5 --logging_steps 50 \
        --per_device_batch_size 24 --grad_accum 1 \
        --lr 5e-5 \
        --dpo_beta 0.1 \
        --group_norm_eps 0.05 --group_norm_warmup 50 \
        --best_metric chosen_score \
        --lora_r 16 --lora_alpha 32 \
        --merge_and_save \
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
# Summary
# ============================================================
echo ""
echo "############################################################"
echo "# DPO-FROM-SFT-50K COMPLETE  (train ${TRAIN_MIN} min + eval ${EVAL_MIN} min = ${TOTAL_MIN} min)"
echo "############################################################"

# Train-time eval trend
echo ""
echo "============================================================"
echo "Train-time eval metrics trend (50k DPO from SFT-50k)"
echo "  Reference points (SFT-50k baseline before DPO Stage 2):"
echo "    chosen_score (ref) = -4.795   ← what we want to NOT undo"
echo "    chosen_recall@96   = 0.0117   ← what we want to keep or improve"
echo "    rejected_recall@96 = 0.0110   ← what DPO is supposed to push down"
echo "    Δrecall            = +0.0007"
echo "============================================================"

if [ -f "$TRAIN_LOG" ]; then
    extract_metric() {
        local key="$1"
        sed -nE "s/.*[\"']${key}[\"']: [\"']?([-+0-9.eE]+)[\"']?.*/\1/p" "$TRAIN_LOG"
    }
    mapfile -t SCORES < <(extract_metric eval_chosen_score)
    mapfile -t REJ < <(extract_metric eval_rejected_score)
    mapfile -t PREF < <(extract_metric eval_pref_acc)
    mapfile -t MARGIN < <(extract_metric eval_margin)
    LABELS=("ckpt 1 (step 1250)" "ckpt 2 (step 2500)" "ckpt 3 (step 3750)" \
            "ckpt 4 (step 5000)" "ckpt 5 (step 6250)" "final (best ckpt)")
    for ((i=0; i<${#SCORES[@]} && i<6; i++)); do
        printf "    %-22s chosen=%s  rejected=%s  margin=%s  pref_acc=%s\n" \
            "${LABELS[$i]}:" "${SCORES[$i]}" "${REJ[$i]:-n/a}" \
            "${MARGIN[$i]:-n/a}" "${PREF[$i]:-n/a}"
    done
fi

# Headline engagement-aware metrics
echo ""
echo "============================================================"
echo "Engagement-aware Recall@K  (n=$EVAL_N, on v1 valid)"
echo "  Baseline (no FT)   : chosen=0.0093  rejected=0.0163  Δ=-0.0070"
echo "  SFT-50k (Stage 1)  : chosen=0.0117  rejected=0.0110  Δ=+0.0007"
echo "  → Target: keep chosen ≥ 0.0117 AND push rejected below 0.0110."
echo "============================================================"
if [ -f "$EVAL_LOG" ]; then
    grep -E "^recall@96|^pass@96" "$EVAL_LOG" | tail -n 2 | sed 's/^/  /'
fi

echo ""
echo "============================================================"
echo "Decision tree for next step"
echo "============================================================"
echo "If chosen ≥ 0.0117 AND rejected < 0.0110 (Δ more positive than +0.0007):"
echo "  → Sequential SFT→DPO works at scale. Project plan validated."
echo "  → For the report: this is the headline result."
echo ""
echo "If chosen drops below 0.0117 (any amount):"
echo "  → DPO partially undid the SFT chosen gain. Consider:"
echo "    (a) lower dpo_beta to 0.05 — softer preference push"
echo "    (b) shorter Stage 2 — best ckpt likely earlier than step 6250"
echo "    (c) re-introduce SFT anchor (joint trainer with sft_weight 0.3)"
echo ""
echo "If chosen and rejected both drop heavily (5k-style collapse):"
echo "  → Same failure mode as 5k SFT→DPO. SFT-50k's lift was apparently"
echo "    not robust enough for DPO to honor. Reconsider ORPO 50k as the"
echo "    single-stage alternative — it can't have this failure mode."
echo ""
echo "For richer trend across the 5 ckpts (~1h):"
echo "  python diagnose/sft_score_trend.py \\"
echo "    --base $SFT_RUN/merged --include_base \\"
echo "    --adapters $RUN/checkpoint-1250 $RUN/checkpoint-2500 \\"
echo "               $RUN/checkpoint-3750 $RUN/checkpoint-5000 \\"
echo "               $RUN/checkpoint-6250 \\"
echo "    --template $TEMPLATE --n 1000 --batch_size 16 \\"
echo "    --output_csv runs/dpo_from_sft_50k_trend.csv"
