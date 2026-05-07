#!/usr/bin/env bash
#
# run_dpo_anchor_from_sft_50k.sh
#
# Stage 2 alternative: DPO + light SFT anchor, starting from SFT-50k.
#
# Counterpart to run_dpo_from_sft_50k.sh (pure DPO). Same starting point
# (runs/sft_only_50k/merged), same hyperparams, ONLY difference is the
# loss adds a low-weight SFT anchor term to keep chosen recall from
# collapsing — the failure mode observed in the 5k SFT→DPO smoke
# (chosen 0.0097 → 0.0043, both sides crashed).
#
# Loss:
#     L_total = L_dpo_grpo  +  sft_weight × L_sft_raw
#             = ~5–7        +  0.3 × ~4.8
#             = DPO ~80%    +  SFT anchor ~20%   (raw mode, predictable)
#
# Why raw mode (not match_dpo) at sft_weight=0.3:
#     match_dpo would multiply by inv_std_mean — but inv_std under
#     "DPO-from-SFT" is unknown (5k smoke saw 3.5–8.0; 5k joint saw ~16).
#     raw mode makes sft_weight literal, so the SFT/DPO ratio is
#     predictable across this run and any sweep.
#
# Why this exists in addition to the pure-DPO arm:
#     The 50k SFT starting point (chosen log P = -4.795, +0.125 nats vs
#     base) is materially different from the 5k starting point (chosen
#     ≈ -4.84, ≈ base). Theory says pure DPO should not collapse this
#     time, but the 5k crash makes a soft anchor a defensible safety
#     measure. Running both arms gives an apples-to-apples comparison
#     of "with anchor vs without" at the 50k scale.
#
# Configuration (all parity with run_dpo_from_sft_50k.sh except SFT anchor):
#     model_path / ref_model_path = runs/sft_only_50k/merged
#     lr=5e-5, dpo_beta=0.1
#     sft_weight=0.3, sft_scale_mode=raw   ← THE difference
#     kl_weight=0  (DPO has implicit KL via ref baseline)
#     group_norm_eps=0.05, group_norm_warmup=50
#     batch=24, grad_accum=1, lora_r=16/alpha=32
#     epochs=1, num_checkpoints=5
#     best_metric=chosen_score  (backup defense even with anchor on)
#
# Wall clock estimate (matches pure DPO arm — anchor adds ~0 cost):
#     ref precompute (cold cache, ref=SFT-50k):    ~5h
#     train (6250 steps × ~18.8s/step):            ~33h
#     final eval n=1000:                            ~6 min
#     TOTAL: ~38–40h
#
# Cache sharing with run_dpo_from_sft_50k.sh:
#     The two trainers use the same _ref_cache_path() hash logic. If you
#     run pure-DPO first, this script's ref precompute is a cache HIT
#     (1 sec) instead of a cold MISS (5h) — but since neither has been
#     run yet, the first launch will pay the 5h cost, the second won't.
#
# Output:
#     runs/dpo_anchor_from_sft_50k/{checkpoint-*, adapter, merged}
#     runs/dpo_anchor_from_sft_50k.log
#     runs/eval_engaged_dpo_anchor_from_sft_50k_n1000.csv

set -e
set -o pipefail

# ---- Environment ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
# Force unbuffered Python stdout/stderr so `tee` sees output line-by-line
# instead of in big chunks. Critical for monitoring a 38h run — otherwise
# the latest [mem] / loss line might lag minutes behind reality, and a
# remote `tail -f $TRAIN_LOG` looks frozen between flushes.
export PYTHONUNBUFFERED=1

# ---- Paths ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_GRPO="data/contrastive_dataset_v1_grpo"
DATA_V1="data/contrastive_dataset_v1"
TEMPLATE="model/qwen3_soft_switch.jinja2"

SFT_RUN="runs/sft_only_50k"          # Stage 1 — must exist
RUN="runs/dpo_anchor_from_sft_50k"
TRAIN_LOG="${RUN}.log"
EVAL_DIR="runs/eval_logs"
EVAL_LOG="$EVAL_DIR/dpo_anchor_from_sft_50k_n1000.log"
EVAL_CSV="runs/eval_engaged_dpo_anchor_from_sft_50k_n1000.csv"

mkdir -p "$EVAL_DIR"

EVAL_N=1000

# ---- SFT anchor knobs (the only thing different from pure DPO arm) ----
# At lr=5e-5 + raw mode, the SFT term contribution to total loss is roughly:
#     ratio = sft_weight × l_sft / l_dpo  ≈  sft_weight × 4.8 / 3.0
# So this knob maps directly to "% of loss attributable to SFT anchor":
#     0.10 → ~16%   (gentle anchor, barely interferes with DPO)
#     0.15 → ~24%   ← current — light protection without dominating
#     0.30 → ~48%   (medium — DPO and SFT roughly equal)
#     0.50 → ~67%   (strong — SFT dominates DPO)
#     1.00 → ~62-80% (heavy — risk of 5k joint failure mode)
# If chosen still drops at 0.15, raise to 0.30 next iteration before going
# higher. Don't jump straight to 1.0 — that recreates the 5k joint smoke's
# 87% SFT-dominance regime that produced eval_pref_acc ≈ 0.546 (random).
SFT_WEIGHT=0.15
SFT_SCALE_MODE=raw

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
# STEP 1/2 — Train DPO+anchor from SFT-50k (~38h)
# ============================================================
if [ -d "$RUN/merged" ]; then
    echo ">>> [skip] $RUN/merged already exists; skipping training."
    echo "    rm -rf $RUN to retrain from scratch."
else
    echo ""
    echo "============================================================"
    echo ">>> STEP 1/2: DPO + light SFT anchor train  (50k groups, ~33h + ~5h ref precompute)"
    echo ">>> base = ref = $SFT_RUN/merged"
    echo ">>> loss = L_dpo_grpo + ${SFT_WEIGHT} × L_sft_raw    (mode=${SFT_SCALE_MODE})"
    echo ">>> dpo_beta=0.1, lora_r=16/alpha=32, batch=24, 1 epoch"
    echo ">>> group_norm_eps=0.05, group_norm_warmup=50"
    echo ">>> best_metric=chosen_score   (backup defense)"
    echo ">>> out: $RUN"
    echo ">>> log: $TRAIN_LOG"
    echo "============================================================"

    if [ -d "$RUN" ]; then
        echo ">>> [warn] $RUN exists without merged/. HF Trainer will resume."
        echo "    rm -rf $RUN for a fresh start. Continuing in 5s..."
        sleep 5
    fi

    # Joint trainer is the right tool here — train_dpo_from_sft.py has no
    # SFT anchor knob (it's a pure-DPO trainer by design). The joint
    # trainer's flags subsume DPO-from-SFT when ref_model_path is passed
    # and sft_weight > 0.
    python train/train_contrastive_dpo_g_normalize.py \
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
        --sft_weight $SFT_WEIGHT \
        --sft_scale_mode $SFT_SCALE_MODE \
        --kl_weight 0 \
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
echo "# DPO+ANCHOR FROM SFT-50K COMPLETE  (train ${TRAIN_MIN} min + eval ${EVAL_MIN} min = ${TOTAL_MIN} min)"
echo "############################################################"

# Train-time eval trend across the 5 ckpts
echo ""
echo "============================================================"
echo "Train-time eval trend (50k DPO + anchor from SFT-50k)"
echo "  Reference points (SFT-50k Stage 1 starting state):"
echo "    chosen_score (start)   = -4.795"
echo "    chosen_recall@96 n=1000 = 0.0117  ← what anchor protects"
echo "    rejected_recall n=1000  = 0.0110  ← what DPO should push down"
echo "    Δrecall n=1000          = +0.0007"
echo "============================================================"

if [ -f "$TRAIN_LOG" ]; then
    extract_metric() {
        local key="$1"
        sed -nE "s/.*[\"']${key}[\"']: [\"']?([-+0-9.eE]+)[\"']?.*/\1/p" "$TRAIN_LOG"
    }
    mapfile -t SCORES < <(extract_metric eval_chosen_score)
    mapfile -t REJ < <(extract_metric eval_rejected_score)
    mapfile -t MARGIN < <(extract_metric eval_margin)
    mapfile -t PREF < <(extract_metric eval_pref_acc)
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
echo "  Baseline (no FT)         : chosen=0.0093  rejected=0.0163  Δ=-0.0070"
echo "  SFT-50k (Stage 1 only)   : chosen=0.0117  rejected=0.0110  Δ=+0.0007"
echo "  → Target: keep chosen ≥ 0.0117 AND push rejected < 0.0110"
echo "============================================================"
if [ -f "$EVAL_LOG" ]; then
    grep -E "^recall@96|^pass@96" "$EVAL_LOG" | tail -n 2 | sed 's/^/  /'
fi

echo ""
echo "============================================================"
echo "Decision tree for next step"
echo "============================================================"
echo "If chosen ≥ 0.0117 AND rejected < 0.0110 (Δ more positive):"
echo "  → SFT anchor preserved chosen + DPO pushed rejected → SUCCESS."
echo "  → Run n=5000 confirmation (~2h):"
echo "      python train/evaluate_engaged.py \\"
echo "        --model_path $RUN/merged \\"
echo "        --valid_parquet $DATA_V1/valid.parquet \\"
echo "        --template $TEMPLATE \\"
echo "        --n 5000 --num_beams 32 --topk 96 \\"
echo "        --output_csv runs/eval_engaged_dpo_anchor_from_sft_50k_n5000.csv"
echo ""
echo "If chosen ≈ 0.0117 AND rejected ≈ 0.0110 (essentially unchanged):"
echo "  → Anchor too strong / DPO no headroom; results match SFT-50k alone."
echo "  → Try sft_weight=0.1 (lower) or run pure-DPO arm for comparison."
echo ""
echo "If chosen drops slightly (e.g. 0.0080–0.0117) but rejected drops more:"
echo "  → Anchor partially holding; DPO partially working. Net win if Δ"
echo "    larger than +0.0007. Likely best ckpt is mid-training; check trend."
echo ""
echo "If chosen collapses (<0.005) and rejected collapses harder:"
echo "  → Same failure mode as 5k SFT→DPO. Anchor=0.3 wasn't enough."
echo "  → Either (a) increase sft_weight to 1.0 raw, or (b) abandon DPO Stage 2"
echo "    and accept SFT-50k as the project headline."
echo ""
echo "For richer trend across the 5 ckpts (~1h):"
echo "  python diagnose/sft_score_trend.py \\"
echo "    --base $SFT_RUN/merged --include_base \\"
echo "    --adapters $RUN/checkpoint-1250 $RUN/checkpoint-2500 \\"
echo "               $RUN/checkpoint-3750 $RUN/checkpoint-5000 \\"
echo "               $RUN/checkpoint-6250 \\"
echo "    --template $TEMPLATE --n 1000 --batch_size 16 \\"
echo "    --output_csv runs/dpo_anchor_from_sft_50k_trend.csv"
