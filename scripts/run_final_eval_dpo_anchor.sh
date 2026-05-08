#!/usr/bin/env bash
#
# run_final_eval_dpo_anchor.sh
#
# Server A launcher: full engagement-aware eval on
# DPO+anchor 50k (= SFT-50k → DPO with light SFT anchor, sft_weight=0.15 raw).
# Reads runs/dpo_anchor_from_sft_50k/merged.
#
# Validation only this round — v1_test (38.8k rows, ~5-6 h Recall@32) is
# skipped. Run it later if needed; the headline ablation table only needs
# engagement-aware Δrecall/Δpass on full v1 valid (~14k rows, ~2 h).
#
# Wall clock estimate (batch=4 unless overridden):
#   STEP 1 engaged full v1 valid (~14k rows): ~2 h
#
# If batch=8 fits VRAM, set BATCH_SIZE=8 — drops to ~1 h.
#
# Outputs:
#   evaluation_results/eval_engaged_dpo_anchor_full.csv
#   evaluation_results/eval_engaged_dpo_anchor_full.json
#   evaluation_results/final_eval_summary_dpo_anchor.json   ← headline JSON
#                                                             (matches name
#                                                              referenced in
#                                                              run_final_eval_baseline_sft.sh)
#
# Usage:
#     bash scripts/run_final_eval_dpo_anchor.sh
#     BATCH_SIZE=8 bash scripts/run_final_eval_dpo_anchor.sh

set -e
set -o pipefail

# ---- Environment ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# ---- Paths ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_V1="data/contrastive_dataset_v1"
TEMPLATE="model/qwen3_soft_switch.jinja2"

# Trained checkpoint produced by scripts/run_dpo_anchor_from_sft_50k.sh
RUN="runs/dpo_anchor_from_sft_50k"
MODEL="$RUN/merged"

EVAL_OUT="evaluation_results"
EVAL_LOGS="evaluation_results/logs"
mkdir -p "$EVAL_OUT" "$EVAL_LOGS"

# ---- Tunable knobs ----
BATCH_SIZE="${BATCH_SIZE:-4}"

START=$(date +%s)

# ---- Sanity ----
if [ ! -f "$DATA_V1/valid.parquet" ]; then
    echo ">>> [error] v1 valid not found at $DATA_V1/valid.parquet"; exit 1
fi
if [ ! -d "$MODEL" ]; then
    echo ">>> [error] DPO+anchor 50k merged checkpoint missing at $MODEL"
    echo "    Training likely still in flight, or run scripts/run_dpo_anchor_from_sft_50k.sh first."
    exit 1
fi

# ============================================================
# STEP 1 — DPO+anchor 50k × engagement-aware on full v1 valid (~14k)
# ============================================================
echo ""
echo "============================================================"
echo ">>> DPO+anchor 50k × engagement-aware on full v1 valid"
echo ">>> model: $MODEL"
echo ">>> rows: ~14,000 (full)  batch=$BATCH_SIZE"
echo "============================================================"

CSV="$EVAL_OUT/eval_engaged_dpo_anchor_full.csv"
JSON="$EVAL_OUT/eval_engaged_dpo_anchor_full.json"
LOG="$EVAL_LOGS/eval_engaged_dpo_anchor_full.log"

if [ -f "$CSV" ] && [ -f "$JSON" ]; then
    echo ">>> [skip] $CSV + $JSON already exist — rm to redo"
else
    python train/evaluate_engaged.py \
        --model_path "$MODEL" \
        --valid_parquet "$DATA_V1/valid.parquet" \
        --template "$TEMPLATE" \
        --n -1 --num_beams 32 --topk 96 \
        --batch_size $BATCH_SIZE \
        --output_csv "$CSV" \
        --summary_json "$JSON" \
        2>&1 | tee "$LOG"
fi

END=$(date +%s)
fmt_min() { echo "$(( ($2 - $1) / 60 )) min"; }

echo ""
echo "############################################################"
echo "# DPO+ANCHOR FINAL EVAL COMPLETE  (total $(fmt_min $START $END))"
echo "############################################################"
echo ""
echo "Outputs:"
echo "  $CSV"
echo "  $JSON"
echo ""

# ============================================================
# Headline JSON — use the per-eval summary as the canonical
# final_eval_summary_dpo_anchor.json (single-arm: engaged only).
# ============================================================
COMBINED_JSON="$EVAL_OUT/final_eval_summary_dpo_anchor.json"

python - <<EOF
import json
from pathlib import Path

eng_path = Path("$JSON")
combined = {
    "dpo_anchor_50k": {
        "engaged": json.loads(eng_path.read_text()) if eng_path.exists() else None,
        "origin":  None,    # v1_test skipped this round
    },
}

out = Path("$COMBINED_JSON")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(combined, f, indent=2)
print(f"saved combined summary → {out}")
print()

eng = combined["dpo_anchor_50k"]["engaged"]
if eng:
    print(f"  dpo_anchor_50k:")
    print(
        f"    engaged (n={eng['n']}): "
        f"chosen={eng['recall_chosen']:.4f}  "
        f"rejected={eng['recall_rejected']:.4f}  "
        f"Δrecall={eng['delta_recall']:+.4f}  "
        f"Δpass={eng['delta_pass']:+.4f}"
    )
    print(f"    origin: (skipped this round — v1_test 38.8k too slow)")
EOF

echo ""
echo ">>> When server B (ORPO 50k) and server C (baseline + SFT-50k) finish,"
echo ">>> combine all three summary JSONs in the notebook to build the final"
echo ">>> ablation table:"
echo "    $EVAL_OUT/final_eval_summary_baseline_sft.json   (server C)"
echo "    $EVAL_OUT/final_eval_summary_orpo.json           (server B)"
echo "    $COMBINED_JSON                                   (this script)"
