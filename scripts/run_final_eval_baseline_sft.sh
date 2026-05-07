#!/usr/bin/env bash
#
# run_final_eval_baseline_sft.sh
#
# Server C launcher: full v1 valid + full v1_test (Recall@32) on
# (a) baseline OneRec-1.7B and (b) SFT-50k merged.
#
# These two arms have models on disk already, so they can run in
# PARALLEL with the in-flight Step 13 training (ORPO 50k on server B,
# DPO+anchor 50k on server A). When all three servers finish, we
# combine their outputs into the final ablation table.
#
# Server A and B will separately handle their own arms after their
# training completes:
#   server A: SFT-50k → DPO+anchor 50k (Step 13B)
#   server B: ORPO 50k                  (Step 13A)
#   server C: baseline + SFT-50k        (this script)
#
# Pre-flight sanity check:
#   STEP 0 — runs n=100 with batch=1 then batch=N on the baseline,
#            confirms the two evaluators agree byte-identically (modulo
#            BF16 fp accumulation) before committing to the full run.
#            Total ~3 min. Skip via SKIP_SANITY=1 env var.
#
# Wall clock estimate (batch=4 unless overridden):
#   STEP 0 sanity        : ~3 min
#   STEP 1 baseline engaged full v1 valid (~14k rows): ~2 h
#   STEP 2 baseline origin full v1_test (~38.8k rows): ~5.5 h
#   STEP 3 SFT-50k engaged full v1 valid:              ~2 h
#   STEP 4 SFT-50k origin full v1_test:                ~5.5 h
#   TOTAL: ~15 h on a single 80GB GPU
#
# If batch=8 fits VRAM, set BATCH_SIZE=8 — total drops to ~8 h.
#
# Usage:
#     bash scripts/run_final_eval_baseline_sft.sh
#     BATCH_SIZE=8 bash scripts/run_final_eval_baseline_sft.sh
#     SKIP_SANITY=1 bash scripts/run_final_eval_baseline_sft.sh

set -e
set -o pipefail

# ---- Environment ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
# Stream Python output line-by-line through `tee` (otherwise stdout is
# block-buffered when piped, hiding progress for many seconds).
export PYTHONUNBUFFERED=1

# ---- Paths ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_V1="data/contrastive_dataset_v1"
DATA_BENCH="data/OpenOneRec/benchmark_data/video/video_test.parquet"
TEMPLATE="model/qwen3_soft_switch.jinja2"
BASE_MODEL="model/OneRec-1.7B"

# Final SFT-50k checkpoint promoted to model/ (treated as a canonical
# model alongside the base, not a transient experiment output).
SFT_MODEL="model/OneRec-1.7B-sft50k"

EVAL_OUT="evaluation_results"           # CSV + summary JSON land here
EVAL_LOGS="evaluation_results/logs"     # stdout transcripts
mkdir -p "$EVAL_OUT" "$EVAL_LOGS"

# ---- Tunable knobs (override via env var) ----
# BATCH_SIZE=8 bash scripts/run_final_eval_baseline_sft.sh   ← 2× speedup
# SKIP_SANITY=1 ...                                          ← skip pre-flight
BATCH_SIZE="${BATCH_SIZE:-4}"
SKIP_SANITY="${SKIP_SANITY:-0}"

START=$(date +%s)

# ---- Sanity (pre-flight) ----
if [ ! -f "$DATA_V1/valid.parquet" ]; then
    echo ">>> [error] v1 valid not found at $DATA_V1/valid.parquet"; exit 1
fi
if [ ! -f "$DATA_BENCH" ]; then
    echo ">>> [error] benchmark not found at $DATA_BENCH"; exit 1
fi
if [ ! -d "$BASE_MODEL" ]; then
    echo ">>> [error] base model not found at $BASE_MODEL"; exit 1
fi
if [ ! -d "$SFT_MODEL" ]; then
    echo ">>> [error] SFT-50k model not found at $SFT_MODEL"
    echo "    Expected location for the canonical SFT-50k checkpoint."
    echo "    If still in runs/, copy/rename: mv runs/sft_only_50k/merged $SFT_MODEL"
    exit 1
fi

# ============================================================
# STEP 0 — sanity: batch=1 vs batch=N on baseline @ n=100 must agree
# ============================================================
if [ "$SKIP_SANITY" = "0" ]; then
    echo ""
    echo "============================================================"
    echo ">>> STEP 0 — sanity check: batch=1 vs batch=$BATCH_SIZE @ n=100"
    echo ">>> any drift > 0.005 in recall_chosen would invalidate the"
    echo ">>> batched runs below. ~3 min total."
    echo "============================================================"

    SAN_B1="$EVAL_LOGS/sanity_baseline_b1.csv"
    SAN_BN="$EVAL_LOGS/sanity_baseline_b${BATCH_SIZE}.csv"

    python train/evaluate_engaged.py \
        --model_path "$BASE_MODEL" \
        --valid_parquet "$DATA_V1/valid.parquet" \
        --template "$TEMPLATE" \
        --n 100 --num_beams 32 --topk 96 \
        --batch_size 1 \
        --output_csv "$SAN_B1" \
        2>&1 | tail -n 10

    python train/evaluate_engaged.py \
        --model_path "$BASE_MODEL" \
        --valid_parquet "$DATA_V1/valid.parquet" \
        --template "$TEMPLATE" \
        --n 100 --num_beams 32 --topk 96 \
        --batch_size $BATCH_SIZE \
        --output_csv "$SAN_BN" \
        2>&1 | tail -n 10

    python - <<EOF
import pandas as pd, sys
b1 = pd.read_csv("$SAN_B1")
bn = pd.read_csv("$SAN_BN")
def summarize(df, label):
    print(f"  {label}: chosen={df['recall_chosen'].mean():.4f}  "
          f"rejected={df['recall_rejected'].mean():.4f}  "
          f"Δ={(df['recall_chosen']-df['recall_rejected']).mean():+.4f}")
print("Sanity-check summary (n=100):")
summarize(b1, "batch=1     ")
summarize(bn, "batch=$BATCH_SIZE     ")
diff_c = abs(b1['recall_chosen'].mean() - bn['recall_chosen'].mean())
diff_r = abs(b1['recall_rejected'].mean() - bn['recall_rejected'].mean())
print(f"  abs diff: chosen={diff_c:.4f}  rejected={diff_r:.4f}")
if diff_c > 0.005 or diff_r > 0.005:
    print(">>> [error] batched evaluator drift > 0.005 — investigate before "
          "committing to the full run")
    sys.exit(1)
print(">>> [ok] batched evaluator agrees with batch=1 within tolerance")
EOF
    rm -f "$SAN_B1" "$SAN_BN"
else
    echo ">>> SKIP_SANITY=1 — skipping pre-flight check"
fi

T_AFTER_SANITY=$(date +%s)

# ============================================================
# STEP 1 — baseline × engagement-aware on full v1 valid (~14k)
# ============================================================
echo ""
echo "============================================================"
echo ">>> STEP 1/4 — baseline × engagement-aware on full v1 valid"
echo ">>> model: $BASE_MODEL"
echo ">>> rows: ~14,000 (full)  batch=$BATCH_SIZE"
echo "============================================================"

CSV1="$EVAL_OUT/eval_engaged_baseline_full.csv"
JSON1="$EVAL_OUT/eval_engaged_baseline_full.json"
LOG1="$EVAL_LOGS/eval_engaged_baseline_full.log"
if [ -f "$CSV1" ] && [ -f "$JSON1" ]; then
    echo ">>> [skip] $CSV1 + $JSON1 already exist — rm to redo"
else
    python train/evaluate_engaged.py \
        --model_path "$BASE_MODEL" \
        --valid_parquet "$DATA_V1/valid.parquet" \
        --template "$TEMPLATE" \
        --n -1 --num_beams 32 --topk 96 \
        --batch_size $BATCH_SIZE \
        --output_csv "$CSV1" \
        --summary_json "$JSON1" \
        2>&1 | tee "$LOG1"
fi

T_AFTER_S1=$(date +%s)

# ============================================================
# STEP 2 — baseline × OneRec-paper Recall@32 on full v1_test (~38.8k)
# ============================================================
echo ""
echo "============================================================"
echo ">>> STEP 2/4 — baseline × OneRec Recall@32 on full v1_test"
echo ">>> model: $BASE_MODEL"
echo ">>> rows: 38,781 (full)  batch=$BATCH_SIZE"
echo "============================================================"

CSV2="$EVAL_OUT/eval_origin_baseline_full.csv"
JSON2="$EVAL_OUT/eval_origin_baseline_full.json"
LOG2="$EVAL_LOGS/eval_origin_baseline_full.log"
if [ -f "$CSV2" ] && [ -f "$JSON2" ]; then
    echo ">>> [skip] $CSV2 + $JSON2 already exist — rm to redo"
else
    python train/evaluate_origin.py \
        --model_path "$BASE_MODEL" \
        --benchmark "$DATA_BENCH" \
        --template "$TEMPLATE" \
        --n -1 --num_beams 32 \
        --batch_size $BATCH_SIZE \
        --output_csv "$CSV2" \
        --summary_json "$JSON2" \
        2>&1 | tee "$LOG2"
fi

T_AFTER_S2=$(date +%s)

# ============================================================
# STEP 3 — SFT-50k × engagement-aware on full v1 valid
# ============================================================
echo ""
echo "============================================================"
echo ">>> STEP 3/4 — SFT-50k × engagement-aware on full v1 valid"
echo ">>> model: $SFT_MODEL"
echo "============================================================"

CSV3="$EVAL_OUT/eval_engaged_sft50k_full.csv"
JSON3="$EVAL_OUT/eval_engaged_sft50k_full.json"
LOG3="$EVAL_LOGS/eval_engaged_sft50k_full.log"
if [ -f "$CSV3" ] && [ -f "$JSON3" ]; then
    echo ">>> [skip] $CSV3 + $JSON3 already exist — rm to redo"
else
    python train/evaluate_engaged.py \
        --model_path "$SFT_MODEL" \
        --valid_parquet "$DATA_V1/valid.parquet" \
        --template "$TEMPLATE" \
        --n -1 --num_beams 32 --topk 96 \
        --batch_size $BATCH_SIZE \
        --output_csv "$CSV3" \
        --summary_json "$JSON3" \
        2>&1 | tee "$LOG3"
fi

T_AFTER_S3=$(date +%s)

# ============================================================
# STEP 4 — SFT-50k × OneRec Recall@32 on full v1_test
# ============================================================
echo ""
echo "============================================================"
echo ">>> STEP 4/4 — SFT-50k × OneRec Recall@32 on full v1_test"
echo ">>> model: $SFT_MODEL"
echo "============================================================"

CSV4="$EVAL_OUT/eval_origin_sft50k_full.csv"
JSON4="$EVAL_OUT/eval_origin_sft50k_full.json"
LOG4="$EVAL_LOGS/eval_origin_sft50k_full.log"
if [ -f "$CSV4" ] && [ -f "$JSON4" ]; then
    echo ">>> [skip] $CSV4 + $JSON4 already exist — rm to redo"
else
    python train/evaluate_origin.py \
        --model_path "$SFT_MODEL" \
        --benchmark "$DATA_BENCH" \
        --template "$TEMPLATE" \
        --n -1 --num_beams 32 \
        --batch_size $BATCH_SIZE \
        --output_csv "$CSV4" \
        --summary_json "$JSON4" \
        2>&1 | tee "$LOG4"
fi

END=$(date +%s)
fmt_min() { echo "$(( ($2 - $1) / 60 )) min"; }

# ============================================================
# Summary
# ============================================================
echo ""
echo "############################################################"
echo "# SERVER C FINAL EVAL COMPLETE  (total $(fmt_min $START $END))"
echo "############################################################"
echo "  STEP 0 sanity         : $(fmt_min $START $T_AFTER_SANITY)"
echo "  STEP 1 baseline engaged: $(fmt_min $T_AFTER_SANITY $T_AFTER_S1)"
echo "  STEP 2 baseline origin : $(fmt_min $T_AFTER_S1 $T_AFTER_S2)"
echo "  STEP 3 SFT-50k engaged : $(fmt_min $T_AFTER_S2 $T_AFTER_S3)"
echo "  STEP 4 SFT-50k origin  : $(fmt_min $T_AFTER_S3 $END)"
echo ""
echo "Outputs:"
echo "  $CSV1"
echo "  $CSV2"
echo "  $CSV3"
echo "  $CSV4"
echo ""

# ============================================================
# Headline numbers — combine 4 per-arm summary JSONs into one file
# ============================================================
COMBINED_JSON="$EVAL_OUT/final_eval_summary_baseline_sft.json"

echo ""
echo "============================================================"
echo "Combining per-arm summary JSONs → $COMBINED_JSON"
echo "============================================================"

python - <<EOF
import json
from pathlib import Path

arms = {
    "baseline": {
        "engaged": "$JSON1",
        "origin":  "$JSON2",
    },
    "sft_50k": {
        "engaged": "$JSON3",
        "origin":  "$JSON4",
    },
}

combined = {}
for arm_name, paths in arms.items():
    arm_data = {}
    for kind, p in paths.items():
        fp = Path(p)
        if fp.exists():
            with open(fp) as f:
                arm_data[kind] = json.load(f)
        else:
            arm_data[kind] = None  # eval was skipped or failed
    combined[arm_name] = arm_data

out = Path("$COMBINED_JSON")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(combined, f, indent=2)
print(f"saved combined summary → {out}")
print()

# Pretty-print the headline numbers for the launcher's stdout
def fmt(arm, data):
    if data is None:
        return f"  {arm}: (missing)"
    eng = data.get("engaged")
    org = data.get("origin")
    parts = [f"  {arm}:"]
    if eng:
        parts.append(
            f"    engaged (n={eng['n']}): "
            f"chosen={eng['recall_chosen']:.4f}  "
            f"rejected={eng['recall_rejected']:.4f}  "
            f"Δrecall={eng['delta_recall']:+.4f}  "
            f"Δpass={eng['delta_pass']:+.4f}"
        )
    if org:
        k = org["num_beams"]
        parts.append(
            f"    origin  (n={org['n']}): "
            f"recall@{k}={org[f'recall_at_{k}']:.4f}  "
            f"pass@{k}={org[f'pass_at_{k}']:.4f}  "
            f"position1={org['position1_pass']:.4f}"
        )
    return "\n".join(parts)

for arm in ("baseline", "sft_50k"):
    print(fmt(arm, combined[arm]))
EOF

echo ""
echo ">>> Notebook usage:"
echo "    import json"
echo "    with open('$COMBINED_JSON') as f: data = json.load(f)"
echo "    data['baseline']['engaged']['recall_chosen']  # → float"
echo ""
echo ">>> Server A and B will produce parallel summary JSONs:"
echo "    $EVAL_OUT/final_eval_summary_dpo_anchor.json  (server A)"
echo "    $EVAL_OUT/final_eval_summary_orpo.json        (server B)"
echo ">>> Combine all three in the notebook to build the final ablation table."
