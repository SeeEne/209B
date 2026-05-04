#!/usr/bin/env bash
#
# run_all_evals.sh
#
# Run engagement-aware eval (evaluate_engaged.py) on three models in sequence
# and print a summary table at the end.
#
# Models compared:
#   1. baseline       — model/OneRec-1.7B (no FT)
#   2. length=3 30k   — runs/contrastive_v1_30k/merged
#   3. GRPO 5k smoke  — runs/contrastive_grpo_smoke/merged
#
# Each eval ~30 min (n=5000 groups × G=3 = 15k pairs at ~0.4s/pair fwd).
# Total wall clock ~1.5 h.

set -e   # bail on any error

# ---- Offline / threading setup ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

# ---- Paths ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VALID_PARQUET="data/contrastive_dataset_v1/valid.parquet"
TEMPLATE="model/qwen3_soft_switch.jinja2"
LOGS_DIR="runs/eval_logs"
mkdir -p "$LOGS_DIR"

# ---- Common eval args ----
N=5000
NUM_BEAMS=32
TOPK=96

run_eval() {
    local NAME="$1"
    local MODEL_PATH="$2"
    local CSV="runs/eval_engaged_${NAME}.csv"
    local LOG="$LOGS_DIR/${NAME}.log"

    if [ ! -d "$MODEL_PATH" ] && [[ ! "$MODEL_PATH" == *"/"* ]]; then
        echo ">>> [skip] ${NAME}: model path not found: ${MODEL_PATH}"
        return 0
    fi

    echo ""
    echo "============================================================"
    echo ">>> EVAL: ${NAME}"
    echo ">>> model: ${MODEL_PATH}"
    echo ">>> n=${N}, num_beams=${NUM_BEAMS}, topk=${TOPK}"
    echo ">>> log:   ${LOG}"
    echo ">>> csv:   ${CSV}"
    echo "============================================================"

    python train/evaluate_engaged.py \
        --model_path "$MODEL_PATH" \
        --valid_parquet "$VALID_PARQUET" \
        --template "$TEMPLATE" \
        --n "$N" \
        --num_beams "$NUM_BEAMS" \
        --topk "$TOPK" \
        --output_csv "$CSV" \
        2>&1 | tee "$LOG"
}

# ---- Run the three evals ----
START_TIME=$(date +%s)

run_eval "baseline_n5000"  "model/OneRec-1.7B"
run_eval "v1_30k_n5000"    "runs/contrastive_v1_30k/merged"
run_eval "grpo_smoke_n5000" "runs/contrastive_grpo_smoke/merged"

END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))

# ---- Aggregate summary table ----
echo ""
echo "############################################################"
echo "# COMBINED SUMMARY  (total wall clock: ${TOTAL_MIN} min)"
echo "############################################################"
echo ""
printf "%-22s | %12s | %12s | %12s\n" "model" "recall_chosen" "recall_reject" "Δ"
printf "%-22s | %12s | %12s | %12s\n" "----------------------" "------------" "------------" "------------"

extract_metric() {
    # Extract a numeric value after a label from an evaluate_engaged.py log.
    # $1 = log path, $2 = "recall@96 " or "pass@96 " or similar
    grep -E "^${2}" "$1" 2>/dev/null | tail -n 1 | awk '{print $(NF-1), $NF, $(NF-2)}' || echo "n/a"
}

for NAME in baseline_n5000 v1_30k_n5000 grpo_smoke_n5000; do
    LOG="$LOGS_DIR/${NAME}.log"
    if [ ! -f "$LOG" ]; then
        printf "%-22s | %12s | %12s | %12s\n" "$NAME" "missing" "missing" "missing"
        continue
    fi

    # Parse the side-by-side table written by evaluate_engaged.py.
    # Format example:
    #   recall@96               0.0093    0.0163        -0.0070
    #   pass@96                 0.0260    0.0460        -0.0200
    RECALL_LINE=$(grep -E "^recall@${TOPK}" "$LOG" | tail -n 1)
    PASS_LINE=$(grep -E "^pass@${TOPK}" "$LOG" | tail -n 1)

    if [ -z "$RECALL_LINE" ]; then
        printf "%-22s | %12s | %12s | %12s\n" "$NAME" "no-output" "no-output" "no-output"
        continue
    fi

    R_CHOSEN=$(echo "$RECALL_LINE" | awk '{print $2}')
    R_REJECT=$(echo "$RECALL_LINE" | awk '{print $3}')
    R_DELTA=$(echo  "$RECALL_LINE" | awk '{print $4}')
    P_CHOSEN=$(echo "$PASS_LINE"   | awk '{print $2}')
    P_REJECT=$(echo "$PASS_LINE"   | awk '{print $3}')
    P_DELTA=$(echo  "$PASS_LINE"   | awk '{print $4}')

    printf "%-22s | %12s | %12s | %12s   (recall)\n" "$NAME" "$R_CHOSEN" "$R_REJECT" "$R_DELTA"
    printf "%-22s | %12s | %12s | %12s   (pass)\n"   ""      "$P_CHOSEN" "$P_REJECT" "$P_DELTA"
done

echo ""
echo "Done. Per-model logs: $LOGS_DIR/*.log"
echo "Per-model CSVs:       runs/eval_engaged_*.csv"
