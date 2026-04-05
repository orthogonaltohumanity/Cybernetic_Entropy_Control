#!/usr/bin/env bash
# Full MATH (5000 problems): baseline vs QEWS hybrid (w_H=1, w_Q=1), side by side
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

DATASET="testbench/data/hendrycks_math.parquet"
MODEL="2b"
MAX_TOKENS=4096
H_TARGET=0.3
QEWS_TARGET=0.0
RESULTS="/mnt/results/testbench/results"
mkdir -p "$RESULTS"

echo "==========================================================="
echo "  Full MATH A/B — Baseline vs QEWS Hybrid (w_H=1, w_Q=1)"
echo "  Model: ${MODEL}  Dataset: 5000 problems"
echo "==========================================================="

# Baseline — no controller
echo ""
echo "-- Starting: Baseline (no controller) --"
python3 testbench/run.py \
    -m $MODEL -d $DATASET --max-tokens $MAX_TOKENS \
    -o "${RESULTS}/full_math_2b_baseline.jsonl" \
    2>&1 | sed -u 's/^/[BASELINE] /' &
PID_BASE=$!

# QEWS hybrid — entropy + QEWS, equal weight
echo "-- Starting: QEWS hybrid (w_H=1.0, w_Q=1.0) --"
python3 testbench/run.py \
    -m $MODEL -d $DATASET --max-tokens $MAX_TOKENS \
    --control --H-target $H_TARGET \
    --K-M 0.005 0.03 0.04 0.08 \
    --K-P 0.005 0.02 0.03 0.05 \
    --K-F 0.002 0.01 0.015 0.025 \
    --qews-mode hybrid \
    --qews-target $QEWS_TARGET \
    --qews-K-M 0.005 0.03 0.04 0.08 \
    --qews-K-P 0.005 0.02 0.03 0.05 \
    --qews-K-F 0.002 0.01 0.015 0.025 \
    --w-H 1.0 --w-Q 1.0 \
    -o "${RESULTS}/full_math_2b_qews_hybrid.jsonl" \
    2>&1 | sed -u 's/^/[HYBRID ] /' &
PID_HYBRID=$!

echo ""
echo "  Baseline PID:  $PID_BASE"
echo "  Hybrid PID:    $PID_HYBRID"
echo "  Results:       ${RESULTS}/full_math_2b_baseline.jsonl"
echo "                 ${RESULTS}/full_math_2b_qews_hybrid.jsonl"
echo "==========================================================="
echo ""

wait $PID_BASE $PID_HYBRID

# Analysis
echo ""
echo "==========================================================="
echo "  Results"
echo "==========================================================="
python3 testbench/analyze.py --sort accuracy \
    "${RESULTS}/full_math_2b_baseline.jsonl" \
    "${RESULTS}/full_math_2b_qews_hybrid.jsonl"
