#!/usr/bin/env bash
# 2B QEWS sweep: replace mode, hybrid mode, hybrid x2
# Baseline reused from sweep_2b_baseline.jsonl

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

DATASET="testbench/data/hendrycks_math.parquet"
MODEL="2b"
LIMIT=200
OFFSET=3000
MAX_TOKENS=4096
H_TARGET=0.3
QEWS_TARGET=0.0
RESULTS="testbench/results"

echo "-----------------------------------------------------------"
echo "  2B QEWS Sweep — H_target=${H_TARGET}, QEWS_target=${QEWS_TARGET}"
echo "-----------------------------------------------------------"

# 1) QEWS replace — QEWS signal is sole process variable
echo ""
echo "-- [1/3] QEWS replace (QEWS only, no entropy controller) --"
python3 testbench/run.py \
    -m $MODEL -d $DATASET --limit $LIMIT --offset $OFFSET --max-tokens $MAX_TOKENS \
    --qews-mode replace \
    --qews-target $QEWS_TARGET \
    --qews-K-M 0.005 0.03 0.04 0.08 \
    --qews-K-P 0.005 0.02 0.03 0.05 \
    --qews-K-F 0.002 0.01 0.015 0.025 \
    -o "${RESULTS}/sweep_2b_qews_replace.jsonl"

# 2) QEWS hybrid — entropy + QEWS controllers, equal weight
echo ""
echo "-- [2/3] QEWS hybrid (w_H=1.0, w_Q=1.0) --"
python3 testbench/run.py \
    -m $MODEL -d $DATASET --limit $LIMIT --offset $OFFSET --max-tokens $MAX_TOKENS \
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
    -o "${RESULTS}/sweep_2b_qews_hybrid.jsonl"

# 3) QEWS hybrid x2 — entropy + QEWS, double QEWS weight
echo ""
echo "-- [3/3] QEWS hybrid x2 (w_H=1.0, w_Q=2.0) --"
python3 testbench/run.py \
    -m $MODEL -d $DATASET --limit $LIMIT --offset $OFFSET --max-tokens $MAX_TOKENS \
    --control --H-target $H_TARGET \
    --K-M 0.005 0.03 0.04 0.08 \
    --K-P 0.005 0.02 0.03 0.05 \
    --K-F 0.002 0.01 0.015 0.025 \
    --qews-mode hybrid \
    --qews-target $QEWS_TARGET \
    --qews-K-M 0.005 0.03 0.04 0.08 \
    --qews-K-P 0.005 0.02 0.03 0.05 \
    --qews-K-F 0.002 0.01 0.015 0.025 \
    --w-H 1.0 --w-Q 2.0 \
    -o "${RESULTS}/sweep_2b_qews_hybrid_x2.jsonl"

# Analysis
echo ""
echo "-----------------------------------------------------------"
echo "  Results"
echo "-----------------------------------------------------------"
echo "  baseline       : no controller"
echo "  qews_replace   : QEWS only, target=${QEWS_TARGET}"
echo "  qews_hybrid    : entropy(w=1.0) + QEWS(w=1.0)"
echo "  qews_hybrid_x2 : entropy(w=1.0) + QEWS(w=2.0)"
echo ""
python3 testbench/analyze.py --sort accuracy \
    "${RESULTS}/sweep_2b_baseline.jsonl" \
    "${RESULTS}/sweep_2b_qews_replace.jsonl" \
    "${RESULTS}/sweep_2b_qews_hybrid.jsonl" \
    "${RESULTS}/sweep_2b_qews_hybrid_x2.jsonl"
