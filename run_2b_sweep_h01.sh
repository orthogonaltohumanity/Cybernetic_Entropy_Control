#!/usr/bin/env bash
# 2B control experiment sweep: PID, 4th-order, 4th-order x2 accel
# H_target=0.1, baseline reused from sweep_2b_baseline.jsonl

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

DATASET="testbench/data/hendrycks_math.parquet"
MODEL="2b"
LIMIT=200
OFFSET=3000
MAX_TOKENS=4096
H_TARGET=0.1
RESULTS="testbench/results"

echo "-----------------------------------------------------------"
echo "  2B Entropy Controller Sweep — H_target=${H_TARGET}"
echo "-----------------------------------------------------------"

# 1) PID — zero out d2_error (acceleration) weight
echo ""
echo "-- [1/3] PID controller (K_dd=0) --"
python3 testbench/run.py \
    -m $MODEL -d $DATASET --limit $LIMIT --offset $OFFSET --max-tokens $MAX_TOKENS --control \
    --H-target $H_TARGET \
    --K-M 0.005 0.03 0.04 0.0 \
    --K-P 0.005 0.02 0.03 0.0 \
    --K-F 0.002 0.01 0.015 0.0 \
    -o "${RESULTS}/sweep_2b_h01_pid.jsonl"

# 2) 4th-order — default acceleration weights
echo ""
echo "-- [2/3] 4th-order controller (default K_dd) --"
python3 testbench/run.py \
    -m $MODEL -d $DATASET --limit $LIMIT --offset $OFFSET --max-tokens $MAX_TOKENS --control \
    --H-target $H_TARGET \
    --K-M 0.005 0.03 0.04 0.08 \
    --K-P 0.005 0.02 0.03 0.05 \
    --K-F 0.002 0.01 0.015 0.025 \
    -o "${RESULTS}/sweep_2b_h01_4th_order.jsonl"

# 3) 4th-order x2 acceleration — double the d2_error weights
echo ""
echo "-- [3/3] 4th-order controller (2x K_dd) --"
python3 testbench/run.py \
    -m $MODEL -d $DATASET --limit $LIMIT --offset $OFFSET --max-tokens $MAX_TOKENS --control \
    --H-target $H_TARGET \
    --K-M 0.005 0.03 0.04 0.16 \
    --K-P 0.005 0.02 0.03 0.10 \
    --K-F 0.002 0.01 0.015 0.05 \
    -o "${RESULTS}/sweep_2b_h01_4th_order_x2accel.jsonl"

# Analysis
echo ""
echo "-----------------------------------------------------------"
echo "  Results"
echo "-----------------------------------------------------------"
python3 testbench/analyze.py --sort accuracy \
    "${RESULTS}/sweep_2b_baseline.jsonl" \
    "${RESULTS}/sweep_2b_h01_pid.jsonl" \
    "${RESULTS}/sweep_2b_h01_4th_order.jsonl" \
    "${RESULTS}/sweep_2b_h01_4th_order_x2accel.jsonl"
