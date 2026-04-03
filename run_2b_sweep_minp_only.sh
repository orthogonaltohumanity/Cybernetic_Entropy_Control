#!/usr/bin/env bash
# 2B sweep: controller only actuates Min-P (top-p and freq penalty gains zeroed)
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
echo "  2B Min-P Only Sweep — H_target=${H_TARGET}"
echo "-----------------------------------------------------------"

# 1) PID min-p only
echo ""
echo "-- [1/3] PID min-p only (K_dd=0) --"
python3 testbench/run.py \
    -m $MODEL -d $DATASET --limit $LIMIT --offset $OFFSET --max-tokens $MAX_TOKENS --control \
    --H-target $H_TARGET \
    --K-M 0.005 0.03 0.04 0.0 \
    --K-P 0.0 0.0 0.0 0.0 \
    --K-F 0.0 0.0 0.0 0.0 \
    -o "${RESULTS}/sweep_2b_minp_pid.jsonl"

# 2) 4th-order min-p only
echo ""
echo "-- [2/3] 4th-order min-p only (default K_dd) --"
python3 testbench/run.py \
    -m $MODEL -d $DATASET --limit $LIMIT --offset $OFFSET --max-tokens $MAX_TOKENS --control \
    --H-target $H_TARGET \
    --K-M 0.005 0.03 0.04 0.08 \
    --K-P 0.0 0.0 0.0 0.0 \
    --K-F 0.0 0.0 0.0 0.0 \
    -o "${RESULTS}/sweep_2b_minp_4th_order.jsonl"

# 3) 4th-order min-p only x2 accel
echo ""
echo "-- [3/3] 4th-order min-p only (2x K_dd) --"
python3 testbench/run.py \
    -m $MODEL -d $DATASET --limit $LIMIT --offset $OFFSET --max-tokens $MAX_TOKENS --control \
    --H-target $H_TARGET \
    --K-M 0.005 0.03 0.04 0.16 \
    --K-P 0.0 0.0 0.0 0.0 \
    --K-F 0.0 0.0 0.0 0.0 \
    -o "${RESULTS}/sweep_2b_minp_4th_order_x2accel.jsonl"

# Analysis
echo ""
echo "-----------------------------------------------------------"
echo "  Results"
echo "-----------------------------------------------------------"
python3 testbench/analyze.py --sort accuracy \
    "${RESULTS}/sweep_2b_baseline.jsonl" \
    "${RESULTS}/sweep_2b_minp_pid.jsonl" \
    "${RESULTS}/sweep_2b_minp_4th_order.jsonl" \
    "${RESULTS}/sweep_2b_minp_4th_order_x2accel.jsonl"
