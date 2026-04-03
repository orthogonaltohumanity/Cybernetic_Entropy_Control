#!/usr/bin/env bash
# Full 2B sweep: baseline, H=0.1 control, min-P only, QEWS
# Low target entropy to reduce uncertainty

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

D="testbench/data/hendrycks_math.parquet"
M="2b"
N=200
O=3000
T=4096
R="testbench/results"

RUN="python3 testbench/run.py -m $M -d $D --limit $N --offset $O --max-tokens $T"

echo "==========================================================="
echo "  Full 2B Sweep"
echo "==========================================================="

# ── Baseline (shared across all sweeps) ──────────────────────────
echo ""
echo "== [BASELINE] No controller =="
$RUN -o "${R}/sweep_2b_baseline.jsonl"

# ── Sweep 1: H_target=0.1 (M+P+F control) ───────────────────────
echo ""
echo "== [SWEEP 1] H_target=0.1 =="

echo "-- PID (K_dd=0) --"
$RUN --control --H-target 0.1 \
    --K-M 0.005 0.03 0.04 0.0 \
    --K-P 0.005 0.02 0.03 0.0 \
    --K-F 0.002 0.01 0.015 0.0 \
    -o "${R}/sweep_2b_h01_pid.jsonl"

echo "-- 4th-order (default) --"
$RUN --control --H-target 0.1 \
    --K-M 0.005 0.03 0.04 0.08 \
    --K-P 0.005 0.02 0.03 0.05 \
    --K-F 0.002 0.01 0.015 0.025 \
    -o "${R}/sweep_2b_h01_4th_order.jsonl"

echo "-- 4th-order (2x accel) --"
$RUN --control --H-target 0.1 \
    --K-M 0.005 0.03 0.04 0.16 \
    --K-P 0.005 0.02 0.03 0.10 \
    --K-F 0.002 0.01 0.015 0.05 \
    -o "${R}/sweep_2b_h01_4th_order_x2accel.jsonl"

# ── Sweep 2: H_target=0.1 min-P only ────────────────────────────
echo ""
echo "== [SWEEP 2] H_target=0.1, min-P only =="

echo "-- PID min-p only --"
$RUN --control --H-target 0.1 \
    --K-M 0.005 0.03 0.04 0.0 \
    --K-P 0.0 0.0 0.0 0.0 \
    --K-F 0.0 0.0 0.0 0.0 \
    -o "${R}/sweep_2b_minp_pid.jsonl"

echo "-- 4th-order min-p only --"
$RUN --control --H-target 0.1 \
    --K-M 0.005 0.03 0.04 0.08 \
    --K-P 0.0 0.0 0.0 0.0 \
    --K-F 0.0 0.0 0.0 0.0 \
    -o "${R}/sweep_2b_minp_4th_order.jsonl"

echo "-- 4th-order min-p only (2x accel) --"
$RUN --control --H-target 0.1 \
    --K-M 0.005 0.03 0.04 0.16 \
    --K-P 0.0 0.0 0.0 0.0 \
    --K-F 0.0 0.0 0.0 0.0 \
    -o "${R}/sweep_2b_minp_4th_order_x2accel.jsonl"

# ── Sweep 3: QEWS ────────────────────────────────────────────────
echo ""
echo "== [SWEEP 3] QEWS =="

echo "-- QEWS replace --"
$RUN \
    --qews-mode replace --qews-target 0.0 \
    --qews-K-M 0.005 0.03 0.04 0.08 \
    --qews-K-P 0.005 0.02 0.03 0.05 \
    --qews-K-F 0.002 0.01 0.015 0.025 \
    -o "${R}/sweep_2b_qews_replace.jsonl"

echo "-- QEWS hybrid (w_H=1, w_Q=1) --"
$RUN --control --H-target 0.1 \
    --K-M 0.005 0.03 0.04 0.08 \
    --K-P 0.005 0.02 0.03 0.05 \
    --K-F 0.002 0.01 0.015 0.025 \
    --qews-mode hybrid --qews-target 0.0 \
    --qews-K-M 0.005 0.03 0.04 0.08 \
    --qews-K-P 0.005 0.02 0.03 0.05 \
    --qews-K-F 0.002 0.01 0.015 0.025 \
    --w-H 1.0 --w-Q 1.0 \
    -o "${R}/sweep_2b_qews_hybrid.jsonl"

echo "-- QEWS hybrid x2 (w_H=1, w_Q=2) --"
$RUN --control --H-target 0.1 \
    --K-M 0.005 0.03 0.04 0.08 \
    --K-P 0.005 0.02 0.03 0.05 \
    --K-F 0.002 0.01 0.015 0.025 \
    --qews-mode hybrid --qews-target 0.0 \
    --qews-K-M 0.005 0.03 0.04 0.08 \
    --qews-K-P 0.005 0.02 0.03 0.05 \
    --qews-K-F 0.002 0.01 0.015 0.025 \
    --w-H 1.0 --w-Q 2.0 \
    -o "${R}/sweep_2b_qews_hybrid_x2.jsonl"

# ── Results ──────────────────────────────────────────────────────
echo ""
echo "==========================================================="
echo "  ALL DONE — running analysis"
echo "==========================================================="
bash testbench/analyze_sweeps.sh
