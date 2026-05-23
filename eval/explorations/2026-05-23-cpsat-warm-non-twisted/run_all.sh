#!/usr/bin/env bash
# Reproducibility script for the 2026-05-23 cpsat-warm non-twisted exploration.
#
# Runs the per-cell driver on both 2x2x4 and 2x4x4, then emits a Pallas
# kernel from each best schedule. All artifacts (schedule fixtures, kernel
# .py files, run logs, results.json) land in stable, predictable paths.
#
# Usage:
#   bash eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_all.sh
#
# Idempotent: rerunning overwrites the kernel .py files and the schedule
# fixtures; results.json is APPENDED to (each row is one (cell, run) pair).
# To start fresh, delete results.json first.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPLO="${ROOT}/eval/explorations/2026-05-23-cpsat-warm-non-twisted"
PY="${ROOT}/.venv/bin/python"

cd "${ROOT}"

# Per-cell parameters: (slice_csv, slice_kern, routing_table_path, time_limit_s)
run_cell() {
    local slice_csv="$1"
    local slice_kern="$2"
    local routing_table="$3"
    local time_limit_s="$4"

    local slice_slug="${slice_csv//,/x}"   # 2,2,4 -> 2x2x4
    local schedule="fixtures/schedule_torus_${slice_slug}_cpsat_literal_warm.json"
    local kernel="pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_${slice_kern}.py"
    local log="${EXPLO}/run_log_${slice_slug}.txt"

    echo "=== ${slice_csv}: driver ==="
    "${PY}" -u "${EXPLO}/run_cpsat_warm.py" \
        --routing-table "${routing_table}" \
        --slice "${slice_csv}" \
        --out-schedule "${schedule}" \
        --time-limit-s "${time_limit_s}" \
        --n-workers 8 \
        --results-json "${EXPLO}/results.json" 2>&1 | tee "${log}"

    echo
    echo "=== ${slice_csv}: kernel ==="
    "${PY}" -u pallas_kernel/gen_orbit_greedy_kernel.py \
        --slice "${slice_csv}" \
        --routing-table "${routing_table}" \
        --schedule-in "${schedule}" \
        --out "${kernel}" \
        --function-name "_ragged_a2a_kernel_cpsat_literal_warm_torus_${slice_kern}"

    echo "wrote ${schedule}"
    echo "wrote ${kernel}"
    echo
}

run_cell "2,2,4" "2_2_4" "fixtures/routing_table_torus_2x2x4.json"  300
run_cell "2,4,4" "2_4_4" "fixtures/routing_table_torus_2x4x4.json" 1800

echo "=== done. See ${EXPLO}/results.json and ${EXPLO}/RESULTS.md ==="
