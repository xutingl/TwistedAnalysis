#!/usr/bin/env bash
# eval/regenerate_8x4x4_orbit_pack_nonpfc_kernels.sh
#
# Regenerate the NON-pfc (no per-step barrier, TPU v5-safe) orbit_pack(K, C=3)
# kernels for the loaded 4x4x8 twisted torus (slice 8,4,4), K in {2, 3, 6}.
#
# Companion to regenerate_8x4x4_orbit_pack_kernels.sh (the TPU v4 "pfc"
# variant). Reuses the schedule JSONs that script produces — run it first if
# fixtures/nonragged/schedule_8x4x4_loaded_orbit_pack_k{K}c3.json is missing.
# Without --per-step-barrier all DMAs are issued up front and drained once at
# the end: fine on TPU v5, hangs on v4's smaller DMA descriptor queue.
set -u

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

ROUTING_TABLE="fixtures/routing/routing_table_8x4x4_twist.json"
SLICE="8,4,4"
C=3
PY=".venv/bin/python"

for K in 2 3 6; do
    SCHED="fixtures/nonragged/schedule_8x4x4_loaded_orbit_pack_k${K}c${C}.json"
    KERN="pallas_kernel/outputs/_ragged_a2a_kernel_orbit_pack_k${K}c${C}_8_4_4.py"
    FUNC="_ragged_a2a_kernel_orbit_pack_k${K}c${C}_8_4_4"

    echo "=== orbit_pack K=${K} C=${C}: non-pfc (no per-step barrier) kernel ==="
    "$PY" pallas_kernel/gen_orbit_greedy_kernel.py \
        --slice "$SLICE" \
        --routing-table "$ROUTING_TABLE" \
        --schedule-in "$SCHED" \
        --capacity-model step \
        --step-edge-cap "$C" \
        --function-name "$FUNC" \
        --out "$KERN"

    echo "OK: $KERN"
    echo
done

echo "All orbit_pack K in {2, 3, 6} (C=${C}) non-pfc kernels regenerated."
