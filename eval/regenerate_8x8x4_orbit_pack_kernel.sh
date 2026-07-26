#!/usr/bin/env bash
# eval/regenerate_8x8x4_orbit_pack_kernel.sh
#
# Regenerate the orbit_pack(K=6, C=3) schedule + NON-pfc (all-up-front) kernel
# for the loaded 4x8x8 twisted torus (N=256, slice 8,8,4 — torus coords with
# largest dim first, same convention as generate_routcache_orbitfull_schedules),
# plus the CNS-renamed schedule copy.
#
# Unlike the 4x4x8 kernels in eval/regenerate_8x4x4_orbit_pack_kernels.sh,
# this emits the all-up-front kernel variant (no --per-step-barrier): all DMAs
# are issued at once and the schedule contributes the per-device issue ORDER
# (sorted by barrier step). The schedule itself is step-model
# (verified with verify_capacity_step at edge cap 3; staggered
# verify_capacity rejects it by design, hence --capacity-model step).
#
# C=3 feasibility on this routing: hottest orbit internal whole-path load = 3
# (26 orbits at 1, 115 at 2, 114 at 3). Routing LB = 184 with EVERY directed
# edge at exactly 184 (perfectly balanced). orbit_greedy_full reference:
# makespan 223, 177 distinct steps.
#
# Requires the generator's deps (pulp, etc.) in .venv.
set -u

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

ROUTING_TABLE="fixtures/routing/routcache_torus_4x8x8_twisted.json"
SLICE="8,8,4"
K=6
C=3
PY=".venv/bin/python"

SCHED="fixtures/nonragged/schedule_8x8x4_loaded_orbit_pack_k${K}c${C}.json"
CNS="fixtures/nonragged/cns_schedules/schedule_orbitpackk${K}c${C}_4x8x8_twisted.json"
KERN="pallas_kernel/outputs/_ragged_a2a_kernel_orbit_pack_k${K}c${C}_8_8_4.py"
FUNC="_ragged_a2a_kernel_orbit_pack_k${K}c${C}_8_8_4"

echo "=== orbit_pack K=${K} C=${C} on 4x8x8 twisted (slice ${SLICE}): schedule ==="
"$PY" scripts/generate_schedule.py \
    --routing-table "$ROUTING_TABLE" \
    --slice "$SLICE" \
    --scheduler orbit_pack --k "$K" --c "$C" \
    --out "$SCHED"

echo "=== orbit_pack K=${K} C=${C}: all-up-front (non-pfc) kernel ==="
"$PY" pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice "$SLICE" \
    --routing-table "$ROUTING_TABLE" \
    --schedule-in "$SCHED" \
    --capacity-model step \
    --step-edge-cap "$C" \
    --function-name "$FUNC" \
    --out "$KERN"

cp "$SCHED" "$CNS"
echo "OK: $SCHED"
echo "OK: $CNS"
echo "OK: $KERN"
