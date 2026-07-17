#!/usr/bin/env bash
# eval/regenerate_8x4x4_orbit_pack_kernels.sh
#
# Regenerate the orbit_pack(K, C=3) schedules + TPU v4 ("pfc") per-step-barrier
# kernels for the loaded 4x4x8 twisted torus (slice 8,4,4), K in {2, 3, 6},
# plus the CNS-renamed schedule copies.
#
# WHY orbit_pack (2026-07-17 analysis):
#   TPU v4 measurements showed orbit_greedy_full (80 barrier steps) slightly
#   beats the P2P rotation baseline (127 steps) while cpsat_literal_warm
#   (78 rounds, but device-jagged: sum_t max-per-device sends = 268 vs 127,
#   incast up to 8 flows -> one device per round) UNDERPERFORMS it. The
#   binding wall-clock terms under --per-step-barrier execution are the
#   barrier-step count and per-step balance/congestion — NOT the staggered
#   simulator makespan cpsat optimizes. orbit_pack packs whole orbits
#   (permutations -> perfect per-device send/recv balance) into few steps
#   under a whole-path congestion cap:
#     K=2, C=3 -> 64 steps   K=3, C=3 -> 43 steps   K=6, C=3 -> 27 steps
#   C=3 = P2P rotation's own worst per-round whole-path edge load; K=6 is
#   v4-queue-safe (the orbit_greedy_full pfc kernel's widest step already
#   fires 6 orbits). Schedules are step-model: verified with
#   verify_capacity_step (staggered verify_capacity rejects them by design),
#   hence --capacity-model step --step-edge-cap 3 at kernel generation.
#
# Requires the generator's deps (pulp, etc.) in .venv.
set -u

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

ROUTING_TABLE="fixtures/routing/routing_table_8x4x4_twist.json"
SLICE="8,4,4"
C=3
PY=".venv/bin/python"

for K in 2 3 6; do
    SCHED="fixtures/nonragged/schedule_8x4x4_loaded_orbit_pack_k${K}c${C}.json"
    CNS="fixtures/nonragged/cns_schedules/schedule_orbitpackk${K}c${C}_4x4x8_twisted.json"
    KERN="pallas_kernel/outputs/_ragged_a2a_kernel_orbit_pack_k${K}c${C}_8_4_4_pfc.py"
    FUNC="_ragged_a2a_kernel_orbit_pack_k${K}c${C}_8_4_4_pfc"

    echo "=== orbit_pack K=${K} C=${C}: schedule ==="
    "$PY" scripts/generate_schedule.py \
        --routing-table "$ROUTING_TABLE" \
        --slice "$SLICE" \
        --scheduler orbit_pack --k "$K" --c "$C" \
        --out "$SCHED"

    echo "=== orbit_pack K=${K} C=${C}: TPU v4 (pfc) per-step-barrier kernel ==="
    "$PY" pallas_kernel/gen_orbit_greedy_kernel.py \
        --slice "$SLICE" \
        --routing-table "$ROUTING_TABLE" \
        --schedule-in "$SCHED" \
        --capacity-model step \
        --step-edge-cap "$C" \
        --per-step-barrier \
        --function-name "$FUNC" \
        --out "$KERN"

    cp "$SCHED" "$CNS"
    echo "OK: $SCHED"
    echo "OK: $CNS"
    echo "OK: $KERN"
    echo
done

echo "All orbit_pack K in {2, 3, 6} (C=${C}) schedules + pfc kernels regenerated."
