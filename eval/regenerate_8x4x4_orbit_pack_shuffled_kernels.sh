#!/usr/bin/env bash
# eval/regenerate_8x4x4_orbit_pack_shuffled_kernels.sh
#
# Regenerate the orbit_pack_shuffled(K=6, C=3) NEGATIVE-CONTROL schedules,
# their non-pfc + pfc kernels, and the CNS-renamed copies, for the loaded
# 4x4x8 twisted torus (slice 8,4,4). Seeds 0, 1, 2.
#
# WHY (2026-07-25):
#   TPU v4 measurements showed orbit_pack_k6c3 beating the P2P rotation
#   WITHOUT the per-step barrier, at parity with orbit_greedy_full, and
#   adding the barrier gained nothing. Without a barrier no step is
#   serialized, so the step count (27 vs orbitfull's 80) is not a physical
#   quantity at all — the schedule reaches the hardware only as a
#   permutation of the 128x127 destination table that the all-up-front
#   kernel iterates. Two mechanisms could then explain the win over P2P:
#
#     (a) orbit-atomicity alone. Every column of the table is a
#         permutation, so no incast and no idle devices — a property P2P's
#         rotation also has, but which the flat-ID rotation pairs with an
#         arbitrary, topology-blind path footprint.
#     (b) the FFD congestion certification. Bins are emitted contiguously,
#         so a sliding window of in-flight DMAs lands on orbits the packer
#         proved co-resident under whole-path edge cap C — the ordering
#         suggests the step structure the barrier would have enforced.
#
#   This control separates them. It holds fixed everything (a) covers —
#   the same 127 orbits, still atomic; the same 27 steps; the same
#   per-step orbit counts, hence the same per-device DMA depth — and
#   forfeits only (b), by assigning orbits to steps at random.
#
#   Read the result as: measures like k6c3   -> (b) buys nothing, and
#   orbit_pack collapses to orbit_greedy_full with extra machinery;
#                     measures worse         -> congestion control is real
#   and carried by the destination ORDERING, not the barrier.
#
#   Three seeds because a single random assignment could be lucky; report
#   the spread, not one draw.
#
# The shuffled schedules do NOT respect C=3 (that is the treatment), so
# --step-edge-cap is set per seed to the cap they actually achieve, read
# back via scripts/schedule_stats.py. Both kernel variants are emitted:
# the non-pfc one is the load-bearing A/B against
# _ragged_a2a_kernel_orbit_pack_k6c3_8_4_4; the pfc one prices the barrier
# when the steps it enforces were never certified.
#
# Run eval/regenerate_8x4x4_orbit_pack_kernels.sh first (or at least keep
# it runnable) — the reference packing defines the step profile reproduced
# here. Requires the generator's deps (pulp, etc.) in .venv.
set -u

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

ROUTING_TABLE="fixtures/routing/routing_table_8x4x4_twist.json"
SLICE="8,4,4"
K=6
C=3
PY=".venv/bin/python"

for SEED in 0 1 2; do
    TAG="k${K}c${C}_shuf${SEED}"
    SCHED="fixtures/nonragged/schedule_8x4x4_loaded_orbit_pack_${TAG}.json"
    CNS="fixtures/nonragged/cns_schedules/schedule_orbitpackk${K}c${C}shuf${SEED}_4x4x8_twisted.json"
    KERN="pallas_kernel/outputs/_ragged_a2a_kernel_orbit_pack_${TAG}_8_4_4.py"
    FUNC="_ragged_a2a_kernel_orbit_pack_${TAG}_8_4_4"
    KERN_PFC="pallas_kernel/outputs/_ragged_a2a_kernel_orbit_pack_${TAG}_8_4_4_pfc.py"
    FUNC_PFC="_ragged_a2a_kernel_orbit_pack_${TAG}_8_4_4_pfc"

    echo "=== orbit_pack_shuffled K=${K} C=${C} seed=${SEED}: schedule ==="
    "$PY" scripts/generate_schedule.py \
        --routing-table "$ROUTING_TABLE" \
        --slice "$SLICE" \
        --scheduler orbit_pack_shuffled --k "$K" --c "$C" --seed "$SEED" \
        --out "$SCHED" || exit 1

    # The achieved cap, not the reference C — the control gives C up.
    CAP="$("$PY" scripts/schedule_stats.py --field edge_cap "$SCHED")" || exit 1
    echo "--- seed ${SEED}: achieved whole-path edge cap = ${CAP} (reference C=${C})"
    "$PY" scripts/schedule_stats.py "$SCHED"

    echo "=== orbit_pack_shuffled seed=${SEED}: non-pfc kernel (primary A/B) ==="
    "$PY" pallas_kernel/gen_orbit_greedy_kernel.py \
        --slice "$SLICE" \
        --routing-table "$ROUTING_TABLE" \
        --schedule-in "$SCHED" \
        --capacity-model step \
        --step-edge-cap "$CAP" \
        --function-name "$FUNC" \
        --out "$KERN" || exit 1

    echo "=== orbit_pack_shuffled seed=${SEED}: pfc per-step-barrier kernel ==="
    "$PY" pallas_kernel/gen_orbit_greedy_kernel.py \
        --slice "$SLICE" \
        --routing-table "$ROUTING_TABLE" \
        --schedule-in "$SCHED" \
        --capacity-model step \
        --step-edge-cap "$CAP" \
        --per-step-barrier \
        --function-name "$FUNC_PFC" \
        --out "$KERN_PFC" || exit 1

    cp "$SCHED" "$CNS"
    echo "OK: $SCHED"
    echo "OK: $CNS"
    echo "OK: $KERN"
    echo "OK: $KERN_PFC"
    echo
done

echo "All orbit_pack_shuffled K=${K} C=${C} seeds {0,1,2} regenerated."
echo
echo "A/B on TPU: _ragged_a2a_kernel_orbit_pack_k6c3_8_4_4        (certified, cap 3)"
echo "        vs  _ragged_a2a_kernel_orbit_pack_k6c3_shuf{0,1,2}_8_4_4  (uncertified)"
echo "        vs  the reference P2P rotation kernel."
