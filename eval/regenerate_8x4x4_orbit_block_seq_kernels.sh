#!/usr/bin/env bash
# eval/regenerate_8x4x4_orbit_block_seq_kernels.sh
#
# Regenerate the orbit_block_seq(W) schedules, kernels and CNS copies for the
# loaded 4x4x8 twisted torus (slice 8,4,4), W in {6, 12}.
#
# WHY (2026-07-26): every scheduler before this one optimised a quantity the
# hardware cannot see. Measurements established, in order:
#   * the per-step barrier gains nothing (pfc == non-pfc), so barrier-step
#     count is not a physical quantity;
#   * orbit_pack_shuffled (whole-path edge load 8) ties orbit_pack (load 3),
#     so the `C` cap — which bounds a SOFTWARE round — is inert;
#   * the benefit of any schedule shrinks as DMA payload grows.
#
# Diagnosis: with no barrier the kernel issues a flat .start() stream and the
# DMA queues decide concurrency. All 128 devices walk their dest tables in
# lockstep, so the concurrent set is a sliding window of W dest-table columns
# — W set by outstanding descriptors (payload / 32 KB packets) and widened by
# the engine's LRU arbitration. Measured max whole-path edge load over W
# consecutive columns, and the advantage over P2P decays exactly as W grows:
#
#   W=          6     12     24     48     96    127
#   P2P        15     27     40     61     75     75
#   orbitfull   8     13     20     36     66     75
#   ratio    1.88   2.08   2.00   1.69   1.14   1.00
#
# All schedules converge to 75 = LB at W=127 (same flows, same routes), which
# IS the payload-size trend. So window load is the objective, with lower bound
# LB(W) = ceil(W * 75 / 127).
#
# orbit_block_seq targets it directly. A flat least-burstiness-next greedy was
# tried first and FAILED — it beat orbitfull only at its tuning window and
# regressed elsewhere (greedy(W=24) scored 22 vs orbitfull's 20 at W=24).
# The structural fix: orbit_pack's ragged bins (sizes 4-6) let a 6-column
# window straddle THREE bins, so three bins at load 3 give 9. Forcing every
# block to size >= W caps the straddle at two, then blocks are ordered so
# adjacent unions stay light.
#
#   W=                6     12     24     48     96
#   orbitfull         8     13     20     36     66
#   k6c3 certified    9     12     21     36     63
#   block(W=6)        7     11     18     32     60
#   block(W=12)       7     11     17     31     60   <- dominates
#
# 13-15% below orbitfull at EVERY window, no tuning-window overfit. W=12 is
# the recommendation; W=6 ships for comparison.
#
# CAVEAT: this optimises a model that has never been validated as a
# QUANTITATIVE wall-clock predictor. It currently gets the ORDERING right
# (P2P worst, shuffled middling, orbitfull/certified best). Before reading
# much into a 13% model gain, check that predicted ratios track measured ones
# across the schedules already benchmarked — that costs nothing.
#
# Requires the generator's deps (pulp, etc.) in .venv.
set -u

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

ROUTING_TABLE="fixtures/routing/routing_table_8x4x4_twist.json"
SLICE="8,4,4"
PY=".venv/bin/python"

for W in 6 12; do
    SCHED="fixtures/nonragged/schedule_8x4x4_loaded_orbit_block_seq_w${W}.json"
    # CNS uses the 4x4x8 dimension labelling, not the 8x4x4 flatten order.
    CNS="fixtures/nonragged/cns_schedules/schedule_orbitblockseqw${W}_4x4x8_twisted.json"
    KERN="pallas_kernel/outputs/_ragged_a2a_kernel_orbit_block_seq_w${W}_8_4_4.py"
    FUNC="_ragged_a2a_kernel_orbit_block_seq_w${W}_8_4_4"

    echo "=== orbit_block_seq W=${W}: schedule ==="
    "$PY" scripts/generate_schedule.py \
        --routing-table "$ROUTING_TABLE" \
        --slice "$SLICE" \
        --scheduler orbit_block_seq --w "$W" \
        --out "$SCHED" || exit 1

    "$PY" scripts/schedule_stats.py "$SCHED"

    # One orbit per round, so the per-round cap is just the hottest single
    # orbit's internal load. Read it back rather than hardcoding.
    CAP="$("$PY" scripts/schedule_stats.py --field edge_cap "$SCHED")" || exit 1

    echo "=== orbit_block_seq W=${W}: non-pfc kernel (all-up-front) ==="
    "$PY" pallas_kernel/gen_orbit_greedy_kernel.py \
        --slice "$SLICE" \
        --routing-table "$ROUTING_TABLE" \
        --schedule-in "$SCHED" \
        --capacity-model step \
        --step-edge-cap "$CAP" \
        --function-name "$FUNC" \
        --out "$KERN" || exit 1

    cp "$SCHED" "$CNS"
    echo "OK: $SCHED"
    echo "OK: $CNS"
    echo "OK: $KERN"
    echo
done

echo "orbit_block_seq W in {6, 12} regenerated."
echo
echo "A/B on TPU (all non-pfc / all-up-front, the only mode that matters):"
echo "  _ragged_a2a_kernel_orbit_greedy_full_8_4_4        (current best)"
echo "  _ragged_a2a_kernel_orbit_block_seq_w12_8_4_4      (predicted -15% window load)"
echo "  _ragged_a2a_kernel_orbit_block_seq_w6_8_4_4"
echo "  reference P2P rotation kernel"
echo
echo "No _pfc variants: one orbit per round would mean 127 barriers, and the"
echo "barrier already measured inert. Use the non-pfc kernels."
