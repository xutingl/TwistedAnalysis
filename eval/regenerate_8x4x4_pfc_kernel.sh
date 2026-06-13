#!/usr/bin/env bash
# eval/regenerate_8x4x4_pfc_kernel.sh
#
# Regenerate the TPU v4 ("pfc") variant of the orbit_greedy_full 8x4x4
# all-to-all kernel.
#
# WHY a separate kernel:
#   The default kernel (_ragged_a2a_kernel_orbit_greedy_full_8_4_4.py) issues
#   ALL ~127*num_packets remote DMAs up front via `.start()` and does a single
#   send_sem/recv_sem wait at the end. That works on TPU v5 (deep DMA queue) but
#   HANGS on TPU v4, whose smaller DMA descriptor queue overflows when the whole
#   schedule is submitted at once.
#
# THE FIX (--per-step-barrier):
#   Issue the DMAs for one OrbitGreedy step at a time, then drain send_sem and
#   recv_sem before the next step. This bounds outstanding DMAs to one step's
#   width (<= 6 orbits * num_packets here, vs ~127 * num_packets), which fits the
#   v4 queue. The orbit_greedy_full schedule is prefix-balanced
#   (incoming(<=t) >= outgoing(<=t) for every device at every step), so the
#   per-step recv drain never blocks on not-yet-sent data -> no deadlock.
#
# The v5 kernel is left untouched; this emits a NEW file with a `_pfc` suffix.
#
# Requires the generator's deps (pulp, etc.) in .venv.
set -u

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

ROUTING_TABLE="fixtures/routing_table_8x4x4_twist.json"
SLICE="8,4,4"
ORDER="lpt_tail_asc"
PY=".venv/bin/python"

OUT_KERN="pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_full_8_4_4_pfc.py"
OUT_SCHED="fixtures/schedule_8x4x4_loaded_orbit_greedy_full_${ORDER}.json"

echo "=== Generating TPU v4 (pfc) per-step-barrier kernel ==="
"$PY" pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice "$SLICE" \
    --routing-table "$ROUTING_TABLE" \
    --scheduler orbit_greedy_full \
    --order "$ORDER" \
    --per-step-barrier \
    --function-name _ragged_a2a_kernel_orbit_greedy_8_4_4_pfc \
    --schedule-out "$OUT_SCHED" \
    --out "$OUT_KERN"

echo "OK: orbit_greedy_full (per-step-barrier / pfc) -> $OUT_KERN"
