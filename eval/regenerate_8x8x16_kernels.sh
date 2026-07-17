#!/usr/bin/env bash
# eval/regenerate_8x8x16_kernels.sh
#
# Regenerate the 8x8x16 twisted-torus (N=1024, slice 16,8,8) non-ragged
# orbit_greedy_full schedule + both kernel variants from
# fixtures/routing/routcache_torus_8x8x16_twisted.json (git-lfs; run
# `git lfs pull` first if the file is a pointer).
#
# Produces:
#   fixtures/nonragged/schedule_16x8x8_loaded_orbit_greedy_full_lpt_tail_asc.json
#   fixtures/nonragged/cns_schedules/schedule_orbitfull_8x8x16_twisted.json
#   pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_full_16_8_8.py       (TPU v5)
#   pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_full_16_8_8_pfc.py   (TPU v4:
#       --per-step-barrier SEND-only throttle; see regenerate_8x4x4_pfc_kernel.sh
#       for why v4 needs it)
#
# The slice (16, 8, 8) — torus coords largest dim first — was verified by
# single-hop topology consistency: every consecutive node_id pair in all
# N*(N-1) routcache paths is a physical edge of Topology(slice=(16, 8, 8));
# the other orientations fail on ~1M paths.
set -u

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

PY=".venv/bin/python"
ROUTING="fixtures/routing/routcache_torus_8x8x16_twisted.json"
SCHED="fixtures/nonragged/schedule_16x8x8_loaded_orbit_greedy_full_lpt_tail_asc.json"

echo "=== [1/3] orbit_greedy_full schedule (+ CNS copy) ==="
"$PY" -u scripts/generate_routcache_orbitfull_schedules.py 8x8x16

echo "=== [2/3] TPU v5 kernel ==="
"$PY" -u pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 16,8,8 \
    --routing-table "$ROUTING" \
    --schedule-in "$SCHED" \
    --function-name _ragged_a2a_kernel_orbit_greedy_16_8_8 \
    --out pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_full_16_8_8.py

echo "=== [3/3] TPU v4 (pfc) per-step-barrier kernel ==="
"$PY" -u pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 16,8,8 \
    --routing-table "$ROUTING" \
    --schedule-in "$SCHED" \
    --per-step-barrier \
    --function-name _ragged_a2a_kernel_orbit_greedy_16_8_8_pfc \
    --out pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_full_16_8_8_pfc.py

echo "OK: schedule + v5 + pfc kernels regenerated for 8x8x16"
