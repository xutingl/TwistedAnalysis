#!/usr/bin/env bash
# eval/regenerate_8x4x4_all_schedulers.sh
#
# Re-run the kernel-generation pipeline against the loaded 8x4x4 routing
# table for every scheduler we support, so each algorithm has fresh
# fixtures + kernel for benchmarking.
#
# orbit_greedy is expected to fail the post-schedule verifier (the whole
# point of this plan); we mark its failure as expected and continue.
#
# ilp_literal is intractable at N=128 within a 10-minute time budget; we
# skip it here but provide a stub line for documentation.
set -u

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

ROUTING_TABLE="fixtures/routing/routing_table_8x4x4_twist.json"
SLICE="8,4,4"
PY=".venv/bin/python"

run() {
  local sched="$1"
  local order="${2:-lpt_tail_asc}"
  local out_kern="pallas_kernel/outputs/_ragged_a2a_kernel_${sched}_8_4_4.py"
  local out_sched="fixtures/nonragged/schedule_8x4x4_loaded_${sched}_${order}.json"
  echo "=== Running scheduler=${sched} order=${order} ==="
  if "$PY" pallas_kernel/gen_orbit_greedy_kernel.py \
        --slice "$SLICE" \
        --routing-table "$ROUTING_TABLE" \
        --scheduler "$sched" \
        --order "$order" \
        --schedule-out "$out_sched" \
        --out "$out_kern"; then
    echo "OK: $sched -> $out_kern"
  else
    echo "FAILED (likely capacity-violation refusal): $sched"
  fi
  echo
}

# Expected to fail: original orbit_greedy on loaded routing.
run orbit_greedy lpt_tail_asc || true

# Expected to succeed.
run orbit_greedy_full lpt_tail_asc
run literal_greedy lpt

# ilp_literal at N=128 has ~16k binary vars * ~80 time slots; CBC won't
# solve in <10 min. Documented as future work.
echo "# Skipping ilp_literal on 8x4x4 (intractable at this scale)."
