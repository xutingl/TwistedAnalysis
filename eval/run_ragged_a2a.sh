#!/usr/bin/env bash
# Reproduce the ragged-A2A scheduling comparison on the loaded 8x4x4 routing.
# Writes schedule fixtures + results/<date>/ragged_a2a.csv.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
RESULTS="results/$(date +%Y-%m-%d)"
mkdir -p "$RESULTS"
CSV="$RESULTS/ragged_a2a.csv"
rm -f "$CSV"

COMMON=(
  --routing-table fixtures/routing_table_8x4x4_twist.json
  --slice 8,4,4
  --workload fixtures/ragged_a2a_workload_node_128_min_32_max_1024_discrete.json
  --csv-append "$CSV"
)

"$PY" -u scripts/generate_ragged_schedule.py "${COMMON[@]}" --scheduler ragged_fluid

for ORDER in lpt spt natural; do
  "$PY" -u scripts/generate_ragged_schedule.py "${COMMON[@]}" \
    --scheduler ragged_greedy --order "$ORDER"
done

"$PY" -u scripts/generate_ragged_schedule.py "${COMMON[@]}" \
  --scheduler ragged_greedy --order lpt --preemptive

echo
echo "=== $CSV ==="
column -s, -t < "$CSV"
