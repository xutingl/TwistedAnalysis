#!/usr/bin/env bash
# Runs every YAML experiment and aggregates summaries into a single table.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATE=$(date +%Y-%m-%d)
RESULTS_DIR="$ROOT/results/$DATE"
mkdir -p "$RESULTS_DIR"

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "Run 'uv venv && uv pip install -e \".[dev]\"' first." >&2
    exit 1
fi

for cfg in "$ROOT"/experiments/*.yaml; do
    name=$(basename "$cfg" .yaml)
    out="$RESULTS_DIR/$name"
    echo "=== $name ==="
    # Patch output_dir for this run
    tmpcfg=$(mktemp --suffix=.yaml)
    sed "s|^output_dir:.*|output_dir: $out|" "$cfg" > "$tmpcfg"
    "$PY" -m twisted_analysis.cli run "$tmpcfg" > "$out.log" 2>&1 || {
        echo "  FAILED — see $out.log"; continue;
    }
    rm "$tmpcfg"
done

# Aggregate summaries
"$PY" - <<EOF
import json, pathlib, csv
root = pathlib.Path("$RESULTS_DIR")
rows = []
for s in sorted(root.rglob("summary.json")):
    rows.append(json.loads(s.read_text()))
with (root / "headlines.csv").open("w", newline="") as f:
    w = csv.writer(f)
    if rows:
        w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow([json.dumps(v) if isinstance(v, (list, dict)) else v
                         for v in r.values()])
print(f"Wrote {root}/headlines.csv with {len(rows)} rows")
EOF
