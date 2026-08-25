#!/usr/bin/env bash
# One-shot v2 dummy-action filter:  source dataset  ->  filtered dataset  ->  tail-10 ranges json  ->  verify.
# Override anything via env:  SRC=... OUT=... FILTER_JSON=... PY=... bash run_all.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$HOME/lerobot_home/franka/tasl_fr3_10task_250ep}        # original dataset (never modified)
OUT=${OUT:-$HOME/lerobot_home/franka/tasl_fr3_10task_v2}           # new filtered dataset (must not exist)
FILTER_JSON=${FILTER_JSON:-$HOME/filters/tasl_fr3_10task_v2/nonidle_ranges_tail10.json}
PY=${PY:-python}
GRIP_GUARD=${GRIP_GUARD:-5}; GRIP_DELTA=${GRIP_DELTA:-0.02}; LIMIT=${LIMIT:-0}

[ -d "$SRC/meta" ] || { echo "SRC not a LeRobot dataset: $SRC"; exit 1; }
[ -e "$OUT" ] && { echo "OUT already exists, refusing to overwrite: $OUT"; exit 1; }
$PY -c "import numpy, pyarrow, PIL" || { echo "need: pip install -r $HERE/requirements.txt"; exit 1; }

echo "== 1/3 export (measured-velocity idle filter + gripper guard +/-$GRIP_GUARD, no tail cut)"
$PY "$HERE/export_filtered_dataset.py" --src "$SRC" --out "$OUT" --grip-guard "$GRIP_GUARD" --grip-delta "$GRIP_DELTA" --limit "$LIMIT"

echo "== 2/3 tail-10 ranges json (official script, official defaults, on the NEW dataset)"
ROOT=$(dirname "$(dirname "$OUT")"); REPO_ID=$(basename "$(dirname "$OUT")")/$(basename "$OUT")
mkdir -p "$(dirname "$FILTER_JSON")"
$PY "$HERE/nonidle_ranges.py" --source lerobot --root "$ROOT" --repo-id "$REPO_ID" --out "$FILTER_JSON"
$PY - "$OUT" "$FILTER_JSON" <<'PYEOF'
import json, sys
eps = {str(json.loads(l)["episode_index"]): json.loads(l)["length"] for l in open(sys.argv[1] + "/meta/episodes.jsonl")}
r = json.load(open(sys.argv[2])); bad = [k for k, v in r.items() if v != [[0, eps[k] - 10]] and not (eps[k] <= 10 and v == [])]
print(f"  ranges json: {len(r)} episodes, {sum(e - s for v in r.values() for s, e in v)} chunk-start anchors;"
      f" episodes not exactly [0, L-10): {len(bad)} {bad[:10]}  (0 expected = no residual idle runs)")
PYEOF

echo "== 3/3 verify"
$PY "$HERE/verify_export.py" --src "$SRC" --out "$OUT"
echo "ALL DONE  dataset=$OUT  filter_json=$FILTER_JSON"
