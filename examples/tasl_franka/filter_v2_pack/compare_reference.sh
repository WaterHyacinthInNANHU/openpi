#!/usr/bin/env bash
# Byte-compare your run against the labserver reference:  OUT=<dataset dir> FILTER_JSON=<json> bash compare_reference.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:?set OUT=<exported dataset dir>}; FILTER_JSON=${FILTER_JSON:?set FILTER_JSON=<tail10 json>}
echo "== dataset parquet + meta vs labserver (392 parquet + 5 meta files)"
( cd "$OUT" && md5sum -c --quiet "$HERE/reference/tasl_fr3_10task_v2.md5" ) && echo "  dataset: ALL IDENTICAL" || echo "  dataset: DIFFERENCES above (source_segments.json / info.json differ only by absolute source path if you used another SRC location; parquet + stats + episodes must match)"
echo "== tail-10 json"
[ "$(md5sum < "$FILTER_JSON" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$HERE/reference/nonidle_ranges_tail10.md5")" ] && echo "  json: IDENTICAL" || { echo "  json: differs, diffing:"; diff <(python -m json.tool "$FILTER_JSON") <(python -m json.tool "$HERE/reference/nonidle_ranges_tail10.json") | head; }
