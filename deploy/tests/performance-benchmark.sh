#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
resolve_image
own_container "$ENGINE_NAME" && engine_health || fail 'Start the engine first'
run="performance-benchmark-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$STATE_DIR/$run"
current=$(cat "$STATE_DIR/current-run.txt")
[[ ! -f "$STATE_DIR/$current/performance.json" ]] || cp "$STATE_DIR/$current/performance.json" "$STATE_DIR/$run/performance.json"
cp "$STATE_DIR/runtime.resolved.json" "$STATE_DIR/$run/config.json"
cp "$STATE_DIR/capacity.json" "$STATE_DIR/$run/capacity.json"
for mode in off high; do
  docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/dspark_benchmark.py \
    --seed dsv41-performance-20260915 --mode "$mode" --k 5 \
    --output "/state/$run/$mode.json" "$@" 2>&1 | tee "$STATE_DIR/$run/$mode.log"
done
printf 'C1/C32 k5 reports and DSpark acceptance counters: %s/%s\n' "$STATE_DIR" "$run"
