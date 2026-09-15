#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
resolve_image
[[ "$PERF_MHC" == 1 ]] || fail 'PERF_MHC=1 is required to test the updated kernels'
run="performance-kernels-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$STATE_DIR/$run"
prepare_performance "$run"
name="dsv41-$run"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
docker run --rm --pull never --name "$name" --label dsv41.role=performance-check \
  --network none --gpus 'device=0' --ipc host --ulimit memlock=-1:-1 \
  "${CONTAINER_USER_ARGS[@]}" "${PERFORMANCE_MOUNTS[@]}" \
  --entrypoint /usr/bin/python3 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$STATE_DIR:/state" -v "$DEPLOY_DIR/tests:/offline-tests:ro" "$RUNTIME_IMAGE" \
  /offline-tests/performance_kernels.py --output "/state/$run/kernels.json" \
  2>&1 | tee "$STATE_DIR/$run/kernels.log"
printf 'Kernel report: %s/%s/kernels.json\n' "$STATE_DIR" "$run"
