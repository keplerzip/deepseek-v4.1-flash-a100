#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
resolve_image
[[ "$PERF_MHC$PERF_INDEXER$PERF_MOE_ALIGN" != 000 ]] || fail 'Enable a performance kernel group to test it'
run="performance-kernels-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$STATE_DIR/$run"
prepare_performance "$run"
name="dsv41-$run"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
# GNU timeout is in the existing Ubuntu coreutils. The EXIT trap also removes
# the container on timeout; no orphan GPU process blocks the old service.
timeout --signal=TERM --kill-after=15s 1800s sudo -n docker run --rm --pull never --name "$name" --label dsv41.role=performance-check \
  --network none --gpus 'device=0' --ipc host --ulimit memlock=-1:-1 \
  "${CONTAINER_USER_ARGS[@]}" "${PERFORMANCE_MOUNTS[@]}" \
  --entrypoint /usr/bin/python3 -e PYTHONDONTWRITEBYTECODE=1 \
  -e VLLM_DSV41_CAND_LOGITS="$PERF_INDEXER" -e VLLM_FUSED_STABLE_MOE_ALIGN="$PERF_MOE_ALIGN" \
  -e VLLM_DETERMINISTIC_MOE_ALIGN=1 \
  -v "$STATE_DIR:/state" -v "$DEPLOY_DIR:/deploy:ro" -v "$DEPLOY_DIR/tests:/offline-tests:ro" "$RUNTIME_IMAGE" \
  -u /offline-tests/performance_kernels.py --output "/state/$run/kernels.json" \
  --mhc "$PERF_MHC" --indexer "$PERF_INDEXER" --moe-align "$PERF_MOE_ALIGN" \
  2>&1 | tee "$STATE_DIR/$run/kernels.log"
printf 'Kernel report: %s/%s/kernels.json\n' "$STATE_DIR" "$run"
