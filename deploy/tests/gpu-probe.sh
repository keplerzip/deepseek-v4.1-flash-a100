#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
exec 9>"$STATE_DIR/operation.lock"
flock -n 9 || fail 'Another deployment operation is active'
resolve_image
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]] || fail 'Run the small GPU probe before loading the model'
run="gpu-probe-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$STATE_DIR/$run"
run_probe() {
  local phase=$1 rc
  shift
  printf '[GPU probe] %s started; log: %s/%s/%s.log\n' "$phase" "$STATE_DIR" "$run" "$phase"
  if "$@" 2>&1 | tee "$STATE_DIR/$run/$phase.log"; then
    printf '[GPU probe] %s passed\n' "$phase"
  else
    rc=$?
    printf '[GPU probe] %s failed (exit %s); last log lines follow\n' "$phase" "$rc" >&2
    tail -n 80 "$STATE_DIR/$run/$phase.log" >&2
    return "$rc"
  fi
}
run_probe probe docker run --rm --pull never --name "$run-basic" --label dsv41.role=gpu-probe \
  "${CONTAINER_USER_ARGS[@]}" --gpus all --network none --ipc host --ulimit memlock=-1:-1 \
  --entrypoint /usr/bin/python3 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET \
  -e "NCCL_DEBUG_FILE=/state/$run/nccl.%h.%p.log" \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
  -e "PROBE_OUTPUT=/state/$run" -v "$STATE_DIR:/state" -v "$DEPLOY_DIR/tests:/offline-tests:ro" \
  "$RUNTIME_IMAGE" -u /offline-tests/probe_supervisor.py --seconds 300 --heartbeat 15 -- \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=8 \
  --node-rank=0 --rdzv-backend=static --master-addr=127.0.0.1 --master-port=29500 \
  --rdzv-conf=timeout=60 --max-restarts=0 /offline-tests/gpu_probe.py
image_python /deploy/tests/check_gpu_probe.py --basic-only "/state/$run"
run_probe engram docker run --rm --pull never --name "$run-engram" --label dsv41.role=gpu-probe \
  "${CONTAINER_USER_ARGS[@]}" --gpus all --network none --ipc host --ulimit memlock=-1:-1 \
  --entrypoint /usr/bin/python3 -e PYTHONDONTWRITEBYTECODE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e "TRITON_CACHE_DIR=/state/$run/triton" -v "$STATE_DIR:/state" -v "$DEPLOY_DIR/tests:/offline-tests:ro" \
  "$RUNTIME_IMAGE" -m pytest -q -p no:cacheprovider /offline-tests/upstream/test_engram.py \
  -k 'lookup_matches_torch or prepared_rows_survive_graph_breaks or lookback_window_reproduces_single_instance' \
  --junitxml="/state/$run/engram.xml"
run_probe fp8-sm80 docker run --rm --pull never --name "$run-fp8" --label dsv41.role=gpu-probe \
  "${CONTAINER_USER_ARGS[@]}" --gpus all --network none --ipc host \
  --entrypoint /usr/bin/python3 -e PYTHONDONTWRITEBYTECODE=1 \
  -e "TRITON_CACHE_DIR=/state/$run/triton" -v "$STATE_DIR:/state" -v "$DEPLOY_DIR/tests:/offline-tests:ro" \
  "$RUNTIME_IMAGE" -m pytest -q -p no:cacheprovider /offline-tests/upstream/test_dsv4_fp8_sm80.py \
  --junitxml="/state/$run/fp8-sm80.xml"
image_python /deploy/tests/check_gpu_probe.py "/state/$run"
printf 'Small GPU probe reports: %s/%s\n' "$STATE_DIR" "$run"
