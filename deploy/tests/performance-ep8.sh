#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
resolve_image
run="performance-ep8-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$STATE_DIR/$run"
prepare_performance "$run"
name="dsv41-$run"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
timeout --signal=TERM --kill-after=15s 300s sudo -n docker run --rm --pull never \
  --name "$name" --label dsv41.role=performance-check --network none \
  --gpus all --ipc host --ulimit memlock=-1:-1 \
  "${CONTAINER_USER_ARGS[@]}" "${PERFORMANCE_MOUNTS[@]}" \
  --entrypoint /usr/bin/python3 -e PYTHONDONTWRITEBYTECODE=1 \
  -e VLLM_SM80_EP8_CUSTOM_AG_RS=1 -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
  -v "$STATE_DIR:/state" -v "$DEPLOY_DIR/tests:/offline-tests:ro" "$RUNTIME_IMAGE" \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=8 \
  --rdzv-backend=static --master-addr=127.0.0.1 --master-port=29579 \
  /offline-tests/performance_ep8.py "/state/$run" 2>&1 | tee "$STATE_DIR/$run/probe.log"
image_python -c 'import json,pathlib,sys; rows=[json.loads(p.read_text()) for p in pathlib.Path(sys.argv[1]).glob("rank-*.json")]; assert len(rows)==8 and {r["rank"] for r in rows}==set(range(8)) and all(r["status"]=="PASS" for r in rows); print("R1.2 EP8 eager/replay/fallback: all 8 ranks passed")' "/state/$run"
