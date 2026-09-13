#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
STATE_DIR="$DEPLOY_DIR/runtime"
MODEL_DIR=""
BIND_HOST=127.0.0.1
API_PORT=8005
START_TIMEOUT_SECONDS=7200
NEWAPI_BASE_URL=""
CONTAINERD_ROOT_DIR=""
while IFS='=' read -r key value; do
  [[ -z "$key" || "$key" == \#* ]] && continue
  case "$key" in
    MODEL_DIR|BIND_HOST|API_PORT|START_TIMEOUT_SECONDS|NEWAPI_BASE_URL|CONTAINERD_ROOT_DIR) printf -v "$key" '%s' "$value" ;;
    *) printf 'Unknown deployment setting: %s\n' "$key" >&2; exit 2 ;;
  esac
done < "$DEPLOY_DIR/deployment.env"
# HF004: the requested local-only policy overrides the old wildcard setting.
BIND_HOST=127.0.0.1
[[ -n "$MODEL_DIR" ]] || MODEL_DIR="$DEPLOY_DIR/../DeepSeek-V4.1-Flash"
MODEL_DIR=$(realpath -e -- "$MODEL_DIR")
[[ "$API_PORT" =~ ^[0-9]+$ ]] && ((API_PORT>=1024 && API_PORT<=65535)) || { echo 'Invalid API_PORT' >&2; exit 2; }
[[ "$START_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || { echo 'Invalid timeout' >&2; exit 2; }
[[ "$BIND_HOST" =~ ^[0-9.]+$ ]] || { echo 'BIND_HOST must be an IPv4 address' >&2; exit 2; }
IMAGE_TAG=deepseek-v4.1-flash-a100:20260913-r1
ENGINE_NAME=deepseek-v4.1-flash-engine
FRONTEND_NAME=deepseek-v4.1-flash-api
NETWORK_NAME=deepseek-v4.1-flash-internal
umask 077
mkdir -p "$STATE_DIR" "$STATE_DIR/home" "$STATE_DIR/tmp"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
docker() { command sudo -n docker "$@"; }
require_operator() {
  [[ -w "$DEPLOY_DIR" && -O "$STATE_DIR" && -w "$STATE_DIR" ]] || fail 'Extract and run this deployment as its ordinary owner; deploy/runtime must be writable by that user.'
  command -v sudo >/dev/null && type -P docker >/dev/null || fail 'This deployment needs the existing Docker CLI and sudo -n docker permission.'
  docker info >/dev/null || fail 'The current user must be allowed to run sudo -n docker without a password.'
}
CONTAINER_USER_ARGS=(--user "$(id -u):$(id -g)" --workdir /state
  -e HOME=/state/home -e TMPDIR=/state/tmp
  -e VLLM_HOST_IP=127.0.0.1
  -e XDG_CACHE_HOME=/state/cache -e UV_CACHE_DIR=/state/cache/uv
  -e TRITON_CACHE_DIR=/state/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/state/cache/inductor
  -e TORCH_EXTENSIONS_DIR=/state/cache/torch-extensions -e CUDA_CACHE_PATH=/state/cache/cuda)
for dsv41_group in $(id -G); do CONTAINER_USER_ARGS+=(--group-add "$dsv41_group"); done

resolve_image() {
  RUNTIME_IMAGE=$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG")
  [[ "$RUNTIME_IMAGE" =~ ^sha256:[a-f0-9]{64}$ ]] || fail 'Invalid local image identity'
  rg_result=1
  while IFS= read -r allowed; do
    [[ "$allowed" != "$RUNTIME_IMAGE" ]] || rg_result=0
  done < "$DEPLOY_DIR/manifests/allowed-image-ids.txt"
  ((rg_result == 0)) || fail 'Local runtime image does not match this offline release'
}

image_python() {
  docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=2g \
    "${CONTAINER_USER_ARGS[@]}" \
    --entrypoint /usr/bin/python3 -e PYTHONDONTWRITEBYTECODE=1 -e HTTP_PROXY= -e HTTPS_PROXY= -e ALL_PROXY= \
    -v "$DEPLOY_DIR:/deploy:ro" -v "$STATE_DIR:/state" \
    -v "$MODEL_DIR:/models/DeepSeek-V4.1-Flash:ro" "$RUNTIME_IMAGE" "$@"
}

op() { image_python /deploy/scripts/offline_ops.py "$@"; }

own_container() {
  local name=$1
  [[ $(docker inspect --format '{{index .Config.Labels "dsv41.owner"}}' "$name" 2>/dev/null || true) == 'offline-delivery' ]]
}

remove_own() {
  local name=$1
  if docker inspect "$name" >/dev/null 2>&1; then
    own_container "$name" || fail "Container name is occupied by a different deployment: $name"
    docker stop -t 90 "$name" >/dev/null
    docker rm "$name" >/dev/null
  fi
}

engine_health() {
  docker exec "$ENGINE_NAME" /usr/bin/python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health",timeout=5).read()' >/dev/null 2>&1
}

frontend_health() {
  local check_host=$BIND_HOST
  [[ "$check_host" != 0.0.0.0 ]] || check_host=127.0.0.1
  [[ $(docker inspect --format '{{.State.Running}}' "$FRONTEND_NAME" 2>/dev/null || true) == true ]] || return 1
  docker exec "$FRONTEND_NAME" /usr/bin/python3 -c \
    'import sys,urllib.request;urllib.request.urlopen("http://"+sys.argv[1]+":"+sys.argv[2]+"/health",timeout=5).read()' \
    "$check_host" "$API_PORT" >/dev/null 2>&1
}

wait_engine() {
  local started=$SECONDS
  while ! engine_health; do
    [[ $(docker inspect --format '{{.State.Running}}' "$ENGINE_NAME") == true ]] || fail "Engine exited; inspect $STATE_DIR startup logs"
    ((SECONDS-started<START_TIMEOUT_SECONDS)) || fail 'Engine startup timed out'
    printf 'Waiting for eight-GPU engine: %s seconds\n' "$((SECONDS-started))"
    sleep 10
  done
}

make_secret() {
  if [[ ! -f "$STATE_DIR/api-key.txt" ]]; then
    { printf 'sk-local-'; od -An -N32 -tx1 /dev/urandom | tr -d ' \n'; printf '\n'; } > "$STATE_DIR/api-key.txt"
    chmod 600 "$STATE_DIR/api-key.txt"
  fi
  # Retain the legacy file for existing acceptance tools; the server is keyless.
  printf 'VLLM_API_KEY=\n' > "$STATE_DIR/engine-secret.env"
  chmod 600 "$STATE_DIR/engine-secret.env"
}

launch_engine() {
  local mode=$1 run=$2
  local args_file="$STATE_DIR/$run/argv.nul"
  op argv "/state/$run/resolved.json" > "$args_file"
  local -a arguments
  mapfile -d '' -t arguments < "$args_file"
  docker run -d --pull never --name "$ENGINE_NAME" --label dsv41.owner=offline-delivery \
    "${CONTAINER_USER_ARGS[@]}" \
    --gpus all --ipc host --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
    --network "$mode" --env-file "$STATE_DIR/engine-secret.env" -e VLLM_API_KEY= \
    -e HTTP_PROXY= -e HTTPS_PROXY= -e ALL_PROXY= -e http_proxy= -e https_proxy= -e all_proxy= \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
    -e HF_HUB_DISABLE_TELEMETRY=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e TOKENIZERS_PARALLELISM=false \
    -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_DSPARK_FUSED_MARKOV=1 \
    -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_IB_DISABLE=1 \
    -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
    -e HF_HOME=/state/cache/hf -e XDG_CACHE_HOME=/state/cache \
    -e TRITON_CACHE_DIR=/state/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/state/cache/inductor \
    -e DSV41_OBSERVATION_DIR="/state/$run/observations" \
    -v "$MODEL_DIR:/models/DeepSeek-V4.1-Flash:ro" -v "$STATE_DIR:/state" \
    -v "$DEPLOY_DIR/tests:/offline-tests:ro" \
    --mount "type=bind,src=$DEPLOY_DIR/overrides/dsv41_guard.py,dst=/usr/local/lib/python3.12/dist-packages/dsv41_guard.py,readonly" \
    "$RUNTIME_IMAGE" "${arguments[@]}" >/dev/null
  printf '%s\n' "$run" > "$STATE_DIR/current-run.txt"
}
