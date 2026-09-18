#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
source "$DEPLOY_DIR/scripts/storage.sh"
require_operator
for tool in sudo docker nvidia-smi nvidia-container-cli bash tar gzip sha256sum flock realpath od sort ss stat df awk sed readlink pgrep id; do
  command -v "$tool" >/dev/null || fail "Missing host prerequisite: $tool; see host-deps/README.zh-CN.md"
done
docker info >/dev/null
nvidia-container-cli --version > "$STATE_DIR/toolkit-version.txt"
storage_preflight
if ! own_container "$FRONTEND_NAME"; then
  [[ -z $(ss -H -ltn "sport = :$API_PORT") ]] || fail "API port $API_PORT is already occupied"
fi
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,mig.mode.current --format=csv,noheader,nounits > "$STATE_DIR/gpus.csv"
[[ $(wc -l < "$STATE_DIR/gpus.csv") -eq 8 ]] || fail 'Exactly eight target GPUs are required'
while IFS=, read -r index name uuid memory driver mig; do
  [[ "$name" == *A100-SXM4-80GB* ]] || fail 'This release is fixed to A100-SXM4-80GB'
  [[ "$mig" == *Disabled* ]] || fail 'MIG must be disabled for this TP4 DP2 EP8 deployment'
  driver=${driver// /}
  [[ $(printf '%s\n%s\n' 580.126.20 "$driver" | sort -V | head -n1) == 580.126.20 ]] || fail 'Driver is below the pinned CUDA toolchain requirement'
done < "$STATE_DIR/gpus.csv"
nvidia-smi topo -m > "$STATE_DIR/gpu-topology.txt"
grep -q NV12 "$STATE_DIR/gpu-topology.txt" || fail 'Expected NVSwitch NV12 topology was not detected'
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
# DP2 has two TP-sharded table replicas (~378 GiB combined), before loading
# temporaries. This guard is a floor for this 2 TiB target, not a peak bound.
((available_kib >= 512*1024*1024)) || fail 'R1.2 DP2 Engram loading budget requires at least 512 GiB currently available host RAM'
if ! own_container "$ENGINE_NAME"; then
  active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z "$active" ]] || fail 'Target GPUs are occupied; release the existing inference service in your maintenance window'
fi
{
  date -Is
  printf 'operator_uid=%s operator_gid=%s\n' "$(id -u)" "$(id -g)"
  uname -r
  docker version --format 'DockerClient={{.Client.Version}} DockerServer={{.Server.Version}}'
  docker info --format 'DockerRootDir={{.DockerRootDir}} Driver={{.Driver}}'
} > "$STATE_DIR/host-operator-profile.txt"
printf 'Target hardware preflight passed.\n'
