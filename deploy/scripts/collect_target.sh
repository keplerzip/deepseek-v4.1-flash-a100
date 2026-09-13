#!/usr/bin/env bash
# Read-only; run on the isolated target and copy the resulting text report back.
set -uo pipefail
OUT=${1:-dsv41-target-profile.txt}
umask 077
run() {
  printf '\n[%s]\n' "$*"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" 2>&1
    local rc=$?
    printf 'exit_code=%s\n' "$rc"
  else
    printf 'NOT_INSTALLED\n'
  fi
}
{
  printf 'DeepSeek-V4.1-Flash target profile; read-only collection\n'
  run date -Is
  run uname -a
  run cat /etc/os-release
  run lscpu
  run numactl --hardware
  run free -b
  run df -B1 / /ai
  run nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,driver_version,mig.mode.current --format=csv,noheader
  run nvidia-smi topo -m
  run nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
  run docker version
  run docker info --format '{{json .Runtimes}}'
  run docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
  run nvidia-container-cli --version
  run dpkg-query -W 'nvidia-container*' 'libnvidia-container*' 'docker*' 'containerd*' 'linux-image*' 'linux-headers*'
  printf '\n[memlock]\n'
  ulimit -l
} > "$OUT"
printf 'Saved: %s\n' "$OUT"
