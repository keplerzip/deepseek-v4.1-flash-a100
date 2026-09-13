#!/usr/bin/env bash
# Read-only supplement. No service restart, package install, GPU allocation or network.
set -uo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run: sudo bash collect_target_privileged.sh [output.txt]' >&2; exit 2; }
OUT=${1:-dsv41-target-privileged.txt}
umask 077
run() {
  printf '\n[%s]\n' "$*"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" 2>&1
    printf 'exit_code=%s\n' "$?"
  else
    printf 'NOT_INSTALLED\n'
  fi
}
{
  printf 'DeepSeek-V4.1-Flash privileged target supplement; read-only, no environment/secrets collected\n'
  run date -Is
  run uname -r
  run docker version --format 'Client={{.Client.Version}} Server={{.Server.Version}} ServerAPI={{.Server.APIVersion}}'
  run docker info --format 'DockerRootDir={{.DockerRootDir}} Driver={{.Driver}} DriverStatus={{json .DriverStatus}} DefaultRuntime={{.DefaultRuntime}} Runtimes={{json .Runtimes}} CgroupDriver={{.CgroupDriver}} CgroupVersion={{.CgroupVersion}}'
  run docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
  docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)
  for path in / /ai /opt /var/lib/docker /var/lib/containerd "$docker_root"; do
    [[ -n "$path" && -e "$path" ]] && run df -B1 -- "$path"
  done
  run containerd --version
  printf '\n[containerd process and effective storage configuration; filtered]\n'
  if command -v containerd >/dev/null 2>&1; then
    containerd config dump 2>/dev/null | awk '/^(root|state|version|imports)[[:space:]]*=/ {print}'
  fi
  for name in containerd dockerd; do
    for pid in $(pgrep -x "$name" 2>/dev/null || true); do
      printf '%s pid=%s ' "$name" "$pid"
      tr '\0' '\n' < "/proc/$pid/cmdline" | awk '
        next_value {print; next_value=0; next}
        /^--?(config|root|state|data-root|containerd)(=|$)/ {print; if ($0 !~ /=/) next_value=1}
        /^-c$/ {print; next_value=1}'
    done
  done
  run dpkg-query -W '-f=${binary:Package}\t${Version}\t${db:Status-Abbrev}\t${Architecture}\n' 'nvidia*' 'libnvidia*' 'libcuda*' 'docker*' 'containerd*' 'runc*' 'linux-image*' 'linux-headers*' 'linux-modules*' 'numactl' 'libnuma1' 'libc6' 'libseccomp2' 'dkms' 'build-essential' 'gcc' 'gcc-11' 'make' 'kmod' 'libelf-dev' 'pkg-config'
  run cat /proc/driver/nvidia/version
  run modinfo -F filename nvidia
  run modinfo -F version nvidia
  run modinfo -F license nvidia
  run systemctl is-active nvidia-fabricmanager
  run systemctl show nvidia-fabricmanager --property=FragmentPath,ActiveState,SubState
  printf '\n[driver installation markers; presence only]\n'
  for path in /var/log/nvidia-installer.log /usr/bin/nvidia-uninstall /usr/bin/nvidia-installer; do
    [[ ! -e "$path" ]] || ls -ld -- "$path"
  done
  printf '\n[NUMA memory and distances]\n'
  for path in /sys/devices/system/node/node*/meminfo /sys/devices/system/node/node*/distance; do
    [[ ! -r "$path" ]] || run cat "$path"
  done
  printf '\n[Fabric state]\n'
  nvidia-smi -q 2>/dev/null | awk '/^GPU / {print} /^[[:space:]]+Fabric$/ {n=6} n>0 {print; n--}'
} > "$OUT"
printf 'Saved: %s\n' "$OUT"
