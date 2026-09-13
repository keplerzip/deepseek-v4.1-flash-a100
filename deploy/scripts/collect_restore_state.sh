#!/usr/bin/env bash
# Small read-only follow-up for OCR-damaged fields; no service or package changes.
set -uo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run with sudo bash' >&2; exit 2; }
OUT=${1:-dsv41-restore-state.txt}
umask 077
{
  printf 'RESTORE_STATE_BEGIN\n'
  printf 'KERNEL='; uname -r
  printf 'LOADED_NVIDIA='; cat /sys/module/nvidia/version 2>/dev/null || true
  printf 'DKMS_STATUS_BEGIN\n'
  if command -v dkms >/dev/null; then dkms status 2>&1; fi
  printf 'DKMS_STATUS_END\n'
  printf 'MODULE_FILES_BEGIN\n'
  find "/lib/modules/$(uname -r)" -type f -name 'nvidia*.ko*' -print 2>/dev/null
  printf 'MODULE_FILES_END\n'
  printf 'MODULE_PATH='; modinfo -k "$(uname -r)" -F filename nvidia 2>&1 || true
  printf 'PACKAGES_BEGIN\n'
  for name in nvidia-dkms-580 nvidia-dkms-580-open nvidia-kernel-source-580 nvidia-kernel-common-580 nvidia-fabricmanager-580 nvidia-fabricmanager nvidia-driver-local-repo-ubuntu2204-580.159.04 dkms; do
    dpkg-query -W '-f=${binary:Package}|${Version}|${db:Status-Status}\n' "$name" 2>/dev/null || true
  done
  printf 'PACKAGES_END\n'
  printf 'INSTALLER_MARKERS_BEGIN\n'
  for path in /var/log/nvidia-installer.log /usr/bin/nvidia-uninstall /usr/bin/nvidia-installer; do
    [[ ! -e "$path" ]] || printf '%s\n' "$path"
  done
  printf 'INSTALLER_MARKERS_END\n'
  printf 'FABRIC_MANAGER_VERSION='; if command -v nv-fabricmanager >/dev/null; then nv-fabricmanager -v 2>&1; fi
  printf 'NEWAPI_IMAGE_ID='; docker inspect --format '{{.Image}}' new-api 2>/dev/null || true
  printf 'NEWAPI_IMAGE_TAG='; docker inspect --format '{{.Config.Image}}' new-api 2>/dev/null || true
  newapi_id=$(docker inspect --format '{{.Image}}' new-api 2>/dev/null || true)
  if [[ "$newapi_id" == sha256:* ]]; then
    docker image inspect --format 'NEWAPI_DIGESTS={{json .RepoDigests}}' "$newapi_id" 2>/dev/null || true
    docker image inspect --format 'NEWAPI_VERSION_LABEL={{index .Config.Labels "org.opencontainers.image.version"}}' "$newapi_id" 2>/dev/null || true
  fi
  printf 'RESTORE_STATE_END\n'
} > "$OUT"
printf 'Saved: %s\n' "$OUT"
cat -- "$OUT"
