#!/usr/bin/env bash
# This file becomes install.sh in the small, source-only offline update.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
[[ $# -ge 1 && $# -le 2 ]] || { echo 'Usage: bash install.sh EXISTING_DEPLOY [--restart]' >&2; exit 2; }
target=$(realpath -e -- "$1")
restart=false
if [[ $# == 2 ]]; then
  [[ "$2" == --restart ]] || { echo 'Only --restart is supported' >&2; exit 2; }
  restart=true
fi
[[ -f "$target/deployment.env" && -w "$target" ]] || { echo 'Target must be an existing deploy directory owned by the operator' >&2; exit 2; }
(cd "$root" && sha256sum --quiet -c SHA256SUMS)
image=$(sudo -n docker image inspect --format '{{.Id}}' deepseek-v4.1-flash-a100:20260913-r1)
allowed=false
while IFS= read -r row; do [[ "$row" != "$image" ]] || allowed=true; done < "$root/deploy/manifests/allowed-image-ids.txt"
[[ "$allowed" == true ]] || { echo 'The installed image is not the pinned offline runtime' >&2; exit 2; }
umask 077
mkdir -p "$target/runtime"
exec 8>"$target/runtime/update-install.lock"
flock -n 8 || { echo 'Another source update is active' >&2; exit 2; }
run_manager() {
  sudo -n docker run --rm --pull never --network none --user "$(id -u):$(id -g)" \
    --entrypoint /usr/bin/python3 -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$root:/update:ro" -v "$target:/target" "$image" \
    /update/deploy/scripts/update_manager.py --target /target "$@"
}
# Verify model implementation compatibility before stopping a running service.
sudo -n docker run --rm --pull never --network none --user "$(id -u):$(id -g)" \
  --entrypoint /usr/bin/python3 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$root/deploy:/deploy:ro" "$image" /deploy/scripts/performance.py \
  --host-deploy "$target" --mhc 1 --indexer 1 --moe-align 1 --nccl auto >/dev/null
if [[ "$restart" == true ]]; then bash "$target/stop.sh"; fi
if ! run_manager --apply /update/deploy; then
  if [[ "$restart" == true ]]; then bash "$target/start.sh" || true; fi
  exit 1
fi
if [[ "$restart" == true ]]; then
  if ! bash "$target/start.sh"; then
    echo 'Updated startup failed. Restoring the backed-up source and restarting the previous deployment.' >&2
    # Even an early configuration/path error may prevent stop.sh from sourcing
    # common.sh. Restore the prior files in that case too; its start.sh owns
    # the final cleanup and restart of any remaining project containers.
    if ! bash "$target/stop.sh"; then
      echo 'Updated stop command failed; continuing source rollback and the previous startup procedure.' >&2
    fi
    run_manager --rollback
    bash "$target/start.sh"
    exit 1
  fi
else
  printf 'Source updated. Start using it with: bash %q/restart.sh\n' "$target"
fi
