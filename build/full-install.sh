#!/usr/bin/env bash
# install.sh in the complete image-inclusive, weight-free delivery.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
target=/ai/services/deepseek-v4.1-flash-a100/deploy
restart=false
target_set=false
for argument in "$@"; do
  case "$argument" in
    --restart) restart=true ;;
    --help|-h)
      printf '%s\n' 'Usage: bash install.sh [TARGET_DEPLOY] [--restart]' \
        'Default: /ai/services/deepseek-v4.1-flash-a100/deploy' \
        'Existing deployment: transactional source update; preserves deployment.env model path and runtime caches.' \
        'Fresh deployment: installs only operator-owned project files and the pinned Docker image. Place the unchanged model snapshot beside deploy.' \
        '--restart applies/restarts an existing deployment, or runs deploy.sh for a fresh installation.'
      exit 0 ;;
    --*) printf 'Unsupported option: %s\n' "$argument" >&2; exit 2 ;;
    *) [[ "$target_set" == false ]] || { echo 'Only one target directory is supported' >&2; exit 2; }
       target=$argument; target_set=true ;;
  esac
done
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $(id -u) != 0 ]] || fail 'Extract and install as the ordinary deployment operator, not root'
[[ ! -L "$target" ]] || fail 'Target deploy directory cannot be a symlink'
target=$(realpath -m -- "$target")
[[ "$target" != "$root/deploy" ]] || fail 'Extract this package outside the target deploy directory before installing'
[[ -f "$root/SHA256SUMS" && -f "$root/update-entry.sh" ]] || fail 'Run install.sh from the extracted complete delivery'
umask 077
verified=false
verify_payload() {
  if [[ "$verified" == false ]]; then
    (cd "$root" && sha256sum --strict --quiet -c SHA256SUMS)
    verified=true
  fi
}
image_tag=deepseek-v4.1-flash-a100:20260913-r1
image_archive="$root/deploy/images/runtime-image.tar"
image_id=''
validate_image() {
  image_id=$(sudo -n docker image inspect --format '{{.Id}}' "$image_tag")
  local allowed=false row
  while IFS= read -r row; do [[ "$row" != "$image_id" ]] || allowed=true; done < "$root/deploy/manifests/allowed-image-ids.txt"
  [[ "$allowed" == true ]] || fail 'The local runtime tag is occupied by an image outside this pinned delivery'
}
ensure_image() {
  if ! sudo -n docker image inspect "$image_tag" >/dev/null 2>&1; then
    verify_payload
    sudo -n docker load -i "$image_archive"
  fi
  validate_image
}
stage=''
image_temporary=''
cleanup() {
  [[ -z "$stage" ]] || rm -rf -- "$stage"
  [[ -z "$image_temporary" ]] || rm -f -- "$image_temporary"
}
trap cleanup EXIT

if [[ -f "$target/deployment.env" ]]; then
  [[ -d "$target" && -O "$target" && -w "$target" ]] || fail 'Existing deploy directory must be owned and writable by the operator'
  ensure_image
  # Older operators may have removed this redistributable archive to free
  # space. Restore only this fixed artifact; do not copy state or settings.
  [[ ! -L "$target/images" && ! -L "$target/images/runtime-image.tar" ]] || fail 'Target image archive cannot be a symlink'
  expected_image_sha=$(awk '$2 == "images/runtime-image.tar" {print $1}' "$root/deploy/manifests/deploy.sha256")
  [[ "$expected_image_sha" =~ ^[a-f0-9]{64}$ ]] || fail 'Pinned image checksum is absent from deploy.sha256'
  target_image_sha=''
  if [[ -f "$target/images/runtime-image.tar" ]]; then
    target_image_sha=$(sha256sum "$target/images/runtime-image.tar" | cut -d ' ' -f1)
  fi
  if [[ "$target_image_sha" != "$expected_image_sha" ]]; then
    verify_payload
    mkdir -p "$target/images"
    image_temporary=$(mktemp "$target/images/.runtime-image.XXXXXX")
    cp --reflink=auto -- "$image_archive" "$image_temporary"
    mv -- "$image_temporary" "$target/images/runtime-image.tar"
    image_temporary=''
  fi
  arguments=("$target")
  [[ "$restart" != true ]] || arguments+=(--restart)
  bash "$root/update-entry.sh" "${arguments[@]}"
  exit 0
fi

if [[ -e "$target" ]]; then
  [[ -d "$target" && -O "$target" && -w "$target" ]] || fail 'Fresh target must be an operator-owned writable directory'
  [[ -z $(find "$target" -mindepth 1 -maxdepth 1 -print -quit) ]] || fail 'Refusing to overwrite a nonempty directory without deployment.env'
fi
verify_payload
ensure_image
parent=$(dirname -- "$target")
mkdir -p "$parent"
exec 7>"$parent/.$(basename -- "$target").full-install.lock"
flock -n 7 || fail 'Another full installation is active'
stage=$(mktemp -d "$parent/.dsv41-full-install.XXXXXX")
cp -a -- "$root/deploy/." "$stage/"
chmod u+rwx "$stage"
# Check again after copying before publishing. deployment.env stays editable.
(cd "$stage" && sha256sum --strict --quiet -c manifests/deploy.sha256)
[[ ! -e "$target" ]] || rmdir -- "$target"
mv -- "$stage" "$target"
stage=''
if [[ "$restart" == true ]]; then
  bash "$target/deploy.sh"
else
  printf 'Installed project files and pinned runtime image. Keep the fixed model snapshot beside deploy, or set MODEL_DIR in deployment.env.\nStart with: bash %q/deploy.sh\n' "$target"
fi
