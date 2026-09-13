#!/usr/bin/env bash
# Run on the connected build machine, never on the isolated target.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ $# == 1 ]] || { printf 'Usage: build/build-runtime.sh NEW_OUTPUT_DIRECTORY\n' >&2; exit 2; }
[[ ! -e "$1" ]] || { printf 'Output directory already exists; choose a new directory.\n' >&2; exit 2; }
mkdir -p -- "$1"
out=$(realpath -- "$1")
base=lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3
tag=deepseek-v4.1-flash-a100:20260913-r1
sudo -n docker pull "$base"
sudo -n docker build --pull=false --network=none -t "$tag" "$root/build/image"
sudo -n docker image inspect "$tag" > "$out/image-inspect.json"
sudo -n docker save "$tag" > "$out/runtime-image.tar.partial"
python3 "$root/build/verify_image.py" --archive "$out/runtime-image.tar.partial" \
  --inspect "$out/image-inspect.json" --output "$out/image-integrity.json"
mv -- "$out/runtime-image.tar.partial" "$out/runtime-image.tar"
printf 'Verified build artifacts: %s . A rebuilt image still requires target GPU acceptance.\n' "$out"
