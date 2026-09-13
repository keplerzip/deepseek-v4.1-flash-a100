#!/usr/bin/env bash
# Sourced after common.sh. Host shell only; runs before importing any image.
# No deletions, Docker configuration changes or service restarts.

containerd_storage_roots() {
  local pid exe root config arg dump i found=0
  local -a argv=()
  if [[ -n "$CONTAINERD_ROOT_DIR" ]]; then
    realpath -e -- "$CONTAINERD_ROOT_DIR"
    return
  fi
  for pid in $(pgrep -x containerd || true); do
    [[ -r "/proc/$pid/cmdline" ]] || continue
    mapfile -d '' -t argv < "/proc/$pid/cmdline"
    root=; config=
    for ((i=1;i<${#argv[@]};i++)); do
      arg=${argv[i]}
      case "$arg" in
        --root=*) root=${arg#*=} ;;
        --root|-r) i=$((i+1)); root=${argv[i]:-} ;;
        --config=*) config=${arg#*=} ;;
        --config|-c) i=$((i+1)); config=${argv[i]:-} ;;
      esac
    done
    if [[ -z "$root" ]]; then
      exe=$(readlink -e "/proc/$pid/exe") || return 1
      if [[ -n "$config" ]]; then
        dump=$("$exe" --config "$config" config dump 2>/dev/null) || return 1
      else
        dump=$("$exe" config dump 2>/dev/null) || return 1
      fi
      root=$(printf '%s\n' "$dump" | sed -nE "s/^root[[:space:]]*=[[:space:]]*['\"]([^'\"]+)['\"][[:space:]]*$/\1/p")
    fi
    [[ "$root" == /* && -d "$root" ]] || return 1
    realpath -e -- "$root" || return 1
    found=1
  done
  ((found == 1))
}

storage_preflight() {
  local docker_root driver_status roots path device free need extra label
  local image_budget=0 failed=0
  local gib=$((1024*1024*1024))
  local -A growth=() reserve=() samples=() labels=()
  docker_root=$(docker info --format '{{.DockerRootDir}}')
  docker_root=$(realpath -e -- "$docker_root") || fail 'Cannot resolve Docker Root Dir'
  driver_status=$(docker info --format '{{json .DriverStatus}}')
  roots=$docker_root
  if [[ "$driver_status" == *io.containerd.snapshotter* ]]; then
    roots=$(containerd_storage_roots) || fail 'Cannot determine active containerd storage. Set its verified CONTAINERD_ROOT_DIR in deployment.env using the existing host inventory; this does not alter Docker configuration.'
  fi
  if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    # Conservative first import: 40 GiB extracted layers + 2 copies of OCI blobs
    # + 8 GiB scratch. Existing shared layers are deliberately not subtracted.
    image_budget=$((40*gib + 2*$(stat -c %s "$DEPLOY_DIR/images/runtime-image.tar") + 8*gib))
  fi
  # Model data is already copied. Group future allocations by actual filesystem,
  # so deploy/cache and image layers on one filesystem must fit together.
  while IFS=$'\t' read -r label path extra need; do
    [[ -d "$path" ]] || fail "Storage path does not exist: $path"
    path=$(realpath -e -- "$path")
    [[ "$path" != *$'\t'* && "$path" != *$'\n'* ]] || fail 'Storage paths cannot contain tabs/newlines'
    device=$(stat -c %d -- "$path")
    growth[$device]=$(( ${growth[$device]:-0} + extra ))
    (( need <= ${reserve[$device]:-0} )) || reserve[$device]=$need
    samples[$device]=$path
    labels[$device]="${labels[$device]:-}${label}:${path} "
  done < <(
    printf 'model\t%s\t0\t%s\n' "$MODEL_DIR" "$((20*gib))"
    printf 'state\t%s\t%s\t%s\n' "$STATE_DIR" "$((64*gib))" "$((20*gib))"
    printf 'docker-metadata\t%s\t%s\t%s\n' "$docker_root" "$((4*gib))" "$((50*gib))"
    while IFS= read -r path; do
      printf 'image-store\t%s\t%s\t%s\n' "$path" "$image_budget" "$((50*gib))"
    done <<< "$roots"
  )
  {
    printf 'DeepSeek-V4.1-Flash storage preflight; byte budgets, no cleanup\n'
    printf 'time=%s\n' "$(date -Is)"
    printf 'driver_status=%s\n' "$driver_status"
    printf 'image_import_budget_bytes=%s\n' "$image_budget"
    for device in "${!samples[@]}"; do
      path=${samples[$device]}
      free=$(df -B1 --output=avail -- "$path" | awk 'NR==2 {print $1}')
      [[ "$free" =~ ^[0-9]+$ ]] || fail 'Cannot read filesystem available bytes'
      need=$(( ${growth[$device]} + ${reserve[$device]} ))
      printf 'filesystem=%s available_bytes=%s growth_bytes=%s reserve_bytes=%s required_bytes=%s paths=%s\n' \
        "$device" "$free" "${growth[$device]}" "${reserve[$device]}" "$need" "${labels[$device]}"
      if ((free < need)); then
        printf 'FAIL: need %s more bytes on filesystem containing %s\n' "$((need-free))" "$path"
        failed=1
      fi
    done
  } > "$STATE_DIR/storage-preflight.txt"
  cat "$STATE_DIR/storage-preflight.txt"
  ((failed == 0)) || fail 'Insufficient storage before image import/start. Keep service files together and free the reported additional bytes on the affected filesystem. The service filesystem needs 64 GiB for runtime/cache plus 20 GiB free reserve; existing Docker storage is checked separately.'
}
