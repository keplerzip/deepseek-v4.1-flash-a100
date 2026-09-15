#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
bash "$root/stop.sh"
exec bash "$root/start.sh" "$@"
