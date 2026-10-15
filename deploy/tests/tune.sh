#!/usr/bin/env bash
set -euo pipefail
# R1.9 fixes all requested serving parameters. Benchmark without retuning them.
printf '%s\n' 'R1.9 is fixed: context=512000, batch=1024, global C64, DSpark k5, GPU=0.92.'
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/performance-benchmark.sh" "$@"
