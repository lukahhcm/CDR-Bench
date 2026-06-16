#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$RELEASE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$#" -eq 0 ]]; then
  set -- "$RELEASE_ROOT/data/benchmark_v3/benchmark_v3_all.jsonl"
fi

python -m cdrbench_v3.validate_benchmark "$@"
