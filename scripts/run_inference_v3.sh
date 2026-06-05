#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RELEASE_ROOT/.." && pwd)"

export PYTHONPATH="$RELEASE_ROOT/src:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m cdrbench_v3.run_inference_v3 "$@"

