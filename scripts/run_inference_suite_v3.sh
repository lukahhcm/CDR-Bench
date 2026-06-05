#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_inference_suite_v3.sh --model MODEL [options]

Options:
  --track-family <core_rule|semantic_extension|all>  Default: all
  --benchmark-root <path>                            Default: data/benchmark_v3
  --output-root <path>                               Default: data/evaluation_v3
  --model <name>                                     Required
  --model-dirname <name>                             Default: sanitized model name
  --base-url <url>                                   Optional
  --api-key <key>                                    Optional
  --prompt-variant-indices <csv|all>                 Default: 0
  --max-samples <int>                                Default: 0
  --concurrency <int>                                Default: 5
  --resume                                           Resume existing outputs
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RELEASE_ROOT/.." && pwd)"

TRACK_FAMILY="all"
BENCHMARK_ROOT="data/benchmark_v3"
OUTPUT_ROOT="data/evaluation_v3"
MODEL=""
MODEL_DIRNAME=""
BASE_URL=""
API_KEY=""
PROMPT_VARIANT_INDICES="0"
MAX_SAMPLES="0"
CONCURRENCY="5"
RESUME_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track-family) TRACK_FAMILY="$2"; shift 2 ;;
    --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --model-dirname) MODEL_DIRNAME="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --prompt-variant-indices) PROMPT_VARIANT_INDICES="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --resume) RESUME_ARGS=(--resume); shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  usage >&2
  exit 1
fi

if [[ -z "$MODEL_DIRNAME" ]]; then
  MODEL_DIRNAME="$(echo "$MODEL" | tr '/:.' '___')"
fi

case "$TRACK_FAMILY" in
  core_rule)
    TRACKS=(atomic_m atomic_f agnostic_m order_m order_f)
    ;;
  semantic_extension)
    TRACKS=(semantic_pii_atomic semantic_pii_compositional semantic_hallu_atomic semantic_hallu_compositional)
    ;;
  all)
    TRACKS=(atomic_m atomic_f agnostic_m order_m order_f semantic_pii_atomic semantic_pii_compositional semantic_hallu_atomic semantic_hallu_compositional)
    ;;
  *)
    echo "Invalid --track-family: $TRACK_FAMILY" >&2
    exit 1
    ;;
esac

for track in "${TRACKS[@]}"; do
  benchmark_path="$REPO_ROOT/$BENCHMARK_ROOT/tracks/$track.jsonl"
  output_path="$REPO_ROOT/$OUTPUT_ROOT/$track/$MODEL_DIRNAME/predictions.jsonl"
  cmd=(
    "$RELEASE_ROOT/scripts/run_inference_v3.sh"
    --benchmark-path "$benchmark_path"
    --output-path "$output_path"
    --model "$MODEL"
    --prompt-variant-indices "$PROMPT_VARIANT_INDICES"
    --max-samples "$MAX_SAMPLES"
    --concurrency "$CONCURRENCY"
    "${RESUME_ARGS[@]}"
  )
  if [[ -n "$BASE_URL" ]]; then cmd+=(--base-url "$BASE_URL"); fi
  if [[ -n "$API_KEY" ]]; then cmd+=(--api-key "$API_KEY"); fi
  echo "[run] inference track=$track output=$output_path"
  "${cmd[@]}"
done

