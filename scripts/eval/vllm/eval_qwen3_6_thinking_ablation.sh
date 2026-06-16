#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  eval_qwen3_6_thinking_ablation.sh [infer|score|all]

Runs the Qwen thinking-mode ablation for:
  - qwen3_6_27b on agnostic_m,order_m,order_f
  - qwen3_6_35b_a3b on agnostic_m,order_m,order_f

Assumes local vLLM OpenAI-compatible servers are already running:
  - qwen3_6_27b:      http://127.0.0.1:8906/v1
  - qwen3_6_35b_a3b:  http://127.0.0.1:8905/v1

Common overrides:
  TARGET_MODELS=27b|35b|both
  TRACKS=agnostic_m,order_m,order_f
  CONCURRENCY=64
  MAX_SAMPLES=100
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_ROOT}"

MODE="${1:-all}"
case "${MODE}" in
  infer|score|all) ;;
  *) echo "Unsupported mode: ${MODE}" >&2; usage; exit 1 ;;
esac

TRACKS="${TRACKS:-agnostic_m,order_m,order_f}"
PROMPT_MODE="${PROMPT_MODE:-direct}"
PROMPT_VARIANT_INDICES="${PROMPT_VARIANT_INDICES:-all}"
PROMPT_VARIANT_SAMPLE_SIZE="${PROMPT_VARIANT_SAMPLE_SIZE:-3}"
PROMPT_VARIANT_SAMPLING_SEED="${PROMPT_VARIANT_SAMPLING_SEED:-0}"
SCORE_PROMPT_VARIANT_SAMPLE_SIZE="${SCORE_PROMPT_VARIANT_SAMPLE_SIZE:-${PROMPT_VARIANT_SAMPLE_SIZE}}"
SCORE_PROMPT_VARIANT_SAMPLING_SEED="${SCORE_PROMPT_VARIANT_SAMPLING_SEED:-${PROMPT_VARIANT_SAMPLING_SEED}}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_INPUT_CHARS="${MAX_INPUT_CHARS:-0}"
MAX_TOKENS="${MAX_TOKENS:-0}"
CONCURRENCY="${CONCURRENCY:-64}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
RESUME="${RESUME:-true}"
API_KEY="${API_KEY:-EMPTY}"
ENABLE_THINKING="true"
TARGET_MODELS="${TARGET_MODELS:-both}"

run_one() {
  local model="$1"
  local model_slug="$2"
  local base_url="$3"

  echo "[ablation] model=${model} model_slug=${model_slug} tracks=${TRACKS} enable_thinking=true mode=${MODE}"
  MODEL="${model}" \
  MODEL_SLUG="${model_slug}" \
  BASE_URL="${base_url}" \
  API_KEY="${API_KEY}" \
  TRACKS="${TRACKS}" \
  EVAL_ROOT="${EVAL_ROOT:-${BENCHMARK_ROOT:-data/benchmark_v3}}" \
  EVALUATION_ROOT="${EVALUATION_ROOT:-${OUTPUT_ROOT:-data/evaluation_v2}}" \
  PROMPT_MODE="${PROMPT_MODE}" \
  PROMPT_VARIANT_INDICES="${PROMPT_VARIANT_INDICES}" \
  PROMPT_VARIANT_SAMPLE_SIZE="${PROMPT_VARIANT_SAMPLE_SIZE}" \
  PROMPT_VARIANT_SAMPLING_SEED="${PROMPT_VARIANT_SAMPLING_SEED}" \
  SCORE_PROMPT_VARIANT_SAMPLE_SIZE="${SCORE_PROMPT_VARIANT_SAMPLE_SIZE}" \
  SCORE_PROMPT_VARIANT_SAMPLING_SEED="${SCORE_PROMPT_VARIANT_SAMPLING_SEED}" \
  MAX_SAMPLES="${MAX_SAMPLES}" \
  MAX_INPUT_CHARS="${MAX_INPUT_CHARS}" \
  MAX_TOKENS="${MAX_TOKENS}" \
  CONCURRENCY="${CONCURRENCY}" \
  PROGRESS_EVERY="${PROGRESS_EVERY}" \
  RESUME="${RESUME}" \
  ENABLE_THINKING="${ENABLE_THINKING}" \
    bash "${RELEASE_ROOT}/scripts/eval/run_model_eval.sh" "${MODE}"
}

case "${TARGET_MODELS}" in
  27b)
    run_one "qwen3_6_27b" "qwen3_6_27b_thinking" "http://127.0.0.1:8906/v1"
    ;;
  35b)
    run_one "qwen3_6_35b_a3b" "qwen3_6_35b_a3b_thinking" "http://127.0.0.1:8905/v1"
    ;;
  both)
    run_one "qwen3_6_27b" "qwen3_6_27b_thinking" "http://127.0.0.1:8906/v1"
    run_one "qwen3_6_35b_a3b" "qwen3_6_35b_a3b_thinking" "http://127.0.0.1:8905/v1"
    ;;
  *)
    echo "Unsupported TARGET_MODELS value: ${TARGET_MODELS}" >&2
    echo "Use 27b, 35b, or both." >&2
    exit 1
    ;;
esac
