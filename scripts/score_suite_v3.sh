#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  score_suite_v3.sh --model-dirname NAME [options]

Options:
  --track-family <core_rule|semantic_extension|all>  Default: all
  --predictions-root <path>                          Default: data/evaluation_v3
  --model-dirname <name>                             Required
  --predictions-filename <name>                      Default: predictions.jsonl
  --score-dirname <name>                             Default: score
  --rs-at-k <int>                                    Default: 3
  --write-csv                                        Write CSV reports
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RELEASE_ROOT/.." && pwd)"

TRACK_FAMILY="all"
PREDICTIONS_ROOT="data/evaluation_v3"
MODEL_DIRNAME=""
PREDICTIONS_FILENAME="predictions.jsonl"
SCORE_DIRNAME="score"
RS_AT_K="3"
WRITE_CSV=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track-family) TRACK_FAMILY="$2"; shift 2 ;;
    --predictions-root) PREDICTIONS_ROOT="$2"; shift 2 ;;
    --model-dirname) MODEL_DIRNAME="$2"; shift 2 ;;
    --predictions-filename) PREDICTIONS_FILENAME="$2"; shift 2 ;;
    --score-dirname) SCORE_DIRNAME="$2"; shift 2 ;;
    --rs-at-k) RS_AT_K="$2"; shift 2 ;;
    --write-csv) WRITE_CSV=(--write-csv); shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL_DIRNAME" ]]; then
  usage >&2
  exit 1
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
  predictions_path="$REPO_ROOT/$PREDICTIONS_ROOT/$track/$MODEL_DIRNAME/$PREDICTIONS_FILENAME"
  output_dir="$REPO_ROOT/$PREDICTIONS_ROOT/$track/$MODEL_DIRNAME/$SCORE_DIRNAME"
  if [[ ! -f "$predictions_path" ]]; then
    echo "Missing predictions for track=$track: $predictions_path" >&2
    exit 1
  fi
  echo "[run] score track=$track predictions=$predictions_path"
  "$RELEASE_ROOT/scripts/score_predictions_v3.sh" \
    --predictions-path "$predictions_path" \
    --output-dir "$output_dir" \
    --rs-at-k "$RS_AT_K" \
    "${WRITE_CSV[@]}"
done

