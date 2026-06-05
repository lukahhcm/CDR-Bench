# CDR-Bench v3 Release Utilities

This folder contains the release-facing v3 utilities. It is intentionally
separate from the v2 research pipeline so the public package can expose a
stable schema and a unified scoring interface without rewriting existing
experiments.

Install minimal dependencies:

```bash
pip install -r release_v3/requirements.txt
```

## Benchmark Layout

Download the v3 JSONL files from Hugging Face:

```bash
./release_v3/scripts/build_benchmark_v3.sh
```

Default output:

```text
data/benchmark_v3/
  manifest.json
  benchmark_v3_all.jsonl
  benchmark_v3_all_full.jsonl
  tracks/
    atomic_m.jsonl
    atomic_f.jsonl
    agnostic_m.jsonl
    order_m.jsonl
    order_f.jsonl
    semantic_pii_atomic.jsonl
    semantic_pii_compositional.jsonl
    semantic_hallu_atomic.jsonl
    semantic_hallu_compositional.jsonl
  tracks_full/
    ...
```

By default, this downloads public data from
`huggingface.co/datasets/lukahh/CDR-Bench`. Maintainers can regenerate the data
from local v2 artifacts with:

```bash
./release_v3/scripts/build_benchmark_v3_from_local.sh
```

The release has two report families:

- `core_rule`: the five deterministic rule-based CDR-Bench tracks.
- `semantic_extension`: two real-scenario semantic extensions, PII redaction
  and hallucination processing.

## v3 Field Policy

The public `tracks/*.jsonl` files use a compact schema:

- `benchmark_track`: report-level track. Semantic splits are grouped into
  `semantic_pii` and `semantic_hallucination`.
- `benchmark_split`: `single`, `atomic`, or `compositional`.
- `source_track`: original file-level track, e.g. `semantic_pii_atomic`.
- `base_sample_id`: source-level sample id before atomic task expansion.
- `track_family`: `core_rule` or `semantic_extension`.
- `scoring_profile`: `text_refinement`, `structured_json`, or reserved
  `mixed_structured_text`.
- `output_format`: `tagged_text`, `json`, or reserved `json_and_tagged_text`.
- `reports_refinement_gain`: whether RG is meaningful for this row.

Public fields:

```text
instance_id, benchmark_track, benchmark_split, track_family, source_track,
base_sample_id, domain, source_record_id, source_benchmark, input_text,
input_length_chars, input_length_bucket, operator, operator_kind,
operator_sequence, semantic_operator, recipe_id, recipe_type, recipe_length,
order_family_id, order_slot, order_group_instance_id, group_success_rule,
reference_status, reference_text, reference_text_full_run, output_format,
scoring_profile, reports_refinement_gain, prompt_variants,
prompt_variant_count, difficulty_score, difficulty_label, pii_meta, hallu_meta
```

The `tracks_full/*.jsonl` files keep construction/debug metadata from the
original research pipeline for maintainers.

## Unified Metric Policy

Recipe Success (RS) is the common success metric:

- Rule-based text refinement: status match and normalized text exact match.
- Semantic PII / text correction: status match and normalized text exact match.
- Semantic JSON subtasks: canonical JSON exact match.
- Future mixed structured+text tasks: all required structured and text
  components must be correct.

Refinement Gain (RG) is reported only for text-output rows. It is not computed
for JSON-only detection, span extraction, or classification subtasks.

RS@K is supported whenever a prediction file contains multiple prompt variants.
Existing semantic appendix runs use a single prompt variant, so those results
should be described as RS@1 / prompt-0 RS.

## Semantic Sample Policy

The semantic extension follows the appendix experiment configuration:

- PII uses 500 base samples. Samples are eligible when they contain at least two
  semantic PII groups, then are stratified by the number of present groups with
  `sampling_seed=42`. The 500 compositional rows expand to 1,394 atomic rows
  because each base sample contributes one atomic row per present PII group.
- Hallucination uses the full available FAVA subset: 460 base samples. These
  produce 460 compositional rows and 1,840 atomic rows because each sample is
  expanded into detection, span extraction, type classification, and correction.

## Validation

```bash
./release_v3/scripts/validate_benchmark_v3.sh
```

or validate specific files:

```bash
./release_v3/scripts/validate_benchmark_v3.sh data/benchmark_v3/tracks/semantic_hallu_atomic.jsonl
```

## Unified Inference

The unified inference runner chooses the prompt/output contract from
`output_format`.

```bash
./release_v3/scripts/run_inference_v3.sh \
  --benchmark-path data/benchmark_v3/tracks/atomic_m.jsonl \
  --output-path data/evaluation_v3/atomic_m/MODEL/predictions.jsonl \
  --model MODEL_NAME \
  --prompt-variant-indices 0
```

For RS@3-style runs:

```bash
./release_v3/scripts/run_inference_v3.sh \
  --benchmark-path data/benchmark_v3/tracks/agnostic_m.jsonl \
  --output-path data/evaluation_v3/agnostic_m/MODEL/predictions.jsonl \
  --model MODEL_NAME \
  --prompt-variant-indices 0,1,2
```

The runner requires the original `cdrbench` package on `PYTHONPATH` for the
OpenAI-compatible backend.

## Unified Scoring

```bash
./release_v3/scripts/score_predictions_v3.sh \
  --predictions-path data/evaluation_v3/atomic_m/MODEL/predictions.jsonl \
  --output-dir data/evaluation_v3/atomic_m/MODEL/score \
  --rs-at-k 3 \
  --write-csv
```

The same scorer can be used for core rule tracks and semantic extension tracks.

Suite-level wrappers:

```bash
./release_v3/scripts/run_inference_suite_v3.sh \
  --track-family semantic_extension \
  --model MODEL_NAME \
  --prompt-variant-indices 0

./release_v3/scripts/score_suite_v3.sh \
  --track-family semantic_extension \
  --model-dirname MODEL_NAME \
  --rs-at-k 3
```
