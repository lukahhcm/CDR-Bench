from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cdrbench_v3.io import read_jsonl, write_json, write_jsonl
from cdrbench_v3.schema import (
    CORE_RULE_TRACKS,
    SEMANTIC_SOURCE_TRACKS,
    is_json_reference,
    order_v3_row,
    public_v3_row,
    validate_v3_row,
)

V3_SCHEMA_VERSION = "3.0.0"
SEMANTIC_SAMPLE_POLICY = {
    "semantic_pii": (
        "Appendix-compatible 500-base-sample subset: eligible samples require at least "
        "2 PII groups, max_input_chars=50000, stratified by number of present PII groups, "
        "sampling_seed=42. Atomic rows expand each base sample into present PII-group subtasks."
    ),
    "semantic_hallucination": (
        "Appendix-compatible full FAVA subset: all 460 available base samples after "
        "max_input_chars=50000 filtering. Atomic rows expand each base sample into four subtasks."
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rule_source_path(root: Path, track: str) -> Path:
    return root / "data" / "benchmark_v2" / track / f"{track}.jsonl"


def _semantic_source_path(root: Path, source_track: str) -> Path:
    return root / "real-scenario" / "data" / "benchmark_v2" / "semantic" / f"{source_track}.jsonl"


def _normalize_common(row: dict[str, Any], source_track: str) -> dict[str, Any]:
    out = dict(row)
    out["v3_schema_version"] = V3_SCHEMA_VERSION
    out["source_track"] = source_track
    out["source_track_label"] = row.get("benchmark_track_label") or source_track
    out["base_sample_id"] = row.get("group_id") or row.get("source_record_id") or row.get("instance_id")
    out.setdefault("reference_status", "KEEP")
    out.setdefault("filter_params_by_name", {})
    out.setdefault("threshold_meta", {})
    out.setdefault("prompt_variants", [])
    out.setdefault("prompt_variant_count", len(out.get("prompt_variants") or []))
    return out


def convert_rule_row(row: dict[str, Any], source_track: str) -> dict[str, Any]:
    label, role = CORE_RULE_TRACKS[source_track]
    out = _normalize_common(row, source_track)
    out["benchmark_track"] = source_track
    out["benchmark_track_label"] = row.get("benchmark_track_label") or label
    out["benchmark_split"] = "single"
    out["track_family"] = "core_rule"
    out["track_role"] = role
    out["source_benchmark"] = "cdrbench_rule_based"
    out["scoring_profile"] = "text_refinement"
    out["output_format"] = "tagged_text"
    out["reports_recipe_success"] = True
    out["reports_refinement_gain"] = True
    out["structured_metric_profile"] = None
    out["required_output_components"] = ["status", "clean_text"]
    out["evaluation_sample_policy"] = "Core deterministic rule-based benchmark release rows."
    out["v3_notes"] = "Core deterministic rule-based CDR-Bench track."
    return order_v3_row(out)


def convert_semantic_row(row: dict[str, Any], source_track: str) -> dict[str, Any]:
    spec = SEMANTIC_SOURCE_TRACKS[source_track]
    out = _normalize_common(row, source_track)
    out["benchmark_track"] = spec["benchmark_track"]
    out["benchmark_track_label"] = spec["benchmark_track_label"]
    out["benchmark_split"] = spec["benchmark_split"]
    out["track_family"] = "semantic_extension"
    out["track_role"] = "real_scenario_extension"

    json_ref = is_json_reference(row.get("reference_text"))
    out["scoring_profile"] = "structured_json" if json_ref else "text_refinement"
    out["output_format"] = "json" if json_ref else "tagged_text"
    out["reports_recipe_success"] = True
    out["reports_refinement_gain"] = not json_ref
    out["structured_metric_profile"] = spec["structured_metric_profile"] if json_ref else None
    out["required_output_components"] = ["json"] if json_ref else ["status", "clean_text"]
    out["evaluation_sample_policy"] = SEMANTIC_SAMPLE_POLICY[spec["benchmark_track"]]
    out["v3_notes"] = (
        "Semantic extension row. RS is canonical JSON exact match for JSON outputs "
        "and normalized text exact match for text outputs; RG is reported only for text outputs."
    )
    return order_v3_row(out)


def build_v3(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    tracks_dir = output_dir / "tracks"
    tracks_full_dir = output_dir / "tracks_full"
    all_rows: list[dict[str, Any]] = []
    all_full_rows: list[dict[str, Any]] = []
    manifest_tracks: dict[str, Any] = {}
    validation_errors: dict[str, list[str]] = defaultdict(list)

    for source_track in CORE_RULE_TRACKS:
        source_path = _rule_source_path(repo_root, source_track)
        rows = [convert_rule_row(row, source_track) for row in read_jsonl(source_path)]
        for index, row in enumerate(rows, start=1):
            errors = validate_v3_row(row)
            if errors:
                validation_errors[source_track].append(f"row {index}: {'; '.join(errors)}")
        public_rows = [public_v3_row(row) for row in rows]
        write_jsonl(tracks_dir / f"{source_track}.jsonl", public_rows)
        write_jsonl(tracks_full_dir / f"{source_track}.jsonl", rows)
        all_rows.extend(public_rows)
        all_full_rows.extend(rows)
        manifest_tracks[source_track] = {
            "track_family": "core_rule",
            "source_path": str(source_path),
            "release_path": f"tracks/{source_track}.jsonl",
            "full_release_path": f"tracks_full/{source_track}.jsonl",
            "num_instances": len(rows),
            "splits": {"single": len(rows)},
            "metric_policy": "RS/RS@k, RG, and OCS for order-sensitive groups.",
            "base_sample_count": len({row.get("base_sample_id") for row in rows}),
        }

    for source_track in SEMANTIC_SOURCE_TRACKS:
        source_path = _semantic_source_path(repo_root, source_track)
        rows = [convert_semantic_row(row, source_track) for row in read_jsonl(source_path)]
        release_name = source_track
        for index, row in enumerate(rows, start=1):
            errors = validate_v3_row(row)
            if errors:
                validation_errors[source_track].append(f"row {index}: {'; '.join(errors)}")
        public_rows = [public_v3_row(row) for row in rows]
        write_jsonl(tracks_dir / f"{release_name}.jsonl", public_rows)
        write_jsonl(tracks_full_dir / f"{release_name}.jsonl", rows)
        all_rows.extend(public_rows)
        all_full_rows.extend(rows)
        spec = SEMANTIC_SOURCE_TRACKS[source_track]
        manifest_tracks[source_track] = {
            "benchmark_track": spec["benchmark_track"],
            "benchmark_split": spec["benchmark_split"],
            "track_family": "semantic_extension",
            "source_path": str(source_path),
            "release_path": f"tracks/{release_name}.jsonl",
            "full_release_path": f"tracks_full/{release_name}.jsonl",
            "num_instances": len(rows),
            "metric_policy": "RS for all rows; RG only for text-output rows; structured diagnostics for JSON rows.",
            "evaluation_sample_policy": SEMANTIC_SAMPLE_POLICY[spec["benchmark_track"]],
            "base_sample_count": len({row.get("base_sample_id") for row in rows}),
        }

    write_jsonl(output_dir / "benchmark_v3_all.jsonl", all_rows)
    write_jsonl(output_dir / "benchmark_v3_all_full.jsonl", all_full_rows)

    family_counts = Counter(row["track_family"] for row in all_rows)
    track_counts = Counter(row["benchmark_track"] for row in all_rows)
    source_counts = Counter(row["source_track"] for row in all_rows)
    scoring_counts = Counter(row["scoring_profile"] for row in all_rows)
    rg_counts = Counter("rg" if row.get("reports_refinement_gain") else "no_rg" for row in all_rows)
    base_counts: dict[str, int] = {}
    for track in sorted(track_counts):
        base_counts[track] = len({row.get("base_sample_id") for row in all_rows if row.get("benchmark_track") == track})

    manifest = {
        "schema_version": V3_SCHEMA_VERSION,
        "description": (
            "CDR-Bench v3 release: five deterministic rule-based core tracks plus "
            "two semantic real-scenario extension tracks."
        ),
        "tracks": manifest_tracks,
        "public_schema_fields": [
            "instance_id",
            "benchmark_track",
            "benchmark_split",
            "track_family",
            "source_track",
            "base_sample_id",
            "domain",
            "source_record_id",
            "source_benchmark",
            "input_text",
            "input_length_chars",
            "input_length_bucket",
            "operator",
            "operator_kind",
            "operator_sequence",
            "semantic_operator",
            "recipe_id",
            "recipe_type",
            "recipe_length",
            "order_family_id",
            "order_slot",
            "order_group_instance_id",
            "group_success_rule",
            "reference_status",
            "reference_text",
            "reference_text_full_run",
            "output_format",
            "scoring_profile",
            "reports_refinement_gain",
            "prompt_variants",
            "prompt_variant_count",
            "difficulty_score",
            "difficulty_label",
            "pii_meta",
            "hallu_meta",
        ],
        "report_groups": {
            "core_rule": ["atomic_m", "atomic_f", "agnostic_m", "order_m", "order_f"],
            "semantic_extension": ["semantic_pii", "semantic_hallucination"],
        },
        "metric_policy": {
            "recipe_success": "Reported for every v3 row.",
            "refinement_gain": "Reported only when scoring_profile=text_refinement.",
            "structured_json": "Canonical JSON exact match is treated as RS; field diagnostics are optional.",
            "rs_at_k": (
                "Supported when predictions contain multiple prompt variants. Existing semantic appendix "
                "runs are single-prompt and should be reported as RS@1."
            ),
        },
        "counts": {
            "total_instances": len(all_rows),
            "by_track_family": dict(sorted(family_counts.items())),
            "by_benchmark_track": dict(sorted(track_counts.items())),
            "by_source_track": dict(sorted(source_counts.items())),
            "by_scoring_profile": dict(sorted(scoring_counts.items())),
            "by_rg_policy": dict(sorted(rg_counts.items())),
            "base_samples_by_benchmark_track": base_counts,
        },
        "validation_errors": dict(validation_errors),
    }
    write_json(output_dir / "manifest.json", manifest)
    if validation_errors:
        raise SystemExit(f"v3 validation failed; see {output_dir / 'manifest.json'}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CDR-Bench v3 release files.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--output-dir", default="data/benchmark_v3")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    output_dir = (repo_root / args.output_dir).resolve()
    manifest = build_v3(repo_root, output_dir)
    print(f"Wrote CDR-Bench v3 to {output_dir}")
    print(f"Total instances: {manifest['counts']['total_instances']}")
    for key, value in manifest["counts"]["by_benchmark_track"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
