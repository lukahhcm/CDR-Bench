from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cdrbench_v3.io import read_jsonl, write_json, write_jsonl
from cdrbench_v3.metrics import compare_json, compare_text
from cdrbench_v3.schema import is_json_reference


def _variant_predictions(row: dict[str, Any]) -> list[dict[str, Any]]:
    variants = row.get("variant_predictions")
    if isinstance(variants, list) and variants:
        return [variant for variant in variants if isinstance(variant, dict)]
    if "predicted_clean_text" in row or "parsed_response" in row or "raw_response" in row:
        return [row]
    return []


def _predicted_text(variant: dict[str, Any]) -> str:
    for key in ("predicted_clean_text", "clean_text", "text", "output"):
        if variant.get(key) is not None:
            return str(variant.get(key))
    parsed = variant.get("parsed_response")
    if isinstance(parsed, dict):
        for key in ("clean_text", "text", "output", "answer"):
            if parsed.get(key) is not None:
                return str(parsed.get(key))
    if variant.get("raw_response") is not None:
        return str(variant.get("raw_response"))
    return ""


def _predicted_status(variant: dict[str, Any]) -> str:
    if variant.get("predicted_status") is not None:
        return str(variant.get("predicted_status"))
    parsed = variant.get("parsed_response")
    if isinstance(parsed, dict) and parsed.get("status") is not None:
        return str(parsed.get("status"))
    return "KEEP"


def _valid_prediction(variant: dict[str, Any]) -> bool:
    if "valid_prediction" in variant:
        return bool(variant.get("valid_prediction"))
    if "prediction_valid_json" in variant:
        return bool(variant.get("prediction_valid_json"))
    return not bool(variant.get("prediction_error"))


def _score_variant(row: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    predicted_text = _predicted_text(variant)
    predicted_status = _predicted_status(variant)
    valid_prediction = _valid_prediction(variant)
    scoring_profile = row.get("scoring_profile")
    if not scoring_profile:
        scoring_profile = "structured_json" if is_json_reference(row.get("reference_text")) else "text_refinement"

    base = {
        "instance_id": row.get("instance_id"),
        "benchmark_track": row.get("benchmark_track"),
        "benchmark_split": row.get("benchmark_split") or ("single" if not str(row.get("benchmark_track", "")).startswith("semantic_") else None),
        "source_track": row.get("source_track") or row.get("benchmark_track"),
        "track_family": row.get("track_family") or ("semantic_extension" if str(row.get("benchmark_track", "")).startswith("semantic_") else "core_rule"),
        "domain": row.get("domain"),
        "operator": row.get("operator"),
        "semantic_operator": row.get("semantic_operator"),
        "scoring_profile": scoring_profile,
        "output_format": row.get("output_format"),
        "prompt_variant_index": int(variant.get("prompt_variant_index", 0) or 0),
        "prompt_style_id": variant.get("prompt_style_id"),
        "prediction_error": variant.get("prediction_error"),
        "valid_prediction": valid_prediction,
        "reports_refinement_gain": bool(row.get("reports_refinement_gain")),
    }
    if not valid_prediction:
        base.update({"recipe_success": False, "primary_score": 0.0})
        return base

    if scoring_profile == "structured_json":
        metrics = compare_json(row.get("reference_text"), predicted_text)
        recipe_success = bool(metrics["json_exact_match"])
        base.update(metrics)
        base.update(
            {
                "recipe_success": recipe_success,
                "primary_score": 1.0 if recipe_success else 0.0,
                "refinement_gain": None,
            }
        )
        return base

    if scoring_profile == "mixed_structured_text":
        # Reserved for future semantic rows that require both structured JSON and text.
        # Existing v3 data does not use this profile, but the report contract supports it.
        json_metrics = compare_json(row.get("reference_json"), predicted_text)
        text_metrics = compare_text(
            input_text=row.get("input_text"),
            reference_status=row.get("reference_status"),
            reference_text=row.get("reference_text"),
            predicted_status=predicted_status,
            predicted_text=predicted_text,
        )
        recipe_success = bool(json_metrics["json_exact_match"]) and bool(text_metrics["recipe_success"])
        base.update(json_metrics)
        base.update(text_metrics)
        base.update({"recipe_success": recipe_success, "primary_score": 1.0 if recipe_success else 0.0})
        return base

    metrics = compare_text(
        input_text=row.get("input_text"),
        reference_status=row.get("reference_status"),
        reference_text=row.get("reference_text"),
        predicted_status=predicted_status,
        predicted_text=predicted_text,
    )
    base.update(metrics)
    base["primary_score"] = 1.0 if metrics["recipe_success"] else 0.0
    return base


def score_rows(rows: list[dict[str, Any]], *, sample_size: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    variant_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    for row in rows:
        scored_variants = [_score_variant(row, variant) for variant in _variant_predictions(row)]
        scored_variants.sort(key=lambda item: int(item.get("prompt_variant_index", 0) or 0))
        if sample_size > 0:
            scored_for_at_k = scored_variants[:sample_size]
        else:
            scored_for_at_k = scored_variants
        variant_rows.extend(scored_variants)
        success_values = [bool(item.get("recipe_success")) for item in scored_variants]
        success_at_k_values = [bool(item.get("recipe_success")) for item in scored_for_at_k]
        rg_values = [
            float(item["refinement_gain"])
            for item in scored_variants
            if item.get("refinement_gain") is not None
        ]
        instance_rows.append(
            {
                "instance_id": row.get("instance_id"),
                "benchmark_track": row.get("benchmark_track"),
                "benchmark_split": row.get("benchmark_split"),
                "source_track": row.get("source_track"),
                "track_family": row.get("track_family"),
                "domain": row.get("domain"),
                "scoring_profile": row.get("scoring_profile"),
                "num_prompt_variants": len(scored_variants),
                "rs": (sum(success_values) / len(success_values) if success_values else 0.0),
                "rs_at_k": (any(success_at_k_values) if success_at_k_values else None),
                "rs_prompt0": (success_values[0] if success_values else None),
                "mean_rg": (sum(rg_values) / len(rg_values) if rg_values else None),
                "reports_refinement_gain": bool(row.get("reports_refinement_gain")),
            }
        )
    summary = aggregate(instance_rows, variant_rows, sample_size=sample_size)
    return variant_rows, instance_rows, summary


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rs_values = [float(row.get("rs", 0.0)) for row in rows]
    rs0_values = [1.0 if row.get("rs_prompt0") else 0.0 for row in rows if row.get("rs_prompt0") is not None]
    rsk_values = [1.0 if row.get("rs_at_k") else 0.0 for row in rows if row.get("rs_at_k") is not None]
    rg_values = [float(row["mean_rg"]) for row in rows if row.get("mean_rg") is not None]
    return {
        "num_instances": len(rows),
        "mean_rs": _mean(rs_values),
        "mean_rs_prompt0": _mean(rs0_values),
        "mean_rs_at_k": _mean(rsk_values),
        "mean_rg": (_mean(rg_values) if rg_values else None),
        "num_rg_instances": len(rg_values),
    }


def aggregate(instance_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]], *, sample_size: int) -> dict[str, Any]:
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scoring: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        by_track[str(row.get("benchmark_track"))].append(row)
        by_source_track[str(row.get("source_track"))].append(row)
        by_family[str(row.get("track_family"))].append(row)
        by_scoring[str(row.get("scoring_profile"))].append(row)

    variant_counts = Counter(row.get("prompt_variant_index") for row in variant_rows)
    return {
        "num_instances": len(instance_rows),
        "num_variant_predictions": len(variant_rows),
        "prompt_variant_sample_size_for_rs_at_k": sample_size if sample_size > 0 else "all",
        "variant_index_counts": {str(key): value for key, value in sorted(variant_counts.items())},
        "overall": _summarize_bucket(instance_rows),
        "by_track_family": {key: _summarize_bucket(value) for key, value in sorted(by_family.items())},
        "by_benchmark_track": {key: _summarize_bucket(value) for key, value in sorted(by_track.items())},
        "by_source_track": {key: _summarize_bucket(value) for key, value in sorted(by_source_track.items())},
        "by_scoring_profile": {key: _summarize_bucket(value) for key, value in sorted(by_scoring.items())},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CDR-Bench v3 predictions.")
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rs-at-k", type=int, default=0, help="Use first K variants for RS@K; 0 means all variants.")
    parser.add_argument("--write-csv", action="store_true")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = read_jsonl(predictions_path)
    variant_rows, instance_rows, summary = score_rows(rows, sample_size=args.rs_at_k)
    summary["predictions_path"] = str(predictions_path)
    summary["output_dir"] = str(output_dir)

    write_jsonl(output_dir / "scored_variant_predictions.jsonl", variant_rows)
    write_jsonl(output_dir / "instance_metrics.jsonl", instance_rows)
    write_json(output_dir / "summary.json", summary)
    if args.write_csv:
        _write_csv(output_dir / "instance_metrics.csv", instance_rows)
        _write_csv(output_dir / "scored_variant_predictions.csv", variant_rows)

    print(f"Scored {len(instance_rows)} instances from {predictions_path}")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
