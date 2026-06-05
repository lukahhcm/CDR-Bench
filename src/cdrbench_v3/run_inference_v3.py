from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cdrbench_v3.io import read_jsonl, write_jsonl

SYSTEM_PROMPT = (
    "You are a careful data refinement engine. Follow the user request exactly. "
    "Return only the required output."
)


def _make_backend(model: str, base_url: str | None, api_key: str | None, concurrency: int, max_tokens: int):
    try:
        from cdrbench.infer.api_model_config import default_base_url_for_model
        from cdrbench.infer.openai_infer import make_api_infer
    except Exception as exc:
        raise SystemExit(
            "run_inference_v3.py requires the original cdrbench package on PYTHONPATH "
            "for OpenAI-compatible inference."
        ) from exc
    resolved_base_url = base_url or default_base_url_for_model(model) or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
    if not resolved_api_key:
        raise SystemExit("No API key. Set --api-key, OPENAI_API_KEY, or DASHSCOPE_API_KEY.")
    return (
        make_api_infer(
            model=model,
            api_base=resolved_base_url,
            api_key=resolved_api_key,
            concurrency=concurrency,
            max_tokens=max_tokens,
            max_retries=3,
            retry_delay=2.0,
        ),
        resolved_base_url,
    )


def _prompt_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    variants = row.get("prompt_variants")
    if isinstance(variants, list) and variants:
        return [variant for variant in variants if isinstance(variant, dict)]
    requirement = row.get("user_requirement")
    if isinstance(requirement, str) and requirement.strip():
        return [{"style_id": "direct", "style_label": "Direct", "user_requirement": requirement}]
    return []


def _indices(row: dict[str, Any], value: str) -> list[int]:
    variants = _prompt_variants(row)
    if value.strip().lower() == "all":
        return list(range(len(variants)))
    out = []
    for token in value.split(","):
        token = token.strip()
        if token:
            index = int(token)
            if index < len(variants):
                out.append(index)
    return sorted(set(out)) or [0]


def _render_prompt(row: dict[str, Any], variant: dict[str, Any]) -> str:
    requirement = variant.get("user_requirement") or ""
    input_text = row.get("input_text") or ""
    output_format = row.get("output_format")
    if output_format == "json":
        return (
            f"Task:\n{requirement}\n\n"
            f"Input text:\n<input>\n{input_text}\n</input>\n\n"
            "Return only the required JSON object or array. Do not use markdown fences."
        )
    if output_format == "json_and_tagged_text":
        return (
            f"Task:\n{requirement}\n\n"
            f"Input text:\n<input>\n{input_text}\n</input>\n\n"
            "Return both the required JSON fields and tagged text exactly as requested. "
            "Do not add explanations."
        )
    return (
        f"Task:\n{requirement}\n\n"
        f"Raw input text:\n<input>\n{input_text}\n</input>\n\n"
        "Return tagged output only using exactly this format: "
        "<status>KEEP</status><clean_text>...</clean_text> or "
        "<status>DROP</status><clean_text>...</clean_text>."
    )


def _parse_tagged(text: str) -> tuple[str, str]:
    status = "KEEP"
    status_match = re.search(r"<status>\s*(KEEP|DROP)\s*</status>", text, re.IGNORECASE)
    if status_match:
        status = status_match.group(1).upper()
    text_match = re.search(r"<clean_text>(.*?)</clean_text>", text, re.DOTALL | re.IGNORECASE)
    if text_match:
        return status, text_match.group(1)
    return status, text.strip()


def _infer_one(backend: Any, row: dict[str, Any], index: int) -> dict[str, Any]:
    variants = _prompt_variants(row)
    variant = variants[index] if index < len(variants) else variants[0]
    prompt = _render_prompt(row, variant)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        raw = backend._call_once(messages)
        if row.get("output_format") == "json":
            predicted_status = "KEEP"
            predicted_clean_text = raw.strip()
        else:
            predicted_status, predicted_clean_text = _parse_tagged(raw)
        return {
            "prompt_variant_index": index,
            "prompt_style_id": variant.get("style_id"),
            "prompt_style_label": variant.get("style_label"),
            "user_requirement": variant.get("user_requirement"),
            "predicted_status": predicted_status,
            "predicted_clean_text": predicted_clean_text,
            "raw_response": raw,
            "prediction_error": None,
            "valid_prediction": True,
        }
    except Exception as exc:
        return {
            "prompt_variant_index": index,
            "prompt_style_id": variant.get("style_id"),
            "prompt_style_label": variant.get("style_label"),
            "user_requirement": variant.get("user_requirement"),
            "predicted_status": "KEEP",
            "predicted_clean_text": "",
            "raw_response": "",
            "prediction_error": str(exc),
            "valid_prediction": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified CDR-Bench v3 inference.")
    parser.add_argument("--benchmark-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt-variant-indices", default="0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_path).resolve()
    output_path = Path(args.output_path).resolve()
    rows = read_jsonl(benchmark_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    existing: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    if args.resume and output_path.exists():
        existing = read_jsonl(output_path)
        done_ids = {str(row.get("instance_id")) for row in existing}
    rows = [row for row in rows if str(row.get("instance_id")) not in done_ids]

    backend, resolved_base_url = _make_backend(args.model, args.base_url, args.api_key, args.concurrency, args.max_tokens)

    def process(row: dict[str, Any]) -> dict[str, Any]:
        indices = _indices(row, args.prompt_variant_indices)
        out = dict(row)
        out["request_model"] = args.model
        out["request_base_url"] = resolved_base_url
        out["prompt_mode"] = "direct"
        out["selected_prompt_variant_indices"] = indices
        out["variant_predictions"] = [_infer_one(backend, row, index) for index in indices]
        return out

    output_rows = list(existing)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process, row): row for row in rows}
        for idx, future in enumerate(as_completed(futures), start=1):
            output_rows.append(future.result())
            if idx % 10 == 0 or idx == len(rows):
                write_jsonl(output_path, output_rows)
                print(f"progress {idx}/{len(rows)}")
    write_jsonl(output_path, output_rows)
    print(f"wrote {len(output_rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()

