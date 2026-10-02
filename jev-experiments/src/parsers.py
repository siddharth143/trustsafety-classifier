"""Shared normalization parsers and cost calculation for Jev and general-purpose LLMs."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD based on model pricing floors.

    Pricing per million tokens (MTok):
    - Jev: $0.042/MTok input, $0 output
    - Sonnet: $3.00/MTok input, $15.00/MTok output
    - Haiku: $0.80/MTok input, $4.00/MTok output
    - Flash: $0.075/MTok input, $0.30/MTok output
    """
    name = model_name.lower().strip()
    if "jev" in name:
        rate_in = 0.042 / 1_000_000.0
        rate_out = 0.0
    elif "sonnet" in name:
        rate_in = 3.00 / 1_000_000.0
        rate_out = 15.00 / 1_000_000.0
    elif "haiku" in name:
        rate_in = 0.80 / 1_000_000.0
        rate_out = 4.00 / 1_000_000.0
    elif "flash" in name:
        rate_in = 0.075 / 1_000_000.0
        rate_out = 0.30 / 1_000_000.0
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return float(input_tokens) * rate_in + float(output_tokens) * rate_out


def _extract_tokens(usage: Any) -> Tuple[int, int]:
    """Safely extract input and output tokens from SDK usage objects or dicts."""
    if usage is None:
        return 0, 0

    if hasattr(usage, "input_tokens"):
        input_tokens = getattr(usage, "input_tokens", 0)
    elif isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", 0)
    else:
        input_tokens = 0

    if hasattr(usage, "output_tokens"):
        output_tokens = getattr(usage, "output_tokens", 0)
    elif isinstance(usage, dict):
        output_tokens = usage.get("output_tokens", 0)
    else:
        output_tokens = 0

    return int(input_tokens or 0), int(output_tokens or 0)


def _serialize_raw_response(response: Any) -> str:
    """Serialize raw API response to JSON string."""
    if isinstance(response, str):
        return response
    if hasattr(response, "model_dump_json") and callable(response.model_dump_json):
        return response.model_dump_json()
    if hasattr(response, "to_json") and callable(response.to_json):
        return response.to_json()
    return json.dumps(response, default=str)


def parse_jev_response(
    comment_id: str,
    response: Any,
    latency_ms: float,
    usage: Any,
    cost_usd: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse Jev SDK response into calls and answers rows.

    - toxic_score matches response.answers["toxic"].score
    - toxic native_confidence matches response.answers["toxic"].confidence natively
    - threat and identity_hate native_confidence is derived as 2 * abs(probability - 0.5)
    """
    input_tokens, output_tokens = _extract_tokens(usage)
    raw_response_json = _serialize_raw_response(response)

    calls_row = {
        "comment_id": comment_id,
        "model": "jev",
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "raw_response_json": raw_response_json,
        "timestamp": now_iso(),
    }

    answers = getattr(response, "answers", None)
    if answers is None and isinstance(response, dict):
        answers = response.get("answers", {})

    toxic = answers["toxic"] if isinstance(answers, dict) else getattr(answers, "toxic")

    if hasattr(toxic, "probabilities"):
        probs = toxic.probabilities
        score = toxic.score
        confidence = getattr(toxic, "confidence", None)
    else:
        probs = toxic["probabilities"]
        score = toxic["score"]
        confidence = toxic.get("confidence")

    p0 = float(probs["0"] if "0" in probs else probs[0])
    p1 = float(probs["1"] if "1" in probs else probs[1])
    p2 = float(probs["2"] if "2" in probs else probs[2])

    answers_rows: List[Dict[str, Any]] = [
        {
            "comment_id": comment_id,
            "model": "jev",
            "category": "toxic",
            "toxic_prob_0": p0,
            "toxic_prob_1": p1,
            "toxic_prob_2": p2,
            "toxic_score": float(score),
            "binary_probability": None,
            "native_confidence": float(confidence) if confidence is not None else None,
        }
    ]

    for category in ["threat", "identity_hate"]:
        noul = answers[category] if isinstance(answers, dict) else getattr(answers, category)
        if hasattr(noul, "noul"):
            binary_p = noul.noul
        elif isinstance(noul, dict):
            binary_p = noul.get("noul", noul.get("probability"))
        elif hasattr(noul, "probability"):
            binary_p = noul.probability
        else:
            binary_p = float(noul)

        binary_p = float(binary_p)
        derived_conf = 2.0 * abs(binary_p - 0.5)

        answers_rows.append(
            {
                "comment_id": comment_id,
                "model": "jev",
                "category": category,
                "toxic_prob_0": None,
                "toxic_prob_1": None,
                "toxic_prob_2": None,
                "toxic_score": None,
                "binary_probability": binary_p,
                "native_confidence": derived_conf,
            }
        )

    return calls_row, answers_rows


def parse_llm_response(
    comment_id: str,
    model_name: str,
    parsed_json: Any,
    latency_ms: float,
    usage: Any,
    cost_usd: float,
    raw_response: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse Claude or Gemini LLM structured output into calls and answers rows.

    - toxic probabilities are normalized if floating-point drift occurs
    - toxic_score is calculated as 0*p0 + 1*p1 + 2*p2 (weighted mean)
    - native_confidence is strictly None for LLMs across all 3 categories
    """
    input_tokens, output_tokens = _extract_tokens(usage)
    raw_response_json = _serialize_raw_response(raw_response)

    calls_row = {
        "comment_id": comment_id,
        "model": model_name,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "raw_response_json": raw_response_json,
        "timestamp": now_iso(),
    }

    if isinstance(parsed_json, str):
        parsed = json.loads(parsed_json)
    else:
        parsed = parsed_json

    # Robust extraction of toxic probabilities:
    toxic_data = parsed.get("toxic", {}) if isinstance(parsed, dict) else {}
    if not toxic_data and isinstance(parsed, dict) and "probabilities" in parsed:
        toxic_data = parsed

    if isinstance(toxic_data, dict):
        if "probabilities" in toxic_data and isinstance(toxic_data["probabilities"], dict):
            probs = toxic_data["probabilities"]
        elif any(k in toxic_data for k in ["0", 0, "1", 1, "2", 2]):
            probs = toxic_data
        elif "probability" in toxic_data:
            try:
                p = float(toxic_data["probability"])
            except (ValueError, TypeError):
                p = 0.0
            probs = {"0": max(0.0, 1.0 - p), "1": p, "2": 0.0}
        else:
            probs = toxic_data.get("probabilities", {})
            if not isinstance(probs, dict):
                probs = {}
    elif isinstance(toxic_data, (int, float, str)):
        try:
            p = float(toxic_data)
        except (ValueError, TypeError):
            p = 0.0
        probs = {"0": max(0.0, 1.0 - p), "1": p, "2": 0.0}
    else:
        probs = {}

    try:
        p0 = float(probs.get("0", probs.get(0, 0.0)))
    except (ValueError, TypeError):
        p0 = 0.0
    try:
        p1 = float(probs.get("1", probs.get(1, 0.0)))
    except (ValueError, TypeError):
        p1 = 0.0
    try:
        p2 = float(probs.get("2", probs.get(2, 0.0)))
    except (ValueError, TypeError):
        p2 = 0.0

    # Normalize probabilities to guarantee sum == 1.0 (corrects floating-point drift)
    total_p = p0 + p1 + p2
    if total_p > 0:
        p0 = p0 / total_p
        p1 = p1 / total_p
        p2 = p2 / total_p
    else:
        p0, p1, p2 = 1.0, 0.0, 0.0

    toxic_score = 0.0 * p0 + 1.0 * p1 + 2.0 * p2

    answers_rows: List[Dict[str, Any]] = [
        {
            "comment_id": comment_id,
            "model": model_name,
            "category": "toxic",
            "toxic_prob_0": p0,
            "toxic_prob_1": p1,
            "toxic_prob_2": p2,
            "toxic_score": toxic_score,
            "binary_probability": None,
            "native_confidence": None,  # Strictly None for LLMs
        }
    ]

    for category in ["threat", "identity_hate"]:
        cat_data = parsed.get(category, {}) if isinstance(parsed, dict) else {}
        prob_val = 0.0
        if isinstance(cat_data, dict):
            raw_p = cat_data.get("probability", cat_data.get("score"))
            if raw_p is not None:
                try:
                    prob_val = float(raw_p)
                except (ValueError, TypeError):
                    prob_val = 0.0
            elif "probabilities" in cat_data and isinstance(cat_data["probabilities"], dict):
                sub_probs = cat_data["probabilities"]
                try:
                    prob_val = float(sub_probs.get("1", sub_probs.get(1, 0.0)))
                except (ValueError, TypeError):
                    prob_val = 0.0
            elif any(k in cat_data for k in ["1", 1]):
                try:
                    prob_val = float(cat_data.get("1", cat_data.get(1, 0.0)))
                except (ValueError, TypeError):
                    prob_val = 0.0
        elif isinstance(cat_data, (int, float, str)):
            try:
                prob_val = float(cat_data)
            except (ValueError, TypeError):
                prob_val = 0.0
        else:
            prob_val = 0.0

        prob_val = max(0.0, min(1.0, prob_val))

        answers_rows.append(
            {
                "comment_id": comment_id,
                "model": model_name,
                "category": category,
                "toxic_prob_0": None,
                "toxic_prob_1": None,
                "toxic_prob_2": None,
                "toxic_score": None,
                "binary_probability": prob_val,
                "native_confidence": None,  # Strictly None for LLMs
            }
        )

    return calls_row, answers_rows
