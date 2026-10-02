import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("jev_parsers")

# Module-level registry tracking parse failures across calls
_PARSE_FAILURES: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
_PARSE_FAILURE_DETAILS: List[Dict[str, Any]] = []


def record_parse_failure(
    model_name: str,
    comment_id: str,
    category: str,
    raw_data: Any,
    reason: str,
) -> None:
    """Record a parse failure with count and structured diagnostics."""
    _PARSE_FAILURES[model_name][category] += 1
    detail = {
        "model": model_name,
        "comment_id": comment_id,
        "category": category,
        "raw_data": str(raw_data)[:200] if raw_data is not None else None,
        "reason": reason,
        "timestamp": now_iso(),
    }
    _PARSE_FAILURE_DETAILS.append(detail)
    logger.warning(
        f"[PARSE FAILURE] Model '{model_name}' comment_id '{comment_id}' "
        f"category '{category}': {reason} (raw: {str(raw_data)[:100]})"
    )


def get_parse_failures() -> Dict[str, Dict[str, int]]:
    """Return dict of parse failures by model and category: {model: {category: count}}."""
    return {m: dict(cats) for m, cats in _PARSE_FAILURES.items()}


def get_parse_failure_details() -> List[Dict[str, Any]]:
    """Return list of detailed parse failure records."""
    return list(_PARSE_FAILURE_DETAILS)


def reset_parse_failures() -> None:
    """Reset parse failure counters and details."""
    _PARSE_FAILURES.clear()
    _PARSE_FAILURE_DETAILS.clear()


def total_parse_failures(model_name: Optional[str] = None) -> int:
    """Return total count of parse failures, optionally filtered by model."""
    if model_name:
        return sum(_PARSE_FAILURES.get(model_name, {}).values())
    return sum(sum(cats.values()) for cats in _PARSE_FAILURES.values())


def now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD based on model pricing floors.

    Pricing per million tokens (MTok):
    - Jev: $0.042/MTok input, $0 output
    - Sonnet: $2.00/MTok input, $10.00/MTok output
    - Haiku: $1.00/MTok input, $5.00/MTok output
    - Flash: $0.75/MTok input, $3.75/MTok output
    """
    name = model_name.lower().strip()
    if "jev" in name:
        rate_in = 0.042 / 1_000_000.0
        rate_out = 0.0
    elif "sonnet" in name:
        rate_in = 2.00 / 1_000_000.0
        rate_out = 10.00 / 1_000_000.0
    elif "haiku" in name:
        rate_in = 1.00 / 1_000_000.0
        rate_out = 5.00 / 1_000_000.0
    elif "flash" in name:
        rate_in = 0.75 / 1_000_000.0
        rate_out = 3.75 / 1_000_000.0
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
    - toxic confidence matches response.answers["toxic"].confidence natively from Score
    - threat and identity_hate confidence is derived mathematically from Noul probability as 2 * abs(p - 0.5)
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
        "parse_failures": 0,
    }

    answers = getattr(response, "answers", None)
    if answers is None and isinstance(response, dict):
        answers = response.get("answers", {})

    if not answers:
        record_parse_failure("jev", comment_id, "all", response, "Missing answers field in Jev response")
        calls_row["parse_failures"] = 3
        # Return fallback rows with parse_error=True
        fallback_answers = [
            {
                "comment_id": comment_id,
                "model": "jev",
                "category": cat,
                "toxic_prob_0": 1.0 if cat == "toxic" else None,
                "toxic_prob_1": 0.0 if cat == "toxic" else None,
                "toxic_prob_2": 0.0 if cat == "toxic" else None,
                "toxic_score": 0.0 if cat == "toxic" else None,
                "binary_probability": 0.0 if cat != "toxic" else None,
                "native_confidence": 0.0,
                "parse_error": True,
            }
            for cat in ["toxic", "threat", "identity_hate"]
        ]
        return calls_row, fallback_answers

    toxic_parse_error = False
    try:
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
        score_val = float(score)
        conf_val = float(confidence) if confidence is not None else None
    except Exception as e:
        record_parse_failure("jev", comment_id, "toxic", answers, f"Failed to parse Jev toxic: {e}")
        calls_row["parse_failures"] += 1
        toxic_parse_error = True
        p0, p1, p2, score_val, conf_val = 1.0, 0.0, 0.0, 0.0, None

    answers_rows: List[Dict[str, Any]] = [
        {
            "comment_id": comment_id,
            "model": "jev",
            "category": "toxic",
            "toxic_prob_0": p0,
            "toxic_prob_1": p1,
            "toxic_prob_2": p2,
            "toxic_score": score_val,
            "binary_probability": None,
            "native_confidence": conf_val,
            "parse_error": toxic_parse_error,
        }
    ]

    for category in ["threat", "identity_hate"]:
        cat_parse_error = False
        try:
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
        except Exception as e:
            record_parse_failure("jev", comment_id, category, answers, f"Failed to parse Jev {category}: {e}")
            calls_row["parse_failures"] += 1
            cat_parse_error = True
            binary_p = 0.0
            derived_conf = 0.0

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
                "parse_error": cat_parse_error,
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
    - records parse failures when required keys or valid probabilities are missing
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
        "parse_failures": 0,
    }

    if isinstance(parsed_json, str):
        try:
            parsed = json.loads(parsed_json)
        except Exception as e:
            record_parse_failure(model_name, comment_id, "all", parsed_json, f"JSON parse error: {e}")
            calls_row["parse_failures"] = 3
            fallback_answers = [
                {
                    "comment_id": comment_id,
                    "model": model_name,
                    "category": cat,
                    "toxic_prob_0": 1.0 if cat == "toxic" else None,
                    "toxic_prob_1": 0.0 if cat == "toxic" else None,
                    "toxic_prob_2": 0.0 if cat == "toxic" else None,
                    "toxic_score": 0.0 if cat == "toxic" else None,
                    "binary_probability": 0.0 if cat != "toxic" else None,
                    "native_confidence": None,
                    "parse_error": True,
                }
                for cat in ["toxic", "threat", "identity_hate"]
            ]
            return calls_row, fallback_answers
    else:
        parsed = parsed_json

    if not isinstance(parsed, dict):
        record_parse_failure(model_name, comment_id, "all", parsed, "Parsed payload is not a dictionary")
        calls_row["parse_failures"] = 3
        fallback_answers = [
            {
                "comment_id": comment_id,
                "model": model_name,
                "category": cat,
                "toxic_prob_0": 1.0 if cat == "toxic" else None,
                "toxic_prob_1": 0.0 if cat == "toxic" else None,
                "toxic_prob_2": 0.0 if cat == "toxic" else None,
                "toxic_score": 0.0 if cat == "toxic" else None,
                "binary_probability": 0.0 if cat != "toxic" else None,
                "native_confidence": None,
                "parse_error": True,
            }
            for cat in ["toxic", "threat", "identity_hate"]
        ]
        return calls_row, fallback_answers

    # Robust extraction of toxic probabilities:
    toxic_parse_error = False
    toxic_data = parsed.get("toxic")
    if toxic_data is None and "probabilities" in parsed:
        toxic_data = parsed

    if toxic_data is None:
        record_parse_failure(model_name, comment_id, "toxic", None, "Missing 'toxic' key in response")
        toxic_parse_error = True
        probs = {}
    elif isinstance(toxic_data, dict):
        if "probabilities" in toxic_data and isinstance(toxic_data["probabilities"], dict):
            probs = toxic_data["probabilities"]
        elif any(k in toxic_data for k in ["0", 0, "1", 1, "2", 2]):
            probs = toxic_data
        elif "probability" in toxic_data:
            try:
                p = float(toxic_data["probability"])
                probs = {"0": max(0.0, 1.0 - p), "1": p, "2": 0.0}
            except (ValueError, TypeError):
                record_parse_failure(model_name, comment_id, "toxic", toxic_data, "Invalid toxic scalar probability")
                toxic_parse_error = True
                probs = {}
        else:
            probs = toxic_data.get("probabilities", {})
            if not isinstance(probs, dict) or not probs:
                record_parse_failure(model_name, comment_id, "toxic", toxic_data, "No valid probabilities found in toxic dict")
                toxic_parse_error = True
                probs = {}
    elif isinstance(toxic_data, (int, float, str)):
        try:
            p = float(toxic_data)
            probs = {"0": max(0.0, 1.0 - p), "1": p, "2": 0.0}
        except (ValueError, TypeError):
            record_parse_failure(model_name, comment_id, "toxic", toxic_data, "Cannot convert toxic scalar to float")
            toxic_parse_error = True
            probs = {}
    else:
        record_parse_failure(model_name, comment_id, "toxic", toxic_data, f"Unexpected type for toxic: {type(toxic_data).__name__}")
        toxic_parse_error = True
        probs = {}

    if not toxic_parse_error:
        try:
            p0 = float(probs.get("0", probs.get(0, 0.0)))
        except (ValueError, TypeError):
            p0 = 0.0
            toxic_parse_error = True
        try:
            p1 = float(probs.get("1", probs.get(1, 0.0)))
        except (ValueError, TypeError):
            p1 = 0.0
            toxic_parse_error = True
        try:
            p2 = float(probs.get("2", probs.get(2, 0.0)))
        except (ValueError, TypeError):
            p2 = 0.0
            toxic_parse_error = True

        total_p = p0 + p1 + p2
        if total_p > 0:
            p0 = p0 / total_p
            p1 = p1 / total_p
            p2 = p2 / total_p
        else:
            record_parse_failure(model_name, comment_id, "toxic", probs, "Sum of toxic probabilities is zero or negative")
            toxic_parse_error = True
            p0, p1, p2 = 1.0, 0.0, 0.0
    else:
        p0, p1, p2 = 1.0, 0.0, 0.0

    if toxic_parse_error:
        calls_row["parse_failures"] += 1

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
            "parse_error": toxic_parse_error,
        }
    ]

    for category in ["threat", "identity_hate"]:
        cat_parse_error = False
        if category not in parsed:
            record_parse_failure(
                model_name,
                comment_id,
                category,
                None,
                f"Missing required key '{category}' in LLM response",
            )
            cat_parse_error = True
            prob_val = 0.0
        else:
            cat_data = parsed[category]
            prob_val = None
            if isinstance(cat_data, dict):
                raw_p = cat_data.get("probability", cat_data.get("score"))
                if raw_p is not None:
                    try:
                        prob_val = float(raw_p)
                    except (ValueError, TypeError):
                        prob_val = None
                elif "probabilities" in cat_data and isinstance(cat_data["probabilities"], dict):
                    sub_probs = cat_data["probabilities"]
                    try:
                        prob_val = float(sub_probs.get("1", sub_probs.get(1, 0.0)))
                    except (ValueError, TypeError):
                        prob_val = None
                elif any(k in cat_data for k in ["1", 1]):
                    try:
                        prob_val = float(cat_data.get("1", cat_data.get(1, 0.0)))
                    except (ValueError, TypeError):
                        prob_val = None
            elif isinstance(cat_data, (int, float, str)):
                try:
                    prob_val = float(cat_data)
                except (ValueError, TypeError):
                    prob_val = None

            if prob_val is None:
                record_parse_failure(
                    model_name,
                    comment_id,
                    category,
                    cat_data,
                    f"Could not extract numeric probability for '{category}'",
                )
                cat_parse_error = True
                prob_val = 0.0
            else:
                prob_val = max(0.0, min(1.0, prob_val))

        if cat_parse_error:
            calls_row["parse_failures"] += 1

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
                "parse_error": cat_parse_error,
            }
        )

    return calls_row, answers_rows
