"""Asynchronous model evaluator for Google Gemini Flash."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from parsers import calculate_cost, parse_llm_response
from prompts import GEMINI_RESPONSE_SCHEMA, SYSTEM_PROMPT_TEMPLATE
from models.base import BaseModelEvaluator

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL_ID", "gemini-3.8-flash")


class GeminiModelEvaluator(BaseModelEvaluator):
    """Evaluator wrapper for Gemini Flash using structured JSON output."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
        max_retries: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        super().__init__(
            model_name="flash",
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
        )
        self.model_id = model_id or os.getenv("GEMINI_MODEL_ID", DEFAULT_GEMINI_MODEL)

        if client is not None:
            self.client = client
        else:
            if genai is None:
                raise ImportError(
                    "Package 'google-genai' is not installed. "
                    "Install it via `pip install google-genai` or pass a mock client."
                )
            resolved_key = api_key or os.getenv("GEMINI_API_KEY")
            self.client = genai.Client(api_key=resolved_key)

    async def _execute_call(
        self,
        comment_id: str,
        comment_text: str,
        latency_ms: float,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Invoke Gemini aio.models.generate_content with structured JSON schema and parse response."""
        contents = SYSTEM_PROMPT_TEMPLATE.format(comment_text=comment_text)

        if types is not None:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GEMINI_RESPONSE_SCHEMA,
            )
        else:
            config = {
                "response_mime_type": "application/json",
                "response_schema": GEMINI_RESPONSE_SCHEMA,
            }

        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config,
        )

        parsed_json = None
        if hasattr(response, "parsed") and isinstance(response.parsed, dict):
            parsed_json = response.parsed
        else:
            raw_text = getattr(response, "text", "")
            if not raw_text:
                raise ValueError(f"Empty text returned from Gemini for {comment_id}")
            parsed_json = json.loads(raw_text)

        usage_metadata = getattr(response, "usage_metadata", None)
        in_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
        out_tokens = getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0
        usage = {"input_tokens": in_tokens, "output_tokens": out_tokens}

        cost_usd = calculate_cost("flash", in_tokens, out_tokens)

        calls_row, answers_rows = parse_llm_response(
            comment_id=comment_id,
            model_name="flash",
            parsed_json=parsed_json,
            latency_ms=latency_ms,
            usage=usage,
            cost_usd=cost_usd,
            raw_response=response,
        )

        return calls_row, answers_rows
