"""Asynchronous model evaluator for Jev (TypeSafe System One)."""

import os
from typing import Any, Dict, List, Optional, Tuple

from parsers import calculate_cost, parse_jev_response
from prompts import JEV_QUESTIONS
from models.base import BaseModelEvaluator

try:
    from typesafe_sdk import AsyncTypeSafeClient
except ImportError:
    AsyncTypeSafeClient = None


class JevModelEvaluator(BaseModelEvaluator):
    """Evaluator wrapper for Jev using TypeSafe async client."""

    def __init__(
        self,
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
        model: str = "jev-latest",
        max_retries: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        super().__init__(
            model_name="jev",
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
        )
        self.model = model
        self.model_id = model

        if client is not None:
            self.client = client
        else:
            if AsyncTypeSafeClient is None:
                raise ImportError(
                    "Package 'typesafe-sdk' is not installed. "
                    "Install it via `pip install typesafe-sdk` or pass a mock client."
                )
            resolved_key = api_key or os.getenv("TYPESAFE_API_KEY")
            self.client = AsyncTypeSafeClient(api_key=resolved_key)

    async def _execute_call(
        self,
        comment_id: str,
        comment_text: str,
        latency_ms: float,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Invoke TypeSafe system_one and parse response."""
        response = await self.client.system_one(
            model=self.model,
            state=comment_text,
            questions=JEV_QUESTIONS,
        )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        cost_usd = calculate_cost("jev", input_tokens, output_tokens)

        calls_row, answers_rows = parse_jev_response(
            comment_id=comment_id,
            response=response,
            latency_ms=latency_ms,
            usage=usage,
            cost_usd=cost_usd,
        )

        return calls_row, answers_rows
