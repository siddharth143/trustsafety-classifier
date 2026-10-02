"""Asynchronous model evaluator for Anthropic Claude (Haiku & Sonnet)."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from parsers import calculate_cost, parse_llm_response
from prompts import CLAUDE_CLASSIFICATION_TOOL, SYSTEM_PROMPT_TEMPLATE
from models.base import BaseModelEvaluator

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

DEFAULT_ANTHROPIC_MODELS = {
    "haiku": os.getenv("ANTHROPIC_HAIKU_MODEL_ID", "claude-haiku-4-5"),
    "sonnet": os.getenv("ANTHROPIC_SONNET_MODEL_ID", "claude-sonnet-5-5"),
}


class AnthropicModelEvaluator(BaseModelEvaluator):
    """Evaluator wrapper for Anthropic models using forced tool-use."""

    def __init__(
        self,
        model_type: str = "haiku",
        model_id: Optional[str] = None,
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
        workspace_id: Optional[str] = None,
        max_retries: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        norm_type = model_type.lower().strip()
        if "sonnet" in norm_type:
            model_name = "sonnet"
        elif "haiku" in norm_type:
            model_name = "haiku"
        else:
            model_name = norm_type

        super().__init__(
            model_name=model_name,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
        )

        if model_id:
            self.model_id = model_id
        elif model_name == "haiku":
            self.model_id = os.getenv("ANTHROPIC_HAIKU_MODEL_ID", "claude-haiku-4-5")
        elif model_name == "sonnet":
            self.model_id = os.getenv("ANTHROPIC_SONNET_MODEL_ID", "claude-sonnet-5-5")
        else:
            self.model_id = model_type
        self.workspace_id = workspace_id or os.getenv("ANTHROPIC_WORKSPACE_ID")

        if client is not None:
            self.client = client
        else:
            if AsyncAnthropic is None:
                raise ImportError(
                    "Package 'anthropic' is not installed. "
                    "Install it via `pip install anthropic` or pass a mock client."
                )
            resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            headers = {}
            if self.workspace_id:
                headers["anthropic-workspace-id"] = self.workspace_id
            self.client = AsyncAnthropic(
                api_key=resolved_key,
                default_headers=headers if headers else None,
            )

    async def _execute_call(
        self,
        comment_id: str,
        comment_text: str,
        latency_ms: float,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Invoke Claude messages.create with forced tool use and parse response."""
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(comment_text=comment_text)

        response = await self.client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": "Please classify the comment using the submit_classification tool."}],
            tools=[CLAUDE_CLASSIFICATION_TOOL],
            tool_choice={"type": "auto"},
        )

        # Extract classification from tool_use block or text fallback
        parsed_json = None
        content_blocks = getattr(response, "content", []) or []
        for block in content_blocks:
            b_type = getattr(block, "type", None)
            b_name = getattr(block, "name", None)
            if b_type == "tool_use" or b_name == "submit_classification":
                parsed_json = getattr(block, "input", None)
                break

        # Fallback: check if model returned JSON in text content block
        if parsed_json is None:
            for block in content_blocks:
                text = getattr(block, "text", "") or ""
                if "{" in text and "}" in text:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    try:
                        parsed_json = json.loads(text[start:end])
                        break
                    except Exception:
                        continue

        if parsed_json is None:
            raise ValueError(f"No classification found in Anthropic response for {comment_id}")

        if isinstance(parsed_json, str):
            parsed_json = json.loads(parsed_json)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        cost_usd = calculate_cost(self.model_name, input_tokens, output_tokens)

        calls_row, answers_rows = parse_llm_response(
            comment_id=comment_id,
            model_name=self.model_name,
            parsed_json=parsed_json,
            latency_ms=latency_ms,
            usage=usage,
            cost_usd=cost_usd,
            raw_response=response,
        )

        return calls_row, answers_rows
