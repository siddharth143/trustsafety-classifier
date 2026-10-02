"""Model evaluator wrappers and factory for Trust & Safety Eval Gate Benchmark."""

from typing import Any

from models.base import BaseModelEvaluator, is_retryable_error, retry_async
from models.jev_model import JevModelEvaluator
from models.anthropic_model import AnthropicModelEvaluator
from models.gemini_model import GeminiModelEvaluator

__all__ = [
    "BaseModelEvaluator",
    "JevModelEvaluator",
    "AnthropicModelEvaluator",
    "GeminiModelEvaluator",
    "get_model_evaluator",
    "is_retryable_error",
    "retry_async",
]


def get_model_evaluator(model_name: str, **kwargs: Any) -> BaseModelEvaluator:
    """Factory returning the appropriate BaseModelEvaluator instance.

    Supported model names:
    - 'jev' / 'jev-latest' -> JevModelEvaluator
    - 'haiku' / 'claude-3-5-haiku' -> AnthropicModelEvaluator (haiku)
    - 'sonnet' / 'claude-3-5-sonnet' -> AnthropicModelEvaluator (sonnet)
    - 'flash' / 'gemini-1.5-flash' / 'gemini-2.0-flash' -> GeminiModelEvaluator
    """
    name = model_name.lower().strip()
    if "jev" in name:
        return JevModelEvaluator(**kwargs)
    elif "haiku" in name:
        return AnthropicModelEvaluator(model_type="haiku", **kwargs)
    elif "sonnet" in name:
        return AnthropicModelEvaluator(model_type="sonnet", **kwargs)
    elif "flash" in name or "gemini" in name:
        return GeminiModelEvaluator(**kwargs)
    else:
        raise ValueError(
            f"Unknown model name: {model_name!r}. "
            "Expected one of 'jev', 'haiku', 'sonnet', or 'flash'."
        )
