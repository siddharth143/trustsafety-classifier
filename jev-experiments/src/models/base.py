"""Base asynchronous model evaluator and retry helper with exponential backoff and jitter."""

import abc
import asyncio
import random
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, TypeVar

T = TypeVar("T")


def is_retryable_error(
    exc: Exception,
    custom_check: Optional[Callable[[Exception], bool]] = None,
) -> bool:
    """Determine whether an exception represents a retryable rate limit or server error.

    Handles:
    - HTTP 429 (Rate Limit / Too Many Requests)
    - HTTP 500, 502, 503 (Service Unavailable / Overloaded), 504 (Gateway Timeout)
    - SDK-specific error types (RateLimitError, ServerError, APIConnectionError, etc.)
    """
    if custom_check is not None:
        return bool(custom_check(exc))

    # Check status_code, code, or http_status attributes
    status_code = getattr(exc, "status_code", getattr(exc, "code", getattr(exc, "http_status", None)))
    if status_code is not None:
        try:
            status_int = int(status_code)
            if status_int in (429, 500, 502, 503, 504):
                return True
            if status_int in (400, 401, 403, 404, 422):
                return False
        except (ValueError, TypeError):
            pass

    # Check class name
    class_name = exc.__class__.__name__
    retryable_names = {
        "RateLimitError",
        "TypeSafeRateLimitError",
        "ServiceUnavailableError",
        "InternalServerError",
        "TypeSafeInternalServerError",
        "APIConnectionError",
        "TypeSafeAPIConnectionError",
        "APITimeoutError",
        "TypeSafeAPITimeoutError",
        "OverloadedError",
        "ServerError",
    }
    if class_name in retryable_names:
        return True

    # Check error message strings
    msg = str(exc).lower()
    retryable_substrings = [
        "429",
        "503",
        "rate limit",
        "too many requests",
        "service unavailable",
        "overloaded",
        "resource exhausted",
        "connection reset",
        "server error",
    ]
    if any(sub in msg for sub in retryable_substrings):
        return True

    return False


async def retry_async(
    coro_fn: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 60.0,
    jitter: bool = True,
    jitter_factor: float = 0.5,
    is_retryable: Optional[Callable[[Exception], bool]] = None,
) -> T:
    """Execute an asynchronous callable with exponential backoff and jitter.

    Formula:
    delay = min(max_delay, base_delay * (2 ** attempt)) + (random.uniform(0, jitter_factor * delay) if jitter else 0)
    """
    attempt = 0
    while True:
        try:
            return await coro_fn()
        except Exception as exc:
            if attempt >= max_retries or not is_retryable_error(exc, is_retryable):
                raise

            delay = min(max_delay, base_delay * (2.0 ** attempt))
            if jitter:
                delay += random.uniform(0.0, jitter_factor * delay)

            await asyncio.sleep(delay)
            attempt += 1


class BaseModelEvaluator(abc.ABC):
    """Abstract base class for asynchronous model evaluators."""

    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    @abc.abstractmethod
    async def _execute_call(
        self,
        comment_id: str,
        comment_text: str,
        latency_ms: float,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Execute a single model invocation and parse results."""
        pass

    async def evaluate(
        self,
        comment_id: str,
        comment_text: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Evaluate a single comment with latency measurement and exponential backoff retry."""
        start_time = time.perf_counter()

        async def _call():
            call_start = time.perf_counter()
            # Pass placeholder latency, will be updated with actual measured latency
            calls_row, answers_rows = await self._execute_call(
                comment_id=comment_id,
                comment_text=comment_text,
                latency_ms=(time.perf_counter() - call_start) * 1000.0,
            )
            return calls_row, answers_rows

        calls_row, answers_rows = await retry_async(
            _call,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            jitter=self.jitter,
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        calls_row["latency_ms"] = total_latency_ms

        return calls_row, answers_rows
