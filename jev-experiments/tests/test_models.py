"""Unit tests for asynchronous model client wrappers in T&S eval gate."""

import asyncio
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
for p in [str(_ROOT / "src"), str(_ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from parsers import calculate_cost
from prompts import (
    CLAUDE_CLASSIFICATION_TOOL,
    GEMINI_RESPONSE_SCHEMA,
    JEV_QUESTIONS,
    SYSTEM_PROMPT_TEMPLATE,
)

# Imports from models package (will fail before models is implemented)
from models import get_model_evaluator
from models.base import BaseModelEvaluator, is_retryable_error, retry_async
from models.jev_model import JevModelEvaluator
from models.anthropic_model import AnthropicModelEvaluator
from models.gemini_model import GeminiModelEvaluator


# =====================================================================
# Mock Helpers and Simulated Errors
# =====================================================================

class MockUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockScoreAnswer:
    def __init__(self, probabilities: dict, score: float, confidence: float):
        self.probabilities = probabilities
        self.score = score
        self.confidence = confidence


class MockNoulAnswer:
    def __init__(self, noul: float):
        self.noul = noul


class MockJevResponse:
    def __init__(
        self,
        toxic_answer: MockScoreAnswer,
        threat_answer: MockNoulAnswer,
        identity_hate_answer: MockNoulAnswer,
        input_tokens: int = 80,
        output_tokens: int = 0,
    ):
        self.answers = {
            "toxic": toxic_answer,
            "threat": threat_answer,
            "identity_hate": identity_hate_answer,
        }
        self.usage = MockUsage(input_tokens, output_tokens)

    def model_dump_json(self) -> str:
        return json.dumps(
            {
                "answers": {
                    "toxic": {
                        "probabilities": self.answers["toxic"].probabilities,
                        "score": self.answers["toxic"].score,
                        "confidence": self.answers["toxic"].confidence,
                    },
                    "threat": {"noul": self.answers["threat"].noul},
                    "identity_hate": {"noul": self.answers["identity_hate"].noul},
                },
                "usage": {
                    "input_tokens": self.usage.input_tokens,
                    "output_tokens": self.usage.output_tokens,
                },
            }
        )


class MockAnthropicToolUseBlock:
    def __init__(self, tool_input: dict, name: str = "submit_classification"):
        self.type = "tool_use"
        self.name = name
        self.input = tool_input


class MockAnthropicMessageResponse:
    def __init__(
        self,
        tool_input: dict,
        input_tokens: int = 150,
        output_tokens: int = 50,
    ):
        self.content = [MockAnthropicToolUseBlock(tool_input)]
        self.usage = MockUsage(input_tokens, output_tokens)

    def model_dump_json(self) -> str:
        return json.dumps(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "submit_classification",
                        "input": self.content[0].input,
                    }
                ],
                "usage": {
                    "input_tokens": self.usage.input_tokens,
                    "output_tokens": self.usage.output_tokens,
                },
            }
        )


class MockGeminiUsageMetadata:
    def __init__(self, prompt_token_count: int = 120, candidates_token_count: int = 45):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class MockGeminiResponse:
    def __init__(
        self,
        json_data: dict,
        prompt_token_count: int = 120,
        candidates_token_count: int = 45,
    ):
        self.text = json.dumps(json_data)
        self.usage_metadata = MockGeminiUsageMetadata(prompt_token_count, candidates_token_count)

    def model_dump_json(self) -> str:
        return json.dumps(
            {
                "text": self.text,
                "usage_metadata": {
                    "prompt_token_count": self.usage_metadata.prompt_token_count,
                    "candidates_token_count": self.usage_metadata.candidates_token_count,
                },
            }
        )


class SimulatedRateLimitError(Exception):
    def __init__(self, message: str = "Rate limit exceeded (429)", status_code: int = 429):
        super().__init__(message)
        self.status_code = status_code
        self.code = status_code


class SimulatedServiceUnavailableError(Exception):
    def __init__(self, message: str = "Service unavailable (503)", status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code
        self.code = status_code


class SimulatedBadRequestError(Exception):
    def __init__(self, message: str = "Bad request (400)", status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code
        self.code = status_code


# =====================================================================
# Unit Tests
# =====================================================================

class TestModelFactory(unittest.TestCase):
    """Test model factory get_model_evaluator."""

    def test_factory_jev(self):
        evaluator = get_model_evaluator("jev", client=MagicMock())
        self.assertIsInstance(evaluator, JevModelEvaluator)
        self.assertIsInstance(evaluator, BaseModelEvaluator)
        self.assertEqual(evaluator.model_name, "jev")

    def test_factory_haiku(self):
        evaluator = get_model_evaluator("haiku", client=MagicMock())
        self.assertIsInstance(evaluator, AnthropicModelEvaluator)
        self.assertIsInstance(evaluator, BaseModelEvaluator)
        self.assertEqual(evaluator.model_name, "haiku")
        self.assertIn("haiku", evaluator.model_id)

    def test_factory_sonnet(self):
        evaluator = get_model_evaluator("sonnet", client=MagicMock())
        self.assertIsInstance(evaluator, AnthropicModelEvaluator)
        self.assertIsInstance(evaluator, BaseModelEvaluator)
        self.assertEqual(evaluator.model_name, "sonnet")
        self.assertIn("sonnet", evaluator.model_id)

    def test_factory_flash(self):
        evaluator = get_model_evaluator("flash", client=MagicMock())
        self.assertIsInstance(evaluator, GeminiModelEvaluator)
        self.assertIsInstance(evaluator, BaseModelEvaluator)
        self.assertEqual(evaluator.model_name, "flash")
        self.assertIn("flash", evaluator.model_id)

    def test_factory_case_insensitive_and_whitespace(self):
        evaluator = get_model_evaluator("  Haiku  ", client=MagicMock())
        self.assertEqual(evaluator.model_name, "haiku")

    def test_factory_unknown_model_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_model_evaluator("gpt-4o")


class TestRetryAsync(unittest.IsolatedAsyncioTestCase):
    """Test exponential backoff retry helper with jitter on simulated 429/503 errors."""

    async def test_retry_succeeds_immediately(self):
        mock_coro = AsyncMock(return_value="success")
        res = await retry_async(mock_coro, max_retries=3, base_delay=0.01)
        self.assertEqual(res, "success")
        self.assertEqual(mock_coro.await_count, 1)

    async def test_retry_on_429_then_succeeds(self):
        mock_coro = AsyncMock(side_effect=[
            SimulatedRateLimitError("Rate limit exceeded"),
            "success",
        ])
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            res = await retry_async(mock_coro, max_retries=3, base_delay=0.1, jitter=False)
            self.assertEqual(res, "success")
            self.assertEqual(mock_coro.await_count, 2)
            self.assertEqual(mock_sleep.await_count, 1)
            # Without jitter, attempt 0 delay is base_delay * (2 ** 0) = 0.1
            mock_sleep.assert_awaited_once_with(0.1)

    async def test_retry_on_503_then_succeeds(self):
        mock_coro = AsyncMock(side_effect=[
            SimulatedServiceUnavailableError("Overloaded"),
            "recovered",
        ])
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            res = await retry_async(mock_coro, max_retries=3, base_delay=0.2, jitter=False)
            self.assertEqual(res, "recovered")
            self.assertEqual(mock_coro.await_count, 2)
            mock_sleep.assert_awaited_once_with(0.2)

    async def test_retry_with_exponential_backoff_and_jitter(self):
        # 3 failures, then success on 4th call
        mock_coro = AsyncMock(side_effect=[
            SimulatedRateLimitError(),
            SimulatedRateLimitError(),
            SimulatedRateLimitError(),
            "final_success",
        ])
        sleep_durations = []

        async def fake_sleep(duration):
            sleep_durations.append(duration)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            res = await retry_async(mock_coro, max_retries=5, base_delay=0.1, jitter=True, jitter_factor=0.5)
            self.assertEqual(res, "final_success")
            self.assertEqual(mock_coro.await_count, 4)
            self.assertEqual(len(sleep_durations), 3)

            # Each delay should be at least base_delay * (2 ** attempt) and bounded
            expected_bases = [0.1 * (2 ** i) for i in range(3)]  # [0.1, 0.2, 0.4]
            for actual, base in zip(sleep_durations, expected_bases):
                self.assertGreaterEqual(actual, base)
                self.assertLessEqual(actual, base * 1.6)  # bounded by base + jitter_factor * base

    async def test_retry_max_retries_exceeded_raises_original_error(self):
        mock_coro = AsyncMock(side_effect=SimulatedRateLimitError("Quota exceeded"))
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaises(SimulatedRateLimitError):
                await retry_async(mock_coro, max_retries=2, base_delay=0.01)
            # Initial call + 2 retries = 3 calls
            self.assertEqual(mock_coro.await_count, 3)

    async def test_non_retryable_error_does_not_retry(self):
        mock_coro = AsyncMock(side_effect=SimulatedBadRequestError("Invalid parameter"))
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with self.assertRaises(SimulatedBadRequestError):
                await retry_async(mock_coro, max_retries=3, base_delay=0.01)
            self.assertEqual(mock_coro.await_count, 1)
            mock_sleep.assert_not_called()


class TestJevModelEvaluator(unittest.IsolatedAsyncioTestCase):
    """Test JevModelEvaluator async evaluate, latency measurement, and retry."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.toxic_answer = MockScoreAnswer(
            probabilities={"0": 0.85, "1": 0.10, "2": 0.05},
            score=0.20,
            confidence=0.91,
        )
        self.threat_answer = MockNoulAnswer(noul=0.02)
        self.identity_hate_answer = MockNoulAnswer(noul=0.01)
        self.jev_response = MockJevResponse(
            toxic_answer=self.toxic_answer,
            threat_answer=self.threat_answer,
            identity_hate_answer=self.identity_hate_answer,
            input_tokens=75,
            output_tokens=0,
        )

    async def test_jev_evaluate_interface_and_schema(self):
        self.mock_client.system_one = AsyncMock(return_value=self.jev_response)
        evaluator = JevModelEvaluator(client=self.mock_client)

        calls_row, answers_rows = await evaluator.evaluate("c_001", "This is a benign comment.")

        # Verify calls_row schema
        self.assertEqual(calls_row["comment_id"], "c_001")
        self.assertEqual(calls_row["model"], "jev")
        self.assertIsInstance(calls_row["latency_ms"], float)
        self.assertGreater(calls_row["latency_ms"], 0.0)
        self.assertEqual(calls_row["input_tokens"], 75)
        self.assertEqual(calls_row["output_tokens"], 0)
        self.assertAlmostEqual(calls_row["cost_usd"], 75 * (0.042 / 1_000_000.0), places=8)
        self.assertIn("answers", calls_row["raw_response_json"])
        self.assertTrue(len(calls_row["timestamp"]) > 0)

        # Verify client.system_one call arguments
        self.mock_client.system_one.assert_awaited_once_with(
            model="jev-latest",
            state="This is a benign comment.",
            questions=JEV_QUESTIONS,
        )

        # Verify answers_rows (3 categories: toxic, threat, identity_hate)
        self.assertEqual(len(answers_rows), 3)
        cat_map = {row["category"]: row for row in answers_rows}

        # Toxic row has native confidence from Score
        self.assertIn("toxic", cat_map)
        self.assertEqual(cat_map["toxic"]["model"], "jev")
        self.assertAlmostEqual(cat_map["toxic"]["toxic_prob_0"], 0.85)
        self.assertAlmostEqual(cat_map["toxic"]["toxic_score"], 0.20)
        self.assertAlmostEqual(cat_map["toxic"]["native_confidence"], 0.91)
        self.assertIsNone(cat_map["toxic"]["binary_probability"])

        # Threat row has derived confidence 2 * |p - 0.5|
        self.assertIn("threat", cat_map)
        self.assertAlmostEqual(cat_map["threat"]["binary_probability"], 0.02)
        self.assertAlmostEqual(cat_map["threat"]["native_confidence"], 2.0 * abs(0.02 - 0.5))

        # Identity hate row
        self.assertIn("identity_hate", cat_map)
        self.assertAlmostEqual(cat_map["identity_hate"]["binary_probability"], 0.01)
        self.assertAlmostEqual(cat_map["identity_hate"]["native_confidence"], 2.0 * abs(0.01 - 0.5))

    async def test_jev_evaluate_latency_measurement(self):
        async def delayed_system_one(**kwargs):
            await asyncio.sleep(0.02)  # 20ms delay
            return self.jev_response

        self.mock_client.system_one = AsyncMock(side_effect=delayed_system_one)
        evaluator = JevModelEvaluator(client=self.mock_client)

        calls_row, _ = await evaluator.evaluate("c_002", "Some text")
        # Latency should be at least 15ms
        self.assertGreaterEqual(calls_row["latency_ms"], 15.0)

    async def test_jev_evaluate_retries_on_rate_limit(self):
        self.mock_client.system_one = AsyncMock(side_effect=[
            SimulatedRateLimitError("429 Too Many Requests"),
            self.jev_response,
        ])
        evaluator = JevModelEvaluator(client=self.mock_client, base_delay=0.01)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            calls_row, answers_rows = await evaluator.evaluate("c_003", "Retried comment")
            self.assertEqual(self.mock_client.system_one.await_count, 2)
            self.assertEqual(mock_sleep.await_count, 1)
            self.assertEqual(calls_row["comment_id"], "c_003")
            self.assertEqual(len(answers_rows), 3)

    def test_jev_missing_sdk_raises_import_error(self):
        with patch("models.jev_model.AsyncTypeSafeClient", None):
            with self.assertRaises(ImportError) as ctx:
                JevModelEvaluator(client=None)
            self.assertIn("typesafe-sdk", str(ctx.exception))


class TestAnthropicModelEvaluator(unittest.IsolatedAsyncioTestCase):
    """Test AnthropicModelEvaluator for Haiku and Sonnet."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.llm_json = {
            "toxic": {"probabilities": {"0": 0.70, "1": 0.25, "2": 0.05}},
            "threat": {"probability": 0.03},
            "identity_hate": {"probability": 0.02},
        }
        self.anthropic_response = MockAnthropicMessageResponse(
            tool_input=self.llm_json,
            input_tokens=100,
            output_tokens=30,
        )

    async def test_haiku_evaluate_interface_and_schema(self):
        self.mock_client.messages = MagicMock()
        self.mock_client.messages.create = AsyncMock(return_value=self.anthropic_response)

        evaluator = AnthropicModelEvaluator(
            model_type="haiku",
            client=self.mock_client,
        )

        calls_row, answers_rows = await evaluator.evaluate("c_haiku_1", "Haiku comment test")

        # Verify calls_row
        self.assertEqual(calls_row["comment_id"], "c_haiku_1")
        self.assertEqual(calls_row["model"], "haiku")
        self.assertIsInstance(calls_row["latency_ms"], float)
        self.assertGreater(calls_row["latency_ms"], 0.0)
        self.assertEqual(calls_row["input_tokens"], 100)
        self.assertEqual(calls_row["output_tokens"], 30)
        # Haiku pricing: $0.80/MTok input, $4.00/MTok output
        expected_cost = (100 * 0.80 + 30 * 4.00) / 1_000_000.0
        self.assertAlmostEqual(calls_row["cost_usd"], expected_cost, places=8)

        # Verify messages.create call
        self.mock_client.messages.create.assert_awaited_once()
        _, kwargs = self.mock_client.messages.create.call_args
        self.assertEqual(kwargs["model"], "claude-haiku-4-5")
        self.assertEqual(kwargs["tools"], [CLAUDE_CLASSIFICATION_TOOL])
        self.assertEqual(kwargs["tool_choice"], {"type": "auto"})
        self.assertIn("Haiku comment test", kwargs["system"])

        # Verify answers_rows (native_confidence is None for all LLMs)
        self.assertEqual(len(answers_rows), 3)
        cat_map = {row["category"]: row for row in answers_rows}
        self.assertIsNone(cat_map["toxic"]["native_confidence"])
        self.assertIsNone(cat_map["threat"]["native_confidence"])
        self.assertIsNone(cat_map["identity_hate"]["native_confidence"])
        self.assertAlmostEqual(cat_map["toxic"]["toxic_prob_0"], 0.70)
        self.assertAlmostEqual(cat_map["toxic"]["toxic_score"], 0.0 * 0.7 + 1.0 * 0.25 + 2.0 * 0.05)

    async def test_sonnet_evaluate_interface_and_schema(self):
        self.mock_client.messages = MagicMock()
        self.mock_client.messages.create = AsyncMock(return_value=self.anthropic_response)

        evaluator = AnthropicModelEvaluator(
            model_type="sonnet",
            client=self.mock_client,
        )

        calls_row, answers_rows = await evaluator.evaluate("c_sonnet_1", "Sonnet comment test")

        self.assertEqual(calls_row["model"], "sonnet")
        # Sonnet pricing: $3.00/MTok input, $15.00/MTok output
        expected_cost = (100 * 3.00 + 30 * 15.00) / 1_000_000.0
        self.assertAlmostEqual(calls_row["cost_usd"], expected_cost, places=8)

        _, kwargs = self.mock_client.messages.create.call_args
        self.assertEqual(kwargs["model"], "claude-sonnet-5-5")

    async def test_anthropic_retries_on_rate_limit(self):
        self.mock_client.messages = MagicMock()
        self.mock_client.messages.create = AsyncMock(side_effect=[
            SimulatedRateLimitError("Anthropic rate limit 429"),
            self.anthropic_response,
        ])
        evaluator = AnthropicModelEvaluator(
            model_type="haiku",
            client=self.mock_client,
            base_delay=0.01,
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            calls_row, answers_rows = await evaluator.evaluate("c_haiku_retry", "Retry text")
            self.assertEqual(self.mock_client.messages.create.await_count, 2)
            self.assertEqual(mock_sleep.await_count, 1)
            self.assertEqual(calls_row["comment_id"], "c_haiku_retry")

    def test_anthropic_missing_sdk_raises_import_error(self):
        with patch("models.anthropic_model.AsyncAnthropic", None):
            with self.assertRaises(ImportError) as ctx:
                AnthropicModelEvaluator(model_type="haiku", client=None)
            self.assertIn("anthropic", str(ctx.exception))

    def test_anthropic_workspace_id_header(self):
        with patch("models.anthropic_model.AsyncAnthropic") as mock_cls:
            evaluator = AnthropicModelEvaluator(
                model_type="haiku",
                client=None,
                api_key="sk-test",
                workspace_id="wrkspc_test_123",
            )
            mock_cls.assert_called_once_with(
                api_key="sk-test",
                default_headers={"anthropic-workspace-id": "wrkspc_test_123"},
            )


class TestGeminiModelEvaluator(unittest.IsolatedAsyncioTestCase):
    """Test GeminiModelEvaluator for Flash."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.llm_json = {
            "toxic": {"probabilities": {"0": 0.90, "1": 0.08, "2": 0.02}},
            "threat": {"probability": 0.01},
            "identity_hate": {"probability": 0.01},
        }
        self.gemini_response = MockGeminiResponse(
            json_data=self.llm_json,
            prompt_token_count=130,
            candidates_token_count=35,
        )

    async def test_flash_evaluate_interface_and_schema(self):
        self.mock_client.aio = MagicMock()
        self.mock_client.aio.models = MagicMock()
        self.mock_client.aio.models.generate_content = AsyncMock(return_value=self.gemini_response)

        evaluator = GeminiModelEvaluator(client=self.mock_client)

        calls_row, answers_rows = await evaluator.evaluate("c_flash_1", "Gemini flash comment test")

        # Verify calls_row
        self.assertEqual(calls_row["comment_id"], "c_flash_1")
        self.assertEqual(calls_row["model"], "flash")
        self.assertIsInstance(calls_row["latency_ms"], float)
        self.assertGreater(calls_row["latency_ms"], 0.0)
        self.assertEqual(calls_row["input_tokens"], 130)
        self.assertEqual(calls_row["output_tokens"], 35)
        # Flash pricing: $0.075/MTok input, $0.30/MTok output
        expected_cost = (130 * 0.075 + 35 * 0.30) / 1_000_000.0
        self.assertAlmostEqual(calls_row["cost_usd"], expected_cost, places=8)

        # Verify generate_content call
        self.mock_client.aio.models.generate_content.assert_awaited_once()
        _, kwargs = self.mock_client.aio.models.generate_content.call_args
        self.assertIn("flash", kwargs["model"].lower())
        self.assertIn("Gemini flash comment test", kwargs["contents"])

        # Check config schema
        config = kwargs["config"]
        if isinstance(config, dict):
            self.assertEqual(config.get("response_mime_type"), "application/json")
            self.assertEqual(config.get("response_schema"), GEMINI_RESPONSE_SCHEMA)
        else:
            self.assertEqual(getattr(config, "response_mime_type", None), "application/json")

        # Verify answers_rows (native_confidence is None for Flash)
        self.assertEqual(len(answers_rows), 3)
        cat_map = {row["category"]: row for row in answers_rows}
        self.assertIsNone(cat_map["toxic"]["native_confidence"])
        self.assertIsNone(cat_map["threat"]["native_confidence"])
        self.assertIsNone(cat_map["identity_hate"]["native_confidence"])
        self.assertAlmostEqual(cat_map["toxic"]["toxic_prob_0"], 0.90)

    async def test_gemini_retries_on_503(self):
        self.mock_client.aio = MagicMock()
        self.mock_client.aio.models = MagicMock()
        self.mock_client.aio.models.generate_content = AsyncMock(side_effect=[
            SimulatedServiceUnavailableError("503 Service Unavailable"),
            self.gemini_response,
        ])
        evaluator = GeminiModelEvaluator(client=self.mock_client, base_delay=0.01)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            calls_row, answers_rows = await evaluator.evaluate("c_flash_retry", "Flash retry text")
            self.assertEqual(self.mock_client.aio.models.generate_content.await_count, 2)
            self.assertEqual(mock_sleep.await_count, 1)
            self.assertEqual(calls_row["comment_id"], "c_flash_retry")

    def test_gemini_missing_sdk_raises_import_error(self):
        with patch("models.gemini_model.genai", None):
            with self.assertRaises(ImportError) as ctx:
                GeminiModelEvaluator(client=None)
            self.assertIn("google-genai", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
