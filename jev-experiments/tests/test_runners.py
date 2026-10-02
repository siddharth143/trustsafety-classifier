"""Unit and integration tests for T&S eval gate benchmark runners (run_pilot.py and run_benchmark.py)."""

import asyncio
import csv
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
for p in [str(_ROOT / "src"), str(_ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.base import BaseModelEvaluator
from run_pilot import build_pilot_arg_parser, format_summary_table, load_prompts, run_pilot
from run_benchmark import (
    build_benchmark_arg_parser,
    get_provider_semaphore,
    run_benchmark,
)


class MockEvaluator(BaseModelEvaluator):
    """Mock model evaluator for testing runner scripts."""

    def __init__(
        self,
        model_name: str,
        latency: float = 20.0,
        should_fail_ids: set = None,
        delay: float = 0.0,
        cost_usd: float = 0.0001,
        input_tokens: int = 100,
        output_tokens: int = 20,
    ):
        super().__init__(model_name=model_name)
        self.latency = latency
        self.should_fail_ids = should_fail_ids or set()
        self.delay = delay
        self.cost_usd = cost_usd
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.call_count = 0
        self.evaluated_ids = []
        self.current_concurrency = 0
        self.max_concurrency = 0

    async def _execute_call(self, comment_id: str, comment_text: str, latency_ms: float):
        self.current_concurrency += 1
        if self.current_concurrency > self.max_concurrency:
            self.max_concurrency = self.current_concurrency
        try:
            if self.delay > 0:
                await asyncio.sleep(self.delay)
            if comment_id in self.should_fail_ids:
                raise RuntimeError(f"Simulated failure for comment {comment_id}")
            self.call_count += 1
            self.evaluated_ids.append(comment_id)
            calls_row = {
                "comment_id": comment_id,
                "model": self.model_name,
                "latency_ms": latency_ms,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": self.cost_usd,
                "raw_response_json": "{}",
                "timestamp": "2026-10-01T20:00:00Z",
            }
            answers_rows = [
                {
                    "comment_id": comment_id,
                    "model": self.model_name,
                    "category": "toxic",
                    "toxic_prob_0": 0.8,
                    "toxic_prob_1": 0.15,
                    "toxic_prob_2": 0.05,
                    "toxic_score": 0.25,
                    "binary_probability": None,
                    "native_confidence": 0.85 if "jev" in self.model_name else None,
                },
                {
                    "comment_id": comment_id,
                    "model": self.model_name,
                    "category": "threat",
                    "toxic_prob_0": None,
                    "toxic_prob_1": None,
                    "toxic_prob_2": None,
                    "toxic_score": None,
                    "binary_probability": 0.02,
                    "native_confidence": 0.96 if "jev" in self.model_name else None,
                },
                {
                    "comment_id": comment_id,
                    "model": self.model_name,
                    "category": "identity_hate",
                    "toxic_prob_0": None,
                    "toxic_prob_1": None,
                    "toxic_prob_2": None,
                    "toxic_score": None,
                    "binary_probability": 0.01,
                    "native_confidence": 0.98 if "jev" in self.model_name else None,
                },
            ]
            return calls_row, answers_rows
        finally:
            self.current_concurrency -= 1


class TestRunPilot(unittest.IsolatedAsyncioTestCase):
    """Tests for run_pilot.py (Task 4)."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.prompts_file = os.path.join(self.test_dir, "test_prompts.csv")
        self.output_dir = os.path.join(self.test_dir, "pilot_results")

        # Create dummy prompts CSV with 10 rows
        with open(self.prompts_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "comment_text"])
            writer.writeheader()
            for i in range(10):
                writer.writerow({
                    "id": f"id_{i:03d}",
                    "comment_text": f"This is comment number {i} - confidential sensitive text.",
                })

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_prompts_sample_size(self):
        """Test loading prompts with limit / sample size."""
        prompts = load_prompts(self.prompts_file, limit=5)
        self.assertEqual(len(prompts), 5)
        self.assertEqual(prompts[0]["id"], "id_000")
        self.assertEqual(prompts[4]["id"], "id_004")

    async def test_pilot_runs_sample_and_records_checkpoints(self):
        """Test running pilot evaluation saves to calls.csv and answers.csv."""
        mock_jev = MockEvaluator("jev")
        evaluators = {"jev": mock_jev}

        stats = await run_pilot(
            prompts_file=self.prompts_file,
            sample_size=3,
            models=["jev"],
            output_dir=self.output_dir,
            concurrency=2,
            evaluators=evaluators,
        )

        calls_file = os.path.join(self.output_dir, "calls.csv")
        answers_file = os.path.join(self.output_dir, "answers.csv")
        self.assertTrue(os.path.exists(calls_file))
        self.assertTrue(os.path.exists(answers_file))

        with open(calls_file, "r", encoding="utf-8") as f:
            calls_rows = list(csv.DictReader(f))
        with open(answers_file, "r", encoding="utf-8") as f:
            answers_rows = list(csv.DictReader(f))

        self.assertEqual(len(calls_rows), 3)
        self.assertEqual(len(answers_rows), 9)  # 3 categories per call
        self.assertEqual(mock_jev.call_count, 3)

        # Check stats summary dictionary
        self.assertIn("jev", stats)
        self.assertEqual(stats["jev"]["processed"], 3)
        self.assertEqual(stats["jev"]["errors"], 0)
        self.assertGreater(stats["jev"]["latency_p50"], 0)
        self.assertGreater(stats["jev"]["latency_p95"], 0)
        self.assertEqual(stats["jev"]["total_tokens"], 3 * (100 + 20))
        self.assertAlmostEqual(stats["jev"]["total_cost"], 3 * 0.0001, places=6)

    async def test_pilot_multi_model(self):
        """Test running pilot across multiple models."""
        mock_jev = MockEvaluator("jev")
        mock_haiku = MockEvaluator("haiku")
        evaluators = {"jev": mock_jev, "haiku": mock_haiku}

        stats = await run_pilot(
            prompts_file=self.prompts_file,
            sample_size=4,
            models=["jev", "haiku"],
            output_dir=self.output_dir,
            concurrency=2,
            evaluators=evaluators,
        )

        self.assertEqual(mock_jev.call_count, 4)
        self.assertEqual(mock_haiku.call_count, 4)
        self.assertEqual(stats["jev"]["processed"], 4)
        self.assertEqual(stats["haiku"]["processed"], 4)

    async def test_pilot_summary_table_generation(self):
        """Test formatted summary table generation output."""
        mock_jev = MockEvaluator("jev")
        evaluators = {"jev": mock_jev}

        stats = await run_pilot(
            prompts_file=self.prompts_file,
            sample_size=3,
            models=["jev"],
            output_dir=self.output_dir,
            evaluators=evaluators,
        )
        table_output = format_summary_table(stats)
        self.assertIn("jev", table_output)
        self.assertIn("p50", table_output.lower())
        self.assertIn("p95", table_output.lower())
        self.assertIn("cost", table_output.lower())

    async def test_pilot_privacy_no_comment_text_leak(self):
        """Verify under no circumstances are raw comment texts printed to stdout/stderr."""
        secret_text = "SECRET_SUPER_CONFIDENTIAL_COMMENT_TEXT_12345"
        with open(self.prompts_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "comment_text"])
            writer.writeheader()
            writer.writerow({"id": "priv_001", "comment_text": secret_text})

        mock_jev = MockEvaluator("jev")
        evaluators = {"jev": mock_jev}

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with patch("sys.stdout", stdout_capture), patch("sys.stderr", stderr_capture):
            await run_pilot(
                prompts_file=self.prompts_file,
                sample_size=1,
                models=["jev"],
                output_dir=self.output_dir,
                evaluators=evaluators,
            )

        captured_stdout = stdout_capture.getvalue()
        captured_stderr = stderr_capture.getvalue()

        self.assertNotIn(secret_text, captured_stdout)
        self.assertNotIn(secret_text, captured_stderr)

    async def test_pilot_error_handling_continues(self):
        """Test that if one comment fails, pilot does not crash and logs error count."""
        mock_jev = MockEvaluator("jev", should_fail_ids={"id_001"})
        evaluators = {"jev": mock_jev}

        stats = await run_pilot(
            prompts_file=self.prompts_file,
            sample_size=3,
            models=["jev"],
            output_dir=self.output_dir,
            evaluators=evaluators,
        )

        self.assertEqual(stats["jev"]["processed"], 2)
        self.assertEqual(stats["jev"]["errors"], 1)

    def test_pilot_arg_parser_defaults_and_custom(self):
        """Test CLI argument parsing for run_pilot."""
        parser = build_pilot_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.sample_size, 50)
        self.assertEqual(args.models, "jev")
        self.assertEqual(args.concurrency, 5)
        self.assertEqual(args.output_dir, "jev-experiments/pilot_results")

        custom_args = parser.parse_args([
            "--prompts-file", "my_prompts.csv",
            "--sample-size", "10",
            "--models", "jev,haiku,flash",
            "--output-dir", "custom_dir",
            "--concurrency", "8",
        ])
        self.assertEqual(custom_args.sample_size, 10)
        self.assertEqual(custom_args.models, "jev,haiku,flash")
        self.assertEqual(custom_args.concurrency, 8)
        self.assertEqual(custom_args.output_dir, "custom_dir")


class TestRunBenchmark(unittest.IsolatedAsyncioTestCase):
    """Tests for run_benchmark.py (Task 5)."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.prompts_file = os.path.join(self.test_dir, "test_prompts.csv")
        self.output_dir = os.path.join(self.test_dir, "benchmark_results")

        # Create dummy prompts CSV with 6 rows
        with open(self.prompts_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "comment_text"])
            writer.writeheader()
            for i in range(6):
                writer.writerow({
                    "id": f"id_{i:03d}",
                    "comment_text": f"Benchmark comment text {i}",
                })

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_benchmark_orchestration_all_models(self):
        """Test orchestrating 4 models across prompts."""
        mock_evaluators = {
            "jev": MockEvaluator("jev"),
            "haiku": MockEvaluator("haiku"),
            "flash": MockEvaluator("flash"),
            "sonnet": MockEvaluator("sonnet"),
        }

        stats = await run_benchmark(
            prompts_file=self.prompts_file,
            models=["jev", "haiku", "flash", "sonnet"],
            output_dir=self.output_dir,
            limit=3,
            evaluators=mock_evaluators,
        )

        calls_file = os.path.join(self.output_dir, "calls.csv")
        with open(calls_file, "r", encoding="utf-8") as f:
            calls = list(csv.DictReader(f))

        # 3 prompts * 4 models = 12 calls
        self.assertEqual(len(calls), 12)
        for model in ["jev", "haiku", "flash", "sonnet"]:
            self.assertEqual(mock_evaluators[model].call_count, 3)
            self.assertEqual(stats[model]["processed"], 3)
            self.assertEqual(stats[model]["errors"], 0)

    async def test_benchmark_resumption_skips_completed(self):
        """Test that already completed (comment_id, model) pairs are skipped."""
        from checkpoint import CheckpointManager

        calls_file = os.path.join(self.output_dir, "calls.csv")
        answers_file = os.path.join(self.output_dir, "answers.csv")
        ckpt = CheckpointManager(calls_file, answers_file)

        # Pre-record id_000 for jev and haiku
        ckpt.record_result(
            {
                "comment_id": "id_000",
                "model": "jev",
                "latency_ms": 15.0,
                "input_tokens": 50,
                "output_tokens": 0,
                "cost_usd": 0.00005,
                "raw_response_json": "{}",
                "timestamp": "2026-10-01T12:00:00Z",
            },
            [],
        )
        ckpt.record_result(
            {
                "comment_id": "id_000",
                "model": "haiku",
                "latency_ms": 45.0,
                "input_tokens": 100,
                "output_tokens": 20,
                "cost_usd": 0.0001,
                "raw_response_json": "{}",
                "timestamp": "2026-10-01T12:00:00Z",
            },
            [],
        )

        mock_evaluators = {
            "jev": MockEvaluator("jev"),
            "haiku": MockEvaluator("haiku"),
        }

        stats = await run_benchmark(
            prompts_file=self.prompts_file,
            models=["jev", "haiku"],
            output_dir=self.output_dir,
            limit=3,
            evaluators=mock_evaluators,
        )

        # Out of 3 items (id_000, id_001, id_002), id_000 was already completed
        # So each should only evaluate 2 items
        self.assertEqual(mock_evaluators["jev"].call_count, 2)
        self.assertEqual(mock_evaluators["haiku"].call_count, 2)
        self.assertNotIn("id_000", mock_evaluators["jev"].evaluated_ids)
        self.assertNotIn("id_000", mock_evaluators["haiku"].evaluated_ids)

    async def test_benchmark_concurrency_control_with_provider_semaphores(self):
        """Test that provider semaphores bound concurrency, including shared Anthropic semaphore."""
        mock_evaluators = {
            "haiku": MockEvaluator("haiku", delay=0.05),
            "sonnet": MockEvaluator("sonnet", delay=0.05),
        }

        # Track active calls across haiku and sonnet
        active_anthropic = 0
        max_active_anthropic = 0
        anthropic_lock = asyncio.Lock()

        orig_haiku_exec = mock_evaluators["haiku"]._execute_call
        orig_sonnet_exec = mock_evaluators["sonnet"]._execute_call

        async def monitored_exec(orig_func, *args, **kwargs):
            nonlocal active_anthropic, max_active_anthropic
            async with anthropic_lock:
                active_anthropic += 1
                if active_anthropic > max_active_anthropic:
                    max_active_anthropic = active_anthropic
            try:
                return await orig_func(*args, **kwargs)
            finally:
                async with anthropic_lock:
                    active_anthropic -= 1

        mock_evaluators["haiku"]._execute_call = lambda *a, **k: monitored_exec(orig_haiku_exec, *a, **k)
        mock_evaluators["sonnet"]._execute_call = lambda *a, **k: monitored_exec(orig_sonnet_exec, *a, **k)

        await run_benchmark(
            prompts_file=self.prompts_file,
            models=["haiku", "sonnet"],
            output_dir=self.output_dir,
            limit=6,
            concurrency_anthropic=2,
            evaluators=mock_evaluators,
        )

        # Max concurrent Anthropic calls must not exceed 2
        self.assertLessEqual(max_active_anthropic, 2)

    async def test_benchmark_error_handling_resilience(self):
        """Test error handling when individual item evaluations fail."""
        mock_evaluators = {
            "jev": MockEvaluator("jev", should_fail_ids={"id_002"}),
        }

        stats = await run_benchmark(
            prompts_file=self.prompts_file,
            models=["jev"],
            output_dir=self.output_dir,
            limit=4,
            evaluators=mock_evaluators,
        )

        self.assertEqual(stats["jev"]["processed"], 3)
        self.assertEqual(stats["jev"]["errors"], 1)

    async def test_benchmark_privacy_no_comment_text_leak(self):
        """Test that benchmark runner does not print comment texts to stdout/stderr even with errors."""
        secret_benchmark_text = "SECRET_BENCHMARK_PROMPT_DO_NOT_LEAK_9999"
        with open(self.prompts_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "comment_text"])
            writer.writeheader()
            writer.writerow({"id": "bm_priv_01", "comment_text": secret_benchmark_text})
            writer.writerow({"id": "bm_priv_err", "comment_text": secret_benchmark_text + "_ERR"})

        mock_evaluators = {
            "jev": MockEvaluator("jev", should_fail_ids={"bm_priv_err"}),
        }

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with patch("sys.stdout", stdout_capture), patch("sys.stderr", stderr_capture):
            await run_benchmark(
                prompts_file=self.prompts_file,
                models=["jev"],
                output_dir=self.output_dir,
                evaluators=mock_evaluators,
            )

        captured_stdout = stdout_capture.getvalue()
        captured_stderr = stderr_capture.getvalue()

        self.assertNotIn(secret_benchmark_text, captured_stdout)
        self.assertNotIn(secret_benchmark_text, captured_stderr)

    async def test_benchmark_graceful_shutdown(self):
        """Test that stop_event / graceful shutdown cleanly terminates loop."""
        stop_event = asyncio.Event()

        # Mock evaluator that sets stop_event after 1 call
        class StoppingEvaluator(MockEvaluator):
            async def _execute_call(self, comment_id: str, comment_text: str, latency_ms: float):
                res = await super()._execute_call(comment_id, comment_text, latency_ms)
                stop_event.set()
                return res

        evaluators = {"jev": StoppingEvaluator("jev", delay=0.01)}

        stats = await run_benchmark(
            prompts_file=self.prompts_file,
            models=["jev"],
            output_dir=self.output_dir,
            limit=5,
            concurrency_jev=1,
            evaluators=evaluators,
            stop_event=stop_event,
        )

        # Should have stopped before finishing all 5 items
        self.assertLess(stats["jev"]["processed"], 5)

    def test_benchmark_arg_parser_defaults_and_custom(self):
        """Test CLI argument parsing for run_benchmark."""
        parser = build_benchmark_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.models, "jev,haiku,flash,sonnet")
        self.assertEqual(args.output_dir, "jev-experiments/results")
        self.assertEqual(args.concurrency_jev, 50)
        self.assertEqual(args.concurrency_flash, 30)
        self.assertEqual(args.concurrency_anthropic, 15)
        self.assertIsNone(args.limit)

        custom_args = parser.parse_args([
            "--prompts-file", "other_prompts.csv",
            "--models", "jev,flash",
            "--output-dir", "custom_res",
            "--limit", "100",
            "--concurrency-jev", "20",
            "--concurrency-flash", "10",
            "--concurrency-anthropic", "5",
        ])
        self.assertEqual(custom_args.prompts_file, "other_prompts.csv")
        self.assertEqual(custom_args.models, "jev,flash")
        self.assertEqual(custom_args.limit, 100)
        self.assertEqual(custom_args.concurrency_jev, 20)
        self.assertEqual(custom_args.concurrency_flash, 10)
        self.assertEqual(custom_args.concurrency_anthropic, 5)


if __name__ == "__main__":
    unittest.main()
