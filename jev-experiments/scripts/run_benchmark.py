"""Full asynchronous benchmark runner for Trust & Safety Eval Gate Benchmark.

Orchestrates 6,000 comments across 4 models (Jev, Claude 3.5 Haiku,
Gemini 1.5 Flash, Claude 3.5 Sonnet) with provider-specific semaphores,
incremental checkpointing, resumption, real-time progress updates,
and graceful signal handling.
"""

import argparse
import asyncio
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src/ and scripts/ are in sys.path
_BASE_DIR = Path(__file__).resolve().parent.parent
_SRC_DIR = _BASE_DIR / "src"
_SCRIPTS_DIR = _BASE_DIR / "scripts"
for p in [str(_SRC_DIR), str(_SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

from checkpoint import CheckpointManager
from models import get_model_evaluator
from models.base import BaseModelEvaluator
from run_pilot import calculate_percentile, format_summary_table, load_prompts


def get_provider_semaphore(
    model_name: str,
    semaphores: Dict[str, asyncio.Semaphore],
) -> asyncio.Semaphore:
    """Return the corresponding provider semaphore for a model.

    Provider mapping:
    - Jev: semaphores['jev']
    - Flash / Gemini: semaphores['flash']
    - Haiku / Sonnet / Anthropic: semaphores['anthropic'] (shared)
    """
    name = model_name.lower().strip()
    if "jev" in name:
        return semaphores["jev"]
    elif "flash" in name or "gemini" in name:
        return semaphores["flash"]
    elif "haiku" in name or "sonnet" in name or "anthropic" in name:
        return semaphores["anthropic"]
    else:
        # Fallback to model-specific key if present or first available semaphore
        return semaphores.get(name, next(iter(semaphores.values())))


def setup_signal_handlers(stop_event: asyncio.Event) -> None:
    """Attach SIGINT and SIGTERM handlers to trigger graceful stop."""
    def _signal_handler(signum, frame):
        sys.stderr.write(
            f"\n[BENCHMARK] Signal {signum} received. Requesting graceful shutdown...\n"
        )
        stop_event.set()

    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
        except (ValueError, AttributeError):
            pass


async def run_benchmark(
    prompts_file: str = "jev-experiments/data/eval_6k_prompts.csv",
    models: Optional[List[str]] = None,
    output_dir: str = "jev-experiments/results",
    limit: Optional[int] = None,
    concurrency_jev: int = 50,
    concurrency_flash: int = 30,
    concurrency_anthropic: int = 15,
    evaluators: Optional[Dict[str, BaseModelEvaluator]] = None,
    stop_event: Optional[asyncio.Event] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run full asynchronous evaluation across all specified models and prompts."""
    load_dotenv(_BASE_DIR / ".env")
    load_dotenv(_BASE_DIR.parent / ".env")
    load_dotenv()

    if models is None:
        models = ["jev", "haiku", "flash", "sonnet"]

    if stop_event is None:
        stop_event = asyncio.Event()

    setup_signal_handlers(stop_event)

    os.makedirs(output_dir, exist_ok=True)
    calls_csv = os.path.join(output_dir, "calls.csv")
    answers_csv = os.path.join(output_dir, "answers.csv")
    checkpoint = CheckpointManager(calls_csv, answers_csv)

    prompts = load_prompts(prompts_file, limit=limit)
    total_comments = len(prompts)
    print(f"[BENCHMARK] Loaded {total_comments} prompt comments from {prompts_file}.")

    # Initialize model evaluators
    active_evaluators: Dict[str, BaseModelEvaluator] = {}
    for model_name in models:
        m = model_name.strip()
        if not m:
            continue
        if evaluators and m in evaluators:
            active_evaluators[m] = evaluators[m]
        else:
            active_evaluators[m] = get_model_evaluator(m)

    # Provider semaphores
    semaphores: Dict[str, asyncio.Semaphore] = {
        "jev": asyncio.Semaphore(concurrency_jev),
        "flash": asyncio.Semaphore(concurrency_flash),
        "anthropic": asyncio.Semaphore(concurrency_anthropic),
    }

    # Tracking per model
    stats: Dict[str, Dict[str, Any]] = {
        m: {
            "processed": 0,
            "errors": 0,
            "latencies": [],
            "total_tokens": 0,
            "total_cost": 0.0,
        }
        for m in active_evaluators
    }

    total_tasks_planned = 0
    tasks_to_run = []

    for model_name, evaluator in active_evaluators.items():
        completed_ids = checkpoint.get_completed_ids(model_name)
        pending = [p for p in prompts if p["id"] not in completed_ids]
        skipped_count = total_comments - len(pending)
        if skipped_count > 0:
            print(
                f"[BENCHMARK] Resuming {model_name}: skipping {skipped_count} completed comments, {len(pending)} pending."
            )
        else:
            print(f"[BENCHMARK] Scheduled {len(pending)} comments for {model_name}.")

        total_tasks_planned += len(pending)
        for item in pending:
            tasks_to_run.append((model_name, evaluator, item))

    print(f"[BENCHMARK] Total evaluations queued: {total_tasks_planned}")

    completed_counter = 0
    lock = asyncio.Lock()
    last_progress_time = time.time()

    async def evaluate_task(model_name: str, evaluator: BaseModelEvaluator, item: Dict[str, str]) -> None:
        nonlocal completed_counter, last_progress_time

        if stop_event.is_set():
            return

        cid = item["id"]
        comment_text = item["comment_text"]
        provider_sem = get_provider_semaphore(model_name, semaphores)

        async with provider_sem:
            if stop_event.is_set():
                return
            try:
                calls_row, answers_rows = await evaluator.evaluate(
                    comment_id=cid,
                    comment_text=comment_text,
                )
                checkpoint.record_result(calls_row, answers_rows)

                async with lock:
                    completed_counter += 1
                    m_stat = stats[model_name]
                    m_stat["processed"] += 1
                    m_stat["latencies"].append(float(calls_row.get("latency_ms", 0.0)))
                    m_stat["total_tokens"] += int(calls_row.get("input_tokens", 0)) + int(
                        calls_row.get("output_tokens", 0)
                    )
                    m_stat["total_cost"] += float(calls_row.get("cost_usd", 0.0))

            except Exception as exc:
                # Privacy: never print or log raw comment_text
                sys.stderr.write(
                    f"[BENCHMARK ERROR] Failed comment_id={cid} for model={model_name}: {exc}\n"
                )
                async with lock:
                    completed_counter += 1
                    stats[model_name]["errors"] += 1

            # Progress update every 1 second or every 50 tasks
            async with lock:
                now = time.time()
                if (now - last_progress_time >= 1.0) or completed_counter == total_tasks_planned:
                    last_progress_time = now
                    pct = (
                        (completed_counter / total_tasks_planned * 100.0)
                        if total_tasks_planned > 0
                        else 100.0
                    )
                    progress_breakdown = ", ".join(
                        f"{m}: {stats[m]['processed']}" for m in active_evaluators
                    )
                    print(
                        f"[PROGRESS] {completed_counter}/{total_tasks_planned} ({pct:.1f}%) | "
                        f"{progress_breakdown} | Errors: {sum(s['errors'] for s in stats.values())}"
                    )

    # Launch evaluation tasks concurrently
    task_futures = [
        asyncio.create_task(evaluate_task(model_name, evaluator, item))
        for (model_name, evaluator, item) in tasks_to_run
    ]

    if task_futures:
        await asyncio.gather(*task_futures, return_exceptions=True)

    # Format final overall stats
    final_stats: Dict[str, Dict[str, Any]] = {}
    for model_name, s in stats.items():
        latencies = s["latencies"]
        p50 = calculate_percentile(latencies, 50)
        p95 = calculate_percentile(latencies, 95)
        final_stats[model_name] = {
            "processed": s["processed"],
            "errors": s["errors"],
            "latency_p50": p50,
            "latency_p95": p95,
            "total_tokens": s["total_tokens"],
            "total_cost": s["total_cost"],
        }

    print("\n==================== BENCHMARK FINAL SUMMARY ====================")
    print(format_summary_table(final_stats))
    print("=================================================================\n")

    return final_stats


def build_benchmark_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser for run_benchmark."""
    parser = argparse.ArgumentParser(
        description="Run full asynchronous evaluation benchmark for Trust & Safety Eval Gate."
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="jev-experiments/data/eval_6k_prompts.csv",
        help="Path to prompts CSV (default: jev-experiments/data/eval_6k_prompts.csv)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="jev,haiku,flash,sonnet",
        help="Comma-separated models to evaluate (default: jev,haiku,flash,sonnet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="jev-experiments/results",
        help="Directory to save calls.csv and answers.csv (default: jev-experiments/results)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of comments to evaluate",
    )
    parser.add_argument(
        "--concurrency-jev",
        type=int,
        default=50,
        help="Concurrency semaphore for Jev (default: 50)",
    )
    parser.add_argument(
        "--concurrency-flash",
        type=int,
        default=30,
        help="Concurrency semaphore for Gemini Flash (default: 30)",
    )
    parser.add_argument(
        "--concurrency-anthropic",
        type=int,
        default=15,
        help="Concurrency semaphore for Anthropic models Haiku/Sonnet (default: 15)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for run_benchmark."""
    parser = build_benchmark_arg_parser()
    args = parser.parse_args(argv)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    asyncio.run(
        run_benchmark(
            prompts_file=args.prompts_file,
            models=model_list,
            output_dir=args.output_dir,
            limit=args.limit,
            concurrency_jev=args.concurrency_jev,
            concurrency_flash=args.concurrency_flash,
            concurrency_anthropic=args.concurrency_anthropic,
        )
    )


if __name__ == "__main__":
    main()
