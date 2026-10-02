"""Pilot dry-run runner for Trust & Safety Eval Gate Benchmark.

Runs a sanity check on a small sample of comments (default 50) across
specified models (default Jev) to verify API connectivity, latency,
cost tracking, and schema alignment before bulk execution.
"""

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src/ is in sys.path regardless of execution directory
_BASE_DIR = Path(__file__).resolve().parent.parent
_SRC_DIR = _BASE_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dotenv import load_dotenv

from checkpoint import CheckpointManager
from models import get_model_evaluator
from models.base import BaseModelEvaluator


def load_prompts(prompts_path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Load prompts from CSV containing 'id' and 'comment_text'."""
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompts file not found at {prompts_path}")

    prompts = []
    with open(prompts_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("id")
            text = row.get("comment_text", "")
            if cid is not None:
                prompts.append({"id": str(cid), "comment_text": str(text)})
                if limit is not None and len(prompts) >= limit:
                    break
    return prompts


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile from a list of numerical values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (percentile / 100.0)
    floor_idx = int(k)
    ceil_idx = floor_idx + 1
    if ceil_idx < len(sorted_vals):
        fraction = k - floor_idx
        return sorted_vals[floor_idx] * (1.0 - fraction) + sorted_vals[ceil_idx] * fraction
    return float(sorted_vals[floor_idx])


def format_summary_table(stats: Dict[str, Dict[str, Any]]) -> str:
    """Format evaluation statistics into a clean text table."""
    headers = [
        "Model",
        "Processed",
        "Errors",
        "Latency p50",
        "Latency p95",
        "Total Tokens",
        "Total Cost ($)",
    ]
    col_widths = [10, 11, 8, 14, 14, 14, 16]

    def format_row(cols: List[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cols, col_widths))

    separator = "-+-".join("-" * w for w in col_widths)
    lines = [
        format_row(headers),
        separator,
    ]

    for model, m_stats in stats.items():
        row = [
            model,
            str(m_stats.get("processed", 0)),
            str(m_stats.get("errors", 0)),
            f"{m_stats.get('latency_p50', 0.0):.1f}ms",
            f"{m_stats.get('latency_p95', 0.0):.1f}ms",
            f"{m_stats.get('total_tokens', 0):,}",
            f"${m_stats.get('total_cost', 0.0):.6f}",
        ]
        lines.append(format_row(row))

    return "\n".join(lines)


async def evaluate_single_comment(
    evaluator: BaseModelEvaluator,
    checkpoint: CheckpointManager,
    item: Dict[str, str],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Evaluate a single comment with concurrency control and checkpoint recording."""
    cid = item["id"]
    comment_text = item["comment_text"]

    async with semaphore:
        try:
            calls_row, answers_rows = await evaluator.evaluate(
                comment_id=cid,
                comment_text=comment_text,
            )
            checkpoint.record_result(calls_row, answers_rows)
            return {
                "success": True,
                "calls_row": calls_row,
                "error": None,
            }
        except Exception as exc:
            # Privacy: never print or log raw comment_text
            sys.stderr.write(
                f"[PILOT WARNING] Error evaluating comment_id={cid} for model={evaluator.model_name}: {exc}\n"
            )
            return {
                "success": False,
                "calls_row": None,
                "error": str(exc),
            }


async def run_pilot(
    prompts_file: str = "jev-experiments/data/eval_6k_prompts.csv",
    sample_size: int = 50,
    models: Optional[List[str]] = None,
    output_dir: str = "jev-experiments/pilot_results",
    concurrency: int = 5,
    evaluators: Optional[Dict[str, BaseModelEvaluator]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Execute pilot dry-run across selected models on a sample of prompts."""
    load_dotenv(_BASE_DIR / ".env")
    load_dotenv(_BASE_DIR.parent / ".env")
    load_dotenv()

    if models is None:
        models = ["jev"]

    os.makedirs(output_dir, exist_ok=True)
    calls_csv = os.path.join(output_dir, "calls.csv")
    answers_csv = os.path.join(output_dir, "answers.csv")
    checkpoint = CheckpointManager(calls_csv, answers_csv)

    prompts = load_prompts(prompts_file, limit=sample_size)
    total_prompts = len(prompts)
    print(f"Loaded {total_prompts} sample prompts from {prompts_file} for pilot evaluation.")

    active_evaluators: Dict[str, BaseModelEvaluator] = {}
    for model_name in models:
        m = model_name.strip()
        if not m:
            continue
        if evaluators and m in evaluators:
            active_evaluators[m] = evaluators[m]
        else:
            active_evaluators[m] = get_model_evaluator(m)

    semaphore = asyncio.Semaphore(concurrency)
    overall_stats: Dict[str, Dict[str, Any]] = {}

    for model_name, evaluator in active_evaluators.items():
        print(f"\n--- Running Pilot for Model: {model_name} (sample size: {total_prompts}) ---")
        completed_ids = checkpoint.get_completed_ids(model_name)
        pending_prompts = [p for p in prompts if p["id"] not in completed_ids]

        if len(pending_prompts) < len(prompts):
            print(f"Skipping {len(prompts) - len(pending_prompts)} already completed prompts for {model_name}.")

        tasks = [
            evaluate_single_comment(evaluator, checkpoint, item, semaphore)
            for item in pending_prompts
        ]

        results = await asyncio.gather(*tasks)

        processed = 0
        errors = 0
        latencies: List[float] = []
        total_tokens = 0
        total_cost = 0.0

        for r in results:
            if r["success"]:
                processed += 1
                row = r["calls_row"]
                latencies.append(float(row.get("latency_ms", 0.0)))
                total_tokens += int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0))
                total_cost += float(row.get("cost_usd", 0.0))
            else:
                errors += 1

        p50 = calculate_percentile(latencies, 50)
        p95 = calculate_percentile(latencies, 95)

        overall_stats[model_name] = {
            "processed": processed,
            "errors": errors,
            "latency_p50": p50,
            "latency_p95": p95,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }

    print("\n======================= PILOT SUMMARY =======================")
    table_str = format_summary_table(overall_stats)
    print(table_str)
    print("=============================================================\n")

    return overall_stats


def build_pilot_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser for run_pilot."""
    parser = argparse.ArgumentParser(
        description="Run dry-run pilot evaluation for Trust & Safety Eval Gate Benchmark."
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="jev-experiments/data/eval_6k_prompts.csv",
        help="Path to prompts CSV file (default: jev-experiments/data/eval_6k_prompts.csv)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of prompt comments to evaluate (default: 50)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="jev",
        help="Comma-separated model names (default: jev, e.g. jev,haiku)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="jev-experiments/pilot_results",
        help="Directory to save checkpoint files (default: jev-experiments/pilot_results)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent asynchronous requests (default: 5)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for run_pilot."""
    parser = build_pilot_arg_parser()
    args = parser.parse_args(argv)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    asyncio.run(
        run_pilot(
            prompts_file=args.prompts_file,
            sample_size=args.sample_size,
            models=model_list,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
