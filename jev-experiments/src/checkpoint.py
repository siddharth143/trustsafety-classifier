"""Incremental, thread-safe checkpoint manager for the T&S eval gate benchmark."""

import csv
import os
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

CALLS_FIELDNAMES = [
    "comment_id",
    "model",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "raw_response_json",
    "timestamp",
]

ANSWERS_FIELDNAMES = [
    "comment_id",
    "model",
    "category",
    "toxic_prob_0",
    "toxic_prob_1",
    "toxic_prob_2",
    "toxic_score",
    "binary_probability",
    "native_confidence",
]


class CheckpointManager:
    """Manages incremental and resumable CSV logging for calls and answers."""

    def __init__(self, calls_path: str, answers_path: str) -> None:
        self.calls_path = calls_path
        self.answers_path = answers_path
        self._lock = threading.Lock()
        self._completed_ids: Dict[str, Set[str]] = defaultdict(set)

        self._ensure_files()
        self._load_completed_ids()

    def _ensure_files(self) -> None:
        """Create target CSV files with headers if they do not already exist."""
        for path, fieldnames in [
            (self.calls_path, CALLS_FIELDNAMES),
            (self.answers_path, ANSWERS_FIELDNAMES),
        ]:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            if not os.path.exists(path) or os.path.getsize(path) == 0:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

    def _load_completed_ids(self) -> None:
        """Load already processed comment IDs per model from calls.csv."""
        if os.path.exists(self.calls_path) and os.path.getsize(self.calls_path) > 0:
            with open(self.calls_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model = row.get("model")
                    cid = row.get("comment_id")
                    if model and cid:
                        self._completed_ids[model].add(cid)

    def get_completed_ids(self, model: str) -> Set[str]:
        """Return the set of comment IDs already processed for the specified model."""
        with self._lock:
            return set(self._completed_ids.get(model, set()))

    def record_result(
        self,
        calls_row: Dict[str, Any],
        answers_rows: List[Dict[str, Any]],
    ) -> None:
        """Atomically append a call record and its category answers to disk."""
        with self._lock:
            with open(self.calls_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CALLS_FIELDNAMES, extrasaction="ignore")
                writer.writerow(calls_row)
                f.flush()

            with open(self.answers_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=ANSWERS_FIELDNAMES, extrasaction="ignore")
                writer.writerows(answers_rows)
                f.flush()

            model = calls_row.get("model")
            cid = calls_row.get("comment_id")
            if model and cid:
                self._completed_ids[model].add(str(cid))
