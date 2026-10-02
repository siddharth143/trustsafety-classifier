import csv
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest

_ROOT = Path(__file__).resolve().parent.parent
for p in [str(_ROOT / "src"), str(_ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from checkpoint import (
    ANSWERS_FIELDNAMES,
    CALLS_FIELDNAMES,
    CheckpointManager,
)


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.calls_path = os.path.join(self.test_dir, "calls.csv")
        self.answers_path = os.path.join(self.test_dir, "answers.csv")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_initialization_creates_files_with_headers(self):
        manager = CheckpointManager(self.calls_path, self.answers_path)

        self.assertTrue(os.path.exists(self.calls_path))
        self.assertTrue(os.path.exists(self.answers_path))

        with open(self.calls_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.assertEqual(headers, CALLS_FIELDNAMES)

        with open(self.answers_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.assertEqual(headers, ANSWERS_FIELDNAMES)

        # Re-initializing should not duplicate headers or corrupt files
        CheckpointManager(self.calls_path, self.answers_path)
        with open(self.calls_path, "r", newline="", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 1)

    def test_creates_parent_directories_if_needed(self):
        nested_calls = os.path.join(self.test_dir, "nested", "sub", "calls.csv")
        nested_answers = os.path.join(self.test_dir, "nested", "sub", "answers.csv")
        CheckpointManager(nested_calls, nested_answers)
        self.assertTrue(os.path.exists(nested_calls))
        self.assertTrue(os.path.exists(nested_answers))

    def test_get_completed_ids_empty(self):
        manager = CheckpointManager(self.calls_path, self.answers_path)
        self.assertEqual(manager.get_completed_ids("jev"), set())
        self.assertEqual(manager.get_completed_ids("haiku"), set())

    def test_record_result_and_get_completed_ids(self):
        manager = CheckpointManager(self.calls_path, self.answers_path)

        calls_row_1 = {
            "comment_id": "c1",
            "model": "jev",
            "latency_ms": 120.5,
            "input_tokens": 50,
            "output_tokens": 10,
            "cost_usd": 0.000002,
            "raw_response_json": '{"dummy": 1}',
            "timestamp": "2026-10-01T20:00:00Z",
        }
        answers_rows_1 = [
            {
                "comment_id": "c1",
                "model": "jev",
                "category": "toxic",
                "toxic_prob_0": 0.9,
                "toxic_prob_1": 0.08,
                "toxic_prob_2": 0.02,
                "toxic_score": 0.12,
                "binary_probability": None,
                "native_confidence": 0.88,
            },
            {
                "comment_id": "c1",
                "model": "jev",
                "category": "threat",
                "toxic_prob_0": None,
                "toxic_prob_1": None,
                "toxic_prob_2": None,
                "toxic_score": None,
                "binary_probability": 0.01,
                "native_confidence": 0.98,
            },
            {
                "comment_id": "c1",
                "model": "jev",
                "category": "identity_hate",
                "toxic_prob_0": None,
                "toxic_prob_1": None,
                "toxic_prob_2": None,
                "toxic_score": None,
                "binary_probability": 0.02,
                "native_confidence": 0.96,
            },
        ]

        manager.record_result(calls_row_1, answers_rows_1)

        self.assertEqual(manager.get_completed_ids("jev"), {"c1"})
        self.assertEqual(manager.get_completed_ids("haiku"), set())

        # Record for another model
        calls_row_2 = {
            "comment_id": "c2",
            "model": "haiku",
            "latency_ms": 300.0,
            "input_tokens": 100,
            "output_tokens": 30,
            "cost_usd": 0.0002,
            "raw_response_json": '{"dummy": 2}',
            "timestamp": "2026-10-01T20:01:00Z",
        }
        answers_rows_2 = [
            {
                "comment_id": "c2",
                "model": "haiku",
                "category": "toxic",
                "toxic_prob_0": 0.1,
                "toxic_prob_1": 0.8,
                "toxic_prob_2": 0.1,
                "toxic_score": 1.0,
                "binary_probability": None,
                "native_confidence": None,
            }
        ]
        manager.record_result(calls_row_2, answers_rows_2)

        self.assertEqual(manager.get_completed_ids("jev"), {"c1"})
        self.assertEqual(manager.get_completed_ids("haiku"), {"c2"})

        # New manager instance reading the existing CSV files
        new_manager = CheckpointManager(self.calls_path, self.answers_path)
        self.assertEqual(new_manager.get_completed_ids("jev"), {"c1"})
        self.assertEqual(new_manager.get_completed_ids("haiku"), {"c2"})

    def test_record_result_data_integrity(self):
        manager = CheckpointManager(self.calls_path, self.answers_path)
        calls_row = {
            "comment_id": "c99",
            "model": "flash",
            "latency_ms": 150.2,
            "input_tokens": 45,
            "output_tokens": 15,
            "cost_usd": 0.000007,
            "raw_response_json": '{"result": "ok"}',
            "timestamp": "2026-10-01T20:05:00Z",
        }
        answers_rows = [
            {
                "comment_id": "c99",
                "model": "flash",
                "category": "toxic",
                "toxic_prob_0": 0.7,
                "toxic_prob_1": 0.2,
                "toxic_prob_2": 0.1,
                "toxic_score": 0.4,
                "binary_probability": None,
                "native_confidence": None,
            }
        ]
        manager.record_result(calls_row, answers_rows)

        with open(self.calls_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["comment_id"], "c99")
            self.assertEqual(rows[0]["model"], "flash")
            self.assertEqual(float(rows[0]["latency_ms"]), 150.2)

        with open(self.answers_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["comment_id"], "c99")
            self.assertEqual(rows[0]["category"], "toxic")
            self.assertEqual(float(rows[0]["toxic_score"]), 0.4)

    def test_thread_safety(self):
        manager = CheckpointManager(self.calls_path, self.answers_path)
        num_threads = 10
        records_per_thread = 20
        total_records = num_threads * records_per_thread

        def worker(thread_idx):
            for i in range(records_per_thread):
                cid = f"t{thread_idx}_c{i}"
                calls_row = {
                    "comment_id": cid,
                    "model": "jev",
                    "latency_ms": 100 + i,
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "cost_usd": 0.000002,
                    "raw_response_json": "{}",
                    "timestamp": "2026-10-01T20:00:00Z",
                }
                answers_rows = [
                    {
                        "comment_id": cid,
                        "model": "jev",
                        "category": cat,
                        "toxic_prob_0": 0.5,
                        "toxic_prob_1": 0.3,
                        "toxic_prob_2": 0.2,
                        "toxic_score": 0.7,
                        "binary_probability": 0.1,
                        "native_confidence": 0.8,
                    }
                    for cat in ["toxic", "threat", "identity_hate"]
                ]
                manager.record_result(calls_row, answers_rows)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        completed_ids = manager.get_completed_ids("jev")
        self.assertEqual(len(completed_ids), total_records)

        with open(self.calls_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            call_rows = list(reader)
            self.assertEqual(len(call_rows), total_records)

        with open(self.answers_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ans_rows = list(reader)
            self.assertEqual(len(ans_rows), total_records * 3)


if __name__ == "__main__":
    unittest.main()
