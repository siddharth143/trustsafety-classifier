"""Unit tests for the Evaluation & Analysis Engine (Task 6)."""

import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent
for p in [str(_ROOT / "src"), str(_ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Ensure matplotlib uses Agg backend and writable config directory before importing evaluate
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import numpy as np
    from evaluate import (
        compute_binary_metrics,
        compute_confidence_calibration,
        compute_cost_and_latency,
        compute_ordinal_toxic_metrics,
        compute_toxic_binary_metrics,
        find_operating_points,
        generate_calibration_plot,
        generate_markdown_summary,
        generate_pr_plots,
        load_and_merge_data,
        run_evaluation,
        sweep_thresholds,
    )
    HAS_EVAL_DEPS = True
except ImportError:
    HAS_EVAL_DEPS = False


class TestEvaluateEngine(unittest.TestCase):
    """Test suite for evaluation data loading, metric computation, threshold sweeping, calibration, and plotting."""

    @classmethod
    def setUpClass(cls):
        if not HAS_EVAL_DEPS:
            raise unittest.SkipTest("Evaluation dependencies not installed in current Python environment")

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.calls_file = os.path.join(self.test_dir, "calls.csv")
        self.answers_file = os.path.join(self.test_dir, "answers.csv")
        self.ground_truth_file = os.path.join(self.test_dir, "ground_truth.csv")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        self._create_sample_datasets()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_sample_datasets(self):
        """Create synthetic ground truth, calls, and answers data for testing."""
        # 1. Ground truth: 6 comments
        # c1: level 0 (not toxic, no threat, no hate)
        # c2: level 1 (toxic, no threat, no hate)
        # c3: level 2 (severe toxic, threat, hate)
        # c4: level 0 (not toxic, threat, no hate)
        # c5: level 1 (toxic, no threat, hate)
        # c6: level 2 (severe toxic, no threat, no hate)
        gt_rows = [
            {"id": "c1", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0, "toxic_level": 0},
            {"id": "c2", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0, "toxic_level": 1},
            {"id": "c3", "toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 1, "insult": 1, "identity_hate": 1, "toxic_level": 2},
            {"id": "c4", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0, "toxic_level": 0},
            {"id": "c5", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 1, "toxic_level": 1},
            {"id": "c6", "toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0, "toxic_level": 2},
        ]
        with open(self.ground_truth_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "toxic_level"])
            writer.writeheader()
            writer.writerows(gt_rows)

        # 2. Calls: 6 comments for jev, 6 comments for haiku
        calls_rows = []
        for cid in ["c1", "c2", "c3", "c4", "c5", "c6"]:
            calls_rows.append({
                "comment_id": cid,
                "model": "jev",
                "latency_ms": 120.0 + (int(cid[1]) * 10),
                "input_tokens": 100,
                "output_tokens": 0,
                "cost_usd": 0.0000042,
                "raw_response_json": "{}",
                "timestamp": "2026-10-01T00:00:00Z",
            })
            calls_rows.append({
                "comment_id": cid,
                "model": "haiku",
                "latency_ms": 350.0 + (int(cid[1]) * 20),
                "input_tokens": 200,
                "output_tokens": 40,
                "cost_usd": 0.00032,
                "raw_response_json": "{}",
                "timestamp": "2026-10-01T00:00:00Z",
            })
        with open(self.calls_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["comment_id", "model", "latency_ms", "input_tokens", "output_tokens", "cost_usd", "raw_response_json", "timestamp"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(calls_rows)

        # 3. Answers: 3 rows per comment per model
        answers_rows = []
        # Jev predictions (mostly accurate with varying confidence)
        jev_preds = {
            "c1": {"toxic": (0.9, 0.08, 0.02, 0.95), "threat": (0.05, 0.90), "identity_hate": (0.02, 0.96)},
            "c2": {"toxic": (0.1, 0.85, 0.05, 0.88), "threat": (0.08, 0.84), "identity_hate": (0.05, 0.90)},
            "c3": {"toxic": (0.02, 0.18, 0.80, 0.92), "threat": (0.95, 0.90), "identity_hate": (0.88, 0.76)},
            "c4": {"toxic": (0.85, 0.10, 0.05, 0.80), "threat": (0.82, 0.64), "identity_hate": (0.10, 0.80)},
            "c5": {"toxic": (0.15, 0.75, 0.10, 0.70), "threat": (0.05, 0.90), "identity_hate": (0.75, 0.50)},
            "c6": {"toxic": (0.05, 0.20, 0.75, 0.85), "threat": (0.02, 0.96), "identity_hate": (0.08, 0.84)},
        }

        for cid, preds in jev_preds.items():
            # toxic
            p0, p1, p2, conf = preds["toxic"]
            score = 0.0 * p0 + 1.0 * p1 + 2.0 * p2
            answers_rows.append({
                "comment_id": cid, "model": "jev", "category": "toxic",
                "toxic_prob_0": p0, "toxic_prob_1": p1, "toxic_prob_2": p2, "toxic_score": score,
                "binary_probability": "", "native_confidence": conf,
            })
            # threat
            p, conf = preds["threat"]
            answers_rows.append({
                "comment_id": cid, "model": "jev", "category": "threat",
                "toxic_prob_0": "", "toxic_prob_1": "", "toxic_prob_2": "", "toxic_score": "",
                "binary_probability": p, "native_confidence": conf,
            })
            # identity_hate
            p, conf = preds["identity_hate"]
            answers_rows.append({
                "comment_id": cid, "model": "jev", "category": "identity_hate",
                "toxic_prob_0": "", "toxic_prob_1": "", "toxic_prob_2": "", "toxic_score": "",
                "binary_probability": p, "native_confidence": conf,
            })

        # Haiku predictions (native_confidence is null)
        haiku_preds = {
            "c1": {"toxic": (0.8, 0.15, 0.05), "threat": 0.10, "identity_hate": 0.05},
            "c2": {"toxic": (0.2, 0.70, 0.10), "threat": 0.15, "identity_hate": 0.10},
            "c3": {"toxic": (0.05, 0.25, 0.70), "threat": 0.85, "identity_hate": 0.90},
            "c4": {"toxic": (0.70, 0.20, 0.10), "threat": 0.75, "identity_hate": 0.15},
            "c5": {"toxic": (0.10, 0.80, 0.10), "threat": 0.10, "identity_hate": 0.80},
            "c6": {"toxic": (0.10, 0.30, 0.60), "threat": 0.05, "identity_hate": 0.10},
        }
        for cid, preds in haiku_preds.items():
            p0, p1, p2 = preds["toxic"]
            score = 0.0 * p0 + 1.0 * p1 + 2.0 * p2
            answers_rows.append({
                "comment_id": cid, "model": "haiku", "category": "toxic",
                "toxic_prob_0": p0, "toxic_prob_1": p1, "toxic_prob_2": p2, "toxic_score": score,
                "binary_probability": "", "native_confidence": "",
            })
            answers_rows.append({
                "comment_id": cid, "model": "haiku", "category": "threat",
                "toxic_prob_0": "", "toxic_prob_1": "", "toxic_prob_2": "", "toxic_score": "",
                "binary_probability": preds["threat"], "native_confidence": "",
            })
            answers_rows.append({
                "comment_id": cid, "model": "haiku", "category": "identity_hate",
                "toxic_prob_0": "", "toxic_prob_1": "", "toxic_prob_2": "", "toxic_score": "",
                "binary_probability": preds["identity_hate"], "native_confidence": "",
            })

        with open(self.answers_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "comment_id", "model", "category",
                "toxic_prob_0", "toxic_prob_1", "toxic_prob_2", "toxic_score",
                "binary_probability", "native_confidence"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(answers_rows)

    def test_load_and_merge_data(self):
        """Test loading and merging calls.csv, answers.csv, and ground_truth.csv on comment_id == id."""
        calls_df, answers_merged_df = load_and_merge_data(
            self.calls_file, self.answers_file, self.ground_truth_file
        )

        # Validate calls_df
        self.assertEqual(len(calls_df), 12)  # 6 comments * 2 models
        self.assertIn("comment_id", calls_df.columns)
        self.assertIn("latency_ms", calls_df.columns)
        self.assertIn("cost_usd", calls_df.columns)
        self.assertIn("model", calls_df.columns)

        # Validate answers_merged_df
        self.assertEqual(len(answers_merged_df), 36)  # 6 comments * 2 models * 3 categories
        self.assertIn("comment_id", answers_merged_df.columns)
        self.assertIn("model", answers_merged_df.columns)
        self.assertIn("category", answers_merged_df.columns)
        # Verify joined ground truth columns
        self.assertIn("toxic_level", answers_merged_df.columns)
        self.assertIn("threat", answers_merged_df.columns)
        self.assertIn("identity_hate", answers_merged_df.columns)
        self.assertIn("toxic", answers_merged_df.columns)
        self.assertIn("severe_toxic", answers_merged_df.columns)

        # Check comment_id == id matching correctness
        c3_threat = answers_merged_df[
            (answers_merged_df["comment_id"] == "c3") &
            (answers_merged_df["model"] == "jev") &
            (answers_merged_df["category"] == "threat")
        ].iloc[0]
        self.assertEqual(c3_threat["threat"], 1)
        self.assertEqual(c3_threat["toxic_level"], 2)
        self.assertAlmostEqual(float(c3_threat["binary_probability"]), 0.95)

    def test_compute_cost_and_latency(self):
        """Test cost per 1,000 items normalized across models and latency percentiles (p50, p90, p95)."""
        calls_df, _ = load_and_merge_data(
            self.calls_file, self.answers_file, self.ground_truth_file
        )
        cost_latency = compute_cost_and_latency(calls_df)

        self.assertIn("jev", cost_latency)
        self.assertIn("haiku", cost_latency)

        # Jev: 6 items, cost = 6 * 0.0000042 = 0.0000252
        # Cost per 1k = 0.0000252 / 6 * 1000 = 0.0042
        self.assertAlmostEqual(cost_latency["jev"]["cost_per_1k"], 0.0042, places=6)
        self.assertEqual(cost_latency["jev"]["count"], 6)

        # Latencies for jev: 130, 140, 150, 160, 170, 180
        # p50 of [130, 140, 150, 160, 170, 180] = 155
        self.assertAlmostEqual(cost_latency["jev"]["p50_latency_ms"], 155.0, places=1)
        self.assertTrue(cost_latency["jev"]["p90_latency_ms"] > cost_latency["jev"]["p50_latency_ms"])
        self.assertTrue(cost_latency["jev"]["p95_latency_ms"] >= cost_latency["jev"]["p90_latency_ms"])

        # Haiku: 6 items, cost = 6 * 0.00032 = 0.00192
        # Cost per 1k = 0.32
        self.assertAlmostEqual(cost_latency["haiku"]["cost_per_1k"], 0.32, places=4)
        self.assertEqual(cost_latency["haiku"]["count"], 6)
        self.assertTrue(cost_latency["haiku"]["p50_latency_ms"] > cost_latency["jev"]["p50_latency_ms"])

    def test_compute_binary_metrics(self):
        """Test precision, recall, and F1 for binary categories (threat and identity_hate)."""
        # Ground truth: [1, 1, 0, 0]
        # Predicted probabilities: [0.9, 0.4, 0.3, 0.1]
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.4, 0.3, 0.1]

        # At threshold 0.5: pred = [1, 0, 0, 0] -> TP=1, FP=0, FN=1, TN=2
        # Precision = 1.0, Recall = 0.5, F1 = 2 * (1.0 * 0.5) / 1.5 = 2/3 ≈ 0.6667
        metrics_50 = compute_binary_metrics(y_true, y_score, threshold=0.5)
        self.assertAlmostEqual(metrics_50["precision"], 1.0, places=4)
        self.assertAlmostEqual(metrics_50["recall"], 0.5, places=4)
        self.assertAlmostEqual(metrics_50["f1"], 2 / 3, places=4)

        # At threshold 0.35: pred = [1, 1, 0, 0] -> TP=2, FP=0, FN=0, TN=2
        # Precision = 1.0, Recall = 1.0, F1 = 1.0
        metrics_35 = compute_binary_metrics(y_true, y_score, threshold=0.35)
        self.assertAlmostEqual(metrics_35["precision"], 1.0, places=4)
        self.assertAlmostEqual(metrics_35["recall"], 1.0, places=4)
        self.assertAlmostEqual(metrics_35["f1"], 1.0, places=4)

    def test_compute_ordinal_toxic_metrics(self):
        """Test 3-level ordinal F1, macro precision, and macro recall for toxic."""
        # True levels: [0, 1, 2, 0, 1, 2]
        y_true = [0, 1, 2, 0, 1, 2]
        # Perfect predictions matrix (shape 6 x 3)
        prob_matrix_perfect = [
            [0.9, 0.05, 0.05],  # 0
            [0.1, 0.8, 0.1],   # 1
            [0.05, 0.05, 0.9],  # 2
            [0.85, 0.1, 0.05],  # 0
            [0.1, 0.75, 0.15],  # 1
            [0.05, 0.1, 0.85],  # 2
        ]
        metrics = compute_ordinal_toxic_metrics(y_true, prob_matrix_perfect)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0, places=4)
        self.assertAlmostEqual(metrics["macro_precision"], 1.0, places=4)
        self.assertAlmostEqual(metrics["macro_recall"], 1.0, places=4)
        self.assertAlmostEqual(metrics["accuracy"], 1.0, places=4)
        self.assertEqual(len(metrics["f1_per_level"]), 3)

        # Imperfect predictions matrix: misclassifies level 1 as level 0
        prob_matrix_imperfect = [
            [0.9, 0.05, 0.05],  # pred 0 (true 0)
            [0.7, 0.2, 0.1],   # pred 0 (true 1) -> error
            [0.05, 0.05, 0.9],  # pred 2 (true 2)
            [0.85, 0.1, 0.05],  # pred 0 (true 0)
            [0.1, 0.75, 0.15],  # pred 1 (true 1)
            [0.05, 0.1, 0.85],  # pred 2 (true 2)
        ]
        metrics_imp = compute_ordinal_toxic_metrics(y_true, prob_matrix_imperfect)
        self.assertLess(metrics_imp["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics_imp["accuracy"], 5 / 6, places=4)

    def test_compute_toxic_binary_metrics(self):
        """Test collapsed binary F1 for toxic where level >= 1 is positive."""
        # True toxic levels: [0, 1, 2, 0] -> binary ground truth: [0, 1, 1, 0]
        y_true_level = [0, 1, 2, 0]
        # Probabilities
        prob_0 = [0.85, 0.10, 0.05, 0.90]
        prob_1 = [0.10, 0.70, 0.20, 0.08]
        prob_2 = [0.05, 0.20, 0.75, 0.02]

        # At threshold 0.5:
        # positive probabilities = [0.15, 0.90, 0.95, 0.10]
        # binary preds = [0, 1, 1, 0] == binary ground truth
        metrics = compute_toxic_binary_metrics(y_true_level, prob_0, prob_1, prob_2, threshold=0.5)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=4)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=4)
        self.assertAlmostEqual(metrics["f1"], 1.0, places=4)

    def test_sweep_thresholds(self):
        """Test threshold sweeping from 0.01 to 0.99 for PR curve generation."""
        y_true = [1, 1, 1, 0, 0, 0]
        y_score = [0.95, 0.85, 0.65, 0.45, 0.25, 0.05]

        sweep_records = sweep_thresholds(y_true, y_score)
        # Ensure default sweeps 99 steps (0.01 to 0.99)
        self.assertEqual(len(sweep_records), 99)
        self.assertAlmostEqual(sweep_records[0]["threshold"], 0.01, places=2)
        self.assertAlmostEqual(sweep_records[-1]["threshold"], 0.99, places=2)

        # At low threshold (0.01): recall should be 1.0
        self.assertAlmostEqual(sweep_records[0]["recall"], 1.0, places=4)
        # At very high threshold (0.99): recall should be 0.0
        self.assertAlmostEqual(sweep_records[-1]["recall"], 0.0, places=4)

        # Monotonicity check: recall should be non-increasing with threshold
        recalls = [rec["recall"] for rec in sweep_records]
        for i in range(len(recalls) - 1):
            self.assertGreaterEqual(recalls[i], recalls[i + 1])

    def test_find_operating_points(self):
        """Test finding recall-constrained (>=90% recall) and max-F1 operating points."""
        # Synthetic sweep results
        records = [
            {"threshold": 0.2, "precision": 0.60, "recall": 1.00, "f1": 0.75},
            {"threshold": 0.4, "precision": 0.80, "recall": 0.95, "f1": 0.87},
            {"threshold": 0.5, "precision": 0.88, "recall": 0.90, "f1": 0.89},
            {"threshold": 0.6, "precision": 0.95, "recall": 0.85, "f1": 0.90},
            {"threshold": 0.8, "precision": 1.00, "recall": 0.50, "f1": 0.67},
        ]

        op_points = find_operating_points(records, min_recall=0.90)

        # Recall-constrained (>= 90% recall): candidates are th=0.2, 0.4, 0.5.
        # Highest precision/F1 among recall >= 0.90 is th=0.5 with precision=0.88, recall=0.90, f1=0.89
        rc_point = op_points["recall_constrained"]
        self.assertAlmostEqual(rc_point["threshold"], 0.5)
        self.assertAlmostEqual(rc_point["recall"], 0.90)
        self.assertAlmostEqual(rc_point["precision"], 0.88)
        self.assertTrue(rc_point["recall"] >= 0.90)

        # Max-F1 point: threshold 0.6 with f1=0.90
        max_f1_point = op_points["max_f1"]
        self.assertAlmostEqual(max_f1_point["threshold"], 0.6)
        self.assertAlmostEqual(max_f1_point["f1"], 0.90)

        # Test fallback when no threshold reaches 90% recall
        low_recall_records = [
            {"threshold": 0.5, "precision": 0.9, "recall": 0.70, "f1": 0.78},
            {"threshold": 0.7, "precision": 1.0, "recall": 0.50, "f1": 0.67},
        ]
        fallback_op = find_operating_points(low_recall_records, min_recall=0.90)
        self.assertAlmostEqual(fallback_op["recall_constrained"]["recall"], 0.70)
        self.assertTrue(fallback_op["recall_constrained"].get("fallback", False))

    def test_compute_confidence_calibration(self):
        """Test Jev confidence calibration: binning predictions into quartiles and computing accuracy per bin."""
        # Create a dataframe with 12 items: 3 in each confidence quartile
        # Higher confidence -> higher accuracy
        data = [
            # Q1 (low confidence 0.1 - 0.25): 1 correct, 2 incorrect -> acc 1/3 ≈ 0.33
            {"toxic_level": 0, "toxic_prob_0": 0.6, "toxic_prob_1": 0.2, "toxic_prob_2": 0.2, "native_confidence": 0.12},
            {"toxic_level": 1, "toxic_prob_0": 0.6, "toxic_prob_1": 0.2, "toxic_prob_2": 0.2, "native_confidence": 0.15},
            {"toxic_level": 2, "toxic_prob_0": 0.5, "toxic_prob_1": 0.3, "toxic_prob_2": 0.2, "native_confidence": 0.20},
            # Q2 (0.35 - 0.50): 2 correct, 1 incorrect -> acc 2/3 ≈ 0.67
            {"toxic_level": 0, "toxic_prob_0": 0.7, "toxic_prob_1": 0.2, "toxic_prob_2": 0.1, "native_confidence": 0.38},
            {"toxic_level": 1, "toxic_prob_0": 0.2, "toxic_prob_1": 0.7, "toxic_prob_2": 0.1, "native_confidence": 0.42},
            {"toxic_level": 2, "toxic_prob_0": 0.6, "toxic_prob_1": 0.2, "toxic_prob_2": 0.2, "native_confidence": 0.48},
            # Q3 (0.60 - 0.75): 2 correct, 1 incorrect -> acc 2/3 ≈ 0.67
            {"toxic_level": 0, "toxic_prob_0": 0.8, "toxic_prob_1": 0.1, "toxic_prob_2": 0.1, "native_confidence": 0.62},
            {"toxic_level": 1, "toxic_prob_0": 0.1, "toxic_prob_1": 0.8, "toxic_prob_2": 0.1, "native_confidence": 0.68},
            {"toxic_level": 2, "toxic_prob_0": 0.5, "toxic_prob_1": 0.1, "toxic_prob_2": 0.4, "native_confidence": 0.72},
            # Q4 (high confidence 0.85 - 0.99): 3 correct, 0 incorrect -> acc 3/3 = 1.0
            {"toxic_level": 0, "toxic_prob_0": 0.95, "toxic_prob_1": 0.03, "toxic_prob_2": 0.02, "native_confidence": 0.88},
            {"toxic_level": 1, "toxic_prob_0": 0.02, "toxic_prob_1": 0.95, "toxic_prob_2": 0.03, "native_confidence": 0.92},
            {"toxic_level": 2, "toxic_prob_0": 0.01, "toxic_prob_1": 0.04, "toxic_prob_2": 0.95, "native_confidence": 0.98},
        ]
        import pandas as pd
        df = pd.DataFrame(data)

        bins = compute_confidence_calibration(df, category="toxic", n_bins=4)
        self.assertEqual(len(bins), 4)

        # Check bin properties
        for b in bins:
            self.assertIn("bin_name", b)
            self.assertIn("count", b)
            self.assertIn("accuracy", b)
            self.assertIn("mean_confidence", b)

        # Lowest bin accuracy should be less than highest bin accuracy
        lowest_acc = bins[0]["accuracy"]
        highest_acc = bins[-1]["accuracy"]
        self.assertLess(lowest_acc, highest_acc)
        self.assertAlmostEqual(highest_acc, 1.0, places=2)

    def test_plot_generation(self):
        """Test generating PR curves and calibration plots without errors (Agg backend)."""
        # Prepare sample data for plotting
        pr_curves = {
            "threat": {
                "jev": [
                    {"threshold": 0.1, "precision": 0.8, "recall": 1.0, "f1": 0.88},
                    {"threshold": 0.5, "precision": 0.9, "recall": 0.9, "f1": 0.90},
                    {"threshold": 0.9, "precision": 1.0, "recall": 0.4, "f1": 0.57},
                ],
                "haiku": [
                    {"threshold": 0.1, "precision": 0.7, "recall": 1.0, "f1": 0.82},
                    {"threshold": 0.5, "precision": 0.85, "recall": 0.85, "f1": 0.85},
                    {"threshold": 0.9, "precision": 0.95, "recall": 0.3, "f1": 0.45},
                ]
            },
            "identity_hate": {
                "jev": [
                    {"threshold": 0.1, "precision": 0.85, "recall": 1.0, "f1": 0.91},
                    {"threshold": 0.5, "precision": 0.92, "recall": 0.92, "f1": 0.92},
                ]
            },
            "toxic": {
                "jev": [
                    {"threshold": 0.1, "precision": 0.88, "recall": 1.0, "f1": 0.93},
                    {"threshold": 0.5, "precision": 0.94, "recall": 0.91, "f1": 0.92},
                ]
            }
        }

        op_points = {
            "threat": {
                "jev": {
                    "recall_constrained": {"threshold": 0.5, "precision": 0.9, "recall": 0.9, "f1": 0.90},
                    "max_f1": {"threshold": 0.5, "precision": 0.9, "recall": 0.9, "f1": 0.90},
                },
                "haiku": {
                    "recall_constrained": {"threshold": 0.1, "precision": 0.7, "recall": 1.0, "f1": 0.82},
                    "max_f1": {"threshold": 0.5, "precision": 0.85, "recall": 0.85, "f1": 0.85},
                }
            },
            "identity_hate": {
                "jev": {
                    "recall_constrained": {"threshold": 0.5, "precision": 0.92, "recall": 0.92, "f1": 0.92},
                    "max_f1": {"threshold": 0.5, "precision": 0.92, "recall": 0.92, "f1": 0.92},
                }
            },
            "toxic": {
                "jev": {
                    "recall_constrained": {"threshold": 0.5, "precision": 0.94, "recall": 0.91, "f1": 0.92},
                    "max_f1": {"threshold": 0.5, "precision": 0.94, "recall": 0.91, "f1": 0.92},
                }
            }
        }

        pr_plot_paths = generate_pr_plots(pr_curves, op_points, self.output_dir)
        self.assertEqual(len(pr_plot_paths), 3)
        for path in [
            os.path.join(self.output_dir, "pr_curve_threat.png"),
            os.path.join(self.output_dir, "pr_curve_identity_hate.png"),
            os.path.join(self.output_dir, "pr_curve_toxic.png"),
        ]:
            self.assertTrue(os.path.exists(path), f"Missing plot: {path}")
            self.assertGreater(os.path.getsize(path), 0)

        # Calibration plot test
        calibration_results = {
            "toxic": [
                {"bin_name": "Q1", "mean_confidence": 0.2, "count": 20, "accuracy": 0.4},
                {"bin_name": "Q2", "mean_confidence": 0.5, "count": 20, "accuracy": 0.7},
                {"bin_name": "Q3", "mean_confidence": 0.75, "count": 20, "accuracy": 0.85},
                {"bin_name": "Q4", "mean_confidence": 0.95, "count": 20, "accuracy": 0.98},
            ],
            "threat": [
                {"bin_name": "Q1", "mean_confidence": 0.25, "count": 15, "accuracy": 0.5},
                {"bin_name": "Q2", "mean_confidence": 0.55, "count": 15, "accuracy": 0.75},
                {"bin_name": "Q3", "mean_confidence": 0.80, "count": 15, "accuracy": 0.90},
                {"bin_name": "Q4", "mean_confidence": 0.98, "count": 15, "accuracy": 1.0},
            ],
            "identity_hate": [
                {"bin_name": "Q1", "mean_confidence": 0.18, "count": 10, "accuracy": 0.45},
                {"bin_name": "Q2", "mean_confidence": 0.48, "count": 10, "accuracy": 0.65},
                {"bin_name": "Q3", "mean_confidence": 0.78, "count": 10, "accuracy": 0.88},
                {"bin_name": "Q4", "mean_confidence": 0.96, "count": 10, "accuracy": 0.95},
            ]
        }
        cal_path = generate_calibration_plot(calibration_results, self.output_dir)
        self.assertTrue(os.path.exists(cal_path))
        self.assertGreater(os.path.getsize(cal_path), 0)

    def test_run_evaluation_end_to_end(self):
        """Test full evaluation pipeline end-to-end generating summary and all outputs."""
        report = run_evaluation(
            calls_file=self.calls_file,
            answers_file=self.answers_file,
            ground_truth_file=self.ground_truth_file,
            output_dir=self.output_dir,
        )

        self.assertIn("summary_markdown", report)
        self.assertIn("cost_latency", report)
        self.assertIn("operating_points", report)
        self.assertIn("calibration", report)
        self.assertIn("plots", report)

        # Check summary markdown has expected tables
        md = report["summary_markdown"]
        self.assertIn("| Model | Cost / 1k Items |", md)
        self.assertIn("jev", md)
        self.assertIn("haiku", md)
        self.assertIn("Operating Points", md)

        # Check all expected output files exist
        for expected_file in [
            "pr_curve_toxic.png",
            "pr_curve_threat.png",
            "pr_curve_identity_hate.png",
            "calibration_jev.png",
            "evaluation_metrics.json",
        ]:
            file_path = os.path.join(self.output_dir, expected_file)
            self.assertTrue(os.path.exists(file_path), f"Expected file not found: {file_path}")

    def test_cli_main(self):
        """Test evaluate.py CLI invocation via main()."""
        import io
        import sys
        from unittest.mock import patch
        from evaluate import main

        cli_output_dir = os.path.join(self.test_dir, "cli_output")
        test_args = [
            "evaluate.py",
            "--calls-file", self.calls_file,
            "--answers-file", self.answers_file,
            "--ground-truth-file", self.ground_truth_file,
            "--output-dir", cli_output_dir,
        ]

        captured_stdout = io.StringIO()
        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout", captured_stdout):
                main()

        output_str = captured_stdout.getvalue()
        self.assertIn("# Trust & Safety Eval Gate Benchmark Report", output_str)
        self.assertIn("jev", output_str)
        self.assertTrue(os.path.exists(os.path.join(cli_output_dir, "summary.md")))
        self.assertTrue(os.path.exists(os.path.join(cli_output_dir, "pr_curve_toxic.png")))


if __name__ == "__main__":
    unittest.main()
