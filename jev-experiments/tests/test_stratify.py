import os
from pathlib import Path
import tempfile
import unittest
import csv
from io import StringIO
import sys

_ROOT = Path(__file__).resolve().parent.parent
for p in [str(_ROOT / "src"), str(_ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import will be tested once module is created
try:
    from stratify_dataset import filter_and_stratify, derive_toxic_level
except ImportError:
    filter_and_stratify = None
    derive_toxic_level = None


class TestStratifyDataset(unittest.TestCase):
    def test_derive_toxic_level(self):
        self.assertIsNotNone(derive_toxic_level, "derive_toxic_level should be defined")
        # Level 0
        self.assertEqual(derive_toxic_level(0, 0), 0)
        # Level 1
        self.assertEqual(derive_toxic_level(1, 0), 1)
        # Level 2
        self.assertEqual(derive_toxic_level(0, 1), 2)
        self.assertEqual(derive_toxic_level(1, 1), 2)

    def test_filter_and_stratify_dummy_data(self):
        self.assertIsNotNone(filter_and_stratify, "filter_and_stratify should be defined")
        
        # Create synthetic data with various strata and edge cases
        rows = [
            # id, comment_text, toxic, severe_toxic, threat, identity_hate
            # 1. Empty/whitespace rows (should be dropped)
            {"id": "drop_1", "comment_text": "", "toxic": "0", "severe_toxic": "0", "threat": "0", "identity_hate": "0"},
            {"id": "drop_2", "comment_text": "   \n\t  ", "toxic": "1", "severe_toxic": "0", "threat": "0", "identity_hate": "0"},
            # 2. Threat positives
            {"id": "threat_1", "comment_text": "This is a direct threat.", "toxic": "1", "severe_toxic": "0", "threat": "1", "identity_hate": "0"},
            {"id": "threat_2", "comment_text": "Another threat message.", "toxic": "0", "severe_toxic": "0", "threat": "1", "identity_hate": "0"},
            # 3. Identity hate positives
            {"id": "hate_1", "comment_text": "Identity attack message.", "toxic": "1", "severe_toxic": "0", "threat": "0", "identity_hate": "1"},
            {"id": "hate_2", "comment_text": "Another identity attack.", "toxic": "0", "severe_toxic": "0", "threat": "0", "identity_hate": "1"},
            # 4. Severe toxic positives (Level 2)
            {"id": "sev_1", "comment_text": "Extremely toxic harassment.", "toxic": "1", "severe_toxic": "1", "threat": "0", "identity_hate": "0"},
            {"id": "sev_2", "comment_text": "More severe toxicity.", "toxic": "1", "severe_toxic": "1", "threat": "0", "identity_hate": "0"},
            # 5. Toxic level 1 positives
            {"id": "tox_1", "comment_text": "Rude and disrespectful.", "toxic": "1", "severe_toxic": "0", "threat": "0", "identity_hate": "0"},
            {"id": "tox_2", "comment_text": "Unpleasant comment.", "toxic": "1", "severe_toxic": "0", "threat": "0", "identity_hate": "0"},
            # 6. Clean negatives (Level 0)
            {"id": "neg_1", "comment_text": "Civil debate.", "toxic": "0", "severe_toxic": "0", "threat": "0", "identity_hate": "0"},
            {"id": "neg_2", "comment_text": "Constructive editing advice.", "toxic": "0", "severe_toxic": "0", "threat": "0", "identity_hate": "0"},
            {"id": "neg_3", "comment_text": "Neutral comment.", "toxic": "0", "severe_toxic": "0", "threat": "0", "identity_hate": "0"},
            {"id": "neg_4", "comment_text": "Thank you for the edit.", "toxic": "0", "severe_toxic": "0", "threat": "0", "identity_hate": "0"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = os.path.join(tmpdir, "input.csv")
            prompts_csv = os.path.join(tmpdir, "prompts.csv")
            gt_csv = os.path.join(tmpdir, "gt.csv")
            fieldnames = ["id", "comment_text", "toxic", "severe_toxic", "threat", "identity_hate"]
            with open(input_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            # Sample size 8 out of 12 valid rows
            stats = filter_and_stratify(
                input_path=input_csv,
                prompts_path=prompts_csv,
                ground_truth_path=gt_csv,
                sample_size=8,
                seed=42,
                skip_lang_filter=True
            )

            self.assertEqual(stats["total_read"], 14)
            self.assertEqual(stats["empty_dropped"], 2)
            self.assertEqual(stats["total_valid"], 12)
            self.assertEqual(stats["total_sampled"], 8)

            # 1. Check prompts file
            with open(prompts_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                prompt_rows = list(reader)
                self.assertEqual(len(prompt_rows), 8)
                # STRICT SEPARATION: prompts file must ONLY have id and comment_text
                self.assertEqual(set(reader.fieldnames), {"id", "comment_text"})
                sampled_ids = {r["id"] for r in prompt_rows}
                self.assertNotIn("drop_1", sampled_ids)
                self.assertNotIn("drop_2", sampled_ids)

            # 2. Check ground truth file
            with open(gt_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                gt_rows = list(reader)
                self.assertEqual(len(gt_rows), 8)
                self.assertIn("id", reader.fieldnames)
                self.assertIn("toxic_level", reader.fieldnames)
                self.assertIn("threat", reader.fieldnames)
                self.assertIn("identity_hate", reader.fieldnames)
                # Confirm IDs match exactly between prompts and ground truth
                gt_ids = {r["id"] for r in gt_rows}
                self.assertEqual(sampled_ids, gt_ids)
                # Threat and hate should be represented in ground truth
                self.assertTrue(any(r["threat"] == "1" for r in gt_rows))
                self.assertTrue(any(r["identity_hate"] == "1" for r in gt_rows))

    def test_determinism_and_seed(self):
        # Verify that running with the same seed produces identical results
        rows = [
            {"id": f"row_{i}", "comment_text": f"Comment {i}", "toxic": str(i % 2),
             "severe_toxic": "1" if i % 10 == 0 else "0",
             "threat": "1" if i % 7 == 0 else "0",
             "identity_hate": "1" if i % 5 == 0 else "0"}
            for i in range(100)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = os.path.join(tmpdir, "input.csv")
            fieldnames = ["id", "comment_text", "toxic", "severe_toxic", "threat", "identity_hate"]
            with open(input_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            out1_p = os.path.join(tmpdir, "out1_p.csv")
            out1_g = os.path.join(tmpdir, "out1_g.csv")
            out2_p = os.path.join(tmpdir, "out2_p.csv")
            out2_g = os.path.join(tmpdir, "out2_g.csv")
            out3_p = os.path.join(tmpdir, "out3_p.csv")
            out3_g = os.path.join(tmpdir, "out3_g.csv")

            filter_and_stratify(input_csv, out1_p, out1_g, sample_size=30, seed=42, skip_lang_filter=True)
            filter_and_stratify(input_csv, out2_p, out2_g, sample_size=30, seed=42, skip_lang_filter=True)
            filter_and_stratify(input_csv, out3_p, out3_g, sample_size=30, seed=999, skip_lang_filter=True)

            with open(out1_p, "r", encoding="utf-8") as f1, open(out2_p, "r", encoding="utf-8") as f2:
                ids1 = [r["id"] for r in csv.DictReader(f1)]
                ids2 = [r["id"] for r in csv.DictReader(f2)]
                self.assertEqual(ids1, ids2, "Same seed must produce identical sampled sequence")

            with open(out3_p, "r", encoding="utf-8") as f3:
                ids3 = [r["id"] for r in csv.DictReader(f3)]
                self.assertNotEqual(ids1, ids3, "Different seed should produce different order/sample")

    def test_privacy_no_text_in_summary(self):
        # Verify that print_summary_table never leaks raw comment text
        from stratify_dataset import print_summary_table
        secret_comment = "CLASSIFIED_TEXT_SHOULD_NEVER_APPEAR"
        stats = {
            "total_read": 100,
            "empty_dropped": 5,
            "non_english_dropped": 2,
            "total_valid": 93,
            "total_sampled": 50,
            "threat_positives": 10,
            "identity_hate_positives": 12,
            "toxic_level_0": 20,
            "toxic_level_1": 20,
            "toxic_level_2": 10,
        }
        captured_output = StringIO()
        sys_stdout_backup = sys.stdout
        try:
            sys.stdout = captured_output
            print_summary_table(stats, prompts_path="/fake/prompts.csv", ground_truth_path="/fake/gt.csv")
        finally:
            sys.stdout = sys_stdout_backup

        summary_text = captured_output.getvalue()
        self.assertNotIn(secret_comment, summary_text)
        self.assertIn("DATASET STRATIFICATION & FILTERING SUMMARY", summary_text)
        self.assertIn("Total sampled items:       50", summary_text)
        self.assertIn("/fake/prompts.csv", summary_text)
        self.assertIn("/fake/gt.csv", summary_text)

    def test_cli_execution(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = os.path.join(tmpdir, "input.csv")
            fieldnames = ["id", "comment_text", "toxic", "severe_toxic", "threat", "identity_hate"]
            rows = [
                {"id": f"row_{i}", "comment_text": f"Valid comment {i}", "toxic": "0",
                 "severe_toxic": "0", "threat": "0", "identity_hate": "0"}
                for i in range(10)
            ]
            with open(input_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            result = subprocess.run(
                [
                    sys.executable,
                    str(_ROOT / "scripts" / "stratify_dataset.py"),
                    "--input", input_csv,
                    "--output-dir", tmpdir,
                    "--prefix", "test_split",
                    "--sample-size", "5",
                    "--seed", "42",
                    "--skip-lang-filter",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("DATASET STRATIFICATION & FILTERING SUMMARY", result.stdout)
            self.assertIn("Total sampled items:       5", result.stdout)
            # Ensure no comment text is printed to stdout or stderr
            self.assertNotIn("Valid comment", result.stdout)
            self.assertNotIn("Valid comment", result.stderr)

            expected_prompts = os.path.join(tmpdir, "test_split_prompts.csv")
            expected_gt = os.path.join(tmpdir, "test_split_ground_truth.csv")
            self.assertTrue(os.path.exists(expected_prompts))
            self.assertTrue(os.path.exists(expected_gt))

            with open(expected_prompts, "r", encoding="utf-8") as f:
                p_reader = csv.DictReader(f)
                self.assertEqual(set(p_reader.fieldnames), {"id", "comment_text"})
            with open(expected_gt, "r", encoding="utf-8") as f:
                g_reader = csv.DictReader(f)
                self.assertNotIn("comment_text", g_reader.fieldnames)
                self.assertIn("toxic_level", g_reader.fieldnames)


if __name__ == "__main__":
    unittest.main()
