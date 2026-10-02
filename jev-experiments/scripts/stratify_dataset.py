#!/usr/bin/env python3
"""
stratify_dataset.py - Filter, stratify, and sample the Jigsaw Toxic Comment dataset
into strictly separated prompts and ground-truth files.

Strict Separation & Privacy Guarantee:
- The prompts file contains ONLY 'id' and 'comment_text' (zero labels, zero leakage risk to models).
- The ground truth file contains 'id' and label columns, completely omitting 'comment_text'.
- Under no circumstances does this script print or log raw comments to stdout, stderr, or log files.
- Only aggregate counts and distribution tables are reported.
"""

import argparse
import csv
import os
import random
import sys
from typing import Dict, List, Any, Optional

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


def derive_toxic_level(toxic: Any, severe_toxic: Any) -> int:
    """
    Derives 3-level ordinal toxicity:
      0 = neither toxic nor severe_toxic
      1 = toxic only
      2 = severe_toxic
    """
    st = int(severe_toxic) if str(severe_toxic).strip() else 0
    tx = int(toxic) if str(toxic).strip() else 0
    if st == 1:
        return 2
    elif tx == 1:
        return 1
    return 0


def is_english(text: str) -> bool:
    """Check if text is English using langdetect if installed."""
    if not LANGDETECT_AVAILABLE:
        return True
    try:
        lang = detect(text)
        return lang == "en"
    except Exception:
        return False


def filter_and_stratify(
    input_path: str,
    prompts_path: str,
    ground_truth_path: str,
    sample_size: int = 6000,
    seed: int = 42,
    skip_lang_filter: bool = False,
) -> Dict[str, Any]:
    """
    Reads the raw Jigsaw CSV, filters out empty/non-English rows,
    applies multi-label stratification, and saves two strictly separated files:
      1. prompts_path: ONLY 'id' and 'comment_text' (for feeding into models)
      2. ground_truth_path: 'id' and labels ('toxic_level', 'threat', etc.), NO comment_text
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rng = random.Random(seed)

    total_read = 0
    empty_dropped = 0
    non_english_dropped = 0

    valid_rows: List[Dict[str, Any]] = []

    # Read and filter
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])

        for row in reader:
            total_read += 1
            comment = row.get("comment_text", "")
            if not comment or not comment.strip():
                empty_dropped += 1
                continue

            if not skip_lang_filter and LANGDETECT_AVAILABLE:
                if not is_english(comment):
                    non_english_dropped += 1
                    continue

            # Parse label columns safely
            toxic_val = row.get("toxic", 0)
            severe_val = row.get("severe_toxic", 0)
            threat_val = int(row.get("threat", 0)) if str(row.get("threat", 0)).strip() else 0
            identity_val = int(row.get("identity_hate", 0)) if str(row.get("identity_hate", 0)).strip() else 0

            toxic_lvl = derive_toxic_level(toxic_val, severe_val)

            row["toxic_level"] = str(toxic_lvl)
            row["_toxic_lvl_int"] = toxic_lvl
            row["_threat_int"] = threat_val
            row["_identity_int"] = identity_val

            valid_rows.append(row)

    total_valid = len(valid_rows)

    if total_valid <= sample_size:
        selected_rows = valid_rows
    else:
        # Separate into pools based on label characteristics
        # Rare categories: threat and identity_hate
        threat_pool = [r for r in valid_rows if r["_threat_int"] == 1]
        hate_pool = [r for r in valid_rows if r["_identity_int"] == 1 and r["_threat_int"] == 0]

        # Toxic severe (level 2) not in rare pools
        sev_pool = [
            r for r in valid_rows
            if r["_toxic_lvl_int"] == 2 and r["_threat_int"] == 0 and r["_identity_int"] == 0
        ]

        # Toxic level 1 not in rare pools
        tox_pool = [
            r for r in valid_rows
            if r["_toxic_lvl_int"] == 1 and r["_threat_int"] == 0 and r["_identity_int"] == 0
        ]

        # Clean negatives (level 0, no threat, no hate)
        neg_pool = [
            r for r in valid_rows
            if r["_toxic_lvl_int"] == 0 and r["_threat_int"] == 0 and r["_identity_int"] == 0
        ]

        # Shuffle each pool with the deterministic RNG
        rng.shuffle(threat_pool)
        rng.shuffle(hate_pool)
        rng.shuffle(sev_pool)
        rng.shuffle(tox_pool)
        rng.shuffle(neg_pool)

        # Allocate targets based on sample_size proportion
        # For sample_size = 6000:
        # threat: ~350, hate: ~450, sev: ~500, tox: ~1200, neg: ~3500
        ratio = sample_size / 6000.0
        target_threat = max(1, min(len(threat_pool), int(350 * ratio)))
        target_hate = max(1, min(len(hate_pool), int(450 * ratio)))
        target_sev = max(1, min(len(sev_pool), int(500 * ratio)))
        target_tox = max(1, min(len(tox_pool), int(1200 * ratio)))

        selected_dict: Dict[str, Dict[str, Any]] = {}

        def add_from_pool(pool: List[Dict[str, Any]], count: int):
            added = 0
            for item in pool:
                if len(selected_dict) >= sample_size:
                    break
                if item["id"] not in selected_dict:
                    selected_dict[item["id"]] = item
                    added += 1
                    if added >= count:
                        break

        # Priority 1: Rare categories
        add_from_pool(threat_pool, target_threat)
        add_from_pool(hate_pool, target_hate)

        # Priority 2: Toxic severe and toxic
        add_from_pool(sev_pool, target_sev)
        add_from_pool(tox_pool, target_tox)

        # Priority 3: Clean negatives
        needed = sample_size - len(selected_dict)
        add_from_pool(neg_pool, needed)

        # Fallback: if quota not reached, draw from any remaining pool
        if len(selected_dict) < sample_size:
            all_remaining = (
                threat_pool + hate_pool + sev_pool + tox_pool + neg_pool
            )
            for item in all_remaining:
                if len(selected_dict) >= sample_size:
                    break
                if item["id"] not in selected_dict:
                    selected_dict[item["id"]] = item

        selected_rows = list(selected_dict.values())

    # Final deterministic shuffle
    rng.shuffle(selected_rows)

    # Compute aggregate statistics
    sampled_threat = sum(1 for r in selected_rows if r["_threat_int"] == 1)
    sampled_hate = sum(1 for r in selected_rows if r["_identity_int"] == 1)
    sampled_tox0 = sum(1 for r in selected_rows if r["_toxic_lvl_int"] == 0)
    sampled_tox1 = sum(1 for r in selected_rows if r["_toxic_lvl_int"] == 1)
    sampled_tox2 = sum(1 for r in selected_rows if r["_toxic_lvl_int"] == 2)

    # Ensure parent directories exist
    for path in [prompts_path, ground_truth_path]:
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    # 1. Write Prompts File: ONLY 'id' and 'comment_text'
    prompts_fields = ["id", "comment_text"]
    with open(prompts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=prompts_fields)
        writer.writeheader()
        for r in selected_rows:
            writer.writerow({
                "id": r.get("id", ""),
                "comment_text": r.get("comment_text", "")
            })

    # 2. Write Ground Truth File: 'id' and labels, NO 'comment_text'
    gt_fields = [f for f in fieldnames if not f.startswith("_") and f != "comment_text"]
    if "toxic_level" not in gt_fields:
        gt_fields.append("toxic_level")
    # Make sure 'id' is first
    if "id" in gt_fields:
        gt_fields.remove("id")
        gt_fields.insert(0, "id")

    with open(ground_truth_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=gt_fields)
        writer.writeheader()
        for r in selected_rows:
            row_data = {k: r.get(k, "") for k in gt_fields}
            writer.writerow(row_data)

    stats = {
        "total_read": total_read,
        "empty_dropped": empty_dropped,
        "non_english_dropped": non_english_dropped,
        "total_valid": total_valid,
        "total_sampled": len(selected_rows),
        "threat_positives": sampled_threat,
        "identity_hate_positives": sampled_hate,
        "toxic_level_0": sampled_tox0,
        "toxic_level_1": sampled_tox1,
        "toxic_level_2": sampled_tox2,
    }
    return stats


def print_summary_table(stats: Dict[str, Any], prompts_path: str, ground_truth_path: str):
    """Prints ONLY aggregate numbers and summary stats. No raw text is exposed."""
    print("=" * 65)
    print("DATASET STRATIFICATION & FILTERING SUMMARY")
    print("=" * 65)
    print(f"Total rows read:           {stats['total_read']:,}")
    print(f"Empty/whitespace dropped:  {stats['empty_dropped']:,}")
    print(f"Non-English dropped:       {stats['non_english_dropped']:,}")
    print(f"Total valid candidates:    {stats['total_valid']:,}")
    print("-" * 65)
    print(f"Total sampled items:       {stats['total_sampled']:,}")
    print(f"  • Toxic Level 0 (None):  {stats['toxic_level_0']:,}")
    print(f"  • Toxic Level 1 (Toxic): {stats['toxic_level_1']:,}")
    print(f"  • Toxic Level 2 (Severe):{stats['toxic_level_2']:,}")
    print(f"  • Threat Positives:      {stats['threat_positives']:,}")
    print(f"  • Identity Hate Pos:     {stats['identity_hate_positives']:,}")
    print("-" * 65)
    print("Output Files (Strict Separation):")
    print(f"  [Prompts / Model Input]   {prompts_path}")
    print(f"                            -> Contains: id, comment_text (NO LABELS)")
    print(f"  [Ground Truth / Eval]     {ground_truth_path}")
    print(f"                            -> Contains: id, toxic_level, threat, identity_hate (NO TEXT)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Filter and stratify Jigsaw Toxic Comment dataset into separated prompts and ground-truth files."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input raw Jigsaw CSV file"
    )
    parser.add_argument(
        "--output-dir", "-o", default="jev-experiments/data",
        help="Directory to save the separated output files (default: jev-experiments/data)"
    )
    parser.add_argument(
        "--prefix", default="eval_6k",
        help="Filename prefix for generated files (default: eval_6k -> eval_6k_prompts.csv & eval_6k_ground_truth.csv)"
    )
    parser.add_argument(
        "--output-prompts",
        help="Explicit path for prompts CSV (id, comment_text)"
    )
    parser.add_argument(
        "--output-ground-truth",
        help="Explicit path for ground truth CSV (id, labels)"
    )
    parser.add_argument(
        "--sample-size", "-n", type=int, default=6000,
        help="Target number of sampled comments (default: 6000)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed for deterministic sampling (default: 42)"
    )
    parser.add_argument(
        "--skip-lang-filter", action="store_true",
        help="Skip language detection filter (useful if langdetect is not installed or for fast dry-run)"
    )

    args = parser.parse_args()

    prompts_path = args.output_prompts or os.path.join(args.output_dir, f"{args.prefix}_prompts.csv")
    gt_path = args.output_ground_truth or os.path.join(args.output_dir, f"{args.prefix}_ground_truth.csv")

    if not args.skip_lang_filter and not LANGDETECT_AVAILABLE:
        print(
            "Note: langdetect is not installed in the current environment. "
            "Proceeding without language filtering. To enable, install with: pip install langdetect",
            file=sys.stderr
        )

    stats = filter_and_stratify(
        input_path=args.input,
        prompts_path=prompts_path,
        ground_truth_path=gt_path,
        sample_size=args.sample_size,
        seed=args.seed,
        skip_lang_filter=args.skip_lang_filter,
    )

    print_summary_table(stats, prompts_path, gt_path)


if __name__ == "__main__":
    main()
