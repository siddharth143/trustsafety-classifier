# Trust & Safety Eval Gate Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-throughput, asynchronous evaluation benchmark comparing Jev against general-purpose LLMs (Claude Haiku 4.5, Claude Sonnet 5.5, and Gemini Flash 3.8) on a 6,000-comment Trust & Safety content moderation task with incremental checkpointing and post-run analysis.

**Architecture:** A modular Python execution pipeline using `asyncio` with provider-specific semaphores to run 1-comment-per-call evaluations concurrently. Model outputs are parsed into a normalized two-table schema (`calls.csv` and `answers.csv`) checkpointed per comment. Analysis joins predictions with an isolated ground-truth file to compute Precision-Recall curves (recall-constrained operating points) and Jev-specific confidence calibration.

**Tech Stack:** Python 3.9+, `asyncio`, TypeSafe Python SDK (`typesafe-sdk`), Anthropic SDK (`anthropic`), Google GenAI SDK (`google-genai`), `python-dotenv`, `matplotlib`, `scikit-learn`.

**Spec:** [`jev-experiments/docs/PRD-tns-eval-gate.md`](../../jev-experiments/docs/PRD-tns-eval-gate.md) and [`jev-experiments/docs/EXECUTION.md`](../../jev-experiments/docs/EXECUTION.md)

---

## Global Constraints

- **Strict Data Isolation:** Models receive ONLY `id` and `comment_text` from `eval_6k_prompts.csv`. No ground-truth labels are ever passed into prompts or model payloads.
- **Privacy & Redaction:** Under no circumstances are raw comment texts or snippets printed to console, stdout, or public run logs.
- **Point-wise Latency:** Every API call evaluates exactly 1 comment to faithfully measure real-world production eval-gate latency (p50/p95).
- **Incremental Resumability:** Every completed call is immediately flushed to disk (`calls.csv` and `answers.csv`) so interrupted runs resume without paying for or duplicating completed work.
- **Provider Pricing Floors:**
  - Jev: $0.042 / MTok input, free output.
  - Sonnet 5.5: $3.00 / MTok input, $15.00 / MTok output.
  - Haiku 4.5: $0.80 / MTok input, $4.00 / MTok output.
  - Gemini Flash 3.8: $0.075 / MTok input, $0.30 / MTok output.

---

## Review Focus

1. **429 Rate Limits / Provider Throttling:** Must implement exponential backoff with jitter on HTTP 429/503 errors so high concurrency does not cause failed runs.
2. **Malformed / Incomplete JSON from LLMs:** Structured output parsers must gracefully catch malformed payloads, log an error row in `calls.csv`, and allow targeted retries without crashing the pipeline.
3. **Distribution Normalization:** Toxic probabilities (`"0"`, `"1"`, `"2"`) from LLMs must sum to 1.0; enforce floating-point normalization in parser before scoring.
4. **Resumable Run Edge Case:** Restarting a partially completed run must read existing `(comment_id, model)` pairs from `calls.csv` and skip them seamlessly.
5. **Jev Native Confidence Nullability:** Ensure `native_confidence` is populated for Jev rows only (direct for `Score`, derived $2\times|p-0.5|$ for `Noul`) and explicitly `None`/null for all LLMs.

---

## Implementation Tasks

### Task 1: Environment Setup, .env Configuration & Checkpointing Core

**Files:**
- Create: `jev-experiments/requirements.txt`
- Create: `.env` (pre-populated with safe placeholder keys)
- Create: `.env.example`
- Create/Modify: `.gitignore` (guaranteeing `.env` is never tracked)
- Create: `jev-experiments/checkpoint.py`
- Test: `jev-experiments/test_checkpoint.py`

**Interfaces:**
- Produces: `CheckpointManager(calls_path, answers_path)` with methods:
  - `get_completed_ids(model: str) -> Set[str]`
  - `record_result(calls_row: dict, answers_rows: list[dict]) -> None`

- [ ] **Step 1: Write failing test for CheckpointManager**
  Test creating checkpoint files, recording results atomically, retrieving already processed IDs for a given model, and verifying that resuming skips processed IDs.
- [ ] **Step 2: Run test to verify it fails**
  Run `python3 -m unittest jev-experiments/test_checkpoint.py`
- [ ] **Step 3: Implement CheckpointManager, requirements, and .env template**
  - Implement `requirements.txt` with required SDK versions (`typesafe-sdk`, `anthropic`, `google-genai`, `python-dotenv`, `pandas`, `matplotlib`, `scikit-learn`).
  - Create `.env` pre-populated with safe placeholders (`TYPESAFE_API_KEY=your_typesafe_key_here`, etc.).
  - Create `.env.example` as a committed reference.
  - Create/Update `.gitignore` to ensure `.env` is strictly ignored.
  - Implement `jev-experiments/checkpoint.py` with CSV thread/coroutine-safe writing and recovery.
- [ ] **Step 4: Run test to verify it passes**
  Run `python3 -m unittest jev-experiments/test_checkpoint.py`

---

### Task 2: Shared Normalization Parsers & Prompt Definitions

**Files:**
- Create: `jev-experiments/prompts.py`
- Create: `jev-experiments/parsers.py`
- Test: `jev-experiments/test_parsers.py`

**Interfaces:**
- Produces:
  - `SYSTEM_PROMPT_TEMPLATE`, `TOXIC_RUBRIC`, `THREAT_DEFINITION`, `IDENTITY_HATE_DEFINITION` in `prompts.py`.
  - `parse_jev_response(...) -> (calls_row, answers_rows)` in `parsers.py`.
  - `parse_llm_response(...) -> (calls_row, answers_rows)` in `parsers.py`.

- [ ] **Step 1: Write failing test for parsers**
  Verify that dummy mock responses from Jev and LLMs are converted into the exact PRD two-table schema (`calls` and `answers`), verifying that `toxic_score` is weighted mean, `native_confidence` is Jev-only, and token cost calculation is accurate.
- [ ] **Step 2: Run test to verify it fails**
  Run `python3 -m unittest jev-experiments/test_parsers.py`
- [ ] **Step 3: Implement prompts and parsers**
  - Transfer verbatim category definitions, worked examples, and noise fallback rule into `prompts.py`.
  - Implement `parse_jev_response` and `parse_llm_response` matching the logic in `EXECUTION.md`.
- [ ] **Step 4: Run test to verify it passes**
  Run `python3 -m unittest jev-experiments/test_parsers.py`

---

### Task 3: Asynchronous Model Client Wrappers

**Files:**
- Create: `jev-experiments/models/base.py`
- Create: `jev-experiments/models/jev_model.py`
- Create: `jev-experiments/models/anthropic_model.py`
- Create: `jev-experiments/models/gemini_model.py`
- Test: `jev-experiments/test_models.py`

**Interfaces:**
- Consumes: `prompts.py`, `parsers.py`
- Produces: Model evaluators with uniform async signature:
  `async def evaluate(comment_id: str, comment_text: str) -> tuple[dict, list[dict]]`
  Includes exponential backoff retry loop on rate limits (429).

- [ ] **Step 1: Write failing test with mock API clients**
  Verify that each model runner invokes its provider with the comment text, records latency in milliseconds, calls the correct parser, and retries with backoff on simulated 429 errors.
- [ ] **Step 2: Run test to verify it fails**
  Run `python3 -m unittest jev-experiments/test_models.py`
- [ ] **Step 3: Implement model clients**
  - Implement Jev System One async call with `Score` and `Noul`.
  - Implement Anthropic tool-use schema for Haiku and Sonnet.
  - Implement Gemini JSON-schema output mode for Flash.
- [ ] **Step 4: Run test to verify it passes**
  Run `python3 -m unittest jev-experiments/test_models.py`

---

### Task 4: Pilot Dry-Run Runner (Sanity Check)

**Files:**
- Create: `jev-experiments/run_pilot.py`

**Goal:** Allow running a cheap 50-comment pilot run (starting with Jev, and 5 comments on LLMs) to verify API keys, live response formats, latency timing, and spot-check label alignment before triggering the 6,000-comment run.

- [ ] **Step 1: Implement `run_pilot.py`**
  - Loads first $N$ rows (default: 50) from `eval_6k_prompts.csv`.
  - Runs inference on selected model(s).
  - Prints summary stats (latency p50/p95, total tokens, total cost).
- [ ] **Step 2: Verification**
  Run dry-run with mock/test mode to verify graceful execution and summary display without text leakage.

---

### Task 5: Full Asynchronous Benchmark Runner

**Files:**
- Create: `jev-experiments/run_benchmark.py`

**Goal:** Orchestrate the full 6,000 items × 4 models (~24,000 evaluations) with concurrent worker pools and real-time progress.

- [ ] **Step 1: Implement `run_benchmark.py`**
  - CLI flags: `--prompts-file`, `--models`, `--concurrency-<provider>`, `--limit`.
  - Semaphores: `asyncio.Semaphore(50)` for Jev, `30` for Gemini, `15` for Anthropic.
  - Integrates `CheckpointManager` so completed items are skipped automatically.
  - Clean signal handling (`Ctrl+C` flushes buffer and halts gracefully).
- [ ] **Step 2: Verification**
  Verify CLI argument parsing, resumption detection, and semaphore pooling.

---

### Task 6: Evaluation & Analysis Engine

**Files:**
- Create: `jev-experiments/evaluate.py`
- Test: `jev-experiments/test_evaluate.py`

**Goal:** Join `answers.csv` and `calls.csv` with `eval_6k_ground_truth.csv` to calculate benchmark metrics and generate PR curve and calibration plots.

- [ ] **Step 1: Write test for evaluation calculations**
  Verify F1 calculation (3-level ordinal and collapsed binary), recall-constrained operating point ($\ge 90\%$ recall threshold search), cost-per-1k aggregation, and confidence calibration bucketing.
- [ ] **Step 2: Run test to verify it fails**
  Run `python3 -m unittest jev-experiments/test_evaluate.py`
- [ ] **Step 3: Implement evaluation calculations and plotting**
  - Compute precision, recall, and F1 across threshold sweeps for all 4 models and 3 categories.
  - Mark operating point ($\ge 90\%$ recall) and max-F1 point.
  - Generate Precision-Recall curve charts (`docs/pr_curves_<category>.png`).
  - Generate Jev confidence calibration chart and quartile table.
  - Generate Latency (p50/p95) and Cost-per-1k summary table.
- [ ] **Step 4: Run test to verify it passes**
  Run `python3 -m unittest jev-experiments/test_evaluate.py`

---

## User Responsibilities & Step-by-Step Guide

Here is the exact step-by-step sequence for you:

### Phase 1: Stratify Your Dataset
1. Run the stratification command on your local Kaggle CSV:
   ```bash
   python3 jev-experiments/stratify_dataset.py \
     --input /path/to/your/train.csv \
     --output-dir jev-experiments/data \
     --prefix eval_6k \
     --sample-size 6000 \
     --seed 42 \
     --skip-lang-filter
   ```
2. Confirm you see the summary table with category counts and that two files were created in `jev-experiments/data/`:
   - `eval_6k_prompts.csv`
   - `eval_6k_ground_truth.csv`

### Phase 2: Configure Secrets & Dependencies
1. Install requirements once created:
   ```bash
   pip install -r jev-experiments/requirements.txt
   ```
2. Create `.env` in the workspace root or `jev-experiments/` and add your API keys:
   ```env
   TYPESAFE_API_KEY=your_typesafe_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   GEMINI_API_KEY=your_gemini_key_here
   ```

### Phase 3: Run the 50-Item Pilot Sanity Check
1. Run Jev on 50 comments to verify connectivity and schema:
   ```bash
   python3 jev-experiments/run_pilot.py --model jev --sample-size 50
   ```
2. Verify latency, token accounting, and that no errors occurred.

### Phase 4: Launch Full Asynchronous Benchmark
1. Run the benchmark across all models (or start with Jev, then LLMs):
   ```bash
   python3 jev-experiments/run_benchmark.py --models jev,haiku,flash,sonnet
   ```
   *Note: If interrupted at any time, re-running the same command automatically resumes from the last completed comment.*

### Phase 5: Generate Final Benchmark Report & PR Curves
1. Run the evaluation script:
   ```bash
   python3 jev-experiments/evaluate.py
   ```
2. Review the generated metrics table, PR curves, and calibration plots.
