# Trust & Safety Eval Gate Benchmark Implementation Plan

**Goal:** Build a high-throughput, asynchronous evaluation benchmark comparing Jev against general-purpose LLMs (Claude Haiku 4.5, Claude Sonnet 5.5, and Gemini Flash 3.8) on a 6,000-comment Trust & Safety content moderation task with incremental checkpointing and post-run analysis.

**Architecture:** A modular Python execution pipeline using `asyncio` with provider-specific semaphores to run 1-comment-per-call evaluations concurrently. Model outputs are parsed into a normalized two-table schema (`calls.csv` and `answers.csv`) checkpointed per comment. Analysis joins predictions with an isolated ground-truth file to compute Precision-Recall curves (recall-constrained operating points) and Jev-specific confidence calibration.

**Tech Stack:** Python 3.9+, `asyncio`, TypeSafe Python SDK (`typesafe-sdk`), Anthropic SDK (`anthropic`), Google GenAI SDK (`google-genai`), `python-dotenv`, `pandas`, `matplotlib`, `scikit-learn`.

**Spec:** [`jev-experiments/PRD-tns-eval-gate.md`](PRD-tns-eval-gate.md) and [`jev-experiments/EXECUTION.md`](EXECUTION.md)

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
- **Files:**
  - Create: `jev-experiments/requirements.txt`
  - Create: `.env` (pre-populated with safe placeholder keys)
  - Create: `.env.example`
  - Create/Modify: `.gitignore` (guaranteeing `.env` is never tracked)
  - Create: `jev-experiments/checkpoint.py`
  - Test: `jev-experiments/test_checkpoint.py`
- **Interfaces:**
  - `CheckpointManager(calls_path, answers_path)` with methods:
    - `get_completed_ids(model: str) -> Set[str]`
    - `record_result(calls_row: dict, answers_rows: list[dict]) -> None`
- **Steps:**
  1. Write failing test for `CheckpointManager` (file creation, atomic appends, recovery of completed IDs).
  2. Run test to verify it fails (`python3 -m unittest jev-experiments/test_checkpoint.py`).
  3. Implement `requirements.txt`, `.env` template, `.gitignore`, and `checkpoint.py`.
  4. Run test to verify it passes.

---

### Task 2: Shared Normalization Parsers & Prompt Definitions
- **Files:**
  - Create: `jev-experiments/prompts.py`
  - Create: `jev-experiments/parsers.py`
  - Test: `jev-experiments/test_parsers.py`
- **Interfaces:**
  - `SYSTEM_PROMPT_TEMPLATE`, `TOXIC_RUBRIC`, `THREAT_DEFINITION`, `IDENTITY_HATE_DEFINITION` in `prompts.py`.
  - `parse_jev_response(...) -> (calls_row, answers_rows)` in `parsers.py`.
  - `parse_llm_response(...) -> (calls_row, answers_rows)` in `parsers.py`.
- **Steps:**
  1. Write failing test for parsers verifying exact schema compliance, probability normalization, weighted mean calculation, and Jev-only native confidence.
  2. Run test to verify it fails.
  3. Implement verbatim category definitions in `prompts.py` and parsers in `parsers.py`.
  4. Run test to verify it passes.

---

### Task 3: Asynchronous Model Client Wrappers
- **Files:**
  - Create: `jev-experiments/models/base.py`
  - Create: `jev-experiments/models/jev_model.py`
  - Create: `jev-experiments/models/anthropic_model.py`
  - Create: `jev-experiments/models/gemini_model.py`
  - Test: `jev-experiments/test_models.py`
- **Interfaces:**
  - Uniform async evaluator interface:
    `async def evaluate(comment_id: str, comment_text: str) -> tuple[dict, list[dict]]`
    Includes exponential backoff with jitter on HTTP 429 rate limits.
- **Steps:**
  1. Write failing test with mock API responses verifying latency measurement, backoff retry logic, and parser integration.
  2. Run test to verify it fails.
  3. Implement Jev System One async call (`Score` & `Noul`), Anthropic tool-use schema (Haiku & Sonnet), and Gemini JSON-schema output (Flash).
  4. Run test to verify it passes.

---

### Task 4: Pilot Dry-Run Runner (Sanity Check)
- **Files:**
  - Create: `jev-experiments/run_pilot.py`
- **Goal:** Allow running a cheap 50-comment pilot run on Jev (and optional 5 items on LLMs) to verify API keys, live response formats, latency timing, and spot-check label alignment before spending the bulk budget.
- **Steps:**
  1. Implement `run_pilot.py` loading $N$ comments from `eval_6k_prompts.csv`, running selected models, and printing latency/token/cost stats without text leakage.
  2. Verify execution on dummy/sample mode.

---

### Task 5: Full Asynchronous Benchmark Runner
- **Files:**
  - Create: `jev-experiments/run_benchmark.py`
- **Goal:** Orchestrate the full 6,000 items × 4 models (~24,000 evaluations) with concurrent worker pools and real-time progress.
- **Steps:**
  1. Implement `run_benchmark.py` with CLI flags, provider-specific semaphores (`asyncio.Semaphore(50)` for Jev, `30` for Gemini, `15` for Anthropic), `CheckpointManager` skipping, and graceful `Ctrl+C` shutdown.
  2. Verify concurrency controls, resumption, and signal handling.

---

### Task 6: Evaluation & Analysis Engine
- **Files:**
  - Create: `jev-experiments/evaluate.py`
  - Test: `jev-experiments/test_evaluate.py`
- **Goal:** Join `answers.csv` and `calls.csv` with `eval_6k_ground_truth.csv` to calculate benchmark metrics and generate PR curves and calibration plots.
- **Steps:**
  1. Write failing test verifying 3-level ordinal F1, collapsed binary F1, recall-constrained ($\ge 90\%$ recall) operating point search, and Jev confidence quartile calibration.
  2. Run test to verify it fails.
  3. Implement evaluation metric computation and plot generation (`matplotlib`).
  4. Run test to verify it passes.

---

## User Responsibilities & Step-by-Step Guide

### Phase 1: Stratify Your Dataset
Run the stratification script on your local Kaggle CSV:
```bash
python3 jev-experiments/stratify_dataset.py \
  --input /path/to/your/train.csv \
  --output-dir jev-experiments/data \
  --prefix eval_6k \
  --sample-size 6000 \
  --seed 42 \
  --skip-lang-filter
```
*Output:* Creates `eval_6k_prompts.csv` (text only) and `eval_6k_ground_truth.csv` (labels only).

### Phase 2: Configure Secrets & Dependencies
1. Install requirements once Task 1 is implemented:
   ```bash
   pip install -r jev-experiments/requirements.txt
   ```
2. Open the auto-generated `.env` file and replace the placeholders with your actual keys:
   ```env
   TYPESAFE_API_KEY=your_typesafe_key
   ANTHROPIC_API_KEY=your_anthropic_key
   GEMINI_API_KEY=your_gemini_key
   ```

### Phase 3: Run the 50-Item Pilot Sanity Check
Test connectivity, latency recording, and schema alignment on 50 comments:
```bash
python3 jev-experiments/run_pilot.py --model jev --sample-size 50
```

### Phase 4: Launch Full Asynchronous Benchmark
Launch the 24,000-call benchmark across all models:
```bash
python3 jev-experiments/run_benchmark.py --models jev,haiku,flash,sonnet
```
*(Automatically saves after every comment and resumes if interrupted).*

### Phase 5: Generate Final Benchmark Report & PR Curves
```bash
python3 jev-experiments/evaluate.py
```
*(Prints headline comparison metrics table and outputs PR curve plots).*
