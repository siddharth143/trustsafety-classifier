# Trust & Safety Eval Gate: Jev vs. LLMs Benchmark

An empirical benchmark comparing **Jev** (TypeSafe System One decision primitive) against general-purpose LLMs (**Claude 3.5 Haiku**, **Claude 3.5 Sonnet**, and **Gemini 1.5 Flash**) on a stratified 6,000-comment subset of the Jigsaw Toxic Comment dataset.

---

## 📁 Repository Structure

```text
jev-experiments/
├── README.md               # Project documentation and quickstart guide
├── requirements.txt        # Pinned Python package dependencies
├── docs/                   # Specifications, PRDs, and architecture plans
│   ├── PRD-tns-eval-gate.md
│   ├── PRD-model-router.md
│   ├── PLAN.md
│   └── EXECUTION.md
├── data/                   # Stratified evaluation datasets
│   ├── eval_6k_prompts.csv         # 6,000 comments (id, comment_text only — strict isolation)
│   ├── eval_6k_ground_truth.csv    # 6,000 labels (id, toxic_level, threat, identity_hate)
│   └── raw/
│       └── train.csv               # Original full Jigsaw dataset
├── src/                    # Core modular library
│   ├── __init__.py
│   ├── checkpoint.py       # Atomic CSV checkpoint manager & crash recovery
│   ├── prompts.py          # Unified rubrics, few-shot examples, JSON schemas
│   ├── parsers.py          # Response normalizers, token pricing, native confidence
│   └── models/             # Async model client wrappers with exponential backoff & jitter
│       ├── base.py
│       ├── jev_model.py
│       ├── anthropic_model.py
│       └── gemini_model.py
├── scripts/                # Execution CLI entrypoints
│   ├── stratify_dataset.py # Deterministic stratification & strict dataset separation
│   ├── run_pilot.py        # 50-item dry-run sanity check
│   ├── run_benchmark.py    # 6,000-item async runner with provider concurrency semaphores
│   └── evaluate.py         # Metrics, PR curves, threshold sweeps, calibration & report
└── tests/                  # Unit and integration test suite (74 tests)
    ├── test_stratify.py
    ├── test_checkpoint.py
    ├── test_parsers.py
    ├── test_models.py
    ├── test_runners.py
    └── test_evaluate.py
```

---

## 🚀 Quickstart

### 1. Environment Setup

Ensure Python $\ge 3.10$ is used (e.g. via virtual environment):

```bash
# From workspace root
source .venv/bin/activate
pip install -r jev-experiments/requirements.txt
```

Set your API credentials in `.env` (at repository root or in your environment):

```bash
TYPESAFE_API_KEY="your-typesafe-key"
ANTHROPIC_API_KEY="your-anthropic-key"
# Required if your Anthropic key is an organization user key (sk-ant-usr...):
ANTHROPIC_WORKSPACE_ID="wrkspc_xxxxxxxxxxxxxxxxxxxxxxxx"
GEMINI_API_KEY="your-gemini-key"
```

---

### 2. Verify with Unit Tests

Run the full test suite (all tests mock external API calls; no live keys required):

```bash
python -m unittest discover -s jev-experiments/tests
```

All 74 tests should pass green.

---

### 3. Step 1: Run Pilot Sanity Check

Run a quick 50-comment pilot check across models to verify API connectivity, latency, pricing, and checkpoint schemas:

```bash
# Pilot with Jev only
python jev-experiments/scripts/run_pilot.py --sample-size 50 --models jev

# Pilot across all models
python jev-experiments/scripts/run_pilot.py --sample-size 50 --models jev,haiku,flash,sonnet
```

---

### 4. Step 2: Run Full Benchmark (6,000 comments)

Execute the full asynchronous benchmark across models with provider-specific concurrency limits and auto-resumption:

```bash
python jev-experiments/scripts/run_benchmark.py \
    --prompts-file jev-experiments/data/eval_6k_prompts.csv \
    --models jev,haiku,flash,sonnet \
    --output-dir jev-experiments/results
```

- **Resumption:** If interrupted, re-running the same command automatically skips already completed comments.
- **Provider Semaphores:**
  - Jev: `50` concurrent requests
  - Gemini 1.5 Flash: `30` concurrent requests
  - Anthropic (Haiku / Sonnet shared): `15` concurrent requests

---

### 5. Step 3: Run Evaluation & Generate Reports

Compute cost, latency percentiles, precision/recall/F1, PR curves, operating points ($\ge 90\%$ recall), and Jev confidence calibration:

```bash
python jev-experiments/scripts/evaluate.py \
    --calls-file jev-experiments/results/calls.csv \
    --answers-file jev-experiments/results/answers.csv \
    --ground-truth-file jev-experiments/data/eval_6k_ground_truth.csv \
    --output-dir jev-experiments/results
```

Generated artifacts in `jev-experiments/results/`:
- `summary.md`: Comprehensive markdown report with benchmark tables.
- `pr_curve_toxic.png`, `pr_curve_threat.png`, `pr_curve_identity_hate.png`: Precision-Recall curves.
- `calibration_jev.png`: Reliability diagram for Jev native confidence scores.
- `evaluation_metrics.json`: Full machine-readable metrics dump.

---

## 🔒 Privacy & Safety Guarantee

- Models only receive `id` and `comment_text` from [eval_6k_prompts.csv](file:///Users/siddharth/Desktop/Traviz/jev-experiments/data/eval_6k_prompts.csv).
- Ground truth labels in [eval_6k_ground_truth.csv](file:///Users/siddharth/Desktop/Traviz/jev-experiments/data/eval_6k_ground_truth.csv) contain **no comment text**.
- Raw comment strings are never printed or logged to stdout, stderr, or telemetry logs.
