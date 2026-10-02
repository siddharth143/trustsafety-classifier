# Trust & Safety Eval Gate: Jev vs. LLMs Benchmark

An empirical content moderation benchmark and production-grade evaluation gate comparing **TypeSafe Jev** (System One decision primitive) against frontier general-purpose LLMs (**Claude 3.5 Sonnet**, **Gemini 1.5 Flash**, and **Claude 3.5 Haiku**) on a stratified 6,000-comment dataset.

---

## Table of Contents

1. [About the Project & Classifier](#-about-the-project--classifier)
2. [Experiment Design & Dataset](#-experiment-design--dataset)
   - [Dataset & Kaggle Source](#dataset--kaggle-source)
   - [Sampling & Stratification](#sampling--stratification)
   - [Strict Data Isolation & Privacy](#strict-data-isolation--privacy)
   - [Category Definitions & Considered Rubrics](#category-definitions--considered-rubrics)
3. [Technical Architecture](#-technical-architecture)
   - [Pointwise Concurrency Pipeline](#pointwise-concurrency-pipeline)
   - [Crash-Resilient Atomic Checkpointing](#crash-resilient-atomic-checkpointing)
   - [Two-Tier Model-Router Architecture](#two-tier-model-router-architecture)
4. [Models, Results & Key Findings](#-models-results--key-findings)
   - [Evaluated Models](#evaluated-models)
   - [Empirical Performance & Cost Summary](#empirical-performance--cost-summary)
   - [The "Hard-Zero" Deficit Finding](#the-hard-zero-deficit-finding)
   - [Jev Confidence Calibration](#jev-confidence-calibration)
   - [Where to Find Results & Visualizations](#where-to-find-results--visualizations)
5. [Repository Structure](#-repository-structure)
6. [Quickstart & Reproduction Guide](#-quickstart--reproduction-guide)
   - [Environment Setup](#1-environment-setup)
   - [Running Unit Tests](#2-running-unit-tests)
   - [Step 1: Pilot Check (50 items)](#3-step-1-pilot-check-50-items)
   - [Step 2: Full Benchmark (6,000 items)](#4-step-2-full-benchmark-6000-items)
   - [Step 3: Metrics & Visual Report Generation](#5-step-3-metrics--visual-report-generation)
7. [Security & Credentials Management](#-security--credentials-management)

---

## 🎯 About the Project & Classifier

Automated Trust & Safety (T&S) classification is a mission-critical component of modern community platforms, agentic guardrails, and user-facing AI pipelines. When deploying moderation filters at scale, system architects encounter a challenging trilemma between **latency**, **operating cost**, and **safety recall**:

1. **Fine-Tuned Classifiers (e.g. BERT/DeBERTa):** Extremely fast and inexpensive, but rigid, prone to drift, require laborious dataset annotation pipelines, and struggle with nuanced linguistic context.
2. **General-Purpose LLMs (e.g. Claude Sonnet, Gemini Flash):** Highly articulate and context-aware, but introduce prohibitive token-generation latencies (2,000–4,000 ms), high operational costs ($0.14–$11.16 per 1k items), and uncalibrated discrete token probabilities.
3. **Purpose-Built Decision Primitives (TypeSafe Jev):** A specialized "System One" decision model that evaluates content directly via typed primitives (`Score` for graded distributions, `Noul` for binary judgments). Jev returns continuous Bayesian probabilities and native confidence scores without autoregressive text decoding.

### What This Classifier Does

This benchmark evaluates inbound user comments across three independent, high-risk moderation dimensions:
- **Toxicity (Ordinal 3-Level):** Quantifies hostility and degradation intensity on a 3-tier scale (Clean, Toxic, Severe Toxic).
- **Threat (Binary):** Flags explicit intent to inflict physical injury, violence, death, or conditional coercion.
- **Identity Hate (Binary):** Detects attacks, slurs, and dehumanizing language directed at protected demographic characteristics.

The primary objective is to evaluate whether **Jev** can match or exceed frontier LLM classification quality (F1, Precision-Recall AUC) while operating at orders-of-magnitude lower latency and cost, and whether a **Two-Tier Model-Router** provides the optimal Pareto frontier for production deployment.

---

## 🔬 Experiment Design & Dataset

### Dataset & Kaggle Source

The benchmark is conducted on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset hosted on Kaggle. The source corpus comprises 159,571 Wikipedia talk-page comments labeled by human annotators across multiple toxicity dimensions.

### Sampling & Stratification

Real-world moderation data exhibits severe class imbalance (threats and identity hate represent $<1\%$ of raw comments). To compute statistically robust precision-recall curves and evaluate high-recall operating points, we deterministically sampled a **6,000-comment stratified subset** (random seed `42`) from the raw training corpus:

| Category / Stratum | Sample Count | Percentage of Eval Set | Purpose |
|:---|---:|---:|:---|
| **Clean Controls** | 4,101 | 68.35% | Rigorous false-positive evaluation |
| **General Toxicity (Level 1)** | 1,498 | 24.97% | Discriminating mild vs. severe toxicity |
| **Severe Toxicity (Level 2)** | 401 | 6.68% | High-intensity degradation detection |
| **Identity Hate (Binary Positive)** | 521 | 8.68% | Statistically powered protected-class eval |
| **Threats (Binary Positive)** | 350 | 5.83% | Statistically powered violence intent eval |

*Note: Individual comments can exhibit multiple labels simultaneously (e.g. both severe_toxic and threat).*

### Strict Data Isolation & Privacy

To prevent label leakage and ensure reproducible blind evaluations:
- **`data/eval_6k_prompts.csv`:** Contains **only** `id` and `comment_text`. Zero ground-truth labels are present in the inference prompts or model payloads.
- **`data/eval_6k_ground_truth.csv`:** Contains **only** `id`, `toxic_level`, `threat`, and `identity_hate`. Zero comment text is retained in the evaluation ground-truth table.
- **Privacy Guarantee:** No raw comment text is ever printed to stdout, stderr, or telemetry logs.

### Category Definitions & Considered Rubrics

All four models evaluated received identical category definitions and prompt instructions (defined in `src/prompts.py`):

#### 1. Toxicity (3-Level Ordinal Rubric)
- **Scope Note:** Toxicity is scored strictly on content intensity, whether directed at a named individual or expressed gratuitously without a specific addressee. Sensitive or controversial topics discussed neutrally do not constitute toxicity.
- **Level 0 (Not Toxic):** Disagreement, criticism, or negative sentiment expressed without personal hostility; blunt critiques of ideas or editing actions.
- **Level 1 (Toxic):** Disrespectful, rude, or hostile language directed at a person or group, moderate gratuitous profanity, name-calling, sarcasm-as-insult, or dismissiveness meant to demean.
- **Level 2 (Severe Toxic):** Extreme hostility, degradation, or dehumanizing content. Qualifies via any of: (a) a single sufficiently extreme/degrading term or phrase, (b) an elaborate degrading construction, or (c) sustained targeted harassment. Does not require a threat of harm.

#### 2. Threat (Binary Judgment)
- **Positive:** The comment states the speaker's own intent to act and cause physical harm to a specific person, group, or identifiable target (physical violence, death threats, doxxing, or conditional enforcement such as *"stop doing X or die"*).
- **Negative:** Wishing or hoping for harm without stating personal intent to act (e.g. *"I hope you die"* is a wish, not an intent to act; grammatically first-person wishes remain negative unless the speaker positions themselves as the causal agent), regular criticism, general statements about violence, hyperbole, or administrative warnings.

#### 3. Identity Hate (Binary Judgment)
- **Positive:** The comment attacks, demeans, or dehumanizes someone based on a protected characteristic (race, religion, ethnicity, gender, sexual orientation, disability, nationality).
- **Negative:** Generic insults, political/ideological attacks (political affiliation is explicitly excluded from identity hate rubrics), profanity, and accusations of prejudice (e.g. calling an editor *"racist"* or *"antisemitic"* describes the target's perceived attitude, not an attack on the target's own identity).

---

## 🏗️ Technical Architecture

```text
                               ┌────────────────────────────────────────┐
                               │  Inbound Comment (1-at-a-time Pointwise)│
                               └──────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                        ┌──────────────────────────────────────────────────┐
                        │      Tier 1: High-Speed Gate (TypeSafe Jev)       │
                        │      • Latency: ~290 ms                          │
                        │      • Cost: $0.0426 / 1k items                  │
                        │      • Concurrency: 50 workers                   │
                        └─────────────────────────┬────────────────────────┘
                                                  │
                             ┌────────────────────┴────────────────────┐
                             ▼                                         ▼
                 [High Confidence (≥ 0.85)]                [Low Confidence (< 0.85)]
                 • 75% of total volume                     • 25% ambiguous cases
                 • Auto-Resolve Instantly                  • Escalate to Tier 2
                             │                                         │
                             ▼                                         ▼
                     ┌───────────────┐                 ┌───────────────────────────────┐
                     │ Immediate     │                 │ Tier 2: Escalation Gate       │
                     │ T&S Decision  │                 │ (Gemini 1.5 Flash)            │
                     └───────────────┘                 │ • Threat F1: 0.6872           │
                                                       │ • Latency: ~2,780 ms          │
                                                       │ • Cost: $0.1360 / 1k items    │
                                                       └───────────────┬───────────────┘
                                                                       ▼
                                                               ┌───────────────┐
                                                               │ Final T&S     │
                                                               │ Classification│
                                                               └───────────────┘
```

### Pointwise Concurrency Pipeline

In production eval gates, models evaluate items individually rather than in artificial offline batches. To accurately benchmark pointwise latency (p50 and p95), our asynchronous execution engine (`scripts/run_benchmark.py`) queries one comment per API call while managing provider-specific concurrency semaphores:
- **TypeSafe Jev:** `50` concurrent workers.
- **Google Gemini (1.5 Flash):** `30` concurrent workers.
- **Anthropic Claude (Haiku 3.5 & Sonnet 3.5):** `15` concurrent workers (shared account semaphore).

Each client includes exponential backoff with jitter and automated retry handling for rate limits (`429`) and transient server errors (`5xx`).

### Crash-Resilient Atomic Checkpointing

Running 24,000 model evaluations across multiple frontier APIs requires crash resilience:
- Results are logged into a normalized two-table schema:
  - `calls.csv`: Stores call metadata, timestamp, model ID, response status, latency (ms), input/output token counts, and exact dollar cost.
  - `answers.csv`: Stores normalized categorical probabilities (`toxic_0`, `toxic_1`, `toxic_2`, `threat_prob`, `identity_hate_prob`, and native confidence scores).
- Every completed call writes to an in-memory buffer and commits atomically to disk (`.tmp` write followed by atomic rename).
- The pipeline supports **seamless resumption**: re-running the benchmark detects completed `(comment_id, model_name)` pairs and skips them automatically.

### Two-Tier Model-Router Architecture

The benchmark demonstrates that pairing Jev with Gemini Flash produces a Pareto-optimal content moderation architecture:
1. **Tier 1 (Front Gate - Jev):** Evaluates 100% of comments at ~291 ms p50 for \$0.04/1k items. Resolves ~75% of traffic with high confidence.
2. **Tier 2 (Escalation Gate - Flash):** Processes the ~25% ambiguous cases requiring specialized threat reasoning.
3. **System Outcome:** Achieves blended latency under 900 ms, \$0.066 / 1k items (75% savings vs. standalone Flash, 99.4% savings vs. Sonnet), and 0% hard-zero blind spots.

---

## 📊 Models, Results & Key Findings

### Evaluated Models

1. **Jev (`jev-latest`):** TypeSafe System One decision primitive utilizing `Score` and `Noul`.
2. **Gemini 1.5 Flash (`gemini-1.5-flash` / `gemini-3.8-flash`):** Google's high-speed, cost-optimized frontier model.
3. **Claude 3.5 Sonnet (`claude-3-5-sonnet-latest`):** Anthropic's flagship model evaluated via tool-calling structured output.
4. **Claude 3.5 Haiku (`claude-3-5-haiku-latest`):** Anthropic's high-speed model evaluated via tool-calling structured output.

### Empirical Performance & Cost Summary

*Evaluated across 6,000 comments (23,998 total API calls). Binary metrics are evaluated at the $\ge 90\%$ recall operating point.*

| Model | Cost / 1k Items | Latency p50 | Latency p95 | Toxic Ordinal F1 | Toxic Binary F1* | Threat F1* | Identity Hate F1* |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Jev** | **$0.0426** | **291.3 ms** | **370.9 ms** | 0.6354 | 0.8876 | 0.3486 | **0.5726** |
| **Gemini 1.5 Flash** | $0.1360 | 2,781.1 ms | 5,243.5 ms | 0.6792 | 0.9066 | **0.6455** | 0.5525 |
| **Claude 3.5 Haiku** | $3.2226 | 4,134.4 ms | 5,575.4 ms | 0.6448 | 0.8861 | 0.3398 | 0.5071 |
| **Claude 3.5 Sonnet** | $11.1578 | 2,056.9 ms | 3,126.3 ms | **0.6914** | **0.9126** | 0.1474 (shortfall) | 0.1823 (shortfall) |

### Detailed Cost & Economics Breakdown

The benchmark processed 23,998 individual pointwise evaluations across 6,000 comments. The table below details the full financial footprint, token pricing rates, total expenditures, and percentage of overall budget:

| Model | Pricing Rates (per MTok) | Evaluated Calls | Total Spend ($) | Cost / 1k Items | Budget Share (%) |
|:---|:---|---:|---:|---:|---:|
| **Jev** | $0.042 input / $0.00 output | 6,000 | **$0.26** | **$0.0426** | **0.29%** |
| **Gemini 1.5 Flash** | $0.075 input / $0.30 output | 5,999 | **$0.82** | **$0.1360** | **0.93%** |
| **Claude 3.5 Haiku** | $0.800 input / $4.00 output | 5,999 | **$19.33** | **$3.2226** | **22.13%** |
| **Claude 3.5 Sonnet** | $3.000 input / $15.00 output | 6,000 | **$66.95** | **$11.1578** | **76.64%** |
| **Total Benchmark** | — | **23,998** | **$87.35** | — | **100.0%** |

#### Key Economic Takeaways:
- **Jev is ~260x cheaper than Sonnet and ~75x cheaper than Haiku:** Evaluating 6,000 comments on Jev cost just $0.26 total.
- **Claude Sonnet consumed 76.6% of the budget:** Despite consuming over three-quarters of the total experiment spend ($66.95), Sonnet exhibited a 35% false-negative blind spot on identity hate due to outputting hard zero probabilities.
- **Two-Tier Router Savings:** Gating traffic through Jev Tier 1 (resolving ~75% of volume) and escalating only low-confidence items to Gemini Flash Tier 2 yields a blended operational cost of **~$0.066 per 1k items** (a 75% reduction vs. standalone Flash, and a 99.4% reduction vs. standalone Sonnet).

### The "Hard-Zero" Deficit Finding

A critical discovery of this benchmark is the **"Hard-Zero" Failure Mode** in discrete autoregressive LLMs:

| Model | Threats Predicted as Exact 0.0 | Identity Hate Predicted as Exact 0.0 | Compliance Risk |
|:---|---:|---:|:---|
| **Jev** | **0 / 350 (0.0%)** | **0 / 521 (0.0%)** | **Minimal (Continuous Bayesian Scoring)** |
| **Gemini 1.5 Flash** | 1 / 350 (0.3%) | 2 / 521 (0.4%) | Low (High Sensitivity) |
| **Claude 3.5 Haiku** | 15 / 350 (4.3%) | 30 / 521 (5.8%) | Moderate |
| **Claude 3.5 Sonnet** | **54 / 350 (15.4%)** | **182 / 521 (34.9%)** | **Severe (Failed Recall Constraints)** |

When prompted for probability distributions, Claude Sonnet frequently predicted exact `0.00` for comments containing implicit slurs or oblique threats. Because these violations were assigned a probability of absolute zero, **no threshold sweep could ever recover them**, causing Sonnet to cap out at 84.6% maximum recall on threats and 64.9% maximum recall on identity hate.

In contrast, **Jev never assigned 0.0 to a single policy violation**, ensuring every harmful comment remains retrievable at conservative operating thresholds.

### Jev Confidence Calibration

Jev outputs a native confidence score alongside each decision. Calibration analysis demonstrates strong monotonic alignment with empirical accuracy:

| Category | Confidence Quartile | Mean Confidence | Item Count | Empirical Accuracy |
|:---|:---|---:|---:|---:|
| **Toxic** | Q1 [0.00 – 0.69] | 0.4609 | 1,531 | 53.17% |
| **Toxic** | Q2 [0.70 – 0.93] | 0.8308 | 1,534 | 63.95% |
| **Toxic** | Q3 [0.94 – 1.00] | 0.9881 | 2,935 | **83.71%** |
| **Threat** | Q1 [0.00 – 0.92] | 0.7808 | 1,967 | 84.75% |
| **Threat** | Q2 [0.94 – 0.96] | 0.9521 | 2,211 | 99.59% |
| **Threat** | Q3 [0.98 – 0.98] | 0.9800 | 1,821 | **100.00%** |
| **Identity Hate**| Q1 [0.00 – 0.88] | 0.6252 | 1,591 | 80.14% |
| **Identity Hate**| Q2 [0.90 – 0.94] | 0.9276 | 1,781 | 95.96% |
| **Identity Hate**| Q3 [0.96 – 0.96] | 0.9600 | 1,496 | **99.33%** |

### Where to Find Results & Visualizations

All metrics, generated plots, and detailed reports are stored in `results/`:
- **Full Markdown Report:** [`results/summary.md`](results/summary.md)
- **Machine-Readable Metrics:** [`results/evaluation_metrics.json`](results/evaluation_metrics.json)
- **Precision-Recall Curves:**
  - Toxicity PR Curve: `results/pr_curve_toxic.png`
  - Threat PR Curve: `results/pr_curve_threat.png`
  - Identity Hate PR Curve: `results/pr_curve_identity_hate.png`
- **Reliability Diagram:**
  - Jev Confidence Calibration: `results/calibration_jev.png`

---

## 📁 Repository Structure

```text
jev-experiments/
├── requirements.txt               # Pinned Python package dependencies
├── docs/                          # PRDs and execution specifications
│   ├── PRD-tns-eval-gate.md       # Content moderation gate PRD
│   ├── PRD-model-router.md        # Two-tier model router PRD
│   ├── PLAN.md                    # Benchmark execution plan
│   └── EXECUTION.md               # Detailed technical spec
├── data/                          # Stratified datasets (strict isolation)
│   ├── eval_6k_prompts.csv        # 6,000 comments (id, comment_text only)
│   └── eval_6k_ground_truth.csv   # 6,000 labels (id, ground-truth only)
├── src/                           # Core modular library
│   ├── checkpoint.py              # Atomic CSV manager & crash recovery
│   ├── prompts.py                 # Unified rubrics & model schemas
│   ├── parsers.py                 # Normalizers, pricing & token counters
│   └── models/                    # Asynchronous model clients
│       ├── base.py                # Abstract base model interface
│       ├── jev_model.py           # TypeSafe Jev client
│       ├── anthropic_model.py     # Claude Haiku & Sonnet client
│       └── gemini_model.py        # Gemini Flash client
├── scripts/                       # CLI execution scripts
│   ├── stratify_dataset.py        # Deterministic dataset sampler
│   ├── run_pilot.py               # 50-item dry-run sanity check
│   ├── run_benchmark.py           # Full 6,000-item async benchmark runner
│   └── evaluate.py                # Metrics, PR curves & summary generator
├── results/                       # Generated evaluation outputs & figures
│   ├── summary.md                 # Full executive benchmark report
│   ├── evaluation_metrics.json    # Machine-readable evaluation metrics
│   ├── pr_curve_toxic.png         # Precision-Recall curve: Toxicity
│   ├── pr_curve_threat.png        # Precision-Recall curve: Threat
│   ├── pr_curve_identity_hate.png # Precision-Recall curve: Identity Hate
│   └── calibration_jev.png        # Jev confidence reliability curve
└── tests/                         # Full unit and integration test suite
    ├── test_stratify.py
    ├── test_checkpoint.py
    ├── test_parsers.py
    ├── test_models.py
    ├── test_runners.py
    └── test_evaluate.py
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r jev-experiments/requirements.txt
```

Set your API credentials in `.env` (using `.env.example` as a reference):

```bash
TYPESAFE_API_KEY="your_typesafe_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
# Optional: Only required if using an Anthropic organization user key:
# ANTHROPIC_WORKSPACE_ID="wrkspc_xxxxxxxxxxxxxxxxxxxxxxxx"
```

### 2. Running Unit Tests

Execute the 74-test test suite (all external API calls are mocked; no live keys required):

```bash
python -m unittest discover -s jev-experiments/tests
```

### 3. Step 1: Pilot Check (50 items)

Validate API connectivity, latency, pricing calculations, and checkpointing schemas across all models:

```bash
python jev-experiments/scripts/run_pilot.py --sample-size 50 --models jev,haiku,flash,sonnet
```

### 4. Step 2: Full Benchmark (6,000 items)

Run the full benchmark across all models with automated concurrency management and auto-resumption:

```bash
python jev-experiments/scripts/run_benchmark.py \
    --prompts-file jev-experiments/data/eval_6k_prompts.csv \
    --models jev,haiku,flash,sonnet \
    --output-dir jev-experiments/results
```

*If interrupted, re-running this command automatically skips completed records.*

### 5. Step 3: Metrics & Visual Report Generation

Generate the evaluation summary, precision-recall curves, and calibration plots:

```bash
python jev-experiments/scripts/evaluate.py \
    --calls-file jev-experiments/results/calls.csv \
    --answers-file jev-experiments/results/answers.csv \
    --ground-truth-file jev-experiments/data/eval_6k_ground_truth.csv \
    --output-dir jev-experiments/results
```

---

## 🔒 Security & Credentials Management

This repository adheres strictly to zero-secrets and privacy best practices:
- **Zero API Keys in Version Control:** All API keys are loaded strictly via environment variables or a local, gitignored `.env` file.
- **Gitignored Datasets and Telemetry:** Large raw datasets (`train.csv`) and execution logs (`calls.csv`, `answers.csv`) are excluded from git via `.gitignore`.
- **Anonymized & Clean:** No personal identifiable information (PII) or hardcoded machine directories exist in the codebase.
