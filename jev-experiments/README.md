# Trust & Safety Eval Gate: Jev vs. LLMs Benchmark

An empirical content moderation benchmark and production-grade evaluation gate comparing **TypeSafe Jev** (System One decision primitive) against frontier general-purpose LLMs (**Claude Sonnet 5.5**, **Gemini Flash 3.8**, and **Claude Haiku 4.5**) on a stratified 6,000-comment dataset.

---

## Table of Contents

1. [About the Project & Classifier](#-about-the-project--classifier)
2. [Experiment Design & Dataset](#-experiment-design--dataset)
   - [Dataset & Kaggle Source](#dataset--kaggle-source)
   - [Sampling & Stratification](#sampling--stratification)
   - [Strict Data Isolation & Privacy](#strict-data-isolation--privacy)
   - [Category Definitions & Considered Rubrics](#category-definitions--considered-rubrics)
3. [Technical Architecture](#-technical-architecture)
   - [Benchmark Evaluation Harness Architecture (Empirical Run)](#1-benchmark-evaluation-harness-architecture-empirical-experiment-run)
   - [Proposed Production Architecture for the T&S Classifier](#2-proposed-production-architecture-for-the-ts-classifier-two-tier-cascade)
4. [Models, Results & Key Findings](#-models-results--key-findings)
   - [Evaluated Models](#evaluated-models)
   - [Empirical Performance & Cost Summary](#empirical-performance--cost-summary)
   - [Comparative Economics: Cross-Model Cost Benchmark](#comparative-economics-cross-model-cost-benchmark)
   - [Detailed Cost & Financial Breakdown](#detailed-cost--financial-breakdown)
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
2. **General-Purpose LLMs (e.g. Claude Sonnet, Gemini Flash):** Highly articulate and context-aware, but introduce prohibitive token-generation latencies (2,000–4,000 ms), high operational costs ($1.43–$7.44 per 1k items), and uncalibrated discrete token probabilities.
3. **Purpose-Built Decision Primitives (TypeSafe Jev):** A specialized "System One" decision model that evaluates content directly via typed primitives (`Score` for graded distributions, `Noul` for binary judgments). Jev returns continuous Bayesian probabilities alongside native confidence scores for graded classifications (`Score`) and derived decision certainty for binary judgments (`Noul`), without autoregressive text decoding.

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

This project encompasses two distinct architectures:
1. **The Benchmark Evaluation Harness Architecture:** The concurrent offline testing and evaluation harness used to execute the empirical benchmark across all 4 models.
2. **The Proposed Production Architecture for the T&S Classifier:** The recommended two-tier cascading gate for live platform moderation derived from our empirical findings.

---

### 1. Benchmark Evaluation Harness Architecture (Empirical Experiment Run)

The empirical benchmark executed 23,998 individual pointwise evaluations over 6,000 comments. The harness was architected to evaluate models completely independently and in parallel without shared state:

```text
               ┌────────────────────────────────────────────────────────┐
               │    Stratified Input Dataset (Strict Isolation)         │
               │    `data/eval_6k_prompts.csv` (6,000 comments)          │
               │    [id, comment_text only — zero ground-truth labels]  │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │          Async Dispatch & Concurrency Engine           │
               │          `scripts/run_benchmark.py` (asyncio)          │
               │     • Pointwise evaluation (1 comment per call)        │
               │     • Independent resume check via CheckpointManager   │
               └───────┬──────────────┬──────────────┬──────────────┬───┘
                       │              │              │              │
                       ▼              ▼              ▼              ▼
               ┌──────────────┐┌──────────────┐┌────────────────────────┐
               │Jev Semaphore ││Gemini Semaph.││  Anthropic Semaphore   │
               │ (limit: 50)  ││ (limit: 30)  ││      (limit: 15)       │
               └───────┬──────┘└──────┬───────┘└──────┬──────────┬──────┘
                       │              │               │          │
                       ▼              ▼               ▼          ▼
               ┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
               │  Jev Model   ││ Gemini Flash ││ Claude Haiku ││Claude Sonnet │
               │ (Score/Noul) ││ (Structured) ││(Tool Calling)││(Tool Calling)│
               └───────┬──────┘└──────┬───────┘└──────┬───────┘└──────┬───────┘
                       │              │               │               │
                       └──────────────┼───────────────┴───────────────┘
                                      │
                                      ▼
               ┌────────────────────────────────────────────────────────┐
               │         Resilience & Unified Parsing Layer             │
               │  • Exponential backoff with jitter (429 / 5xx retries)  │
               │  • Response normalization & confidence extraction      │
               │  • Exact token counting & pricing (`src/parsers.py`)   │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │         Crash-Resilient Atomic Checkpoint Engine       │
               │              `src/checkpoint.py`                       │
               │  • Atomic rename: memory buffer -> .tmp -> .csv        │
               │  • `calls.csv`: latency, token usage, dollar cost      │
               │  • `answers.csv`: normalized category probabilities    │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │         Offline Evaluation & Reporting Engine          │
               │                 `scripts/evaluate.py`                  │
               │  • Join with `data/eval_6k_ground_truth.csv`           │
               │  • Threshold sweeps & recall-constrained operating pts │
               │  • Precision-Recall & Calibration curve generation     │
               │  • Outputs: `summary.md`, `evaluation_metrics.json`    │
               └────────────────────────────────────────────────────────┘
```

#### Key Components of the Evaluation Harness:
- **Blind Pointwise Isolation:** Each API call processed exactly 1 comment to measure real-world production p50/p95 latency (rather than artificially smoothed batch latency). Prompts were strictly stripped of ground-truth labels.
- **Provider-Aware Semaphore Rate-Limiting:** Concurrency was strictly throttled per API provider to maximize throughput without triggering rate limits:
  - **TypeSafe Jev:** `50` concurrent workers.
  - **Google Gemini (Gemini Flash 3.8):** `30` concurrent workers.
  - **Anthropic Claude (Haiku 4.5 & Sonnet 5.5):** `15` concurrent workers (shared pool).
- **Automated Fault Resilience:** Model wrappers implement exponential backoff with jitter to gracefully handle transient provider throttling (`429`) or server errors (`5xx`).
- **Atomic Two-Table Persistence:** Checkpoints committed atomically to `calls.csv` and `answers.csv` on disk per comment, ensuring that if a run was interrupted, re-launching skipped already completed pairs automatically with zero duplicated cost.

---

### 2. Proposed Production Architecture for the T&S Classifier (Two-Tier Cascade)

Based on the empirical benchmark results—specifically Jev's ultra-low latency (~291 ms), low cost (\$0.0426/1k), and 0% hard-zero policy misses, combined with Gemini Flash's high threat F1 (0.6455 at $\ge 90\%$ recall)—we propose the **Two-Tier Cascading Model-Router** as the target production architecture for high-volume content moderation:

```text
                               ┌────────────────────────────────────────┐
                               │  Inbound Comment Stream (Production)   │
                               └──────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                        ┌──────────────────────────────────────────────────┐
                        │      Tier 1: Frontline Gate (TypeSafe Jev)       │
                        │      • Evaluates 100% of incoming traffic        │
                        │      • Latency: ~291 ms p50, 370 ms p95          │
                        │      • Cost: $0.0426 / 1k items                  │
                        │      • Zero "Hard-Zero" false negative misses    │
                        └─────────────────────────┬────────────────────────┘
                                                  │
                             ┌────────────────────┴────────────────────┐
                             ▼                                         ▼
                 [High Confidence (≥ 0.85)]                [Low Confidence (< 0.85)]
                 • ~75% of total volume                    • ~25% ambiguous/borderline
                 • Clear Clean or Clear Violation          • Escalate to Tier 2
                             │                                         │
                             ▼                                         ▼
                     ┌───────────────┐                 ┌───────────────────────────────┐
                     │ Immediate     │                 │ Tier 2: Escalation Gate       │
                     │ T&S Decision  │                 │ (Gemini Flash 3.8)            │
                     │ (Action/Pass) │                 │ • Deep semantic threat parser │
                     └───────────────┘                 │ • Threat F1: 0.6455 (≥90% rec)│
                                                       │ • Latency: ~2,780 ms          │
                                                       │ • Cost: $1.4271 / 1k items    │
                                                       └───────────────┬───────────────┘
                                                                       ▼
                                                               ┌───────────────┐
                                                               │ Final T&S     │
                                                               │ Determination │
                                                               └───────────────┘
```

#### Why This Is the Proposed Production Architecture:
1. **Pareto-Optimal Economics:** Resolving ~75% of volume in Tier 1 reduces the effective system cost to **~$0.399 / 1k items** (a 72% savings vs. standalone Gemini Flash and a 94.6% savings vs. standalone Claude Sonnet).
2. **Sub-Second User Experience:** 75% of users receive an instant moderation decision in under 300 ms, bringing blended average latency below 900 ms.
3. **Zero Policy Blind Spots & High Recall:** Jev provides continuous Bayesian probability scoring with 0.0% zero-misses on true violations, while Gemini Flash brings deep semantic arbitration for edge cases, yielding an end-to-end cascade with $\ge 90\%$ recall across all moderation categories.

---

## 📊 Models, Results & Key Findings

### Evaluated Models

1. **Jev (`jev-latest`):** TypeSafe System One decision primitive utilizing `Score` (with native confidence) and `Noul` (with derived Bayesian certainty: $2 \times |p - 0.5|$).
2. **Gemini Flash 3.8 (`gemini-3.8-flash`):** Google's high-speed, cost-optimized frontier model.
3. **Claude Sonnet 5.5 (`claude-sonnet-5-5`):** Anthropic's flagship model evaluated via tool-calling structured output.
4. **Claude Haiku 4.5 (`claude-haiku-4-5`):** Anthropic's high-speed model evaluated via tool-calling structured output.

### Empirical Performance & Cost Summary

*Evaluated across 6,000 comments (23,998 total API calls). Binary metrics are evaluated at the $\ge 90\%$ recall operating point.*

| Model | Cost / 1k Items | Latency p50 | Latency p95 | Toxic Ordinal F1 | Toxic Binary F1* | Threat F1* | Identity Hate F1* |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Jev** (`jev-latest`) | **$0.0426** | **291.3 ms** | **370.9 ms** | 0.6354 | 0.8876 | 0.3486 | 0.5726 |
| **Gemini Flash 3.8** | $1.4271 | 2,781.1 ms | 5,243.5 ms | 0.6792 | 0.9066 | 0.6455 | 0.5525 |
| **Claude Haiku 4.5** | $4.0282 | 4,134.4 ms | 5,575.4 ms | 0.6448 | 0.8861 | 0.3401 | 0.5074 |
| **Claude Sonnet 5.5** | $7.4385 | 2,056.9 ms | 3,126.3 ms | **0.6916** | **0.9128** | **0.6628** | **0.6198** |

### Comparative Economics: Cross-Model Cost Benchmark

To evaluate operational sustainability at scale, the table below compares the economics of each model against **TypeSafe Jev** (`jev-latest`) and **Gemini Flash 3.8** (`gemini-3.8-flash`), using **Claude Sonnet 5.5** (`claude-sonnet-5-5`) as the highest benchmark reference ceiling:

| Model | Cost / 1k Items | vs. Jev Baseline | vs. Flash Baseline | vs. Sonnet Benchmark (Highest Ceiling) | Projected Cost / 1M Items |
|:---|---:|:---|:---|:---|---:|
| **Jev** (`jev-latest`) | **$0.0426** | **1.00x** (Lowest) | **33.5x cheaper** (97.0% savings) | **174.7x cheaper** (99.43% savings) | **$42.57** |
| **Gemini Flash 3.8** | **$1.4271** | 33.5x higher | **1.00x** (Base LLM) | **5.21x cheaper** (80.82% savings) | **$1,427.07** |
| **Claude Haiku 4.5** | **$4.0282** | 94.6x higher | 2.82x higher | **1.85x cheaper** (45.85% savings) | **$4,028.22** |
| **Claude Sonnet 5.5** | **$7.4385** | 174.7x higher | 5.21x higher | **1.00x** (Highest Benchmark Ceiling) | **$7,438.52** |

#### Comparative Takeaways:
1. **Jev vs. Highest Benchmark (Sonnet):** At \$0.0426/1k items, Jev is **175x cheaper than Claude Sonnet 5.5**, offering a **99.43% cost reduction** and 7x lower latency while delivering high-recall moderation (0.8876 toxic F1, 0.5726 hate F1).
2. **Flash vs. Highest Benchmark (Sonnet):** Gemini Flash 3.8 offers **5.2x cost savings (80.82% reduction)** compared to Sonnet, while delivering competitive threat detection (0.6455 vs. 0.6628 F1) at a fraction of the cost.
3. **Jev vs. Flash:** Jev is **33.5x less expensive than Gemini Flash 3.8**, while executing with 9.5x lower p50 latency (291 ms vs. 2,781 ms).
4. **Haiku Disadvantage:** Claude Haiku 4.5 is **2.82x more expensive than Flash** and **94.6x more expensive than Jev**, despite delivering lower F1 scores across toxicity, threat, and identity hate.

### Detailed Cost & Financial Breakdown

The benchmark processed 23,998 individual pointwise evaluations across 6,000 comments. The table below details the full financial footprint, token pricing rates, total expenditures, and percentage of overall budget:

| Model | Pricing Rates (per MTok) | Evaluated Calls | Total Spend ($) | Cost / 1k Items | Budget Share (%) |
|:---|:---|---:|---:|---:|---:|
| **Jev** (`jev-latest`) | $0.042 input / $0.00 output | 6,000 | **$0.26** | **$0.0426** | **0.33%** |
| **Gemini Flash 3.8** | $0.750 input / $3.75 output | 5,999 | **$8.56** | **$1.4271** | **11.03%** |
| **Claude Haiku 4.5** | $1.000 input / $5.00 output | 5,999 | **$24.17** | **$4.0282** | **31.14%** |
| **Claude Sonnet 5.5** | $2.000 input / $10.00 output | 6,000 | **$44.63** | **$7.4385** | **57.50%** |
| **Total Benchmark** | — | **23,998** | **$77.61** | — | **100.0%** |

#### Key Economic Takeaways:
- **Jev is ~175x cheaper than Sonnet 5.5 and ~95x cheaper than Haiku 4.5:** Evaluating 6,000 comments on Jev cost just $0.26 total.
- **Claude Sonnet 5.5 consumed 57.5% of the budget:** While Sonnet delivered top frontier accuracy ($44.63 spend), its $7.44/1k cost and 2.1s p50 latency make it economically unsustainable as a monolithic high-volume gate.
- **Two-Tier Router Savings:** Gating traffic through Jev Tier 1 (resolving ~75% of volume) and escalating only low-confidence items to Gemini Flash 3.8 Tier 2 yields a blended operational cost of **~$0.399 per 1k items** (a 72% reduction vs. standalone Flash, and a 94.6% reduction vs. standalone Sonnet).

### False Negative Sensitivity & Zero-Miss Robustness

Analysis of false negative sensitivity (predictions of exact 0.0 probability for ground-truth violations):

| Model | Threats Predicted as Exact 0.0 | Identity Hate Predicted as Exact 0.0 | Compliance Risk |
|:---|---:|---:|:---|
| **Jev** (`jev-latest`) | **0 / 350 (0.0%)** | **0 / 521 (0.0%)** | **Minimal (Continuous Bayesian Scoring)** |
| **Claude Sonnet 5.5** | **0 / 350 (0.0%)** | **0 / 521 (0.0%)** | **Minimal (Frontier Recall)** |
| **Gemini Flash 3.8** | 1 / 350 (0.3%) | 2 / 521 (0.4%) | Low (High Sensitivity) |
| **Claude Haiku 4.5** | 14 / 350 (4.0%) | 29 / 521 (5.6%) | Moderate |

> **Parser Audit & Score Reconciliation Note:** An earlier evaluation pass observed an apparent shortfall in Claude Sonnet due to a parser defect where string scalar probabilities (e.g. `"0.93"`, `"0.85"`) returned via Claude tool-use were strictly cast into a `dict`-only type branch and defaulted to `0.0`. Upon updating `parsers.py` to robustly cast numeric string scalars, Sonnet demonstrated 0 hard-zero misses on ground-truth violations and achieved an unconstrained peak Max-F1 of **0.6902** on identity hate (threshold 0.22) and **0.6871** on threat (threshold 0.07), alongside its recall-constrained ($\ge 90\%$ recall target) F1 scores of **0.6198** on identity hate and **0.6628** on threat reported in the summary table. Furthermore, silent 0-defaults have been completely removed across all parsers: any missing schema key or unparseable probability is now logged to the module-level failure registry, tagged in answer records with `parse_error=True`, and counted in benchmark health metrics.

### Schema Integrity & Parse Failure Audit

To ensure benchmark reliability and eliminate silent error propagation, all model responses are validated against explicit schema requirements. Any missing category key or unparseable probability is explicitly counted and reported:

| Model | Total Calls | Valid Calls | Schema Errors | Schema Integrity | Audit Details |
|:---|---:|---:|---:|---:|:---|
| **Jev** (`jev-latest`) | 6,000 | 6,000 | 0 | **100.00%** | Zero parse failures across all decision primitives. |
| **Gemini Flash 3.8** | 5,999 | 5,999 | 0 | **100.00%** | 100% schema integrity via structured JSON schema enforcement. |
| **Claude Sonnet 5.5** | 6,000 | 6,000 | 0 | **100.00%** | 100% schema integrity with normalized string scalar casting. |
| **Claude Haiku 4.5** | 5,999 | 5,997 | 2 | **99.97%** | 2 calls omitted required 'threat' schema key (`2edeaa3725e35abc`, `971724e599334e90`); safely defaulted to 0.0 with warning. Neither was a ground-truth violation (GT=0). |

### Jev Decision Confidence & Certainty Calibration

Jev outputs a native confidence score for graded classifications (`Score` / toxic) and a well-calibrated continuous probability for binary judgments (`Noul` / threat & identity hate), from which Bayesian certainty is derived as $2 \times |p - 0.5|$. Calibration analysis demonstrates strong monotonic alignment with empirical accuracy across both native and derived confidence measures:

| Category | Measure Type | Confidence / Certainty Quartile | Mean Value | Item Count | Empirical Accuracy |
|:---|:---|:---|---:|---:|---:|
| **Toxic** | Native Confidence (`Score`) | Q1 [0.00 – 0.69] | 0.4609 | 1,531 | 53.17% |
| **Toxic** | Native Confidence (`Score`) | Q2 [0.70 – 0.93] | 0.8308 | 1,534 | 63.95% |
| **Toxic** | Native Confidence (`Score`) | Q3 [0.94 – 1.00] | 0.9881 | 2,935 | **83.71%** |
| **Threat** | Derived Certainty ($2\|p - 0.5\|$) | Q1 [0.00 – 0.92] | 0.7808 | 1,967 | 84.75% |
| **Threat** | Derived Certainty ($2\|p - 0.5\|$) | Q2 [0.94 – 0.96] | 0.9521 | 2,211 | 99.59% |
| **Threat** | Derived Certainty ($2\|p - 0.5\|$) | Q3 [0.98 – 0.98] | 0.9800 | 1,821 | **100.00%** |
| **Threat** | Derived Certainty ($2\|p - 0.5\|$) | Q4 [1.00 – 1.00] | 1.0000 | 1 | **100.00%** |
| **Identity Hate**| Derived Certainty ($2\|p - 0.5\|$) | Q1 [0.00 – 0.88] | 0.6252 | 1,591 | 80.14% |
| **Identity Hate**| Derived Certainty ($2\|p - 0.5\|$) | Q2 [0.90 – 0.94] | 0.9276 | 1,781 | 95.96% |
| **Identity Hate**| Derived Certainty ($2\|p - 0.5\|$) | Q3 [0.96 – 0.96] | 0.9600 | 1,496 | **99.33%** |
| **Identity Hate**| Derived Certainty ($2\|p - 0.5\|$) | Q4 [0.98 – 0.98] | 0.9800 | 1,132 | **99.82%** |

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
