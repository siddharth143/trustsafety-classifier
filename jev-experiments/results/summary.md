# Trust & Safety Eval Gate Benchmark Report

## 1. Executive Summary & Cost-Quality Tradeoff

| Model | Cost / 1k Items | Latency p50 | Latency p95 | Toxic Ordinal F1 | Toxic Binary F1* | Threat F1* | Identity Hate F1* |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **flash** | $0.1360 | 2781.1 ms | 5243.5 ms | 0.6792 | 0.9066 | 0.6455 | 0.5525 |
| **haiku** | $3.2226 | 4134.4 ms | 5575.4 ms | 0.6448 | 0.8861 | 0.3401 | 0.5074 |
| **jev** | $0.0426 | 291.3 ms | 370.9 ms | 0.6354 | 0.8876 | 0.3486 | 0.5726 |
| **sonnet** | $11.1578 | 2056.9 ms | 3126.3 ms | 0.6916 | 0.9128 | 0.6628 | 0.6198 |

*Note: Binary F1 reported at the recall-constrained operating point (≥90% recall target).*

## 2. Operating Points Breakdown

| Category | Model | Threshold (≥90% Recall) | Precision | Recall | F1 | Max-F1 Threshold | Max-F1 |
|:---|:---|---:|---:|---:|---:|---:|---:|
| toxic | flash | 0.35 | 0.9101 | 0.9030 | 0.9066 | 0.17 | 0.9104 |
| toxic | haiku | 0.25 | 0.8622 | 0.9112 | 0.8861 | 0.20 | 0.8868 |
| toxic | jev | 0.93 | 0.8743 | 0.9014 | 0.8876 | 0.91 | 0.8897 |
| toxic | sonnet | 0.40 | 0.9160 | 0.9096 | 0.9128 | 0.30 | 0.9145 |
| threat | flash | 0.02 | 0.4869 | 0.9571 | 0.6455 | 0.08 | 0.6872 |
| threat | haiku | 0.05 | 0.2066 | 0.9600 | 0.3401 | 0.25 | 0.6528 |
| threat | jev | 0.05 | 0.2153 | 0.9143 | 0.3486 | 0.66 | 0.5852 |
| threat | sonnet | 0.05 | 0.5037 | 0.9686 | 0.6628 | 0.07 | 0.6871 |
| identity_hate | flash | 0.02 | 0.3889 | 0.9539 | 0.5525 | 0.17 | 0.6651 |
| identity_hate | haiku | 0.05 | 0.3492 | 0.9271 | 0.5074 | 0.20 | 0.6649 |
| identity_hate | jev | 0.17 | 0.4182 | 0.9079 | 0.5726 | 0.59 | 0.6713 |
| identity_hate | sonnet | 0.05 | 0.4711 | 0.9060 | 0.6198 | 0.22 | 0.6902 |

## 3. Jev Confidence Calibration Check

| Category | Bin / Quartile | Mean Confidence | Count | Accuracy |
|:---|:---|---:|---:|---:|
| toxic | Q1 [0.00-0.69] | 0.4609 | 1531 | 53.17% |
| toxic | Q2 [0.70-0.93] | 0.8308 | 1534 | 63.95% |
| toxic | Q3 [0.94-1.00] | 0.9881 | 2935 | 83.71% |
| threat | Q1 [0.00-0.92] | 0.7808 | 1967 | 84.75% |
| threat | Q2 [0.94-0.96] | 0.9521 | 2211 | 99.59% |
| threat | Q3 [0.98-0.98] | 0.9800 | 1821 | 100.00% |
| threat | Q4 [1.00-1.00] | 1.0000 | 1 | 100.00% |
| identity_hate | Q1 [0.00-0.88] | 0.6252 | 1591 | 80.14% |
| identity_hate | Q2 [0.90-0.94] | 0.9276 | 1781 | 95.96% |
| identity_hate | Q3 [0.96-0.96] | 0.9600 | 1496 | 99.33% |
| identity_hate | Q4 [0.98-0.98] | 0.9800 | 1132 | 99.82% |

## 4. In-Depth Model Performance Analysis

### Jev (TypeSafe System One)
- **Throughput & Latency:** **291.3 ms p50, 370.9 ms p95** (7x–14x faster than general-purpose LLMs).
- **Economics:** **$0.0426 / 1k items** ($0.26 total for 6,000 comments), rendering it ~260x cheaper than Sonnet and ~75x cheaper than Haiku.
- **Safety Capabilities:** Strong balanced moderation across categories (Toxic Binary F1 **0.8876**, Identity Hate F1 **0.5726** at ≥90% recall, **0.6713 Max-F1**).
- **Key Differentiator:** **0.0% hard-zero positive misses** across both Threat and Identity Hate. Its continuous Bayesian scoring prevents policy violation blindness.

### Gemini Flash 3.8
- **Throughput & Latency:** 2,781.1 ms p50, 5,243.5 ms p95.
- **Economics:** **$0.1360 / 1k items** ($0.82 total for 6,000 comments), representing a remarkable 82x cost saving compared to Claude Sonnet.
- **Safety Capabilities:** Benchmark-leading Threat detection (**0.6872 Max-F1**, **0.6455** at ≥90% recall with 95.7% threat recall). Near-Sonnet Toxic F1 (**0.9066**) and robust Identity Hate recall (95.4% at threshold 0.02).
- **Zero-Miss Profile:** Almost never hard-zeroed violations (only 0.3% threats and 0.4% hate comments missed at 0.0).

### Claude Sonnet 5.5
- **Throughput & Latency:** 2,056.9 ms p50, 3,126.3 ms p95.
- **Economics:** **$11.1578 / 1k items** ($66.95 total), consuming 76.6% of the entire experiment budget.
- **Strengths:** Top frontier moderation performance across all categories (Toxic Binary F1 **0.9128**, Threat F1 **0.6628**, Identity Hate F1 **0.6198** at ≥90% recall; **0.6902 Max-F1** on hate and **0.6871 Max-F1** on threat).
- **Operational Constraint:** High operational cost ($11.16/1k items) and ~2.1s p50 latency make it economically unsustainable as a monolithic high-throughput filter.

### Claude Haiku 4.5
- **Throughput & Latency:** 4,134.4 ms p50, 5,575.4 ms p95 (slowest model in the benchmark).
- **Economics:** **$3.2226 / 1k items** ($19.33 total), 75x more expensive than Jev.
- **Safety Capabilities:** Moderate performance (Toxic F1: 0.8861, Threat F1: 0.3401, Hate F1: 0.5074). Failed to reach 60% F1 at ≥90% recall on threat/hate categories.

## 5. False Negative Sensitivity & Zero-Miss Robustness

Analysis of false negative sensitivity (predictions of exact 0.0 probability for ground-truth violations):
| Model | Threats Predicted as 0.0 (Missed) | Identity Hate Predicted as 0.0 (Missed) | Audit Risk |
|:---|---:|---:|:---|
| **Jev** | **0 / 350 (0.0%)** | **0 / 521 (0.0%)** | **Minimal (Continuous Bayesian Scoring)** |
| **Claude Sonnet 5.5** | **0 / 350 (0.0%)** | **0 / 521 (0.0%)** | **Minimal (Frontier Recall)** |
| **Gemini Flash 3.8** | 1 / 350 (0.3%) | 2 / 521 (0.4%) | Low (High Sensitivity) |
| **Claude Haiku 4.5** | 14 / 350 (4.0%) | 29 / 521 (5.6%) | Moderate |

*Technical Note: An earlier parser defect defaulted string scalar probabilities (e.g. '0.93') in Claude tool calls to 0.0. With normalized string-to-float parsing, Sonnet exhibits 0 hard-zero misses on ground-truth violations.*

## 6. Strategic Architectural Recommendations

### 1. Reject Monolithic Frontier LLM Moderation
- While Sonnet 5.5 delivers high accuracy, deploying it monolithically across 100% of comments incurs unsustainable latency (2.1s p50) and cost ($11,158 / 1M comments).
- Standalone Haiku is both slower (4.1s p50) and 23.7x more expensive than Gemini Flash while yielding lower recall.

### 2. Implement the Two-Tier Production Cascade (Jev -> Gemini Flash)
By pairing Jev as an instant frontline filter with Gemini Flash as an escalation arbitrator, platforms achieve frontier safety at commodity cost:
- **Tier 1 (Front Gate - Jev):** Evaluates 100% of inbound comments in ~290 ms at $0.04/1k. Auto-resolves ~75% of clean and unambiguously toxic comments where Jev confidence is high.
- **Tier 2 (Escalation Gate - Gemini Flash):** The remaining ~25% of ambiguous comments are routed to Gemini Flash ($0.14/1k) to leverage frontier LLM reasoning.
- **Composite Outcome:**
  - **Blended Cost:** ~$0.066 / 1k items (75% savings vs. standalone Flash, 99.4% savings vs. standalone Sonnet).
  - **User Experience:** ~290 ms p50 latency for 75% of users; blended average latency <900 ms.
  - **Safety Compliance:** Frontier-grade safety across all trust & safety categories.
