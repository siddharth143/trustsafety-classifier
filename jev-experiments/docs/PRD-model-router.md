# PRD: Domain-Agnostic Model Router — Jev vs. LLM-as-Router

**Status:** Draft
**Owner:** Siddharth
**Type:** Experiment / research spike, not a shipped feature
**Target output:** Open-sourceable router library + write-up of findings

---

## Problem Statement

Every agentic workflow that calls multiple LLMs for different tasks currently either (a) hardcodes which model handles which task, or (b) burns a full LLM call just to *decide* which model should handle the task — using a general-purpose model as a classifier is wasteful and slow. Jev is a purpose-built "System 1" decision model (Choice/Score/Noul primitives, returns typed values + confidence, no free-text parsing) that TypeSafe markets specifically for this — their own "Intent Routing" and "Confidence-Gated Routing" patterns describe exactly this use case. If Jev-as-router genuinely beats an LLM-as-router (e.g. Haiku or Flash prompted to classify) on cost, latency, and accuracy, that's a reusable primitive for every future agentic product — including Aaro's own agent orchestration.

## Hypothesis

**Primary hypothesis:** A Jev-based router (Choice + Score primitives, confidence-gated fallback) matches or exceeds an LLM-based router (Haiku or Flash doing classification via prompt) on routing accuracy against labeled ground truth, while costing and running meaningfully less per routing decision.

**Secondary hypothesis:** Confidence-gating (escalating low-confidence Jev decisions to a bigger model) recovers most of the accuracy gap at a fraction of the cost of routing everything through an LLM classifier.

## Goals

1. Determine whether Jev-as-router beats Haiku/Flash-as-router on **routing accuracy** (does it pick the model that was actually sufficient for the task?) against a labeled ground-truth set.
2. Quantify **cost per routing decision** and **added latency** for both router mechanisms.
3. Measure **quality-matched cost savings**: for tasks routed by each mechanism, does the resulting end-to-end task success rate match a fixed baseline (e.g. "always Sonnet"), and at what $ and ms cost?
4. Test whether **confidence-gated fallback** (Jev routes low-confidence cases to Sonnet directly) closes most of the accuracy gap cheaply.
5. Produce a reusable, domain-agnostic router module/interface that can be dropped into any agentic workflow.

## Non-Goals

- **Not building a production router service.** This is a benchmarking harness first; a pluggable library is the artifact, not a hosted service.
- **Not tuning task-specific prompts for Haiku/Sonnet/Flash.** We use each model's default/simple prompting for the downstream task itself — the experiment is about routing, not about squeezing max quality out of each model.
- **Not covering multi-turn or streaming routing decisions.** Single-shot task routing only for v1.
- **Not including local/open-source models in the target pool.** Adds infra overhead without changing the core Jev-vs-LLM-router question; can be a fast-follow.
- **Not building router logic that changes mid-task (dynamic re-routing).** Router decides once, upfront, per task.

## Experiment Design

### Task taxonomy (domain-agnostic, so routing has real signal)
Define 5–6 task categories spanning a genuine difficulty/cost gradient so routing decisions aren't trivial:
1. Simple extraction/classification (e.g. "extract the date from this text")
2. Short factual Q&A
3. Summarization (short document)
4. Multi-step reasoning / math
5. Creative or open-ended generation
6. Code generation/debugging

For each category, source or write ~50–80 representative tasks (public benchmark subsets where available — e.g. pull from existing eval sets like BIG-bench-lite, HumanEval, GSM8K, CNN/DailyMail — rather than hand-authoring all of them, to save time and add credibility).

### Ground truth ("correct" model per task)
For a subset of tasks, run **all three downstream models (Haiku, Sonnet, Flash)** and label the cheapest model that produced an acceptable answer (acceptable = passes a task-specific correctness check or a rubric-based pass/fail). This "oracle" pass is the expensive part of the experiment — budget accordingly (see Open Questions).

### Router mechanisms under test
- **Router A (LLM-as-router):** Haiku or Flash prompted with the task and asked to output a JSON classification (task type + suggested model) via structured output/function calling.
- **Router B (Jev-as-router):** Jev `Choice` for task type + `Score` for complexity, per the Intent Routing pattern, mapped to a routing rule (`confidence < threshold` → escalate to Sonnet directly).

Both routers see identical task inputs and are scored against the same ground truth.

## Requirements

### Must-Have (P0)
- Task set of ≥300 items across the 5–6 categories, each with an oracle-labeled "sufficient model."
  - *Acceptance:* every task has exactly one ground-truth label and a reproducible pass/fail check.
- Router A and Router B implementations behind a common interface (`route(task) -> model_choice, confidence, latency, cost`).
  - *Acceptance:* both routers can be run over the full task set with one command; outputs logged to a common schema.
- Benchmarking harness that computes: routing accuracy, cost per decision, latency per decision, end-to-end task success rate, and total pipeline cost (routing + downstream execution).
  - *Acceptance:* harness produces a single comparison table/report across both routers.
- Confidence-gated fallback logic for Router B, with threshold as a configurable parameter.
  - *Acceptance:* sweeping the threshold produces a cost/accuracy tradeoff curve.

### Nice-to-Have (P1)
- Sweep the confidence threshold for Router A too (if the LLM router can also emit a confidence/probability), for an apples-to-apples tradeoff curve.
- Add Jev as a *terminal* answer option (not just router) for tasks that are themselves Choice/Score-shaped (e.g. category 1).
- Package the winning router design as a standalone pip-installable module with a clean interface, ready for open-sourcing.

### Future Considerations (P2)
- Expand target model pool to include local/open-source models.
- Dynamic re-routing mid-task based on intermediate confidence signals.
- Multi-label routing (task needs more than one model, e.g. a tool call + a reasoning step).

## Success Metrics

**Primary:**
- Routing accuracy (Router B vs. Router A) against ground truth, target: within 5pp or better, at ≤50% of Router A's per-decision cost.
- Quality-matched pipeline cost: total $ to achieve the same end-to-end task success rate as an "always Sonnet" baseline, for each router.

**Secondary:**
- Latency per routing decision (Jev should win decisively here given it's not a general LLM completion).
- Cost/accuracy tradeoff curve shape as confidence threshold varies (does gating actually pay for itself, or does it just shift cost around?).

## Open Questions

- **[You/eng]** What's Jev's actual pricing model (per-call, per-token, subscription)? Needed to compute cost-per-decision — blocking for the cost comparison.
- **[You]** Sample size / budget: recommend starting with a **~300-item pilot** (50–60 per category), oracle-labeled by running all 3 downstream models once each (~900 downstream calls total, dominated by Sonnet cost — roughly $10–25 depending on task length, using public Anthropic/Gemini pricing as a rough guide since Jev pricing is unknown). Scale to 1,500–2,000 items only if the pilot shows a signal worth firming up statistically.
- **[Eng]** What correctness check do we use per task category? (Exact-match for extraction, unit tests for code, rubric-scored pass/fail for summarization/creative — needs per-category definition before oracle labeling starts.)
- **[You]** Do we want Router A implemented with Haiku or Flash specifically, or both (doubles the LLM-router baseline work but gives a cleaner "cheapest general model vs. purpose-built model" comparison)?

## Timeline Considerations

- No hard deadline; sequence after or alongside the T&S eval-gate experiment since both need Jev API access validated first.
- Suggested phasing: (1) task set + oracle labels, (2) Router A/B implementation, (3) benchmark run + threshold sweep, (4) write-up, (5) package for open-source if results are positive.
