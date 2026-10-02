# Execution Notes — T&S Eval Gate Experiment

*This is a runbook, not a spec — implementation/tooling decisions that don't affect the experiment's design or validity. For the rubric, prompt design, and why things are structured the way they are, see [PRD-tns-eval-gate.md](PRD-tns-eval-gate.md). Expect this file to change as reality intervenes during actual implementation; the PRD shouldn't need to.*

## Execution environment & workflow

**Resolved:** prototype in a notebook (Colab or local Jupyter), then move the actual bulk run to a plain Python script — not a single notebook end-to-end. Reasoning:

- **Prototyping (dry run, rubric/schema sanity-checks):** a notebook's cell-by-cell iteration is the right tool — run one comment, inspect the raw response, adjust, rerun. This is where the `{what, examples}` field-syntax check (flagged as unverified in the PRD) happens.
- **The bulk 6,000 × 4-model run (~24,000 API calls):** moved to a local Python script, not run inside the notebook, because:
  - **Session fragility** — Colab disconnects on idle timeouts or browser hiccups; losing an in-progress run partway through wastes money already spent and requires a resumable design that most notebooks don't have anyway.
  - **Secrets handling** — 4 provider API keys (TypeSafe, Anthropic, Gemini) are easier to keep out of notebook cell output/history via a script + `.env` file (gitignored) than via Colab's secrets manager.
  - **Reproducibility/review** — a script is easier to diff, test, and hand off than a notebook, relevant since the router experiment (separate PRD) is meant to be open-sourced and habits carry over.
- **Analysis (PR curves, calibration charts, cost rollups):** back in a notebook — interactive plotting with immediate visual feedback is the right tool here, and this is genuinely Colab/Jupyter's strength.

**Results storage:** incremental checkpointing to CSV — append results to disk after every N comments (e.g. every 50-100), rather than holding everything in memory until the end. A crash, rate limit, or network blip then only costs the last unsaved batch, not the whole run. One CSV per table (`calls.csv`, `answers.csv`, matching the PRD's two-table schema), keyed so a resumed run can skip comments already processed.

## Jev API call implementation

```python
from typesafe_sdk import Score, Noul, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        model="jev-latest",
        state=comment_text,  # raw comment, plain string
        questions={
            "toxic": Score(
                instructions="How toxic is this comment?",
                criteria=[
                    {
                        "what": "Not toxic: disagreement, criticism, or negative sentiment without hostility toward a person, and no gratuitous extreme content. Sensitive topics discussed without hostility still count here.",
                        "examples": [
                            "I'm sure if you worked on articles other than Arcayne you'd do fine...",
                            "are you Jew? if so how are you really thinking there is any resemblance between holocaust and Armenian deportations?"
                        ]
                    },
                    {
                        "what": "Toxic: disrespectful, rude, or hostile language directed at a person or group, or moderate gratuitous profanity — name-calling, contempt, dismissiveness — that stops short of the extreme bar below.",
                        "examples": [
                            "Why are you changing the SOMALI PEOPLE WIKI... YOU STUPID CRIMINAL"
                        ]
                    },
                    {
                        "what": "Severe_toxic: extreme hostility, degradation, or dehumanizing/gratuitously extreme content, directed at a person or not. Qualifies via a single extreme term, an elaborate degrading construction, or sustained harassment — any one is sufficient. Does not require a threat of harm.",
                        "examples": [
                            "COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK",
                            "Tony Sidaway is obviously a fistfuckee. He loves an arm up his ass.",
                            "Hi... Im a fucking bitch... [gratuitous, no addressee]"
                        ]
                    }
                ]
            ),
            "threat": Noul(
                instructions="Does the comment state the speaker's own intent to act and cause harm to a specific person, group, or identifiable target?",
                criteria={
                    "true": "Direct statements ('I will...', 'I'm going to...') or conditional demands implying the speaker will enforce a harmful consequence ('stop or die'). E.g. 'im going to petrol bomb your house...', 'Stop undoing my edits or die!'",
                    "false": "Wishing or hoping harm without stating intent to personally cause it (even first-person: 'I sincerely hope you die'), criticism, insults, hyperbole, general violence statements with no target, political attacks, sarcasm."
                }
            ),
            "identity_hate": Noul(
                instructions="Does the comment attack, demean, or dehumanize someone based on a protected/identity characteristic (race, religion, ethnicity, gender, sexual orientation, disability, nationality)? Political affiliation/ideology does not count.",
                criteria={
                    "true": "Hostility aimed at the target's own identity. E.g. 'Are you a faggot or what?', 'ameriKKKans like to get their anusses raped'.",
                    "false": "Accusations that the target holds prejudiced views (e.g. 'I suspect that editor is a racist', 'jew-hating nazi schmucks') are about the target's attitude, not their own identity — stays false. Also false: general insults, political/ideological attacks, ordinary disagreement."
                }
            )
        }
    )
```

Open items still pending before this is run-ready (see PRD for the underlying design reasoning):
- Whether `criteria` examples should include *all* worked examples per level/category or a trimmed subset (token cost is trivial at Jev's pricing, so leaning toward including all of them — but worth confirming this doesn't make the `instructions`/`criteria` unwieldy).
- Confirming `Score`'s structured `{what, examples}` form (used above) is in fact the correct field-level syntax — the "Advanced Structure" doc page that would confirm this returned a 404; worth a sanity-check dry run against the real API before the full run.

## LLM structured-output implementation (Haiku / Sonnet / Flash)

The shared prompt template (the actual rubric text embedded per model) lives in the PRD, since that's fairness-critical design, not implementation. What differs per model is purely the *mechanism* used to force structured output:

- **Claude (Haiku/Sonnet):** forced tool-use with a single tool (e.g. `submit_classification`) whose `input_schema` matches the shared output schema (see PRD → Shared output schema).
- **Gemini Flash:** native JSON-schema-constrained output mode (`response_mime_type: application/json` + `response_schema`) with the same shape.

Shared output JSON shape (identical across all 3 LLMs, mapped onto Jev's native output for scoring):
```json
{
  "toxic": {"probabilities": {"0": 0.7, "1": 0.25, "2": 0.05}},
  "threat": {"probability": 0.02},
  "identity_hate": {"probability": 0.01}
}
```

## Parsing raw responses into the shared schema, then writing to CSV

Each of the 4 models returns a differently-shaped raw response. This is the step that normalizes all of them into the PRD's `calls`/`answers` schema before anything hits disk. One parser per provider, all producing the same row shapes.

**Jev parser** — `Score` and `Noul` already return most of what's needed natively:

```python
def parse_jev_response(comment_id, response, latency_ms, usage, cost_usd):
    calls_row = {
        "comment_id": comment_id, "model": "jev",
        "latency_ms": latency_ms,
        "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "cost_usd": cost_usd, "raw_response_json": response.model_dump_json(),
        "timestamp": now_iso(),
    }

    toxic = response.answers["toxic"]  # type "score"
    answers_rows = [{
        "comment_id": comment_id, "model": "jev", "category": "toxic",
        "toxic_prob_0": toxic.probabilities["0"],
        "toxic_prob_1": toxic.probabilities["1"],
        "toxic_prob_2": toxic.probabilities["2"],
        "toxic_score": toxic.score,
        "binary_probability": None,
        "native_confidence": toxic.confidence,  # Score returns this natively
    }]

    for category in ["threat", "identity_hate"]:
        noul = response.answers[category]  # type "noul"
        answers_rows.append({
            "comment_id": comment_id, "model": "jev", "category": category,
            "toxic_prob_0": None, "toxic_prob_1": None, "toxic_prob_2": None,
            "toxic_score": None,
            "binary_probability": noul.noul,
            # Noul has no native confidence field — derive it (Jev-only; see PRD)
            "native_confidence": 2 * abs(noul.noul - 0.5),
        })

    return calls_row, answers_rows
```

**Claude and Gemini parser** — same shared JSON shape (per the structured-output schema above), so one parser covers both; only how the raw JSON is extracted from each SDK's response object differs upstream of this function:

```python
def parse_llm_response(comment_id, model_name, parsed_json, latency_ms, usage, cost_usd, raw_response):
    calls_row = {
        "comment_id": comment_id, "model": model_name,
        "latency_ms": latency_ms,
        "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "cost_usd": cost_usd, "raw_response_json": raw_response,
        "timestamp": now_iso(),
    }

    probs = parsed_json["toxic"]["probabilities"]
    toxic_score = sum(int(level) * p for level, p in probs.items())  # weighted mean, mirrors Jev's `score`
    answers_rows = [{
        "comment_id": comment_id, "model": model_name, "category": "toxic",
        "toxic_prob_0": probs["0"], "toxic_prob_1": probs["1"], "toxic_prob_2": probs["2"],
        "toxic_score": toxic_score,
        "binary_probability": None,
        "native_confidence": None,  # LLMs are out of scope for calibration — see PRD
    }]

    for category in ["threat", "identity_hate"]:
        answers_rows.append({
            "comment_id": comment_id, "model": model_name, "category": category,
            "toxic_prob_0": None, "toxic_prob_1": None, "toxic_prob_2": None,
            "toxic_score": None,
            "binary_probability": parsed_json[category]["probability"],
            "native_confidence": None,
        })

    return calls_row, answers_rows
```

**Writing to CSV (incremental, resumable):**

```python
import csv, os

def append_rows(path, rows, fieldnames):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

def process_comment(comment_id, comment_text):
    calls_row, answers_rows = parse_jev_response(...)   # or parse_llm_response(...)
    append_rows("calls.csv", [calls_row], CALLS_FIELDNAMES)
    append_rows("answers.csv", answers_rows, ANSWERS_FIELDNAMES)
```

Calling `append_rows` after *every* comment (not batched in memory) is what makes the run resumable — on restart, read `calls.csv`, collect the `comment_id`s already present per model, and skip them rather than re-calling (and re-paying for) completed work. `ground_truth_label` isn't written at collection time — it's joined from the Jigsaw source file during analysis, keeping collection and scoring cleanly separated.
