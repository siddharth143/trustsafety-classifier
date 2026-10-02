"""Category definitions, worked examples, system prompt templates, and schema configurations.

Matches PRD-tns-eval-gate.md and EXECUTION.md specifications for the Trust & Safety Eval Gate.
"""

# Noise default instruction (PRD line 250)
NOISE_DEFAULT_INSTRUCTION = (
    "If the comment contains no interpretable content relevant to a category "
    "(e.g. random characters, pure noise), default to negative / Level 0 for that "
    "category — don't infer hostility, threat, or identity-targeting from noise alone."
)

# Category definitions (PRD lines 63-126)
TOXIC_SCOPE_NOTE = (
    "toxicity is scored on content intensity, whether or not a specific person is the target. "
    "This covers both directed hostility (attacks, contempt, or degradation aimed at a person or group) "
    "and gratuitous extreme content (self-referential, non-sequitur, or vandalism-style extreme language "
    "with no clear addressee) — the latter is scored on the same intensity axis as the former; "
    "absence of a named target doesn't lower severity."
)

TOXIC_RUBRIC = {
    "0": (
        "Level 0 — Not toxic: Disagreement, criticism, or negative sentiment expressed without hostility "
        "toward a person, and no gratuitous extreme content. Argues a position, critiques content/ideas/actions, "
        "even bluntly, but doesn't attack who someone is. Sensitive or historically-charged topics discussed "
        "without hostility (e.g. a clumsy but non-hostile question referencing religion or a historical atrocity) "
        "are still Level 0 — topic sensitivity alone is not toxicity; the bar is hostility/degradation of language, "
        "not subject matter."
    ),
    "1": (
        "Level 1 — Toxic: Disrespectful, rude, or hostile language directed at a person or group, or moderate "
        "gratuitous profanity, that stops short of the extreme bar below — name-calling, contempt, sarcasm-as-insult, "
        "dismissiveness meant to demean. Would make a reasonable person feel attacked or disrespected, but doesn't "
        "reach extreme degradation, dehumanization, or target protected characteristics."
    ),
    "2": (
        "Level 2 — Severe_toxic: Extreme hostility, degradation, or dehumanizing/gratuitously extreme content — "
        "whether directed at a specific person or not. Qualifies via any of: (a) a single sufficiently extreme/degrading "
        "term or phrase, (b) an elaborate/specific degrading construction (even without repetition or one standout word "
        "— length and specificity of a degrading characterization can clear the bar on its own), or (c) sustained/repeated "
        "targeted harassment. None of the three is a strict requirement over the others — any one is sufficient. "
        "Does not require a threat of harm to qualify. Comments that do threaten harm are captured separately under "
        "the threat category below (a comment can be both severe_toxic and a threat simultaneously — the two categories "
        "are judged independently, not mutually exclusively, matching how Jigsaw's own labels can co-occur)."
    ),
}

TOXIC_WORKED_EXAMPLES = [
    {
        "id": 1,
        "comment": "I'm sure if you worked on articles other than Arcayne you'd do fine...",
        "level": 0,
        "rationale": "Criticism of behavior/editing pattern, no hostility toward the person.",
    },
    {
        "id": 2,
        "comment": "LMAO, what a n00b. Go and listen to manele!",
        "level": 0,
        "rationale": "Dismissive/mocking but below the directed-hostility bar; borderline.",
    },
    {
        "id": 3,
        "comment": "know the sex of the foetus",
        "level": 0,
        "rationale": "Fragment, no hostility or extreme content.",
    },
    {
        "id": 4,
        "comment": "COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK",
        "level": 2,
        "rationale": "Single extreme/degrading term directed at a person — clears the bar via condition (a).",
    },
    {
        "id": 5,
        "comment": "Tony Sidaway is obviously a fistfuckee. He loves an arm up his ass.",
        "level": 2,
        "rationale": "Elaborate, specific degrading sexual characterization of a named individual — clears the bar via condition (b), independent of single-word intensity.",
    },
    {
        "id": 6,
        "comment": "Hi... Im a fucking bitch... [IP]",
        "level": 2,
        "rationale": "Gratuitous extreme content, no clear addressee — scored on intensity alone per the scope note; absence of a target doesn't reduce severity.",
    },
    {
        "id": 7,
        "comment": "closedmouth is a DICK-FUCK... this guy is a jackass who sux cock...",
        "level": 2,
        "rationale": "Sustained degrading language directed at a named person — clears the bar via condition (a)/(c).",
    },
    {
        "id": 8,
        "comment": "are you Jew? if so how are you really thinking there is any resemblance between holocaust and Armenian deportations?",
        "level": 0,
        "rationale": "Clumsy and topically sensitive, but no hostility or degradation — topic sensitivity alone doesn't meet the toxicity bar. Not a candidate for identity_hate either.",
    },
    {
        "id": 9,
        "comment": "Why are you changing the SOMALI PEOPLE WIKI...YOU STUPID CRIMINAL",
        "level": 1,
        "rationale": "Directed name-calling/contempt, doesn't reach extreme degradation.",
    },
]

THREAT_DEFINITION = {
    "positive": (
        "Positive if the comment states the speaker's own intent to act and cause harm to a specific person, "
        "group, or identifiable target — physical violence, death threats, doxxing-as-threat, or credible-sounding "
        "intimidation where the speaker is expressing what they will do. "
        "Threats can be stated directly ('I will...', 'I'm going to...') or as a conditional demand where the "
        "speaker implies they will personally enforce the consequence ('stop doing X or die,' 'do this or else') "
        "— both count as positive. The conditional form doesn't require the literal word 'I will'; the implied "
        "causal agency is enough."
    ),
    "negative": (
        "Negative for everything else, including but not limited to: wishing harm on someone without stating intent "
        "to act ('I hope something bad happens to them,' 'they deserve to suffer,' 'I sincerely hope you die'), "
        "regular criticism or disagreement, insults or rudeness, hyperbolic language, general statements about "
        "violence with no directed intent, political/ideological attacks, and sarcasm. The bar is specifically "
        "stated intent to act — a wish, hope, or desire for someone's harm, without the speaker positioning themselves "
        "as the causal agent of that harm, is negative. Note: this holds even when the wish is grammatically "
        "first-person ('I hope you die') — the test is whether the speaker claims they will cause the harm, not "
        "whether the sentence starts with 'I.' Nearly all wishes are naturally phrased in first person, so grammatical "
        "person alone can't be the deciding signal without collapsing this entire negative category."
    ),
}

THREAT_WORKED_EXAMPLES = [
    {
        "id": 1,
        "comment": "...Artie Lange is a can't-miss candidate for the 2007 Dead pool!... I will repost it!!!",
        "label": "Not a threat",
        "rationale": "Joking reference to a third party; 'I will repost it' isn't harm-related.",
    },
    {
        "id": 2,
        "comment": "Christian arabs category discussion (civil, no violence language)",
        "label": "Not a threat",
        "rationale": "Ordinary disagreement/discussion, no hostility or harm language at all.",
    },
    {
        "id": 3,
        "comment": "Please do not vandalize pages... you will be blocked from editing.",
        "label": "Not a threat",
        "rationale": "Administrative warning about a platform consequence, not personal harm.",
    },
    {
        "id": 4,
        "comment": "Don't be evasive... Don't lock pages when you don't get your own way.",
        "label": "Not a threat",
        "rationale": "Criticism/frustration, no harm language.",
    },
    {
        "id": 5,
        "comment": "Stop undoing my edits or die!",
        "label": "Threat",
        "rationale": "Conditional-demand pattern — implies the speaker will enforce the consequence, even without literal 'I will.'",
    },
    {
        "id": 6,
        "comment": "I sincerely hope you die.",
        "label": "Not a threat",
        "rationale": "Wish for harm, not a stated intent to personally cause it — grammatically first-person, but the speaker isn't the claimed causal agent.",
    },
    {
        "id": 7,
        "comment": "im going to petrol bomb your house and kill all 3 of your inbred parent cunts... i will find a way to...",
        "label": "Threat",
        "rationale": "Explicit first-person stated intent to act, specific target and method.",
    },
]

IDENTITY_HATE_DEFINITION = {
    "positive": (
        "Positive if the comment attacks, demeans, or dehumanizes someone based on a protected/identity "
        "characteristic — race, religion, ethnicity, gender, sexual orientation, disability, nationality. "
        "Political affiliation/ideology is explicitly out of scope — attacks based on being 'conservative,' "
        "'liberal,' or similar are not identity_hate under this rubric, even if hostile. "
        "The operative test is whether the hostility is aimed at the target's own identity, not whether the comment "
        "merely contains identity-related vocabulary or accuses the target of holding prejudiced views."
    ),
    "negative": (
        "Negative for everything else, including but not limited to: regular criticism or disagreement, "
        "personal insults with no identity-based targeting, political/ideological attacks (regardless of hostility), "
        "general rudeness, and profanity. 'You're a racist,' 'you're antisemitic,' 'you're a nazi' are accusations "
        "about the target's attitude, not attacks on the target's own race/religion/ethnicity — these stay negative, "
        "same as a generic character insult, unless the comment also independently attacks the target's own identity."
    ),
}

IDENTITY_HATE_WORKED_EXAMPLES = [
    {
        "id": 1,
        "comment": "...I suspect that editor is a racist.",
        "label": "0",
        "rationale": "Accusation about the target's attitude/behavior, not an attack on the target's own identity.",
    },
    {
        "id": 2,
        "comment": "Czech Republic/Caucasus geography dispute",
        "label": "0",
        "rationale": "Editorial disagreement, no identity attack.",
    },
    {
        "id": 3,
        "comment": "Plagiarism dispute referencing the 'Jewish Action Taskforce' (an org name)",
        "label": "0",
        "rationale": "References an organization's name in a content dispute; no attack on a person's identity.",
    },
    {
        "id": 4,
        "comment": "A pair of jew-hating weiner nazi schmucks.",
        "label": "0",
        "rationale": "Accuses the targets of antisemitism/bigotry (their attitude), not an attack on the targets' own race/religion.",
    },
    {
        "id": 5,
        "comment": "...are you gay? Do you frequent the Hellfire Club? Do you shake your little tusch on the catwalk?",
        "label": "1",
        "rationale": "Mocks by insinuating sexual orientation as an insult — attacks the target's identity.",
    },
    {
        "id": 6,
        "comment": "Are you a faggot or what?",
        "label": "1",
        "rationale": "Direct slur targeting sexual orientation.",
    },
    {
        "id": 7,
        "comment": "ameriKKKans like to get their anusses raped",
        "label": "1",
        "rationale": "Degrading language attacking nationality (Americans), styled on a hate-group reference.",
    },
]

# Shared LLM system prompt template (PRD lines 153-194)
SYSTEM_PROMPT_TEMPLATE = f"""You are evaluating a single user comment against three independent content-moderation categories. Apply each category definition exactly as written below. Do not consider categories not defined here. Respond only in the required structured output format.

{NOISE_DEFAULT_INSTRUCTION}

## Category: toxic (severity, 3 levels)
Level 0 — Not toxic: Disagreement, criticism, or negative sentiment expressed without hostility toward a person, and no gratuitous extreme content. Argues a position, critiques content/ideas/actions, even bluntly, but doesn't attack who someone is. Sensitive or historically-charged topics discussed without hostility (e.g. a clumsy but non-hostile question referencing religion or a historical atrocity) are still Level 0 — topic sensitivity alone is not toxicity; the bar is hostility/degradation of language, not subject matter.
  Examples: "I'm sure if you worked on articles other than Arcayne you'd do fine...", "are you Jew? if so how are you really thinking there is any resemblance between holocaust and Armenian deportations?"
Level 1 — Toxic: Disrespectful, rude, or hostile language directed at a person or group, or moderate gratuitous profanity, that stops short of the extreme bar below — name-calling, contempt, sarcasm-as-insult, dismissiveness meant to demean. Would make a reasonable person feel attacked or disrespected, but doesn't reach extreme degradation, dehumanization, or target protected characteristics.
  Examples: "Why are you changing the SOMALI PEOPLE WIKI... YOU STUPID CRIMINAL"
Level 2 — Severe_toxic: Extreme hostility, degradation, or dehumanizing/gratuitously extreme content — whether directed at a specific person or not. Qualifies via any of: (a) a single sufficiently extreme/degrading term or phrase, (b) an elaborate/specific degrading construction (even without repetition or one standout word — length and specificity of a degrading characterization can clear the bar on its own), or (c) sustained/repeated targeted harassment. None of the three is a strict requirement over the others — any one is sufficient. Does not require a threat of harm to qualify. Comments that do threaten harm are captured separately under the threat category below (a comment can be both severe_toxic and a threat simultaneously — the two categories are judged independently, not mutually exclusively, matching how Jigsaw's own labels can co-occur).
  Examples: "COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK", "Tony Sidaway is obviously a fistfuckee. He loves an arm up his ass.", "Hi... Im a fucking bitch... [gratuitous, no addressee]"

Return a probability distribution across levels 0, 1, and 2 (must sum to 1.0) reflecting your belief in each level assignment for this comment.

## Category: threat (yes/no)
Positive: Positive if the comment states the speaker's own intent to act and cause harm to a specific person, group, or identifiable target — physical violence, death threats, doxxing-as-threat, or credible-sounding intimidation where the speaker is expressing what they will do. Threats can be stated directly ('I will...', 'I'm going to...') or as a conditional demand where the speaker implies they will personally enforce the consequence ('stop doing X or die,' 'do this or else') — both count as positive. The conditional form doesn't require the literal word 'I will'; the implied causal agency is enough.
  Examples: "im going to petrol bomb your house...", "Stop undoing my edits or die!"
Negative: Negative for everything else, including but not limited to: wishing harm on someone without stating intent to act ('I hope something bad happens to them,' 'they deserve to suffer,' 'I sincerely hope you die'), regular criticism or disagreement, insults or rudeness, hyperbolic language, general statements about violence with no directed intent, political/ideological attacks, and sarcasm. The bar is specifically stated intent to act — a wish, hope, or desire for someone's harm, without the speaker positioning themselves as the causal agent of that harm, is negative. Note: this holds even when the wish is grammatically first-person ('I hope you die') — the test is whether the speaker claims they will cause the harm, not whether the sentence starts with 'I.' Nearly all wishes are naturally phrased in first person, so grammatical person alone can't be the deciding signal without collapsing this entire negative category.
  Example: "I sincerely hope you die." (wish, not stated intent to act — negative)

Return a single probability (0 to 1) that this comment is a threat.

## Category: identity_hate (yes/no)
Positive: Positive if the comment attacks, demeans, or dehumanizes someone based on a protected/identity characteristic — race, religion, ethnicity, gender, sexual orientation, disability, nationality. Political affiliation/ideology is explicitly out of scope — attacks based on being 'conservative,' 'liberal,' or similar are not identity_hate under this rubric, even if hostile. The operative test is whether the hostility is aimed at the target's own identity, not whether the comment merely contains identity-related vocabulary or accuses the target of holding prejudiced views.
  Examples: "Are you a faggot or what?", "ameriKKKans like to get their anusses raped"
Negative: Negative for everything else, including but not limited to: regular criticism or disagreement, personal insults with no identity-based targeting, political/ideological attacks (regardless of hostility), general rudeness, and profanity. 'You're a racist,' 'you're antisemitic,' 'you're a nazi' are accusations about the target's attitude, not attacks on the target's own race/religion/ethnicity — these stay negative, same as a generic character insult, unless the comment also independently attacks the target's own identity.
  Example: "A pair of jew-hating weiner nazi schmucks." (accusation of the target's attitude, not an attack on the target's own identity — negative)

Return a single probability (0 to 1) that this comment is identity_hate.

Comment to evaluate:
\"\"\"{{comment_text}}\"\"\""""

# Jev Question Configuration (EXECUTION.md lines 27-69)
try:
    from typesafe_sdk import Noul, Score
except ImportError:
    class Score:
        """Fallback Score representation when typesafe_sdk is not installed."""

        def __init__(self, instructions: str, criteria: list):
            self.instructions = instructions
            self.criteria = criteria

        def __repr__(self) -> str:
            return f"Score(instructions={self.instructions!r}, criteria={self.criteria!r})"

    class Noul:
        """Fallback Noul representation when typesafe_sdk is not installed."""

        def __init__(self, instructions: str, criteria: dict):
            self.instructions = instructions
            self.criteria = criteria

        def __repr__(self) -> str:
            return f"Noul(instructions={self.instructions!r}, criteria={self.criteria!r})"

JEV_QUESTIONS = {
    "toxic": Score(
        instructions="How toxic is this comment?",
        criteria=[
            {
                "what": (
                    "Not toxic: disagreement, criticism, or negative sentiment without hostility "
                    "toward a person, and no gratuitous extreme content. Sensitive topics discussed "
                    "without hostility still count here."
                ),
                "examples": [
                    "I'm sure if you worked on articles other than Arcayne you'd do fine...",
                    "are you Jew? if so how are you really thinking there is any resemblance between holocaust and Armenian deportations?",
                ],
            },
            {
                "what": (
                    "Toxic: disrespectful, rude, or hostile language directed at a person or group, "
                    "or moderate gratuitous profanity — name-calling, contempt, dismissiveness — "
                    "that stops short of the extreme bar below."
                ),
                "examples": [
                    "Why are you changing the SOMALI PEOPLE WIKI... YOU STUPID CRIMINAL"
                ],
            },
            {
                "what": (
                    "Severe_toxic: extreme hostility, degradation, or dehumanizing/gratuitously "
                    "extreme content, directed at a person or not. Qualifies via a single extreme "
                    "term, an elaborate degrading construction, or sustained harassment — any one "
                    "is sufficient. Does not require a threat of harm."
                ),
                "examples": [
                    "COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK",
                    "Tony Sidaway is obviously a fistfuckee. He loves an arm up his ass.",
                    "Hi... Im a fucking bitch... [gratuitous, no addressee]",
                ],
            },
        ],
    ),
    "threat": Noul(
        instructions=(
            "Does the comment state the speaker's own intent to act and cause harm to a specific person, "
            "group, or identifiable target?"
        ),
        criteria={
            "true": (
                "Direct statements ('I will...', 'I'm going to...') or conditional demands implying "
                "the speaker will enforce a harmful consequence ('stop or die'). "
                "E.g. 'im going to petrol bomb your house...', 'Stop undoing my edits or die!'"
            ),
            "false": (
                "Wishing or hoping harm without stating intent to personally cause it (even first-person: "
                "'I sincerely hope you die'), criticism, insults, hyperbole, general violence statements "
                "with no target, political attacks, sarcasm."
            ),
        },
    ),
    "identity_hate": Noul(
        instructions=(
            "Does the comment attack, demean, or dehumanize someone based on a protected/identity "
            "characteristic (race, religion, ethnicity, gender, sexual orientation, disability, nationality)? "
            "Political affiliation/ideology does not count."
        ),
        criteria={
            "true": (
                "Hostility aimed at the target's own identity. E.g. 'Are you a faggot or what?', "
                "'ameriKKKans like to get their anusses raped'."
            ),
            "false": (
                "Accusations that the target holds prejudiced views (e.g. 'I suspect that editor is a racist', "
                "'jew-hating nazi schmucks') are about the target's attitude, not their own identity — "
                "stays false. Also false: general insults, political/ideological attacks, ordinary disagreement."
            ),
        },
    ),
}

# Shared output JSON schemas for LLMs (PRD line 200, EXECUTION.md lines 83-90)
SHARED_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "toxic": {
            "type": "object",
            "properties": {
                "probabilities": {
                    "type": "object",
                    "properties": {
                        "0": {"type": "number", "description": "Probability of Level 0 (Not toxic)"},
                        "1": {"type": "number", "description": "Probability of Level 1 (Toxic)"},
                        "2": {"type": "number", "description": "Probability of Level 2 (Severe_toxic)"},
                    },
                    "required": ["0", "1", "2"],
                }
            },
            "required": ["probabilities"],
        },
        "threat": {
            "type": "object",
            "properties": {
                "probability": {
                    "type": "number",
                    "description": "Probability that the comment is a threat (0.0 to 1.0)",
                }
            },
            "required": ["probability"],
        },
        "identity_hate": {
            "type": "object",
            "properties": {
                "probability": {
                    "type": "number",
                    "description": "Probability that the comment is identity_hate (0.0 to 1.0)",
                }
            },
            "required": ["probability"],
        },
    },
    "required": ["toxic", "threat", "identity_hate"],
}

# Anthropic tool use definition for Haiku and Sonnet
CLAUDE_CLASSIFICATION_TOOL = {
    "name": "submit_classification",
    "description": (
        "Submit content moderation classification probabilities for toxic (3-level distribution), "
        "threat (binary probability), and identity_hate (binary probability)."
    ),
    "input_schema": SHARED_OUTPUT_SCHEMA,
}

# Gemini structured output response schema
GEMINI_RESPONSE_SCHEMA = SHARED_OUTPUT_SCHEMA
