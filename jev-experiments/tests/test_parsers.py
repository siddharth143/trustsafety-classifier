"""Unit tests for prompts, parsers, and cost calculations in T&S eval gate."""

import json
from pathlib import Path
import sys
import unittest
from datetime import datetime

_ROOT = Path(__file__).resolve().parent.parent
for p in [str(_ROOT / "src"), str(_ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from parsers import calculate_cost, parse_jev_response, parse_llm_response
from prompts import (
    CLAUDE_CLASSIFICATION_TOOL,
    GEMINI_RESPONSE_SCHEMA,
    IDENTITY_HATE_DEFINITION,
    JEV_QUESTIONS,
    NOISE_DEFAULT_INSTRUCTION,
    SYSTEM_PROMPT_TEMPLATE,
    THREAT_DEFINITION,
    TOXIC_RUBRIC,
)


class MockScoreAnswer:
    def __init__(self, probabilities, score, confidence):
        self.probabilities = probabilities
        self.score = score
        self.confidence = confidence


class MockNoulAnswer:
    def __init__(self, noul):
        self.noul = noul


class MockJevResponse:
    def __init__(self, toxic_answer, threat_answer, identity_hate_answer):
        self.answers = {
            "toxic": toxic_answer,
            "threat": threat_answer,
            "identity_hate": identity_hate_answer,
        }

    def model_dump_json(self):
        return json.dumps(
            {
                "answers": {
                    "toxic": {
                        "probabilities": self.answers["toxic"].probabilities,
                        "score": self.answers["toxic"].score,
                        "confidence": self.answers["toxic"].confidence,
                    },
                    "threat": {"noul": self.answers["threat"].noul},
                    "identity_hate": {"noul": self.answers["identity_hate"].noul},
                }
            }
        )


class MockUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class TestCalculateCost(unittest.TestCase):
    def test_calculate_cost_jev(self):
        # Jev: $0.042/MTok input, $0 output
        cost = calculate_cost("jev", 1_000_000, 500)
        self.assertAlmostEqual(cost, 0.042, places=6)

        cost_partial = calculate_cost("jev-latest", 500_000, 0)
        self.assertAlmostEqual(cost_partial, 0.021, places=6)

    def test_calculate_cost_sonnet(self):
        # Sonnet: $3.00/MTok input, $15.00/MTok output
        cost_in = calculate_cost("sonnet", 1_000_000, 0)
        self.assertAlmostEqual(cost_in, 3.00, places=6)

        cost_out = calculate_cost("claude-3-5-sonnet", 0, 1_000_000)
        self.assertAlmostEqual(cost_out, 15.00, places=6)

        cost_sonnet_55 = calculate_cost("claude-sonnet-5-5", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost_sonnet_55, 18.00, places=6)

        cost_both = calculate_cost("claude-3-5-sonnet-20241022", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost_both, 18.00, places=6)

    def test_calculate_cost_haiku(self):
        # Haiku: $0.80/MTok input, $4.00/MTok output
        cost_in = calculate_cost("haiku", 1_000_000, 0)
        self.assertAlmostEqual(cost_in, 0.80, places=6)

        cost_out = calculate_cost("claude-3-5-haiku", 0, 1_000_000)
        self.assertAlmostEqual(cost_out, 4.00, places=6)

        cost_haiku_45 = calculate_cost("claude-haiku-4-5", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost_haiku_45, 4.80, places=6)

        cost_both = calculate_cost("claude-3-5-haiku-20241022", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost_both, 4.80, places=6)

    def test_calculate_cost_flash(self):
        # Flash: $0.075/MTok input, $0.30/MTok output
        cost_in = calculate_cost("flash", 1_000_000, 0)
        self.assertAlmostEqual(cost_in, 0.075, places=6)

        cost_out = calculate_cost("gemini-1.5-flash", 0, 1_000_000)
        self.assertAlmostEqual(cost_out, 0.30, places=6)

        cost_flash_38 = calculate_cost("gemini-3.8-flash", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost_flash_38, 0.375, places=6)

        cost_both = calculate_cost("gemini-1.5-flash-latest", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost_both, 0.375, places=6)

    def test_calculate_cost_unknown_model(self):
        with self.assertRaises(ValueError):
            calculate_cost("unknown-model", 100, 100)


class TestParseJevResponse(unittest.TestCase):
    def setUp(self):
        self.comment_id = "test_comment_123"
        self.latency_ms = 450.5
        self.usage = MockUsage(input_tokens=150, output_tokens=0)
        self.cost_usd = 0.0000063
        self.response = MockJevResponse(
            toxic_answer=MockScoreAnswer(
                probabilities={"0": 0.1, "1": 0.7, "2": 0.2},
                score=1.1,
                confidence=0.85,
            ),
            threat_answer=MockNoulAnswer(noul=0.15),
            identity_hate_answer=MockNoulAnswer(noul=0.9),
        )

    def test_parse_jev_response_calls_row(self):
        calls_row, answers_rows = parse_jev_response(
            self.comment_id,
            self.response,
            self.latency_ms,
            self.usage,
            self.cost_usd,
        )

        self.assertEqual(calls_row["comment_id"], self.comment_id)
        self.assertEqual(calls_row["model"], "jev")
        self.assertEqual(calls_row["latency_ms"], self.latency_ms)
        self.assertEqual(calls_row["input_tokens"], 150)
        self.assertEqual(calls_row["output_tokens"], 0)
        self.assertEqual(calls_row["cost_usd"], self.cost_usd)
        self.assertIsInstance(calls_row["raw_response_json"], str)
        # raw_response_json should be valid JSON
        parsed_raw = json.loads(calls_row["raw_response_json"])
        self.assertIn("answers", parsed_raw)

        # Check timestamp is ISO format
        datetime.fromisoformat(calls_row["timestamp"].replace("Z", "+00:00"))

    def test_parse_jev_response_answers_rows(self):
        _, answers_rows = parse_jev_response(
            self.comment_id,
            self.response,
            self.latency_ms,
            self.usage,
            self.cost_usd,
        )

        self.assertEqual(len(answers_rows), 3)

        # Check toxic row
        toxic_row = next(r for r in answers_rows if r["category"] == "toxic")
        self.assertEqual(toxic_row["comment_id"], self.comment_id)
        self.assertEqual(toxic_row["model"], "jev")
        self.assertEqual(toxic_row["toxic_prob_0"], 0.1)
        self.assertEqual(toxic_row["toxic_prob_1"], 0.7)
        self.assertEqual(toxic_row["toxic_prob_2"], 0.2)
        self.assertEqual(toxic_row["toxic_score"], 1.1)
        self.assertIsNone(toxic_row["binary_probability"])
        self.assertEqual(toxic_row["native_confidence"], 0.85)

        # Check threat row: derived native_confidence = 2 * abs(p - 0.5)
        # 2 * abs(0.15 - 0.5) = 2 * 0.35 = 0.70
        threat_row = next(r for r in answers_rows if r["category"] == "threat")
        self.assertEqual(threat_row["comment_id"], self.comment_id)
        self.assertEqual(threat_row["model"], "jev")
        self.assertIsNone(threat_row["toxic_prob_0"])
        self.assertIsNone(threat_row["toxic_prob_1"])
        self.assertIsNone(threat_row["toxic_prob_2"])
        self.assertIsNone(threat_row["toxic_score"])
        self.assertEqual(threat_row["binary_probability"], 0.15)
        self.assertAlmostEqual(threat_row["native_confidence"], 0.70, places=6)

        # Check identity_hate row: 2 * abs(0.9 - 0.5) = 2 * 0.4 = 0.80
        idh_row = next(r for r in answers_rows if r["category"] == "identity_hate")
        self.assertEqual(idh_row["comment_id"], self.comment_id)
        self.assertEqual(idh_row["model"], "jev")
        self.assertIsNone(idh_row["toxic_prob_0"])
        self.assertIsNone(idh_row["toxic_prob_1"])
        self.assertIsNone(idh_row["toxic_prob_2"])
        self.assertIsNone(idh_row["toxic_score"])
        self.assertEqual(idh_row["binary_probability"], 0.9)
        self.assertAlmostEqual(idh_row["native_confidence"], 0.80, places=6)


class TestParseLLMResponse(unittest.TestCase):
    def setUp(self):
        self.comment_id = "test_comment_456"
        self.model_name = "haiku"
        self.latency_ms = 820.0
        self.usage = MockUsage(input_tokens=220, output_tokens=45)
        self.cost_usd = 0.000356
        self.raw_response = '{"id": "msg_123", "role": "assistant"}'

    def test_parse_llm_response_schema_and_types(self):
        parsed_json = {
            "toxic": {"probabilities": {"0": 0.7, "1": 0.25, "2": 0.05}},
            "threat": {"probability": 0.02},
            "identity_hate": {"probability": 0.01},
        }

        calls_row, answers_rows = parse_llm_response(
            self.comment_id,
            self.model_name,
            parsed_json,
            self.latency_ms,
            self.usage,
            self.cost_usd,
            self.raw_response,
        )

        self.assertEqual(calls_row["comment_id"], self.comment_id)
        self.assertEqual(calls_row["model"], self.model_name)
        self.assertEqual(calls_row["latency_ms"], self.latency_ms)
        self.assertEqual(calls_row["input_tokens"], 220)
        self.assertEqual(calls_row["output_tokens"], 45)
        self.assertEqual(calls_row["cost_usd"], self.cost_usd)
        self.assertEqual(calls_row["raw_response_json"], self.raw_response)
        datetime.fromisoformat(calls_row["timestamp"].replace("Z", "+00:00"))

        self.assertEqual(len(answers_rows), 3)

        toxic_row = next(r for r in answers_rows if r["category"] == "toxic")
        self.assertAlmostEqual(toxic_row["toxic_prob_0"], 0.7)
        self.assertAlmostEqual(toxic_row["toxic_prob_1"], 0.25)
        self.assertAlmostEqual(toxic_row["toxic_prob_2"], 0.05)
        # 0*0.7 + 1*0.25 + 2*0.05 = 0.35
        self.assertAlmostEqual(toxic_row["toxic_score"], 0.35, places=6)
        self.assertIsNone(toxic_row["binary_probability"])
        self.assertIsNone(toxic_row["native_confidence"])

        threat_row = next(r for r in answers_rows if r["category"] == "threat")
        self.assertEqual(threat_row["binary_probability"], 0.02)
        self.assertIsNone(threat_row["native_confidence"])

        idh_row = next(r for r in answers_rows if r["category"] == "identity_hate")
        self.assertEqual(idh_row["binary_probability"], 0.01)
        self.assertIsNone(idh_row["native_confidence"])

    def test_parse_llm_response_normalizes_drift(self):
        # Unnormalized probabilities due to floating point drift (sum = 1.05)
        parsed_json = {
            "toxic": {"probabilities": {"0": 0.70, "1": 0.25, "2": 0.10}},
            "threat": {"probability": 0.05},
            "identity_hate": {"probability": 0.03},
        }

        _, answers_rows = parse_llm_response(
            self.comment_id,
            self.model_name,
            parsed_json,
            self.latency_ms,
            self.usage,
            self.cost_usd,
            self.raw_response,
        )

        toxic_row = next(r for r in answers_rows if r["category"] == "toxic")
        total_p = (
            toxic_row["toxic_prob_0"]
            + toxic_row["toxic_prob_1"]
            + toxic_row["toxic_prob_2"]
        )
        self.assertAlmostEqual(total_p, 1.0, places=6)

        expected_p0 = 0.70 / 1.05
        expected_p1 = 0.25 / 1.05
        expected_p2 = 0.10 / 1.05
        expected_score = 0 * expected_p0 + 1 * expected_p1 + 2 * expected_p2
        self.assertAlmostEqual(toxic_row["toxic_score"], expected_score, places=6)
        self.assertIsNone(toxic_row["native_confidence"])

    def test_parse_llm_response_json_string_input(self):
        # parsed_json passed as raw string instead of dict
        json_str = json.dumps(
            {
                "toxic": {"probabilities": {"0": 0.8, "1": 0.2, "2": 0.0}},
                "threat": {"probability": 0.0},
                "identity_hate": {"probability": 0.0},
            }
        )

        _, answers_rows = parse_llm_response(
            self.comment_id,
            self.model_name,
            json_str,
            self.latency_ms,
            self.usage,
            self.cost_usd,
            self.raw_response,
        )

        toxic_row = next(r for r in answers_rows if r["category"] == "toxic")
        self.assertAlmostEqual(toxic_row["toxic_score"], 0.2, places=6)
        self.assertIsNone(toxic_row["native_confidence"])

    def test_parse_llm_response_schema_resilience(self):
        # Case 1: flattened toxic without nested "probabilities"
        flat_json = {
            "toxic": {"0": 0.6, "1": 0.3, "2": 0.1},
            "threat": 0.05,
            "identity_hate": {"score": 0.12},
        }
        _, answers_rows = parse_llm_response(
            self.comment_id, self.model_name, flat_json, self.latency_ms, self.usage, self.cost_usd, self.raw_response
        )
        toxic_row = next(r for r in answers_rows if r["category"] == "toxic")
        self.assertAlmostEqual(toxic_row["toxic_score"], 0.5, places=6)
        threat_row = next(r for r in answers_rows if r["category"] == "threat")
        self.assertAlmostEqual(threat_row["binary_probability"], 0.05, places=6)
        idh_row = next(r for r in answers_rows if r["category"] == "identity_hate")
        self.assertAlmostEqual(idh_row["binary_probability"], 0.12, places=6)

        # Case 2: scalar toxic probability
        scalar_json = {
            "toxic": 0.25,
            "threat": {"probability": 0.0},
            "identity_hate": {"probability": 0.0},
        }
        _, answers_rows2 = parse_llm_response(
            self.comment_id, self.model_name, scalar_json, self.latency_ms, self.usage, self.cost_usd, self.raw_response
        )
        toxic_row2 = next(r for r in answers_rows2 if r["category"] == "toxic")
        self.assertAlmostEqual(toxic_row2["toxic_prob_0"], 0.75, places=6)
        self.assertAlmostEqual(toxic_row2["toxic_prob_1"], 0.25, places=6)


class TestPromptsAndDefinitions(unittest.TestCase):
    def test_category_definitions_exist_and_non_empty(self):
        self.assertIsInstance(TOXIC_RUBRIC, dict)
        self.assertIn("0", TOXIC_RUBRIC)
        self.assertIn("1", TOXIC_RUBRIC)
        self.assertIn("2", TOXIC_RUBRIC)
        self.assertTrue(len(TOXIC_RUBRIC["0"]) > 50)
        self.assertTrue(len(TOXIC_RUBRIC["1"]) > 50)
        self.assertTrue(len(TOXIC_RUBRIC["2"]) > 50)

        self.assertIsInstance(THREAT_DEFINITION, dict)
        self.assertIn("positive", THREAT_DEFINITION)
        self.assertIn("negative", THREAT_DEFINITION)
        self.assertTrue(len(THREAT_DEFINITION["positive"]) > 50)
        self.assertTrue(len(THREAT_DEFINITION["negative"]) > 50)

        self.assertIsInstance(IDENTITY_HATE_DEFINITION, dict)
        self.assertIn("positive", IDENTITY_HATE_DEFINITION)
        self.assertIn("negative", IDENTITY_HATE_DEFINITION)
        self.assertTrue(len(IDENTITY_HATE_DEFINITION["positive"]) > 50)
        self.assertTrue(len(IDENTITY_HATE_DEFINITION["negative"]) > 50)

    def test_noise_default_instruction(self):
        expected_noise_instruction = (
            "If the comment contains no interpretable content relevant to a category "
            "(e.g. random characters, pure noise), default to negative / Level 0 for that "
            "category — don't infer hostility, threat, or identity-targeting from noise alone."
        )
        self.assertEqual(NOISE_DEFAULT_INSTRUCTION.strip(), expected_noise_instruction)

    def test_system_prompt_template(self):
        self.assertIn("{comment_text}", SYSTEM_PROMPT_TEMPLATE)
        self.assertIn("## Category: toxic", SYSTEM_PROMPT_TEMPLATE)
        self.assertIn("## Category: threat", SYSTEM_PROMPT_TEMPLATE)
        self.assertIn("## Category: identity_hate", SYSTEM_PROMPT_TEMPLATE)
        self.assertIn(NOISE_DEFAULT_INSTRUCTION, SYSTEM_PROMPT_TEMPLATE)

        # Check that formatting with a comment works
        formatted = SYSTEM_PROMPT_TEMPLATE.format(comment_text="Sample test comment")
        self.assertIn('"""Sample test comment"""', formatted)

    def test_jev_questions_configuration(self):
        self.assertIn("toxic", JEV_QUESTIONS)
        self.assertIn("threat", JEV_QUESTIONS)
        self.assertIn("identity_hate", JEV_QUESTIONS)

        toxic_q = JEV_QUESTIONS["toxic"]
        threat_q = JEV_QUESTIONS["threat"]
        idh_q = JEV_QUESTIONS["identity_hate"]

        # Check instructions and criteria
        self.assertTrue(hasattr(toxic_q, "instructions") or "instructions" in toxic_q)
        self.assertTrue(hasattr(threat_q, "instructions") or "instructions" in threat_q)
        self.assertTrue(hasattr(idh_q, "instructions") or "instructions" in idh_q)

    def test_structured_output_schemas(self):
        # Claude tool definition
        self.assertEqual(CLAUDE_CLASSIFICATION_TOOL["name"], "submit_classification")
        claude_props = CLAUDE_CLASSIFICATION_TOOL["input_schema"]["properties"]
        self.assertIn("toxic", claude_props)
        self.assertIn("threat", claude_props)
        self.assertIn("identity_hate", claude_props)

        # Gemini JSON schema
        gemini_props = GEMINI_RESPONSE_SCHEMA["properties"]
        self.assertIn("toxic", gemini_props)
        self.assertIn("threat", gemini_props)
        self.assertIn("identity_hate", gemini_props)


if __name__ == "__main__":
    unittest.main()
