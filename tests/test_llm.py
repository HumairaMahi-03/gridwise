"""Operator-note interpretation.

The provider is mocked at the transport level, so these tests are deterministic
and never make a network call. A small opt-in suite at the bottom runs against a
real provider when ``RUN_LIVE_LLM_TESTS=1`` and credentials are present; that is
where semantic paraphrase behaviour is genuinely exercised.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

from app.config import Settings
from app.exceptions import LLMInterpretationError, LLMUnavailableError
from app.guardrails.validator import validate_interpretations
from app.llm.interpreter import (
    OpenAICompatibleInterpreter,
    _extract_interpretations,
    build_interpreter,
)
from app.llm.prompts import build_response_schema, build_system_prompt, build_user_prompt
from app.models.directives import DirectiveType
from tests.helpers import directive_payload


# ------------------------------------------------------------- response parsing


def test_parses_plain_object_payload():
    content = json.dumps({"interpretations": [{"note_index": 0, "directive_type": "no_op"}]})
    assert _extract_interpretations(content)[0]["directive_type"] == "no_op"


def test_parses_fenced_json():
    content = '```json\n{"interpretations": [{"note_index": 0, "directive_type": "no_op"}]}\n```'
    assert len(_extract_interpretations(content)) == 1


def test_parses_bare_array():
    assert len(_extract_interpretations('[{"note_index": 0, "directive_type": "no_op"}]')) == 1


def test_parses_json_embedded_in_prose():
    content = 'Sure! Here is the result:\n{"interpretations": [{"note_index": 0, "directive_type": "no_op"}]}'
    assert len(_extract_interpretations(content)) == 1


def test_accepts_alternative_wrapper_keys():
    content = json.dumps({"results": [{"note_index": 0, "directive_type": "no_op"}]})
    assert len(_extract_interpretations(content)) == 1


@pytest.mark.parametrize("content", ["", "   ", None, "I cannot help with that request."])
def test_unusable_responses_raise(content):
    with pytest.raises(LLMInterpretationError):
        _extract_interpretations(content)


def test_non_object_entries_raise():
    with pytest.raises(LLMInterpretationError):
        _extract_interpretations('["not an object"]')


# ------------------------------------------------------------------- transport


def make_settings(**overrides):
    defaults = {
        "llm_api_key": "test-key",
        "llm_model": "test-model",
        "llm_base_url": "https://provider.test/v1",
        "llm_timeout_seconds": 5.0,
        "llm_temperature": 0.0,
        "llm_max_output_tokens": 500,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class FakeClient:
    """Stands in for httpx.Client, recording requests and replaying responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append({"url": url, "json": json, "headers": headers})
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_response(status_code=200, content="{}"):
    return httpx.Response(
        status_code=status_code,
        json={"choices": [{"message": {"content": content}}]},
    )


def test_request_is_well_formed():
    payload = json.dumps({"interpretations": [{"note_index": 0, "directive_type": "no_op"}]})
    client = FakeClient([make_response(content=payload)])
    interpreter = OpenAICompatibleInterpreter(make_settings(), client=client)

    interpreter.interpret(["A note."])
    sent = client.requests[0]

    assert sent["url"] == "https://provider.test/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert sent["json"]["model"] == "test-model"
    assert sent["json"]["temperature"] == 0.0
    assert sent["json"]["messages"][0]["role"] == "system"
    assert "A note." in sent["json"]["messages"][1]["content"]
    assert sent["json"]["response_format"]["type"] == "json_schema"


def test_falls_back_to_json_object_when_schema_is_rejected():
    payload = json.dumps({"interpretations": [{"note_index": 0, "directive_type": "no_op"}]})
    client = FakeClient([make_response(status_code=400), make_response(content=payload)])
    interpreter = OpenAICompatibleInterpreter(make_settings(), client=client)

    assert len(interpreter.interpret(["A note."])) == 1
    assert client.requests[0]["json"]["response_format"]["type"] == "json_schema"
    assert client.requests[1]["json"]["response_format"]["type"] == "json_object"


def test_repair_includes_the_reported_problems():
    payload = json.dumps({"interpretations": [{"note_index": 0, "directive_type": "no_op"}]})
    client = FakeClient([make_response(content=payload)])
    interpreter = OpenAICompatibleInterpreter(make_settings(), client=client)

    interpreter.repair(["A note."], [{"broken": True}], ["note_index 0: 'factor' must be a number."])
    messages = client.requests[0]["json"]["messages"]

    assert messages[2]["role"] == "assistant"
    assert "'factor' must be a number" in messages[3]["content"]


def test_missing_api_key_reports_unavailable():
    interpreter = OpenAICompatibleInterpreter(make_settings(llm_api_key=""), client=FakeClient([]))
    with pytest.raises(LLMUnavailableError):
        interpreter.interpret(["A note."])


def test_timeout_reports_unavailable():
    client = FakeClient([httpx.TimeoutException("timed out")])
    interpreter = OpenAICompatibleInterpreter(make_settings(), client=client)
    with pytest.raises(LLMUnavailableError):
        interpreter.interpret(["A note."])


def test_server_error_reports_unavailable():
    client = FakeClient([make_response(status_code=500)])
    interpreter = OpenAICompatibleInterpreter(make_settings(), client=client)
    with pytest.raises(LLMUnavailableError):
        interpreter.interpret(["A note."])


def test_unexpected_envelope_reports_interpretation_failure():
    client = FakeClient([httpx.Response(status_code=200, json={"unexpected": True})])
    interpreter = OpenAICompatibleInterpreter(make_settings(), client=client)
    with pytest.raises(LLMInterpretationError):
        interpreter.interpret(["A note."])


def test_unknown_provider_is_reported():
    with pytest.raises(LLMUnavailableError):
        build_interpreter(make_settings(llm_provider="carrier-pigeon"))


# --------------------------------------------------------------------- prompts


def test_system_prompt_states_the_closed_vocabulary():
    prompt = build_system_prompt()
    for directive in DirectiveType:
        assert directive.value in prompt
    assert "Never invent a new type" in prompt
    assert "no_op" in prompt


def test_system_prompt_teaches_percentage_conversion():
    prompt = build_system_prompt()
    assert "drop BY 80%" in prompt
    assert "fraction that REMAINS" in prompt


def test_hour_convention_is_configurable():
    half_open = build_system_prompt(inclusive_end=False)
    inclusive = build_system_prompt(inclusive_end=True)

    assert "[13, 14]" in half_open
    assert "[13, 14, 15]" in inclusive


def test_user_prompt_numbers_every_note():
    prompt = build_user_prompt(["first", "second", "third"])
    assert "0:" in prompt and "1:" in prompt and "2:" in prompt
    assert "note_index\n0 through 2" in prompt or "0 through 2" in prompt


def test_response_schema_pins_the_note_count():
    schema = build_response_schema(3)
    interpretations = schema["properties"]["interpretations"]
    assert interpretations["minItems"] == 3
    assert interpretations["maxItems"] == 3
    enum = interpretations["items"]["properties"]["directive_type"]["enum"]
    assert set(enum) == {directive.value for directive in DirectiveType}


# ---------------------------------------------------- paraphrase contract tests


PARAPHRASE_GROUPS = {
    "no_charge_window": [
        "Do not charge the battery from 2 PM until 4 PM.",
        "Battery charging should be disabled during the early afternoon window.",
        "Keep the battery from charging between 14:00 and 16:00.",
    ],
    "solar_reduction": [
        "Solar output will fall to 20% between 1 PM and 3 PM.",
        "Only one fifth of normal solar generation will be available in the early afternoon.",
        "Expect an 80% drop in PV output in the early afternoon.",
    ],
}


def test_equivalent_directives_compile_identically(battery):
    """Whatever wording produced them, identical directives must behave identically."""
    from app.guardrails.validator import compile_directives

    first = validate_interpretations(
        [directive_payload(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2})],
        ["Solar output will fall to 20% between 1 PM and 3 PM."],
        battery,
    )
    second = validate_interpretations(
        [directive_payload(0, "solar_reduction", {"hours": [14, 13], "factor": 0.2})],
        ["Only one fifth of normal solar generation will be available."],
        battery,
    )

    assert first.ok and second.ok
    assert compile_directives(first.interpretations, battery).solar_factor == compile_directives(
        second.interpretations, battery
    ).solar_factor


# ------------------------------------------------------------ live provider run

LIVE = os.getenv("RUN_LIVE_LLM_TESTS") == "1" and os.getenv("LLM_API_KEY")
live_only = pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_LLM_TESTS=1 and LLM_API_KEY to run")


@live_only
@pytest.mark.parametrize("expected_type, notes", PARAPHRASE_GROUPS.items())
def test_live_paraphrases_map_to_the_same_directive(expected_type, notes, battery):
    interpreter = build_interpreter()
    for note in notes:
        payload = interpreter.interpret([note])
        result = validate_interpretations(payload, [note], battery)
        assert result.ok, result.problems
        assert result.interpretations[0].directive_type.value == expected_type, note


@live_only
@pytest.mark.parametrize(
    "note",
    [
        "The cafeteria menu changes tomorrow.",
        "The library will host a book fair in the main hall.",
        "Please remind staff to submit timesheets by Friday.",
    ],
)
def test_live_irrelevant_notes_become_no_op(note, battery):
    interpreter = build_interpreter()
    result = validate_interpretations(interpreter.interpret([note]), [note], battery)

    assert result.ok, result.problems
    assert result.interpretations[0].directive_type is DirectiveType.NO_OP
    assert result.interpretations[0].applies is False
    assert result.interpretations[0].structured_adjustment is None
