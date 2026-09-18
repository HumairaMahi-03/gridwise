"""The degraded-mode rule interpreter.

These tests pin the fallback's behaviour so that enabling it during an outage
is a known quantity rather than a gamble. They are not a substitute for the
LLM: the rules are checked against wording they are expected to handle, and
against wording they are expected to refuse.
"""

from __future__ import annotations

import json
import os

import pytest

from app.exceptions import LLMUnavailableError
from app.llm.fallback import (
    ResilientInterpreter,
    RuleBasedInterpreter,
    classify,
    extract_fraction,
    extract_hours,
    extract_kwh,
)
from tests.helpers import ExplodingInterpreter, MockNoteInterpreter, no_op_payload

CASES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "samples", "public_cases.json")
with open(CASES_PATH) as handle:
    CASES = json.load(handle)["cases"]


# ------------------------------------------------------------------ primitives


@pytest.mark.parametrize(
    "note, expected",
    [
        ("from 1 PM to 3 PM", [13, 14]),
        ("between 2 PM and 4 PM", [14, 15]),
        ("from 14:00 until 16:00", [14, 15]),
        ("from noon until 2 PM", [12, 13]),
        ("from 2 AM until 5 AM", [2, 3, 4]),
        ("from 6 PM until 10 PM", [18, 19, 20, 21]),
        ("from 10 AM until noon", [10, 11]),
        ("during the early afternoon", [12, 13, 14]),
        ("at hour 9", [9]),
    ],
)
def test_hour_windows_are_half_open(note, expected):
    assert extract_hours(note) == expected


@pytest.mark.parametrize(
    "note, expected",
    [
        ("solar will drop to 20%", 0.2),
        ("expect an 80% reduction in rooftop solar", 0.2),
        ("output reduced by 75 percent", 0.25),
        ("treated as roughly 25% of the forecast", 0.25),
        ("about half of the forecast solar output", 0.5),
        ("only one fifth of normal generation", 0.2),
        ("the array will be offline with no output", 0.0),
    ],
)
def test_solar_fraction_is_the_remaining_share(note, expected):
    assert extract_fraction(note) == pytest.approx(expected, abs=1e-6)


def test_kwh_extraction_handles_absolute_and_proportional():
    assert extract_kwh("keep at least 90 kWh in the battery", 250) == 90
    assert extract_kwh("keep at least 50% of the battery capacity", 200) == 100
    assert extract_kwh("keep the battery a quarter full", 240) == 60
    assert extract_kwh("keep some charge available", 200) is None


# ----------------------------------------------------------------- classification


def test_discharge_prohibition_is_not_read_as_charging():
    """'discharge' contains 'charg'; the classifier must not confuse the two."""
    directive_type, adjustment, _ = classify(
        "Do not discharge the battery from 5 PM until 7 PM during relay testing.", 200
    )
    assert directive_type == "no_discharge_window"
    assert adjustment["hours"] == [17, 18]


def test_irrelevant_notes_become_no_op():
    for note in [
        "The cafeteria menu changes tomorrow.",
        "The sports office moved next month's registration deadline.",
        "The library is extending book-return hours next week.",
        "A seminar room booking was moved to next week.",
    ]:
        directive_type, adjustment, _ = classify(note, 200)
        assert directive_type == "no_op"
        assert adjustment is None


def test_energy_note_without_a_usable_number_becomes_no_op():
    """Better to do nothing than to invent a constraint."""
    directive_type, _, _ = classify("Please keep grid import sensible this evening.", 200)
    assert directive_type == "no_op"


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_fallback_matches_ground_truth_on_the_public_cases(case):
    capacity = case["input"]["battery"]["capacity_kwh"]
    payload = RuleBasedInterpreter().interpret(
        case["input"]["operator_notes"], {"battery_capacity_kwh": capacity}
    )

    for produced, expected in zip(payload, case["expected_directives"]):
        assert produced["directive_type"] == expected["directive_type"]
        if expected["directive_type"] == "no_op":
            assert produced["applies"] is False
            assert produced["structured_adjustment"] is None
        else:
            wanted = {key: value for key, value in expected.items() if key != "directive_type"}
            assert produced["structured_adjustment"] == wanted


# -------------------------------------------------------------------- resilience


def test_resilient_interpreter_prefers_the_llm():
    primary = MockNoteInterpreter(no_op_payload(1))
    resilient = ResilientInterpreter(primary, RuleBasedInterpreter())

    resilient.interpret(["Do not charge the battery from 2 PM until 4 PM."])
    assert primary.interpret_calls == 1


def test_resilient_interpreter_falls_back_only_on_an_outage():
    resilient = ResilientInterpreter(
        ExplodingInterpreter(LLMUnavailableError()), RuleBasedInterpreter()
    )
    payload = resilient.interpret(
        ["Do not charge the battery from 2 PM until 4 PM."], {"battery_capacity_kwh": 200}
    )

    assert payload[0]["directive_type"] == "no_charge_window"
    assert payload[0]["structured_adjustment"]["hours"] == [14, 15]


def test_resilient_interpreter_does_not_mask_a_bad_response():
    """A reachable provider returning nonsense must reach the guardrails, not the rules."""
    from app.exceptions import LLMInterpretationError

    resilient = ResilientInterpreter(
        ExplodingInterpreter(LLMInterpretationError()), RuleBasedInterpreter()
    )
    with pytest.raises(LLMInterpretationError):
        resilient.interpret(["Do not charge the battery from 2 PM until 4 PM."])


def test_fallback_is_disabled_by_default():
    from app.config import Settings

    assert Settings().llm_fallback_to_rules is False
