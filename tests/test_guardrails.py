"""Deterministic guardrails over LLM output."""

from __future__ import annotations

import pytest

from app.exceptions import LLMInterpretationError
from app.guardrails.validator import (
    compile_directives,
    salvage_interpretations,
    validate_interpretations,
)
from app.models.directives import DirectiveType
from tests.helpers import directive_payload, no_op_payload


def test_valid_directive_is_accepted(battery):
    payload = [directive_payload(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2})]
    result = validate_interpretations(payload, ["note"], battery)

    assert result.ok, result.problems
    interpretation = result.interpretations[0]
    assert interpretation.directive_type is DirectiveType.SOLAR_REDUCTION
    assert interpretation.applies is True
    assert interpretation.structured_adjustment == {"hours": [13, 14], "factor": 0.2}


def test_no_op_is_accepted_and_carries_no_adjustment(battery):
    result = validate_interpretations(no_op_payload(1), ["note"], battery)
    assert result.ok
    assert result.interpretations[0].directive_type is DirectiveType.NO_OP
    assert result.interpretations[0].structured_adjustment is None


@pytest.mark.parametrize(
    "payload, reason",
    [
        ([directive_payload(0, "solar_reduction", {"hours": [13], "factor": 1.4})], "factor > 1"),
        ([directive_payload(0, "solar_reduction", {"hours": [13], "factor": -0.1})], "factor < 0"),
        ([directive_payload(0, "solar_reduction", {"hours": [13], "factor": "0.2"})], "factor is a string"),
        ([directive_payload(0, "solar_reduction", {"hours": [13]})], "factor missing"),
        ([directive_payload(0, "no_charge_window", {"hours": [24]})], "hour out of range"),
        ([directive_payload(0, "no_charge_window", {"hours": [-1]})], "negative hour"),
        ([directive_payload(0, "no_charge_window", {"hours": []})], "empty hours"),
        ([directive_payload(0, "no_charge_window", {"hours": [13.5]})], "fractional hour"),
        ([directive_payload(0, "no_charge_window", {"hours": "14-16"})], "hours not a list"),
        ([directive_payload(0, "shed_load", {"hours": [13]})], "unsupported directive"),
        ([directive_payload(0, "max_grid_window", {"hours": [18], "max_grid_kwh": -5})], "negative cap"),
        ([directive_payload(0, "max_grid_window", {"hours": [18]})], "cap missing"),
        (
            [directive_payload(0, "minimum_battery_reserve", {"hours": [18], "minimum_energy_kwh": 9000})],
            "reserve above capacity",
        ),
        (
            [directive_payload(0, "minimum_battery_reserve", {"hours": [18], "minimum_energy_kwh": -1})],
            "negative reserve",
        ),
        ([directive_payload(0, "no_charge_window", None)], "adjustment missing for active directive"),
    ],
)
def test_invalid_directives_are_rejected(payload, reason, battery):
    result = validate_interpretations(payload, ["note"], battery)
    assert not result.ok, f"{reason} should have been rejected"
    assert result.problems
    assert result.interpretations == []


def test_unsorted_and_duplicated_hours_are_normalised(battery):
    payload = [directive_payload(0, "no_discharge_window", {"hours": [16, 14, 14, 15]})]
    result = validate_interpretations(payload, ["note"], battery)

    assert result.ok, result.problems
    assert result.interpretations[0].structured_adjustment["hours"] == [14, 15, 16]


def test_integral_float_hours_are_accepted(battery):
    payload = [directive_payload(0, "no_charge_window", {"hours": [14.0, 15.0]})]
    result = validate_interpretations(payload, ["note"], battery)
    assert result.ok
    assert result.interpretations[0].structured_adjustment["hours"] == [14, 15]


def test_unknown_adjustment_keys_are_dropped(battery):
    payload = [
        directive_payload(0, "no_charge_window", {"hours": [14], "reason": "maintenance", "factor": 9})
    ]
    result = validate_interpretations(payload, ["note"], battery)
    assert result.ok
    assert result.interpretations[0].structured_adjustment == {"hours": [14]}


def test_wrong_number_of_interpretations_is_rejected(battery):
    assert not validate_interpretations(no_op_payload(2), ["a", "b", "c"], battery).ok
    assert not validate_interpretations(no_op_payload(3), ["a", "b"], battery).ok


def test_duplicate_note_index_is_rejected(battery):
    payload = no_op_payload(2)
    payload[1]["note_index"] = 0
    assert not validate_interpretations(payload, ["a", "b"], battery).ok


def test_missing_note_index_is_rejected(battery):
    payload = no_op_payload(2)
    payload[1]["note_index"] = 7
    result = validate_interpretations(payload, ["a", "b"], battery)
    assert not result.ok
    assert "Missing interpretation" in result.problems[0]


def test_note_order_is_restored(battery):
    payload = [
        directive_payload(1, "no_charge_window", {"hours": [14]}),
        directive_payload(0, "no_op", None),
    ]
    result = validate_interpretations(payload, ["a", "b"], battery)

    assert result.ok, result.problems
    assert [item.note_index for item in result.interpretations] == [0, 1]
    assert result.interpretations[0].directive_type is DirectiveType.NO_OP


def test_no_op_claiming_to_apply_is_rejected(battery):
    payload = [{"note_index": 0, "applies": True, "directive_type": "no_op",
                "structured_adjustment": None, "explanation": ""}]
    assert not validate_interpretations(payload, ["note"], battery).ok


def test_active_directive_marked_not_applying_is_rejected(battery):
    payload = [{"note_index": 0, "applies": False, "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": [14]}, "explanation": ""}]
    assert not validate_interpretations(payload, ["note"], battery).ok


def test_non_object_entry_is_rejected(battery):
    assert not validate_interpretations(["just a string"], ["note"], battery).ok


def test_salvage_downgrades_only_notes_declared_irrelevant(battery):
    broken = [{"note_index": 0, "applies": False, "directive_type": "nonsense",
               "structured_adjustment": None, "explanation": ""}]
    salvaged = salvage_interpretations(broken, ["note"], battery)
    assert salvaged[0].directive_type is DirectiveType.NO_OP


def test_salvage_refuses_to_drop_a_relevant_directive(battery):
    broken = [directive_payload(0, "solar_reduction", {"hours": [13], "factor": 5})]
    with pytest.raises(LLMInterpretationError):
        salvage_interpretations(broken, ["note"], battery)


def test_salvage_refuses_when_an_interpretation_is_missing(battery):
    with pytest.raises(LLMInterpretationError):
        salvage_interpretations([], ["note"], battery)


def test_compile_merges_overlapping_directives_to_the_strictest(battery):
    payload = [
        directive_payload(0, "solar_reduction", {"hours": [13], "factor": 0.5}),
        directive_payload(1, "solar_reduction", {"hours": [13], "factor": 0.2}),
        directive_payload(2, "minimum_battery_reserve", {"hours": [18], "minimum_energy_kwh": 120}),
    ]
    result = validate_interpretations(payload, ["a", "b", "c"], battery)
    assert result.ok, result.problems

    directives = compile_directives(result.interpretations, battery)
    assert directives.solar_factor[13] == 0.2  # least solar remaining wins
    assert directives.min_reserve[18] == 120


def test_compile_merges_reserves_and_caps(battery):
    payload = [
        directive_payload(0, "minimum_battery_reserve", {"hours": [18], "minimum_energy_kwh": 80}),
        directive_payload(1, "minimum_battery_reserve", {"hours": [18], "minimum_energy_kwh": 150}),
        directive_payload(2, "max_grid_window", {"hours": [18], "max_grid_kwh": 200}),
    ]
    result = validate_interpretations(payload, ["a", "b", "c"], battery)
    directives = compile_directives(result.interpretations, battery)

    assert directives.min_reserve[18] == 150  # highest floor wins
    assert directives.max_grid[18] == 200


def test_compile_ignores_no_ops(battery):
    result = validate_interpretations(no_op_payload(2), ["a", "b"], battery)
    assert compile_directives(result.interpretations, battery).is_empty()
