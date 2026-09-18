"""The ten official public sample cases, run end to end.

Directives come from the case pack's own ground truth, so these tests measure
the optimizer and validator rather than the LLM. Two things are asserted per
case: every GridWise constraint holds on the returned plan, and the cost is at
least as low as the pack's reference optimum.

The pack accepts any schedule that is feasible and equivalently optimal, so the
cost check is ``<=`` rather than an exact schedule comparison.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pytest

from app.models.request import OptimizationRequest
from app.services.energy_service import EnergyService
from tests.helpers import MockNoteInterpreter

CASES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "samples", "public_cases.json")

with open(CASES_PATH) as handle:
    CASES: List[Dict[str, Any]] = json.load(handle)["cases"]

TOLERANCE = 0.01  # the pack's stated equivalence tolerance


def payload_from(expected_directives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn the pack's ground-truth directives into an interpreter payload."""
    payload = []
    for index, directive in enumerate(expected_directives):
        directive_type = directive["directive_type"]
        if directive_type == "no_op":
            payload.append({
                "note_index": index,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "The note does not affect energy scheduling.",
            })
        else:
            payload.append({
                "note_index": index,
                "applies": True,
                "directive_type": directive_type,
                "structured_adjustment": {
                    key: value for key, value in directive.items() if key != "directive_type"
                },
                "explanation": "ground-truth directive from the public case pack",
            })
    return payload


def solve(case: Dict[str, Any]):
    request = OptimizationRequest(**case["input"])
    interpreter = MockNoteInterpreter(payload_from(case["expected_directives"]))
    return request, EnergyService(interpreter).optimize(request)


def ids(cases):
    return [case["id"] for case in cases]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_case_meets_or_beats_the_reference_cost(case):
    _, response = solve(case)
    reference = case["reference_totals"]["total_cost_bdt"]
    assert response.total_cost_bdt <= reference + TOLERANCE, (
        f"{case['id']}: our cost {response.total_cost_bdt} exceeds the reference optimum "
        f"{reference}"
    )


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_case_totals_are_internally_consistent(case):
    request, response = solve(case)
    tariffs = {record.hour: record.tariff_bdt_per_kwh for record in request.hours}

    grid = sum(entry.grid_kwh for entry in response.hourly_plan)
    cost = sum(entry.grid_kwh * tariffs[entry.hour] for entry in response.hourly_plan)
    peak = max(entry.grid_kwh for entry in response.hourly_plan)

    assert response.total_grid_kwh == pytest.approx(grid, abs=TOLERANCE)
    assert response.total_cost_bdt == pytest.approx(cost, abs=TOLERANCE)
    assert response.peak_grid_kwh == pytest.approx(peak, abs=TOLERANCE)


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_case_plan_satisfies_every_constraint(case):
    """Independent replay of the pack's own constraint reminders."""
    request, response = solve(case)
    battery = request.battery
    hours = {record.hour: record for record in request.hours}

    plan = response.hourly_plan
    assert len(plan) == 24
    assert [entry.hour for entry in plan] == list(range(24))

    directives = {}
    for directive in case["expected_directives"]:
        directives.setdefault(directive["directive_type"], []).append(directive)

    solar_factor = {}
    for directive in directives.get("solar_reduction", []):
        for hour in directive["hours"]:
            solar_factor[hour] = directive["factor"]

    previous = battery.initial_energy_kwh
    for entry in plan:
        charge = entry.battery_kwh if entry.battery_action.value == "charge" else 0.0
        discharge = entry.battery_kwh if entry.battery_action.value == "discharge" else 0.0
        record = hours[entry.hour]

        # Energy balance.
        assert entry.grid_kwh + entry.solar_used_kwh + discharge == pytest.approx(
            record.demand_kwh + charge, abs=TOLERANCE
        ), f"{case['id']} hour {entry.hour}: energy balance"

        # Effective solar ceiling.
        effective = record.solar_kwh * solar_factor.get(entry.hour, 1.0)
        assert entry.solar_used_kwh <= effective + TOLERANCE

        # Grid and battery rate limits.
        assert entry.grid_kwh >= -TOLERANCE
        assert charge <= battery.max_charge_kwh_per_hour + TOLERANCE
        assert discharge <= battery.max_discharge_kwh_per_hour + TOLERANCE
        if entry.battery_action.value == "idle":
            assert entry.battery_kwh == 0

        # State of charge.
        assert entry.battery_energy_after_kwh == pytest.approx(
            previous + charge - discharge, abs=TOLERANCE
        )
        assert battery.minimum_energy_kwh - TOLERANCE <= entry.battery_energy_after_kwh
        assert entry.battery_energy_after_kwh <= battery.capacity_kwh + TOLERANCE
        previous = entry.battery_energy_after_kwh

    # Day-end neutrality.
    assert plan[23].battery_energy_after_kwh == pytest.approx(
        battery.initial_energy_kwh, abs=TOLERANCE
    )

    # Directive-specific rules.
    by_hour = {entry.hour: entry for entry in plan}
    for directive in directives.get("no_charge_window", []):
        for hour in directive["hours"]:
            assert by_hour[hour].battery_action.value != "charge"
    for directive in directives.get("no_discharge_window", []):
        for hour in directive["hours"]:
            assert by_hour[hour].battery_action.value != "discharge"
    for directive in directives.get("minimum_battery_reserve", []):
        for hour in directive["hours"]:
            assert by_hour[hour].battery_energy_after_kwh >= directive["minimum_energy_kwh"] - TOLERANCE
    for directive in directives.get("max_grid_window", []):
        for hour in directive["hours"]:
            assert by_hour[hour].grid_kwh <= directive["max_grid_kwh"] + TOLERANCE


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_case_returns_one_interpretation_per_note(case):
    _, response = solve(case)
    notes = case["input"]["operator_notes"]

    assert len(response.directive_interpretation) == len(notes)
    assert [item.note_index for item in response.directive_interpretation] == list(range(len(notes)))

    for item, expected in zip(response.directive_interpretation, case["expected_directives"]):
        assert item.directive_type.value == expected["directive_type"]
        if expected["directive_type"] == "no_op":
            assert item.applies is False
            assert item.structured_adjustment is None
        else:
            assert item.applies is True
            assert item.structured_adjustment is not None


def test_every_case_in_the_pack_is_covered():
    assert len(CASES) == 10
    assert {case["id"] for case in CASES} == {f"SAMPLE-{n:02d}" for n in range(1, 11)}
