"""End-to-end behaviour through the HTTP layer with a mocked interpreter."""

from __future__ import annotations

import pytest

from app.exceptions import LLMInterpretationError, LLMUnavailableError
from tests.helpers import (
    DEMAND,
    SOLAR,
    ExplodingInterpreter,
    MockNoteInterpreter,
    directive_payload,
    make_scenario,
    no_op_payload,
)


def post(client, scenario):
    response = client.post("/optimize-energy", json=scenario)
    assert response.status_code == 200, response.text
    return response.json()


def by_hour(body):
    return {entry["hour"]: entry for entry in body["hourly_plan"]}


def test_full_pipeline_with_three_notes(client_factory):
    notes = [
        "Solar output will drop to about 20% from 1 PM to 3 PM.",
        "Do not charge the battery between 2 PM and 4 PM.",
        "The cafeteria menu changes tomorrow.",
    ]
    payload = [
        directive_payload(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2}),
        directive_payload(1, "no_charge_window", {"hours": [14, 15]}),
        directive_payload(2, "no_op", None),
    ]
    body = post(client_factory(payload), make_scenario(notes=notes))
    plan = by_hour(body)

    # Every hour balances.
    for hour, entry in plan.items():
        charge = entry["battery_kwh"] if entry["battery_action"] == "charge" else 0.0
        discharge = entry["battery_kwh"] if entry["battery_action"] == "discharge" else 0.0
        supply = entry["grid_kwh"] + entry["solar_used_kwh"] + discharge
        assert supply == pytest.approx(DEMAND[hour] + charge, abs=1e-2)

    # Directives are actually applied.
    assert plan[13]["solar_used_kwh"] <= SOLAR[13] * 0.2 + 1e-2
    assert plan[14]["solar_used_kwh"] <= SOLAR[14] * 0.2 + 1e-2
    assert plan[14]["battery_action"] != "charge"
    assert plan[15]["battery_action"] != "charge"

    # Day-end neutrality.
    assert plan[23]["battery_energy_after_kwh"] == pytest.approx(200, abs=1e-2)

    # The irrelevant note changed nothing.
    assert body["directive_interpretation"][2]["directive_type"] == "no_op"


def test_battery_state_chain_is_consistent(client, scenario):
    body = post(client, scenario)
    previous = 200.0

    for entry in body["hourly_plan"]:
        charge = entry["battery_kwh"] if entry["battery_action"] == "charge" else 0.0
        discharge = entry["battery_kwh"] if entry["battery_action"] == "discharge" else 0.0
        assert entry["battery_energy_after_kwh"] == pytest.approx(
            previous + charge - discharge, abs=1e-2
        )
        previous = entry["battery_energy_after_kwh"]


def test_idle_hours_report_zero_movement(client, scenario):
    body = post(client, scenario)
    for entry in body["hourly_plan"]:
        if entry["battery_action"] == "idle":
            assert entry["battery_kwh"] == 0


def test_irrelevant_notes_do_not_change_the_schedule(client_factory):
    plain = post(client_factory(no_op_payload(1)), make_scenario(notes=["Nothing to report."]))
    chatty = post(
        client_factory(no_op_payload(3)),
        make_scenario(notes=["The menu changes.", "Book fair on Friday.", "Submit timesheets."]),
    )
    assert plain["total_cost_bdt"] == pytest.approx(chatty["total_cost_bdt"], abs=1e-6)


def test_repair_round_recovers_a_bad_payload(client_factory):
    broken = [directive_payload(0, "solar_reduction", {"hours": [12], "factor": 50})]
    fixed = [directive_payload(0, "solar_reduction", {"hours": [12], "factor": 0.5})]
    interpreter = MockNoteInterpreter(broken, repair_payload=fixed)

    client = client_factory(interpreter=interpreter)
    body = post(client, make_scenario(notes=["Solar halves at noon."]))

    assert interpreter.interpret_calls == 1
    assert interpreter.repair_calls == 1
    assert body["directive_interpretation"][0]["structured_adjustment"]["factor"] == 0.5
    assert by_hour(body)[12]["solar_used_kwh"] <= SOLAR[12] * 0.5 + 1e-2


def test_only_one_interpretation_round_when_the_payload_is_valid(client_factory):
    interpreter = MockNoteInterpreter(no_op_payload(1))
    client = client_factory(interpreter=interpreter)
    post(client, make_scenario())

    assert interpreter.interpret_calls == 1
    assert interpreter.repair_calls == 0


def test_persistently_invalid_relevant_directive_is_refused(client_factory):
    broken = [directive_payload(0, "solar_reduction", {"hours": [12], "factor": 50})]
    client = client_factory(MockNoteInterpreter(broken))

    response = client.post("/optimize-energy", json=make_scenario(notes=["Solar halves."]))
    assert response.status_code == 502
    assert "detail" in response.json()


def test_infeasible_constraints_return_422(client_factory):
    payload = [directive_payload(0, "max_grid_window", {"hours": [19], "max_grid_kwh": 0})]
    client = client_factory(payload)

    response = client.post("/optimize-energy", json=make_scenario(notes=["No import at 7 PM."]))
    assert response.status_code == 422
    assert "feasible" in response.json()["detail"].lower()


def test_llm_unavailable_returns_503(client_factory):
    client = client_factory(interpreter=ExplodingInterpreter(LLMUnavailableError()))
    response = client.post("/optimize-energy", json=make_scenario())
    assert response.status_code == 503


def test_llm_malformed_output_returns_502(client_factory):
    client = client_factory(interpreter=ExplodingInterpreter(LLMInterpretationError()))
    response = client.post("/optimize-energy", json=make_scenario())
    assert response.status_code == 502


def test_unexpected_interpreter_failure_returns_500_without_details(client_factory):
    client = client_factory(interpreter=ExplodingInterpreter(RuntimeError("secret-key-12345")))
    response = client.post("/optimize-energy", json=make_scenario())

    assert response.status_code == 500
    assert "secret-key-12345" not in response.text


def test_scenario_id_is_echoed_unchanged(client_factory):
    client = client_factory(no_op_payload(1))
    body = post(client, make_scenario(scenario_id="CAMPUS-2026-XYZ"))
    assert body["scenario_id"] == "CAMPUS-2026-XYZ"


def test_a_different_scenario_produces_a_different_plan(client_factory):
    """Guards against any hardcoding of the sample scenario."""
    client = client_factory(no_op_payload(1))

    baseline = post(client, make_scenario())

    altered = make_scenario()
    for record in altered["hours"]:
        record["demand_kwh"] = record["demand_kwh"] * 2
        record["solar_kwh"] = 0
    altered["scenario_id"] = "OTHER-1"

    doubled = post(client_factory(no_op_payload(1)), altered)

    assert doubled["total_grid_kwh"] > baseline["total_grid_kwh"]
    assert doubled["total_cost_bdt"] > baseline["total_cost_bdt"]
    assert doubled["scenario_id"] == "OTHER-1"


def test_flat_tariff_still_produces_a_valid_neutral_day(client_factory):
    """With no price spread there is nothing to arbitrage, but the day must still close."""
    scenario = make_scenario()
    for record in scenario["hours"]:
        record["tariff_bdt_per_kwh"] = 10

    body = post(client_factory(no_op_payload(1)), scenario)
    assert by_hour(body)[23]["battery_energy_after_kwh"] == pytest.approx(200, abs=1e-2)


def test_summary_mentions_grid_cost_and_directives(client_factory):
    payload = [directive_payload(0, "no_charge_window", {"hours": [14, 15, 16]})]
    body = post(client_factory(payload), make_scenario(notes=["No charging in the afternoon."]))

    summary = body["plan_summary"]
    assert "kWh from the grid" in summary
    assert "no charging in hour(s) 14-16" in summary
