"""HTTP surface: health, happy path, and input validation."""

from __future__ import annotations

import pytest

from tests.helpers import TARIFFS, directive_payload, make_scenario, no_op_payload


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_requires_no_credentials(client_factory):
    # Built without any LLM configuration at all.
    client = client_factory(no_op_payload(1))
    assert client.get("/health").status_code == 200


def test_optimize_returns_full_response_shape(client, scenario):
    response = client.post("/optimize-energy", json=scenario)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {
        "scenario_id",
        "directive_interpretation",
        "hourly_plan",
        "total_grid_kwh",
        "total_cost_bdt",
        "peak_grid_kwh",
        "plan_summary",
    }
    assert body["scenario_id"] == "GRID-101"
    assert len(body["hourly_plan"]) == 24
    assert [entry["hour"] for entry in body["hourly_plan"]] == list(range(24))
    assert body["plan_summary"]

    for entry in body["hourly_plan"]:
        assert set(entry) == {
            "hour",
            "grid_kwh",
            "solar_used_kwh",
            "battery_action",
            "battery_kwh",
            "battery_energy_after_kwh",
        }
        assert entry["battery_action"] in {"charge", "discharge", "idle"}


def test_totals_match_the_returned_plan(client, scenario):
    body = client.post("/optimize-energy", json=scenario).json()
    plan = body["hourly_plan"]

    expected_grid = round(sum(entry["grid_kwh"] for entry in plan), 3)
    expected_cost = round(sum(entry["grid_kwh"] * TARIFFS[entry["hour"]] for entry in plan), 2)
    expected_peak = round(max(entry["grid_kwh"] for entry in plan), 3)

    assert body["total_grid_kwh"] == pytest.approx(expected_grid, abs=1e-6)
    assert body["total_cost_bdt"] == pytest.approx(expected_cost, abs=1e-6)
    assert body["peak_grid_kwh"] == pytest.approx(expected_peak, abs=1e-6)


def test_directive_interpretation_is_echoed_in_note_order(client_factory):
    payload = [
        directive_payload(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2}),
        directive_payload(1, "no_charge_window", {"hours": [14, 15]}),
        directive_payload(2, "no_op", None),
    ]
    client = client_factory(payload)
    scenario = make_scenario(notes=["Solar drops.", "No charging.", "Menu changes."])

    body = client.post("/optimize-energy", json=scenario).json()
    interpretations = body["directive_interpretation"]

    assert [item["note_index"] for item in interpretations] == [0, 1, 2]
    assert interpretations[0]["directive_type"] == "solar_reduction"
    assert interpretations[2]["directive_type"] == "no_op"
    assert interpretations[2]["applies"] is False
    assert interpretations[2]["structured_adjustment"] is None


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda s: s.__setitem__("hours", s["hours"][:23]), "23 hours"),
        (lambda s: s.__setitem__("hours", s["hours"] + [dict(s["hours"][0])]), "25 hours"),
        (lambda s: s["hours"][5].__setitem__("hour", 6), "duplicate hour"),
        (lambda s: s["hours"][5].__setitem__("hour", 24), "hour out of range"),
        (lambda s: s["hours"][5].__setitem__("hour", -1), "negative hour"),
        (lambda s: s["hours"][3].__setitem__("demand_kwh", -5), "negative demand"),
        (lambda s: s["hours"][3].__setitem__("solar_kwh", -1), "negative solar"),
        (lambda s: s["hours"][3].__setitem__("tariff_bdt_per_kwh", -2), "negative tariff"),
        (lambda s: s["hours"][3].pop("demand_kwh"), "missing field"),
        (lambda s: s.__setitem__("scenario_id", ""), "blank scenario id"),
        (lambda s: s.__setitem__("operator_notes", []), "no notes"),
        (lambda s: s.__setitem__("operator_notes", ["a", "b", "c", "d"]), "four notes"),
        (lambda s: s["battery"].__setitem__("minimum_energy_kwh", 400), "minimum above initial"),
        (lambda s: s["battery"].__setitem__("initial_energy_kwh", 900), "initial above capacity"),
        (lambda s: s["battery"].__setitem__("capacity_kwh", 0), "zero capacity"),
        (lambda s: s["battery"].__setitem__("max_charge_kwh_per_hour", -10), "negative charge rate"),
        (lambda s: s.pop("battery"), "missing battery"),
    ],
)
def test_invalid_scenarios_return_422(client, mutate, reason):
    scenario = make_scenario()
    mutate(scenario)
    response = client.post("/optimize-energy", json=scenario)
    assert response.status_code == 422, f"{reason} should be rejected"
    assert "detail" in response.json()


def test_malformed_json_is_rejected(client):
    response = client.post(
        "/optimize-energy",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_unknown_fields_are_rejected(client):
    scenario = make_scenario()
    scenario["unexpected_field"] = True
    assert client.post("/optimize-energy", json=scenario).status_code == 422


def test_hours_may_arrive_out_of_order(client):
    scenario = make_scenario()
    scenario["hours"] = list(reversed(scenario["hours"]))
    body = client.post("/optimize-energy", json=scenario).json()
    assert [entry["hour"] for entry in body["hourly_plan"]] == list(range(24))


def test_error_bodies_never_leak_internals(client_factory):
    from app.exceptions import LLMUnavailableError
    from tests.helpers import ExplodingInterpreter

    client = client_factory(interpreter=ExplodingInterpreter(LLMUnavailableError()))
    response = client.post("/optimize-energy", json=make_scenario())

    assert response.status_code == 503
    body = response.json()
    assert set(body) == {"detail"}
    assert "Traceback" not in body["detail"]
    assert "api_key" not in body["detail"].lower()
