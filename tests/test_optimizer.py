"""LP model, constraint derivation and post-solve validation."""

from __future__ import annotations

import pytest

from app.exceptions import InfeasibleScheduleError
from app.guardrails.schedule_validator import validate_final_plan
from app.guardrails.validator import compile_directives, validate_interpretations
from app.models.request import OptimizationRequest
from app.models.response import BatteryAction
from app.optimizer.energy_optimizer import solve_schedule
from app.optimizer.model import derive_hour_bounds
from app.utils.calculations import build_hour_plans
from tests.helpers import DEMAND, SOLAR, TARIFFS, directive_payload, make_scenario


def build(notes=None, payload=None, **battery_overrides):
    """Compile directives and solve, returning (request, directives, schedule)."""
    scenario = make_scenario(notes=notes or ["note"], **battery_overrides)
    request = OptimizationRequest(**scenario)

    interpretations = []
    if payload:
        result = validate_interpretations(payload, request.operator_notes, request.battery)
        assert result.ok, result.problems
        interpretations = result.interpretations

    directives = compile_directives(interpretations, request.battery)
    return request, directives, solve_schedule(request, directives)


def plan_of(schedule):
    return {entry.hour: entry for entry in build_hour_plans(schedule, rounded=False)}


# ------------------------------------------------------------ bound derivation


def test_bounds_apply_solar_factor_and_windows():
    payload = [
        directive_payload(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2}),
        directive_payload(1, "no_charge_window", {"hours": [14, 15]}),
        directive_payload(2, "minimum_battery_reserve", {"hours": [18], "minimum_energy_kwh": 150}),
    ]
    request, directives, _ = build(notes=["a", "b", "c"], payload=payload)
    bounds = derive_hour_bounds(request, directives)

    assert bounds[13].effective_solar_kwh == pytest.approx(SOLAR[13] * 0.2)
    assert bounds[12].effective_solar_kwh == pytest.approx(SOLAR[12])
    assert bounds[14].max_charge_kwh == 0
    assert bounds[15].max_charge_kwh == 0
    assert bounds[13].max_charge_kwh == 100
    assert bounds[18].energy_min_kwh == 150
    assert bounds[17].energy_min_kwh == 50  # the battery's own permanent floor


def test_no_discharge_window_zeroes_the_discharge_bound():
    payload = [directive_payload(0, "no_discharge_window", {"hours": [17, 18]})]
    request, directives, _ = build(payload=payload)
    bounds = derive_hour_bounds(request, directives)

    assert bounds[17].max_discharge_kwh == 0
    assert bounds[18].max_discharge_kwh == 0
    assert bounds[19].max_discharge_kwh == 100


def test_max_grid_bound_is_recorded_only_where_directed():
    payload = [directive_payload(0, "max_grid_window", {"hours": [19], "max_grid_kwh": 250})]
    request, directives, _ = build(payload=payload)
    bounds = derive_hour_bounds(request, directives)

    assert bounds[19].max_grid_kwh == 250
    assert bounds[18].max_grid_kwh is None


# ------------------------------------------------------------------- solving


def test_baseline_schedule_is_valid():
    request, directives, schedule = build()
    plan = build_hour_plans(schedule, rounded=False)

    assert len(plan) == 24
    violations = validate_final_plan(plan, schedule.bounds, request.battery, directives, 1e-6)
    assert violations == [], violations


def test_energy_balance_holds_every_hour():
    _, _, schedule = build()
    for hour in schedule.hours:
        supply = hour.grid_kwh + hour.solar_used_kwh + hour.discharge_kwh
        consumption = DEMAND[hour.hour] + hour.charge_kwh
        assert supply == pytest.approx(consumption, abs=1e-6)


def test_day_ends_on_the_opening_state_of_charge():
    request, _, schedule = build()
    assert schedule.hours[23].energy_after_kwh == pytest.approx(
        request.battery.initial_energy_kwh, abs=1e-6
    )


def test_battery_stays_inside_its_limits():
    _, _, schedule = build()
    for hour in schedule.hours:
        assert 50 - 1e-6 <= hour.energy_after_kwh <= 500 + 1e-6
        assert hour.charge_kwh <= 100 + 1e-6
        assert hour.discharge_kwh <= 100 + 1e-6


def test_charging_and_discharging_never_happen_together():
    _, _, schedule = build()
    for hour in schedule.hours:
        assert hour.charge_kwh < 1e-6 or hour.discharge_kwh < 1e-6


def test_battery_charges_cheaply_and_discharges_expensively():
    _, _, schedule = build()
    charge_hours = [h.hour for h in schedule.hours if h.charge_kwh > 1e-6]
    discharge_hours = [h.hour for h in schedule.hours if h.discharge_kwh > 1e-6]

    assert charge_hours and discharge_hours
    average_charge = sum(TARIFFS[h] for h in charge_hours) / len(charge_hours)
    average_discharge = sum(TARIFFS[h] for h in discharge_hours) / len(discharge_hours)
    assert average_discharge > average_charge


def test_using_the_battery_beats_leaving_it_idle():
    from app.utils.calculations import total_cost_bdt

    request, _, schedule = build()
    idle_request, _, idle_schedule = build(max_charge_kwh_per_hour=0, max_discharge_kwh_per_hour=0)

    active_cost = total_cost_bdt(build_hour_plans(schedule), request)
    idle_cost = total_cost_bdt(build_hour_plans(idle_schedule), idle_request)
    assert active_cost < idle_cost


def test_solar_reduction_limits_solar_use_and_raises_cost():
    from app.utils.calculations import total_cost_bdt

    payload = [directive_payload(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2})]
    request, _, reduced = build(payload=payload)
    baseline_request, _, baseline = build()

    reduced_plan = plan_of(reduced)
    assert reduced_plan[13].solar_used_kwh <= SOLAR[13] * 0.2 + 1e-6
    assert reduced_plan[14].solar_used_kwh <= SOLAR[14] * 0.2 + 1e-6
    assert total_cost_bdt(build_hour_plans(reduced), request) > total_cost_bdt(
        build_hour_plans(baseline), baseline_request
    )


def test_zero_factor_removes_solar_entirely():
    payload = [directive_payload(0, "solar_reduction", {"hours": [12], "factor": 0.0})]
    _, _, schedule = build(payload=payload)
    assert plan_of(schedule)[12].solar_used_kwh == pytest.approx(0.0, abs=1e-6)


def test_no_charge_window_is_respected():
    payload = [directive_payload(0, "no_charge_window", {"hours": [2, 3, 4]})]
    _, _, schedule = build(payload=payload)
    for hour in (2, 3, 4):
        assert schedule.hours[hour].charge_kwh == pytest.approx(0.0, abs=1e-6)


def test_no_discharge_window_is_respected():
    payload = [directive_payload(0, "no_discharge_window", {"hours": [17, 18]})]
    _, _, schedule = build(payload=payload)
    for hour in (17, 18):
        assert schedule.hours[hour].discharge_kwh == pytest.approx(0.0, abs=1e-6)


def test_minimum_reserve_is_respected():
    payload = [
        directive_payload(0, "minimum_battery_reserve", {"hours": [18, 19, 20], "minimum_energy_kwh": 260})
    ]
    _, _, schedule = build(payload=payload)
    for hour in (18, 19, 20):
        assert schedule.hours[hour].energy_after_kwh >= 260 - 1e-6


def test_max_grid_window_is_respected():
    payload = [
        directive_payload(0, "max_grid_window", {"hours": [18, 19, 20], "max_grid_kwh": 250})
    ]
    _, _, schedule = build(payload=payload)
    for hour in (18, 19, 20):
        assert schedule.hours[hour].grid_kwh <= 250 + 1e-6


def test_multiple_simultaneous_directives_all_hold():
    payload = [
        directive_payload(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2}),
        directive_payload(1, "no_charge_window", {"hours": [14, 15]}),
        directive_payload(2, "max_grid_window", {"hours": [18, 19, 20], "max_grid_kwh": 260}),
    ]
    request, directives, schedule = build(notes=["a", "b", "c"], payload=payload)
    plan = build_hour_plans(schedule, rounded=False)

    assert validate_final_plan(plan, schedule.bounds, request.battery, directives, 1e-6) == []

    by_hour = {entry.hour: entry for entry in plan}
    assert by_hour[13].solar_used_kwh <= SOLAR[13] * 0.2 + 1e-6
    assert by_hour[14].battery_action is not BatteryAction.CHARGE
    assert by_hour[15].battery_action is not BatteryAction.CHARGE
    for hour in (18, 19, 20):
        assert by_hour[hour].grid_kwh <= 260 + 1e-6
    assert by_hour[23].battery_energy_after_kwh == pytest.approx(200, abs=1e-6)


def test_directives_do_not_overwrite_one_another():
    payload = [
        directive_payload(0, "no_charge_window", {"hours": [2, 3]}),
        directive_payload(1, "no_discharge_window", {"hours": [18, 19]}),
    ]
    _, _, schedule = build(notes=["a", "b"], payload=payload)
    for hour in (2, 3):
        assert schedule.hours[hour].charge_kwh == pytest.approx(0.0, abs=1e-6)
    for hour in (18, 19):
        assert schedule.hours[hour].discharge_kwh == pytest.approx(0.0, abs=1e-6)


def test_infeasible_grid_cap_raises():
    # Hour 19 needs 300 kWh, has no solar, and the battery can supply only 100.
    payload = [directive_payload(0, "max_grid_window", {"hours": [19], "max_grid_kwh": 0})]
    with pytest.raises(InfeasibleScheduleError):
        build(payload=payload)


def test_infeasible_reserve_raises():
    payload = [
        directive_payload(
            0, "minimum_battery_reserve", {"hours": list(range(24)), "minimum_energy_kwh": 500}
        )
    ]
    with pytest.raises(InfeasibleScheduleError):
        build(payload=payload)


# ----------------------------------------------------------- final validator


@pytest.mark.parametrize(
    "tamper, expected",
    [
        (lambda plan: setattr(plan[5], "grid_kwh", plan[5].grid_kwh + 40), "energy balance"),
        (lambda plan: setattr(plan[8], "battery_energy_after_kwh", plan[8].battery_energy_after_kwh + 25),
         "does not follow"),
        (lambda plan: setattr(plan[23], "battery_energy_after_kwh", 123.0), "day-end"),
        (lambda plan: setattr(plan[10], "solar_used_kwh", plan[10].solar_used_kwh + 500),
         "exceeds effective solar"),
        (lambda plan: setattr(plan[10], "battery_energy_after_kwh", 999.0), "exceeds capacity"),
    ],
)
def test_validator_rejects_tampered_plans(tamper, expected):
    request, directives, schedule = build()
    plan = build_hour_plans(schedule, rounded=False)
    tamper(plan)

    violations = validate_final_plan(plan, schedule.bounds, request.battery, directives, 1e-6)
    assert any(expected in violation for violation in violations), violations


def test_validator_rejects_an_idle_hour_that_moves_energy():
    request, directives, schedule = build()
    plan = build_hour_plans(schedule, rounded=False)
    idle = next(entry for entry in plan if entry.battery_action is BatteryAction.IDLE)
    idle.battery_kwh = 10.0

    violations = validate_final_plan(plan, schedule.bounds, request.battery, directives, 1e-6)
    assert any("idle" in violation for violation in violations), violations


def test_validator_rejects_a_plan_with_the_wrong_number_of_hours():
    request, directives, schedule = build()
    plan = build_hour_plans(schedule, rounded=False)[:23]

    violations = validate_final_plan(plan, schedule.bounds, request.battery, directives, 1e-6)
    assert violations and "expected 24" in violations[0]


def test_validator_rejects_directive_breaches_directly():
    payload = [directive_payload(0, "max_grid_window", {"hours": [19], "max_grid_kwh": 250})]
    request, directives, schedule = build(payload=payload)
    plan = build_hour_plans(schedule, rounded=False)

    by_hour = {entry.hour: entry for entry in plan}
    by_hour[19].grid_kwh = 400.0

    violations = validate_final_plan(plan, schedule.bounds, request.battery, directives, 1e-6)
    assert any("max_grid" in violation or "cap" in violation for violation in violations), violations
