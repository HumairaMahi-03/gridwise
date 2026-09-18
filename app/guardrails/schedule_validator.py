"""Independent validation of a finished schedule.

This layer assumes the optimizer may be wrong. It re-derives every constraint
from the original scenario and the compiled directives, then checks the plan
that is about to be returned to the client — including the rounded, published
numbers rather than only the solver's internal floats.

It returns a list of human-readable violations; an empty list means the plan is
safe to return.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from app.config import HOURS_IN_DAY
from app.models.directives import DirectiveSet
from app.models.request import BatteryConfig
from app.models.response import BatteryAction, HourPlan
from app.optimizer.model import HourBounds


def _exceeds(value: float, limit: float, tolerance: float) -> bool:
    return value > limit + tolerance


def _below(value: float, limit: float, tolerance: float) -> bool:
    return value < limit - tolerance


def validate_final_plan(
    plan: Sequence[HourPlan],
    bounds: Sequence[HourBounds],
    battery: BatteryConfig,
    directives: Optional[DirectiveSet] = None,
    tolerance: float = 1e-6,
) -> List[str]:
    """Check a completed 24-hour plan. Returns every violation found."""
    violations: List[str] = []

    if len(plan) != HOURS_IN_DAY:
        violations.append(f"plan has {len(plan)} hours, expected {HOURS_IN_DAY}")
        return violations
    if [entry.hour for entry in plan] != list(range(HOURS_IN_DAY)):
        violations.append("plan hours are not exactly 0-23 in ascending order")
        return violations
    if len(bounds) != HOURS_IN_DAY:
        violations.append("bounds do not cover all 24 hours")
        return violations

    bounds_by_hour = {bound.hour: bound for bound in bounds}
    previous_energy = battery.initial_energy_kwh

    for entry in plan:
        hour = entry.hour
        bound = bounds_by_hour[hour]

        charge, discharge = _split_battery_movement(entry, violations)

        # 1. Energy balance.
        supply = entry.grid_kwh + entry.solar_used_kwh + discharge
        consumption = bound.demand_kwh + charge
        if abs(supply - consumption) > tolerance:
            violations.append(
                f"hour {hour}: energy balance off by {supply - consumption:+.6f} kWh "
                f"(supply {supply:.6f} vs consumption {consumption:.6f})"
            )

        # 2. Solar availability, after any solar_reduction directive.
        if _exceeds(entry.solar_used_kwh, bound.effective_solar_kwh, tolerance):
            violations.append(
                f"hour {hour}: solar_used {entry.solar_used_kwh:.6f} kWh exceeds "
                f"effective solar {bound.effective_solar_kwh:.6f} kWh"
            )

        # 3. Grid import.
        if _below(entry.grid_kwh, 0.0, tolerance):
            violations.append(f"hour {hour}: grid import is negative ({entry.grid_kwh:.6f} kWh)")
        if bound.max_grid_kwh is not None and _exceeds(entry.grid_kwh, bound.max_grid_kwh, tolerance):
            violations.append(
                f"hour {hour}: grid import {entry.grid_kwh:.6f} kWh exceeds the "
                f"{bound.max_grid_kwh:.6f} kWh cap"
            )

        # 4. Battery rate limits, including no-charge / no-discharge windows.
        if _exceeds(charge, bound.max_charge_kwh, tolerance):
            violations.append(
                f"hour {hour}: charge {charge:.6f} kWh exceeds the limit of "
                f"{bound.max_charge_kwh:.6f} kWh"
            )
        if _exceeds(discharge, bound.max_discharge_kwh, tolerance):
            violations.append(
                f"hour {hour}: discharge {discharge:.6f} kWh exceeds the limit of "
                f"{bound.max_discharge_kwh:.6f} kWh"
            )

        # 5. Battery state transition.
        expected_energy = previous_energy + charge - discharge
        if abs(entry.battery_energy_after_kwh - expected_energy) > tolerance:
            violations.append(
                f"hour {hour}: stored energy {entry.battery_energy_after_kwh:.6f} kWh does not "
                f"follow from the previous state ({expected_energy:.6f} kWh expected)"
            )

        # 6. State-of-charge window, including minimum_battery_reserve.
        if _below(entry.battery_energy_after_kwh, bound.energy_min_kwh, tolerance):
            violations.append(
                f"hour {hour}: stored energy {entry.battery_energy_after_kwh:.6f} kWh is below "
                f"the required minimum of {bound.energy_min_kwh:.6f} kWh"
            )
        if _exceeds(entry.battery_energy_after_kwh, battery.capacity_kwh, tolerance):
            violations.append(
                f"hour {hour}: stored energy {entry.battery_energy_after_kwh:.6f} kWh exceeds "
                f"capacity {battery.capacity_kwh:.6f} kWh"
            )

        previous_energy = entry.battery_energy_after_kwh

    # 7. Day-end neutrality.
    final_energy = plan[HOURS_IN_DAY - 1].battery_energy_after_kwh
    if abs(final_energy - battery.initial_energy_kwh) > tolerance:
        violations.append(
            f"day-end battery energy {final_energy:.6f} kWh does not match the initial "
            f"{battery.initial_energy_kwh:.6f} kWh"
        )

    # 8. Directive-level re-check, derived independently of the bounds above.
    if directives is not None:
        violations.extend(_check_directives_directly(plan, directives, tolerance))

    return violations


def _split_battery_movement(entry: HourPlan, violations: List[str]) -> tuple[float, float]:
    """Decode ``battery_action``/``battery_kwh`` into charge and discharge."""
    if entry.battery_action is BatteryAction.CHARGE:
        if entry.battery_kwh <= 0:
            violations.append(f"hour {entry.hour}: action 'charge' with battery_kwh {entry.battery_kwh}")
        return entry.battery_kwh, 0.0
    if entry.battery_action is BatteryAction.DISCHARGE:
        if entry.battery_kwh <= 0:
            violations.append(
                f"hour {entry.hour}: action 'discharge' with battery_kwh {entry.battery_kwh}"
            )
        return 0.0, entry.battery_kwh
    if entry.battery_kwh != 0:
        violations.append(f"hour {entry.hour}: action 'idle' with battery_kwh {entry.battery_kwh}")
    return 0.0, 0.0


def _check_directives_directly(
    plan: Sequence[HourPlan],
    directives: DirectiveSet,
    tolerance: float,
) -> List[str]:
    """Re-assert each compiled directive straight against the published plan."""
    violations: List[str] = []
    by_hour = {entry.hour: entry for entry in plan}

    for hour in sorted(directives.no_charge_hours):
        entry = by_hour[hour]
        if entry.battery_action is BatteryAction.CHARGE and entry.battery_kwh > tolerance:
            violations.append(f"hour {hour}: charging during a no_charge_window")

    for hour in sorted(directives.no_discharge_hours):
        entry = by_hour[hour]
        if entry.battery_action is BatteryAction.DISCHARGE and entry.battery_kwh > tolerance:
            violations.append(f"hour {hour}: discharging during a no_discharge_window")

    for hour, reserve in sorted(directives.min_reserve.items()):
        entry = by_hour[hour]
        if _below(entry.battery_energy_after_kwh, reserve, tolerance):
            violations.append(
                f"hour {hour}: stored energy {entry.battery_energy_after_kwh:.6f} kWh breaches "
                f"the {reserve:.6f} kWh reserve directive"
            )

    for hour, cap in sorted(directives.max_grid.items()):
        entry = by_hour[hour]
        if _exceeds(entry.grid_kwh, cap, tolerance):
            violations.append(
                f"hour {hour}: grid import {entry.grid_kwh:.6f} kWh breaches the "
                f"{cap:.6f} kWh max_grid directive"
            )

    return violations
