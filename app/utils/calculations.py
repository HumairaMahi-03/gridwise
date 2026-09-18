"""Turning a solved schedule into the published plan, totals and summary.

Totals are always recomputed from the final, validated, rounded plan rather
than read back from the solver's objective, so the numbers a client sees are
guaranteed to be consistent with the hourly rows next to them.
"""

from __future__ import annotations

from typing import List, Sequence

from app.config import ROUND_DP
from app.models.directives import DirectiveSet, NoteInterpretation
from app.models.request import BatteryConfig, OptimizationRequest
from app.models.response import BatteryAction, HourPlan
from app.optimizer.model import HourBounds, Schedule

#: Battery movements below this are solver noise, not a real action.
ACTION_EPSILON = 1e-6


def build_hour_plans(schedule: Schedule, rounded: bool = True) -> List[HourPlan]:
    """Convert raw solver output into client-facing hourly rows.

    Charging and discharging are netted before rounding. The MILP already
    forbids doing both in the same hour; netting makes that guarantee explicit
    in the output and cannot disturb either the energy balance or the state of
    charge, since only the difference enters both equations.

    With ``rounded=False`` the solver's full-precision values are kept, which is
    what the strict validation pass checks before any rounding is applied.
    """
    adjust = _round if rounded else float
    plans: List[HourPlan] = []

    for hour in schedule.hours:
        net = hour.charge_kwh - hour.discharge_kwh

        if net > ACTION_EPSILON:
            action, magnitude = BatteryAction.CHARGE, net
        elif net < -ACTION_EPSILON:
            action, magnitude = BatteryAction.DISCHARGE, -net
        else:
            action, magnitude = BatteryAction.IDLE, 0.0

        plans.append(
            HourPlan(
                hour=hour.hour,
                grid_kwh=adjust(max(hour.grid_kwh, 0.0)),
                solar_used_kwh=adjust(max(hour.solar_used_kwh, 0.0)),
                battery_action=action,
                battery_kwh=adjust(magnitude),
                battery_energy_after_kwh=adjust(max(hour.energy_after_kwh, 0.0)),
            )
        )

    return plans


def total_grid_kwh(plan: Sequence[HourPlan]) -> float:
    return _round(sum(entry.grid_kwh for entry in plan))


def total_cost_bdt(plan: Sequence[HourPlan], request: OptimizationRequest) -> float:
    tariffs = {record.hour: record.tariff_bdt_per_kwh for record in request.hours}
    return round(sum(entry.grid_kwh * tariffs[entry.hour] for entry in plan), 2)


def peak_grid_kwh(plan: Sequence[HourPlan]) -> float:
    return _round(max(entry.grid_kwh for entry in plan))


def build_plan_summary(
    plan: Sequence[HourPlan],
    bounds: Sequence[HourBounds],
    battery: BatteryConfig,
    directives: DirectiveSet,
    interpretations: Sequence[NoteInterpretation],
) -> str:
    """Describe the plan in plain language, computed from the plan itself.

    This is generated deterministically by the application. The LLM's only role
    in the pipeline is interpreting operator notes.
    """
    grid_total = total_grid_kwh(plan)
    peak = peak_grid_kwh(plan)
    peak_hour = max(plan, key=lambda entry: entry.grid_kwh).hour

    solar_total = _round(sum(entry.solar_used_kwh for entry in plan))
    available_solar = _round(sum(bound.effective_solar_kwh for bound in bounds))

    charge_hours = [e.hour for e in plan if e.battery_action is BatteryAction.CHARGE]
    discharge_hours = [e.hour for e in plan if e.battery_action is BatteryAction.DISCHARGE]
    charged = _round(sum(e.battery_kwh for e in plan if e.battery_action is BatteryAction.CHARGE))
    discharged = _round(
        sum(e.battery_kwh for e in plan if e.battery_action is BatteryAction.DISCHARGE)
    )

    parts: List[str] = [
        f"The plan imports {grid_total:g} kWh from the grid across the day, "
        f"peaking at {peak:g} kWh in hour {peak_hour}.",
        f"It consumes {solar_total:g} kWh of the {available_solar:g} kWh of solar "
        "generation available after operator adjustments.",
    ]

    if charge_hours or discharge_hours:
        parts.append(
            f"The battery charges {charged:g} kWh over {len(charge_hours)} hour(s) "
            f"and discharges {discharged:g} kWh over {len(discharge_hours)} hour(s), "
            f"returning to its opening level of {battery.initial_energy_kwh:g} kWh by hour 23."
        )
    else:
        parts.append(
            f"The battery stays idle all day and holds its opening level of "
            f"{battery.initial_energy_kwh:g} kWh."
        )

    constraints = _describe_constraints(directives)
    if constraints:
        parts.append("Operator directives applied: " + "; ".join(constraints) + ".")
    else:
        applied = any(interpretation.applies for interpretation in interpretations)
        parts.append(
            "No operator directives constrained the schedule."
            if not applied
            else "Operator directives were interpreted but imposed no binding limits."
        )

    return " ".join(parts)


def _describe_constraints(directives: DirectiveSet) -> List[str]:
    descriptions: List[str] = []
    if directives.solar_factor:
        hours = _format_hours(sorted(directives.solar_factor))
        descriptions.append(f"reduced solar in hour(s) {hours}")
    if directives.no_charge_hours:
        descriptions.append(f"no charging in hour(s) {_format_hours(sorted(directives.no_charge_hours))}")
    if directives.no_discharge_hours:
        descriptions.append(
            f"no discharging in hour(s) {_format_hours(sorted(directives.no_discharge_hours))}"
        )
    if directives.min_reserve:
        highest = max(directives.min_reserve.values())
        descriptions.append(
            f"a battery reserve of up to {highest:g} kWh in hour(s) "
            f"{_format_hours(sorted(directives.min_reserve))}"
        )
    if directives.max_grid:
        tightest = min(directives.max_grid.values())
        descriptions.append(
            f"a grid ceiling as low as {tightest:g} kWh in hour(s) "
            f"{_format_hours(sorted(directives.max_grid))}"
        )
    return descriptions


def _format_hours(hours: Sequence[int]) -> str:
    """Collapse consecutive hours into ranges: [13, 14, 15, 18] -> '13-15, 18'."""
    if not hours:
        return "-"
    groups: List[str] = []
    start = previous = hours[0]
    for hour in hours[1:]:
        if hour == previous + 1:
            previous = hour
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = hour
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(groups)


def _round(value: float) -> float:
    rounded = round(float(value), ROUND_DP)
    return 0.0 if rounded == 0 else rounded
