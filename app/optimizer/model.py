"""Solver-agnostic description of the scheduling problem.

These types carry no dependency on PuLP, which keeps the constraint derivation
and the post-solve validator testable in isolation from the solver backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.models.directives import DirectiveSet
from app.models.request import OptimizationRequest


@dataclass(frozen=True)
class HourBounds:
    """Everything the model needs to know about one hour."""

    hour: int
    demand_kwh: float
    tariff_bdt_per_kwh: float
    original_solar_kwh: float
    effective_solar_kwh: float
    max_charge_kwh: float
    max_discharge_kwh: float
    energy_min_kwh: float
    energy_max_kwh: float
    max_grid_kwh: Optional[float]


@dataclass
class HourSolution:
    """Raw solver values for one hour, before rounding or validation."""

    hour: int
    grid_kwh: float
    solar_used_kwh: float
    charge_kwh: float
    discharge_kwh: float
    energy_after_kwh: float


@dataclass
class Schedule:
    """A full 24-hour solution plus the bounds it was solved against."""

    hours: List[HourSolution]
    bounds: List[HourBounds]


def derive_hour_bounds(
    request: OptimizationRequest,
    directives: DirectiveSet,
) -> List[HourBounds]:
    """Fold scenario data and compiled directives into per-hour bounds.

    Pure function: no solver, no I/O. Directive merge rules were already applied
    when the :class:`DirectiveSet` was compiled, so each hour has at most one
    solar factor, one reserve floor and one grid ceiling.

    Note that the battery's own ``minimum_energy_kwh`` acts as a permanent floor
    on stored energy; a ``minimum_battery_reserve`` directive can raise that
    floor for specific hours but never lowers it.
    """
    battery = request.battery
    bounds: List[HourBounds] = []

    for record in request.hours_in_order():
        hour = record.hour
        factor = directives.solar_factor.get(hour, 1.0)
        effective_solar = record.solar_kwh * factor

        max_charge = 0.0 if hour in directives.no_charge_hours else battery.max_charge_kwh_per_hour
        max_discharge = (
            0.0 if hour in directives.no_discharge_hours else battery.max_discharge_kwh_per_hour
        )

        energy_min = max(battery.minimum_energy_kwh, directives.min_reserve.get(hour, 0.0))

        bounds.append(
            HourBounds(
                hour=hour,
                demand_kwh=record.demand_kwh,
                tariff_bdt_per_kwh=record.tariff_bdt_per_kwh,
                original_solar_kwh=record.solar_kwh,
                effective_solar_kwh=effective_solar,
                max_charge_kwh=max_charge,
                max_discharge_kwh=max_discharge,
                energy_min_kwh=min(energy_min, battery.capacity_kwh),
                energy_max_kwh=battery.capacity_kwh,
                max_grid_kwh=directives.max_grid.get(hour),
            )
        )

    return bounds
