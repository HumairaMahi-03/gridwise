"""Mathematical scheduling model for the 24-hour campus energy problem.

Formulated as a mixed-integer linear program and solved with PuLP/CBC.

Decision variables, for each hour ``h`` in 0..23:

===================  ==========================================================
``grid[h]``          energy imported from the grid, kWh
``solar_used[h]``    solar generation actually consumed, kWh
``charge[h]``        energy pushed into the battery, kWh
``discharge[h]``     energy drawn from the battery, kWh
``energy[h]``        battery state of charge at the *end* of the hour, kWh
``mode[h]``          binary; 1 permits charging, 0 permits discharging
===================  ==========================================================

Objective: minimise total grid cost, ``sum(grid[h] * tariff[h])``. A negligible
per-kWh throughput term breaks ties between equal-cost schedules so the battery
is not cycled for no reason; it is orders of magnitude too small to change
which grid schedule is cheapest.

The per-hour constraint derivation lives in :mod:`app.optimizer.model`, which
is free of solver types so it can be verified independently of PuLP.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pulp

from app.config import HOURS_IN_DAY, Settings, get_settings
from app.exceptions import InfeasibleScheduleError, OptimizerError
from app.models.directives import DirectiveSet
from app.models.request import OptimizationRequest
from app.optimizer.model import HourBounds, HourSolution, Schedule, derive_hour_bounds

logger = logging.getLogger(__name__)


def solve_schedule(
    request: OptimizationRequest,
    directives: DirectiveSet,
    settings: Settings | None = None,
) -> Schedule:
    """Build and solve the MILP, returning the raw optimal schedule."""
    settings = settings or get_settings()
    bounds = derive_hour_bounds(request, directives)
    battery = request.battery

    problem = pulp.LpProblem("campus_energy_schedule", pulp.LpMinimize)

    grid: Dict[int, pulp.LpVariable] = {}
    solar_used: Dict[int, pulp.LpVariable] = {}
    charge: Dict[int, pulp.LpVariable] = {}
    discharge: Dict[int, pulp.LpVariable] = {}
    energy: Dict[int, pulp.LpVariable] = {}

    for bound in bounds:
        h = bound.hour
        grid[h] = pulp.LpVariable(f"grid_{h}", lowBound=0, upBound=bound.max_grid_kwh)
        solar_used[h] = pulp.LpVariable(f"solar_used_{h}", lowBound=0, upBound=bound.effective_solar_kwh)
        charge[h] = pulp.LpVariable(f"charge_{h}", lowBound=0, upBound=bound.max_charge_kwh)
        discharge[h] = pulp.LpVariable(f"discharge_{h}", lowBound=0, upBound=bound.max_discharge_kwh)
        energy[h] = pulp.LpVariable(
            f"energy_{h}", lowBound=bound.energy_min_kwh, upBound=bound.energy_max_kwh
        )

    # Objective: grid cost, with a tie-break on battery throughput.
    penalty = settings.battery_cycle_penalty
    problem += (
        pulp.lpSum(grid[b.hour] * b.tariff_bdt_per_kwh for b in bounds)
        + penalty * pulp.lpSum(charge[b.hour] + discharge[b.hour] for b in bounds)
    ), "total_grid_cost_bdt"

    for bound in bounds:
        h = bound.hour

        # Energy balance: supply equals consumption, every hour.
        problem += (
            grid[h] + solar_used[h] + discharge[h] == bound.demand_kwh + charge[h],
            f"energy_balance_{h}",
        )

        # Battery state transition.
        previous = battery.initial_energy_kwh if h == 0 else energy[h - 1]
        problem += (
            energy[h] == previous + charge[h] - discharge[h],
            f"battery_state_{h}",
        )

        # Mutual exclusion of charging and discharging. The binary is only
        # introduced where both directions are actually possible, which keeps
        # most hours in the LP relaxation and the solve fast.
        if bound.max_charge_kwh > 0 and bound.max_discharge_kwh > 0:
            mode = pulp.LpVariable(f"mode_{h}", cat=pulp.LpBinary)
            problem += charge[h] <= bound.max_charge_kwh * mode, f"charge_mode_{h}"
            problem += discharge[h] <= bound.max_discharge_kwh * (1 - mode), f"discharge_mode_{h}"

    # Day-end neutrality: the day must close on the opening state of charge.
    problem += (
        energy[HOURS_IN_DAY - 1] == battery.initial_energy_kwh,
        "day_end_neutrality",
    )

    try:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=settings.solver_time_limit_seconds)
        status = problem.solve(solver)
    except pulp.PulpSolverError as exc:
        logger.exception("Solver invocation failed")
        raise OptimizerError() from exc

    status_name = pulp.LpStatus.get(status, "Unknown")
    if status_name == "Infeasible":
        logger.info("Scenario %s is infeasible under the supplied directives", request.scenario_id)
        raise InfeasibleScheduleError()
    if status_name != "Optimal":
        logger.error("Solver finished with non-optimal status: %s", status_name)
        raise OptimizerError()

    hours: List[HourSolution] = []
    for bound in bounds:
        h = bound.hour
        hours.append(
            HourSolution(
                hour=h,
                grid_kwh=_value(grid[h]),
                solar_used_kwh=_value(solar_used[h]),
                charge_kwh=_value(charge[h]),
                discharge_kwh=_value(discharge[h]),
                energy_after_kwh=_value(energy[h]),
            )
        )

    return Schedule(hours=hours, bounds=bounds)


def _value(variable: pulp.LpVariable) -> float:
    """Read a solved variable, snapping solver noise to exact zero."""
    raw = pulp.value(variable)
    if raw is None:
        raise OptimizerError()
    return 0.0 if abs(raw) < 1e-9 else float(raw)
