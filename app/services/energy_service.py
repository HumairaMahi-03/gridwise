"""End-to-end orchestration of the optimization pipeline.

    request -> LLM interpretation -> deterministic guardrails
            -> directive compilation -> MILP optimizer
            -> independent schedule validation -> totals -> response

Each stage is a plain function call on a validated, typed object, so a failure
at any stage produces a clean domain exception rather than a partial response.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from app.config import ROUNDED_TOLERANCE, SOLVER_TOLERANCE, Settings, get_settings
from app.exceptions import ScheduleValidationError
from app.guardrails.schedule_validator import validate_final_plan
from app.guardrails.validator import (
    compile_directives,
    salvage_interpretations,
    validate_interpretations,
)
from app.llm.interpreter import NoteInterpreter
from app.models.directives import DirectiveSet, NoteInterpretation
from app.models.request import OptimizationRequest
from app.models.response import HourPlan, OptimizationResponse
from app.optimizer.energy_optimizer import solve_schedule
from app.utils.calculations import (
    build_hour_plans,
    build_plan_summary,
    peak_grid_kwh,
    total_cost_bdt,
    total_grid_kwh,
)

logger = logging.getLogger(__name__)


def build_scenario_context(request: OptimizationRequest) -> Dict[str, float]:
    """Scenario facts the interpreter may need to resolve proportional wording.

    Only battery ratings are shared. Demand, solar and tariff data are withheld
    so the model has nothing to schedule with, by construction.
    """
    battery = request.battery
    return {
        "battery_capacity_kwh": battery.capacity_kwh,
        "battery_initial_energy_kwh": battery.initial_energy_kwh,
        "battery_minimum_energy_kwh": battery.minimum_energy_kwh,
        "battery_max_charge_kwh_per_hour": battery.max_charge_kwh_per_hour,
        "battery_max_discharge_kwh_per_hour": battery.max_discharge_kwh_per_hour,
    }


class EnergyService:
    """Runs the optimization pipeline for a single scenario."""

    def __init__(self, interpreter: NoteInterpreter, settings: Settings | None = None) -> None:
        self._interpreter = interpreter
        self._settings = settings or get_settings()

    def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        interpretations = self.interpret_notes(request)
        directives = compile_directives(interpretations, request.battery)

        logger.info(
            "scenario=%s notes=%d applied_directives=%d",
            request.scenario_id,
            len(request.operator_notes),
            sum(1 for item in interpretations if item.applies),
        )

        schedule = solve_schedule(request, directives, self._settings)

        # Pass 1: the solver's full-precision answer, at solver tolerance.
        raw_plan = build_hour_plans(schedule, rounded=False)
        self._assert_valid(raw_plan, schedule, request, directives, SOLVER_TOLERANCE)

        # Pass 2: the exact numbers about to be published, at the tolerance
        # that three-decimal rounding can introduce. A plan ships only if it
        # survives both passes.
        published_plan = build_hour_plans(schedule, rounded=True)
        self._assert_valid(published_plan, schedule, request, directives, ROUNDED_TOLERANCE)

        return OptimizationResponse(
            scenario_id=request.scenario_id,
            directive_interpretation=interpretations,
            hourly_plan=published_plan,
            total_grid_kwh=total_grid_kwh(published_plan),
            total_cost_bdt=total_cost_bdt(published_plan, request),
            peak_grid_kwh=peak_grid_kwh(published_plan),
            plan_summary=build_plan_summary(
                published_plan, schedule.bounds, request.battery, directives, interpretations
            ),
        )

    # ------------------------------------------------------------------ stages

    def interpret_notes(self, request: OptimizationRequest) -> List[NoteInterpretation]:
        """Interpret the operator notes, with exactly one repair attempt.

        LLM call -> guardrails -> (on failure) one repair round -> guardrails
        -> (on failure) salvage, which downgrades only notes the model itself
        declared irrelevant and otherwise refuses to produce a schedule.
        """
        notes = request.operator_notes
        battery = request.battery
        context = build_scenario_context(request)

        raw_payload = self._interpreter.interpret(notes, context)
        result = validate_interpretations(raw_payload, notes, battery)
        if result.ok:
            return result.interpretations

        logger.warning(
            "Guardrails rejected the initial interpretation for scenario %s: %s",
            request.scenario_id,
            "; ".join(result.problems),
        )

        repaired_payload = self._interpreter.repair(notes, raw_payload, result.problems, context)
        repaired = validate_interpretations(repaired_payload, notes, battery)
        if repaired.ok:
            return repaired.interpretations

        logger.warning(
            "Repair attempt still invalid for scenario %s: %s",
            request.scenario_id,
            "; ".join(repaired.problems),
        )
        return salvage_interpretations(repaired_payload, notes, battery)

    # ------------------------------------------------------------------ checks

    @staticmethod
    def _assert_valid(
        plan: Sequence[HourPlan],
        schedule,
        request: OptimizationRequest,
        directives: DirectiveSet,
        tolerance: float,
    ) -> None:
        violations = validate_final_plan(
            plan=plan,
            bounds=schedule.bounds,
            battery=request.battery,
            directives=directives,
            tolerance=tolerance,
        )
        if violations:
            logger.error(
                "Schedule for scenario %s failed validation at tolerance %s: %s",
                request.scenario_id,
                tolerance,
                "; ".join(violations[:10]),
            )
            raise ScheduleValidationError(violations)
