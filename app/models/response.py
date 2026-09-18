"""Response models for POST /optimize-energy."""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from app.models.directives import NoteInterpretation


class BatteryAction(str, Enum):
    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"


class HourPlan(BaseModel):
    """The plan for a single hour.

    ``battery_kwh`` is the magnitude of the battery movement for the hour: the
    amount charged when ``battery_action`` is ``charge``, the amount discharged
    when it is ``discharge``, and 0 when ``idle``.
    """

    model_config = ConfigDict(extra="forbid")

    hour: int = Field(..., ge=0, le=23)
    grid_kwh: float = Field(..., ge=0)
    solar_used_kwh: float = Field(..., ge=0)
    battery_action: BatteryAction
    battery_kwh: float = Field(..., ge=0)
    battery_energy_after_kwh: float = Field(..., ge=0)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


class OptimizationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    directive_interpretation: List[NoteInterpretation]
    hourly_plan: List[HourPlan] = Field(..., min_length=24, max_length=24)
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class ErrorResponse(BaseModel):
    """Client-safe error envelope; never carries internals or credentials."""

    model_config = ConfigDict(extra="forbid")

    detail: str
