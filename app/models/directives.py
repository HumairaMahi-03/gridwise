"""Directive vocabulary shared by the LLM layer, guardrails and optimizer.

Two representations exist on purpose:

* ``NoteInterpretation`` is the *wire* form. It mirrors one operator note, is
  echoed back to the client, and keeps ``structured_adjustment`` as a plain
  mapping so that a malformed LLM payload can be inspected rather than crash
  parsing.
* ``DirectiveSet`` is the *compiled* form. Guardrails turn a list of validated
  interpretations into per-hour constraints that the optimizer consumes
  directly. Compilation is where overlapping directives are merged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class DirectiveType(str, Enum):
    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


#: Directive types that constrain the optimization problem.
ACTIVE_DIRECTIVE_TYPES = frozenset(t for t in DirectiveType if t is not DirectiveType.NO_OP)

#: Required keys of ``structured_adjustment`` for each active directive type.
REQUIRED_ADJUSTMENT_FIELDS: Dict[DirectiveType, frozenset] = {
    DirectiveType.SOLAR_REDUCTION: frozenset({"hours", "factor"}),
    DirectiveType.MINIMUM_BATTERY_RESERVE: frozenset({"hours", "minimum_energy_kwh"}),
    DirectiveType.NO_CHARGE_WINDOW: frozenset({"hours"}),
    DirectiveType.NO_DISCHARGE_WINDOW: frozenset({"hours"}),
    DirectiveType.MAX_GRID_WINDOW: frozenset({"hours", "max_grid_kwh"}),
}


class NoteInterpretation(BaseModel):
    """One interpreted operator note, in the form returned to the client."""

    model_config = ConfigDict(extra="forbid")

    note_index: int = Field(..., ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[Dict[str, Any]] = None
    explanation: str = ""

    @classmethod
    def no_op(cls, note_index: int, explanation: str) -> "NoteInterpretation":
        return cls(
            note_index=note_index,
            applies=False,
            directive_type=DirectiveType.NO_OP,
            structured_adjustment=None,
            explanation=explanation,
        )


@dataclass
class DirectiveSet:
    """Per-hour constraints compiled from all validated interpretations.

    Merge rules when several directives touch the same hour (documented in the
    README): the strictest requirement always wins.

    * ``solar_factor``  -> smallest factor (least solar available)
    * ``min_reserve``   -> largest reserve (highest battery floor)
    * ``max_grid``      -> smallest cap (tightest grid ceiling)
    * no-charge / no-discharge are simple set unions
    """

    solar_factor: Dict[int, float] = field(default_factory=dict)
    min_reserve: Dict[int, float] = field(default_factory=dict)
    max_grid: Dict[int, float] = field(default_factory=dict)
    no_charge_hours: set = field(default_factory=set)
    no_discharge_hours: set = field(default_factory=set)

    def add_solar_reduction(self, hour: int, factor: float) -> None:
        current = self.solar_factor.get(hour)
        self.solar_factor[hour] = factor if current is None else min(current, factor)

    def add_min_reserve(self, hour: int, minimum_energy_kwh: float) -> None:
        current = self.min_reserve.get(hour)
        self.min_reserve[hour] = (
            minimum_energy_kwh if current is None else max(current, minimum_energy_kwh)
        )

    def add_max_grid(self, hour: int, max_grid_kwh: float) -> None:
        current = self.max_grid.get(hour)
        self.max_grid[hour] = max_grid_kwh if current is None else min(current, max_grid_kwh)

    def is_empty(self) -> bool:
        return not (
            self.solar_factor
            or self.min_reserve
            or self.max_grid
            or self.no_charge_hours
            or self.no_discharge_hours
        )
