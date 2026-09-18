"""Shared fixtures data and test doubles.

The LLM is always mocked at the :class:`NoteInterpreter` seam so the suite is
deterministic, offline and fast. No test ever contacts a provider.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.llm.interpreter import NoteInterpreter

TARIFFS = [7, 7, 7, 7, 7, 7, 9, 9, 11, 11, 11, 11, 12, 12, 12, 12, 14, 16, 16, 16, 14, 11, 9, 7]
SOLAR = [0, 0, 0, 0, 0, 0, 10, 40, 90, 140, 180, 210, 220, 210, 180, 140, 90, 40, 5, 0, 0, 0, 0, 0]
DEMAND = [180, 170, 165, 160, 160, 170, 200, 240, 300, 340, 360, 370, 380, 375, 360, 340, 330, 320,
          310, 300, 280, 250, 220, 195]

DEFAULT_BATTERY: Dict[str, float] = {
    "capacity_kwh": 500,
    "initial_energy_kwh": 200,
    "minimum_energy_kwh": 50,
    "max_charge_kwh_per_hour": 100,
    "max_discharge_kwh_per_hour": 100,
}


def make_scenario(
    notes: Optional[List[str]] = None,
    scenario_id: str = "GRID-101",
    **battery_overrides: Any,
) -> Dict[str, Any]:
    """Build a complete, valid 24-hour request body."""
    battery = dict(DEFAULT_BATTERY)
    battery.update(battery_overrides)
    return {
        "scenario_id": scenario_id,
        "operator_notes": notes if notes is not None else ["Nothing unusual is expected today."],
        "hours": [
            {
                "hour": hour,
                "demand_kwh": DEMAND[hour],
                "solar_kwh": SOLAR[hour],
                "tariff_bdt_per_kwh": TARIFFS[hour],
            }
            for hour in range(24)
        ],
        "battery": battery,
    }


def no_op_payload(count: int = 1) -> List[Dict[str, Any]]:
    return [
        {
            "note_index": index,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "The note does not affect energy scheduling.",
        }
        for index in range(count)
    ]


def directive_payload(
    note_index: int,
    directive_type: str,
    adjustment: Optional[Dict[str, Any]],
    explanation: str = "test directive",
) -> Dict[str, Any]:
    return {
        "note_index": note_index,
        "applies": adjustment is not None,
        "directive_type": directive_type,
        "structured_adjustment": adjustment,
        "explanation": explanation,
    }


class MockNoteInterpreter(NoteInterpreter):
    """Returns canned payloads and records how it was called."""

    def __init__(
        self,
        payload: Sequence[Dict[str, Any]],
        repair_payload: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        self.payload = list(payload)
        self.repair_payload = list(repair_payload) if repair_payload is not None else None
        self.interpret_calls = 0
        self.repair_calls = 0
        self.seen_notes: List[str] = []
        self.seen_problems: List[str] = []
        self.seen_context: Optional[Dict[str, Any]] = None

    def interpret(self, notes, context=None):
        self.interpret_calls += 1
        self.seen_notes = list(notes)
        self.seen_context = context
        return self.payload

    def repair(self, notes, previous_payload, problems, context=None):
        self.repair_calls += 1
        self.seen_problems = list(problems)
        return self.repair_payload if self.repair_payload is not None else self.payload


class ExplodingInterpreter(NoteInterpreter):
    """Raises a chosen exception, to exercise LLM failure paths."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def interpret(self, notes, context=None):
        raise self.error

    def repair(self, notes, previous_payload, problems, context=None):
        raise self.error
