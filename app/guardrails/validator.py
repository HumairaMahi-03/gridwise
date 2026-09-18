"""Deterministic guardrails over LLM output.

Nothing the model produces reaches the optimizer unchecked. This module:

1. validates every field of every interpretation against the closed directive
   vocabulary and against the scenario's own battery limits;
2. normalises harmless formatting problems (unsorted or duplicated hours,
   integral floats, missing explanations);
3. reports precise, human-readable problems that can be fed back to the model
   in a single repair attempt;
4. compiles surviving interpretations into per-hour constraints, merging
   overlapping directives by taking the strictest requirement.

The module is pure and synchronous: given the same input it always produces the
same output, with no network access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import HOURS_IN_DAY
from app.exceptions import LLMInterpretationError
from app.models.directives import (
    ACTIVE_DIRECTIVE_TYPES,
    REQUIRED_ADJUSTMENT_FIELDS,
    DirectiveSet,
    DirectiveType,
    NoteInterpretation,
)
from app.models.request import BatteryConfig

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Outcome of validating a full interpretation payload."""

    interpretations: List[NoteInterpretation] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


# --------------------------------------------------------------------- helpers


def _coerce_number(value: Any) -> Optional[float]:
    """Return a float for int/float input, rejecting bools and strings-as-numbers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            return None
        return numeric
    return None


def _coerce_hour(value: Any) -> Optional[int]:
    """Return an integer hour, accepting integral floats like ``13.0``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _validate_hours(raw_hours: Any, label: str, problems: List[str]) -> Optional[List[int]]:
    """Validate and normalise an ``hours`` array to sorted unique integers."""
    if not isinstance(raw_hours, list) or not raw_hours:
        problems.append(f"{label}: 'hours' must be a non-empty array of integers 0-23.")
        return None

    hours: List[int] = []
    for item in raw_hours:
        hour = _coerce_hour(item)
        if hour is None:
            problems.append(f"{label}: hour value {item!r} is not an integer.")
            return None
        if not 0 <= hour <= HOURS_IN_DAY - 1:
            problems.append(f"{label}: hour {hour} is outside the valid range 0-23.")
            return None
        hours.append(hour)

    normalised = sorted(set(hours))
    if normalised != hours:
        # Formatting only: the intended window is unambiguous, so repair it
        # locally instead of spending another model round-trip on it.
        logger.info("%s: normalised hours %s -> %s", label, hours, normalised)
    return normalised


def _validate_entry(
    entry: Any,
    expected_index: int,
    battery: BatteryConfig,
) -> Tuple[Optional[NoteInterpretation], List[str]]:
    """Validate one interpretation object.

    Returns the validated interpretation (or ``None``) and any problems found.
    """
    problems: List[str] = []
    label = f"note_index {expected_index}"

    if not isinstance(entry, dict):
        return None, [f"{label}: interpretation must be a JSON object."]

    raw_type = entry.get("directive_type")
    if not isinstance(raw_type, str):
        return None, [f"{label}: 'directive_type' is missing or not a string."]
    try:
        directive_type = DirectiveType(raw_type.strip().lower())
    except ValueError:
        supported = ", ".join(t.value for t in DirectiveType)
        return None, [
            f"{label}: '{raw_type}' is not a supported directive_type. "
            f"Use one of: {supported}."
        ]

    explanation = entry.get("explanation")
    explanation = explanation.strip() if isinstance(explanation, str) else ""

    if directive_type is DirectiveType.NO_OP:
        if entry.get("applies") is True:
            problems.append(f"{label}: a no_op directive must have 'applies': false.")
            return None, problems
        return (
            NoteInterpretation(
                note_index=expected_index,
                applies=False,
                directive_type=DirectiveType.NO_OP,
                structured_adjustment=None,
                explanation=explanation or "The note does not affect energy scheduling.",
            ),
            problems,
        )

    if entry.get("applies") is False:
        problems.append(
            f"{label}: 'applies' is false but directive_type is "
            f"'{directive_type.value}'. Use no_op for notes that do not apply."
        )
        return None, problems

    adjustment = entry.get("structured_adjustment")
    if not isinstance(adjustment, dict):
        problems.append(
            f"{label}: '{directive_type.value}' requires a structured_adjustment object."
        )
        return None, problems

    required = REQUIRED_ADJUSTMENT_FIELDS[directive_type]
    missing = sorted(required - set(adjustment))
    if missing:
        problems.append(
            f"{label}: structured_adjustment is missing required field(s): {', '.join(missing)}."
        )
        return None, problems

    hours = _validate_hours(adjustment.get("hours"), label, problems)
    if hours is None:
        return None, problems

    clean: Dict[str, Any] = {"hours": hours}

    if directive_type is DirectiveType.SOLAR_REDUCTION:
        factor = _coerce_number(adjustment.get("factor"))
        if factor is None:
            problems.append(f"{label}: 'factor' must be a number.")
            return None, problems
        if not 0.0 <= factor <= 1.0:
            problems.append(
                f"{label}: 'factor' is {factor}, but it must be the fraction of solar "
                "that REMAINS, between 0 and 1 (a drop of 80% means factor 0.2)."
            )
            return None, problems
        clean["factor"] = factor

    elif directive_type is DirectiveType.MINIMUM_BATTERY_RESERVE:
        reserve = _coerce_number(adjustment.get("minimum_energy_kwh"))
        if reserve is None:
            problems.append(f"{label}: 'minimum_energy_kwh' must be a number.")
            return None, problems
        if reserve < 0:
            problems.append(f"{label}: 'minimum_energy_kwh' must not be negative.")
            return None, problems
        if reserve > battery.capacity_kwh:
            problems.append(
                f"{label}: 'minimum_energy_kwh' is {reserve}, which exceeds the battery "
                f"capacity of {battery.capacity_kwh} kWh."
            )
            return None, problems
        clean["minimum_energy_kwh"] = reserve

    elif directive_type is DirectiveType.MAX_GRID_WINDOW:
        cap = _coerce_number(adjustment.get("max_grid_kwh"))
        if cap is None:
            problems.append(f"{label}: 'max_grid_kwh' must be a number.")
            return None, problems
        if cap < 0:
            problems.append(f"{label}: 'max_grid_kwh' must not be negative.")
            return None, problems
        clean["max_grid_kwh"] = cap

    return (
        NoteInterpretation(
            note_index=expected_index,
            applies=True,
            directive_type=directive_type,
            structured_adjustment=clean,
            explanation=explanation or f"{directive_type.value} applied to hours {hours}.",
        ),
        problems,
    )


# ------------------------------------------------------------------ public API


def validate_interpretations(
    raw_payload: Sequence[Any],
    notes: Sequence[str],
    battery: BatteryConfig,
) -> GuardrailResult:
    """Validate a complete LLM payload against the notes it should describe."""
    result = GuardrailResult()
    expected = len(notes)

    if len(raw_payload) != expected:
        result.problems.append(
            f"Expected exactly {expected} interpretation object(s), one per note, "
            f"but received {len(raw_payload)}."
        )
        return result

    by_index: Dict[int, Any] = {}
    for position, entry in enumerate(raw_payload):
        index = entry.get("note_index") if isinstance(entry, dict) else None
        index = _coerce_hour(index)
        if index is None:
            index = position  # fall back to array position
        if index in by_index:
            result.problems.append(
                f"note_index {index} appears more than once; each note needs exactly one "
                "interpretation."
            )
            return result
        by_index[index] = entry

    missing = sorted(set(range(expected)) - set(by_index))
    if missing:
        result.problems.append(
            "Missing interpretation(s) for note_index "
            + ", ".join(str(index) for index in missing)
            + "."
        )
        return result

    for index in range(expected):
        interpretation, problems = _validate_entry(by_index[index], index, battery)
        result.problems.extend(problems)
        if interpretation is not None:
            result.interpretations.append(interpretation)

    if result.problems:
        result.interpretations = []
    return result


def salvage_interpretations(
    raw_payload: Sequence[Any],
    notes: Sequence[str],
    battery: BatteryConfig,
) -> List[NoteInterpretation]:
    """Last-resort handling after a failed repair attempt.

    Entries that validate are kept. An entry is downgraded to ``no_op`` only
    when the model itself signalled that the note does not apply — dropping a
    directive the model believed was relevant would silently produce a schedule
    that ignores a real operating constraint, so that case raises instead.
    """
    salvaged: List[NoteInterpretation] = []

    for index in range(len(notes)):
        entry = raw_payload[index] if index < len(raw_payload) else None
        interpretation, _ = _validate_entry(entry, index, battery)

        if interpretation is not None:
            salvaged.append(interpretation)
            continue

        claims_no_effect = (
            isinstance(entry, dict)
            and entry.get("applies") is False
            and not isinstance(entry.get("structured_adjustment"), dict)
        )
        if claims_no_effect:
            logger.warning("note_index %s downgraded to no_op after failed repair", index)
            salvaged.append(
                NoteInterpretation.no_op(
                    index, "The note could not be mapped onto a supported directive."
                )
            )
            continue

        logger.error(
            "note_index %s could not be interpreted safely; rejecting the request", index
        )
        raise LLMInterpretationError(
            f"Operator note {index} could not be interpreted reliably; no schedule was produced."
        )

    return salvaged


def compile_directives(
    interpretations: Sequence[NoteInterpretation],
    battery: BatteryConfig,
) -> DirectiveSet:
    """Merge validated interpretations into per-hour optimizer constraints."""
    directives = DirectiveSet()

    for interpretation in interpretations:
        if not interpretation.applies or interpretation.directive_type not in ACTIVE_DIRECTIVE_TYPES:
            continue
        adjustment = interpretation.structured_adjustment or {}
        hours = adjustment.get("hours", [])

        if interpretation.directive_type is DirectiveType.SOLAR_REDUCTION:
            for hour in hours:
                directives.add_solar_reduction(hour, float(adjustment["factor"]))
        elif interpretation.directive_type is DirectiveType.MINIMUM_BATTERY_RESERVE:
            for hour in hours:
                directives.add_min_reserve(hour, float(adjustment["minimum_energy_kwh"]))
        elif interpretation.directive_type is DirectiveType.NO_CHARGE_WINDOW:
            directives.no_charge_hours.update(hours)
        elif interpretation.directive_type is DirectiveType.NO_DISCHARGE_WINDOW:
            directives.no_discharge_hours.update(hours)
        elif interpretation.directive_type is DirectiveType.MAX_GRID_WINDOW:
            for hour in hours:
                directives.add_max_grid(hour, float(adjustment["max_grid_kwh"]))

    return directives
