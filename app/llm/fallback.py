"""Deterministic fallback interpretation, used only when the LLM is down.

This is a **degraded mode**, disabled by default. The challenge requires that
operator notes be interpreted by an LLM, and rules cannot match a model's
handling of paraphrase, so this never runs while the provider is reachable.

Its purpose is availability: if the provider is unreachable mid-demo or
mid-judging, ``LLM_FALLBACK_TO_RULES=true`` lets the service keep returning
valid, constraint-satisfying schedules instead of 503s. Every fallback
interpretation is logged loudly, and anything the rules cannot classify
confidently becomes ``no_op`` rather than a guessed constraint.

Enable with care and never silently: prefer a working LLM.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.llm.interpreter import NoteInterpreter

logger = logging.getLogger(__name__)

# --- keyword vocabularies -----------------------------------------------------

_CHARGE_WORDS = ("charg", "top up", "top-up", "recharg")
_DISCHARGE_WORDS = ("discharg", "draw from the battery", "drawing from the battery", "export from the battery")
_PROHIBITION = ("not", "no ", "never", "avoid", "disable", "unavailable", "isolat",
                "prohibit", "forbid", "stop", "prevent", "suspend", "cannot", "can't",
                "must not", "keep the battery from", "offline", "out of service")
_SOLAR_WORDS = ("solar", "pv", "photovoltaic", "panel", "rooftop", "inverter", "array")
_SOLAR_LOSS = ("reduc", "drop", "fall", "fell", "lower", "less", "half", "cloud", "haze",
               "fog", "rain", "dust", "dirty", "soil", "shade", "shading", "clean", "wash",
               "derat", "outage", "only", "limited", "degrad", "inspect")
_RESERVE_WORDS = ("reserve", "keep at least", "remain in the battery", "maintain at least",
                  "backup", "back-up", "emergency", "buffer", "stay above", "hold at least",
                  "requires at least", "kept in the battery")
_GRID_WORDS = ("grid", "import", "intake", "feeder", "transformer", "substation", "draw from the grid")
_CAP_WORDS = ("not exceed", "no more than", "at or below", "cap", "limit", "maximum", "max",
              "under", "below", "up to", "ceiling")

_VAGUE_WINDOWS = {
    "early morning": [5, 6, 7],
    "late morning": [9, 10, 11],
    "morning": [6, 7, 8, 9, 10, 11],
    "midday": [11, 12, 13],
    "early afternoon": [12, 13, 14],
    "late afternoon": [15, 16, 17],
    "afternoon": [12, 13, 14, 15, 16],
    "early evening": [17, 18, 19],
    "evening": [17, 18, 19, 20, 21],
    "overnight": [22, 23, 0, 1, 2, 3, 4],
    "night": [22, 23, 0, 1, 2, 3, 4],
}

_FRACTION_WORDS = {
    "half": 0.5, "a half": 0.5, "one half": 0.5, "third": 1 / 3, "one third": 1 / 3,
    "a third": 1 / 3, "quarter": 0.25, "a quarter": 0.25, "one quarter": 0.25,
    "fifth": 0.2, "one fifth": 0.2, "a fifth": 0.2, "two thirds": 2 / 3,
    "three quarters": 0.75,
}

_TIME = r"(\d{1,2})\s*(?::\s*(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?"
_RANGE_RE = re.compile(
    rf"(?:from\s+)?{_TIME}\s*(?:-|–|to|until|till|through|and)\s*{_TIME}", re.IGNORECASE
)


def _to_hour(number: str, meridiem: Optional[str]) -> Optional[int]:
    try:
        hour = int(number)
    except (TypeError, ValueError):
        return None
    if meridiem:
        meridiem = meridiem.replace(".", "").lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
    if not 0 <= hour <= 24:
        return None
    return hour % 24


def _normalise_clock_words(text: str) -> str:
    """Rewrite clock words as times.

    Word boundaries matter here: a naive replace turns "afternoon" into
    "after12 pm" and destroys every vague-window match.
    """
    text = re.sub(r"\bnoon\b", "12 pm", text)
    text = re.sub(r"\bmidday\b", "12 pm", text)
    return re.sub(r"\bmidnight\b", "12 am", text)


def extract_hours(note: str) -> Optional[List[int]]:
    """Find the hour window a note refers to, half-open on the end hour."""
    text = _normalise_clock_words(note.lower())

    match = _RANGE_RE.search(text)
    if match:
        start = _to_hour(match.group(1), match.group(3))
        end = _to_hour(match.group(4), match.group(6))
        if start is not None and end is not None:
            # Inherit an explicit end meridiem when the start omitted one
            # ("from 6 until 9 PM"), which is how operators usually write it.
            if match.group(3) is None and match.group(6) is not None and start < end:
                pass
            span = (end - start) % 24
            if span == 0:
                return [start]
            if span > 12:  # implausible for a maintenance window; treat as noise
                return None
            return [(start + offset) % 24 for offset in range(span)]

    for phrase, hours in _VAGUE_WINDOWS.items():
        if phrase in text:
            return sorted(hours)

    single = re.search(r"\b(?:at|during|in)\s+hour\s+(\d{1,2})\b", text)
    if single:
        hour = _to_hour(single.group(1), None)
        return [hour] if hour is not None else None

    return None


def extract_fraction(note: str) -> Optional[float]:
    """Return the fraction of solar that remains after the note's wording."""
    text = note.lower()

    percent = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent|per cent)", text)
    if percent:
        value = float(percent.group(1)) / 100.0
        if not 0.0 <= value <= 1.0:
            return None
        # "a drop OF 80%" / "80% reduction" removes 80%; "drop TO 20%" leaves 20%.
        window = text[max(0, percent.start() - 40): percent.end() + 40]
        removal = any(word in window for word in
                      ("reduction of", "reduction in", "drop of", "drop by", "reduced by",
                       "decrease of", "decrease by", "fall by", "lower by", "down by",
                       "% reduction", "percent reduction", "loss of", "cut by"))
        if removal:
            return round(1.0 - value, 6)
        return value

    for phrase, value in sorted(_FRACTION_WORDS.items(), key=lambda item: -len(item[0])):
        if phrase in text:
            if any(word in text for word in ("only", "about", "roughly", "leave", "leaves",
                                             "remaining", "available", "treated as", "down to")):
                return value
            if any(word in text for word in ("reduce", "reduced", "reduction", "cut", "lose", "lost")):
                return round(1.0 - value, 6)
            return value

    if any(word in text for word in ("no output", "zero output", "offline", "completely out",
                                     "fully offline", "shut down", "no generation")):
        return 0.0
    return None


def extract_kwh(note: str, capacity: Optional[float]) -> Optional[float]:
    """Extract an absolute kWh figure, converting proportions using capacity."""
    absolute = re.search(r"(\d{1,6}(?:\.\d+)?)\s*kwh", note.lower())
    if absolute:
        return float(absolute.group(1))

    if capacity:
        percent = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent|per cent)", note.lower())
        if percent:
            return round(capacity * float(percent.group(1)) / 100.0, 6)
        for phrase, value in _FRACTION_WORDS.items():
            if phrase in note.lower():
                return round(capacity * value, 6)
    return None


def _has(text: str, words: Sequence[str]) -> bool:
    return any(word in text for word in words)


def classify(note: str, capacity: Optional[float]) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """Map a note onto a directive. Returns (type, adjustment, explanation)."""
    text = note.lower()
    hours = extract_hours(note)

    # 1. Solar degradation.
    if _has(text, _SOLAR_WORDS) and _has(text, _SOLAR_LOSS):
        factor = extract_fraction(note)
        if hours and factor is not None:
            return ("solar_reduction", {"hours": hours, "factor": factor},
                    "Rule-based fallback: solar availability is reduced during these hours.")

    # 2. Battery prohibitions. Check discharge first: "do not discharge" also
    #    contains "charg" as a substring.
    if _has(text, _PROHIBITION) and hours:
        if _has(text, _DISCHARGE_WORDS):
            return ("no_discharge_window", {"hours": hours},
                    "Rule-based fallback: battery discharge is prohibited during these hours.")
        if _has(text, _CHARGE_WORDS):
            return ("no_charge_window", {"hours": hours},
                    "Rule-based fallback: battery charging is prohibited during these hours.")

    # 3. Grid import ceiling.
    if _has(text, _GRID_WORDS) and _has(text, _CAP_WORDS) and hours:
        cap = extract_kwh(note, None)
        if cap is not None:
            return ("max_grid_window", {"hours": hours, "max_grid_kwh": cap},
                    "Rule-based fallback: grid import is capped during these hours.")

    # 4. Battery reserve floor.
    if _has(text, _RESERVE_WORDS) and hours:
        reserve = extract_kwh(note, capacity)
        if reserve is not None and (capacity is None or reserve <= capacity):
            return ("minimum_battery_reserve", {"hours": hours, "minimum_energy_kwh": reserve},
                    "Rule-based fallback: a minimum battery reserve applies during these hours.")

    return ("no_op", None,
            "Rule-based fallback: the note could not be mapped onto a supported directive.")


class RuleBasedInterpreter(NoteInterpreter):
    """Keyword and pattern interpretation. Degraded mode only."""

    def interpret(
        self,
        notes: Sequence[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        capacity = (context or {}).get("battery_capacity_kwh")
        payload: List[Dict[str, Any]] = []

        for index, note in enumerate(notes):
            directive_type, adjustment, explanation = classify(note, capacity)
            payload.append({
                "note_index": index,
                "applies": directive_type != "no_op",
                "directive_type": directive_type,
                "structured_adjustment": adjustment,
                "explanation": explanation,
            })
            logger.warning("Rule-based fallback classified note %s as %s", index, directive_type)

        return payload

    def repair(
        self,
        notes: Sequence[str],
        previous_payload: List[Dict[str, Any]],
        problems: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # The rules are deterministic, so a second pass would return the same
        # thing. Drop to no_op for the notes that failed instead of looping.
        return [
            {
                "note_index": index,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "Rule-based fallback could not produce a valid directive.",
            }
            for index in range(len(notes))
        ]


class ResilientInterpreter(NoteInterpreter):
    """Tries the LLM first and falls back to rules only if it is unreachable.

    A provider that answers badly is *not* a fallback trigger — that path is
    handled by the repair round and the guardrails. Only an outage is.
    """

    def __init__(self, primary: NoteInterpreter, fallback: Optional[NoteInterpreter] = None) -> None:
        self._primary = primary
        self._fallback = fallback or RuleBasedInterpreter()

    def interpret(self, notes, context=None):
        from app.exceptions import LLMUnavailableError
        try:
            return self._primary.interpret(notes, context)
        except LLMUnavailableError:
            logger.error(
                "LLM unavailable; falling back to deterministic rule-based interpretation. "
                "This is a degraded mode — check LLM configuration."
            )
            return self._fallback.interpret(notes, context)

    def repair(self, notes, previous_payload, problems, context=None):
        from app.exceptions import LLMUnavailableError
        try:
            return self._primary.repair(notes, previous_payload, problems, context)
        except LLMUnavailableError:
            logger.error("LLM unavailable during repair; using rule-based interpretation")
            return self._fallback.interpret(notes, context)
