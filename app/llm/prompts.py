"""Prompt construction for operator-note interpretation.

The prompt teaches the model a closed vocabulary of directives and a single,
explicit hour-range convention. It deliberately contains no scheduling advice:
the model classifies language, the solver makes every energy decision.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from app.models.directives import DirectiveType

_HALF_OPEN_RULE = """HOUR RANGE CONVENTION (half-open intervals)
A time range covers every hour that STARTS inside the range; the end hour
itself is NOT included.
  "from 1 PM to 3 PM"        -> [13, 14]
  "between 2 PM and 4 PM"    -> [14, 15]
  "from 14:00 until 16:00"   -> [14, 15]
  "during hour 9"            -> [9]
  "9 AM through 11 AM"       -> [9, 10]"""

_INCLUSIVE_RULE = """HOUR RANGE CONVENTION (inclusive end)
A time range covers the start hour through the end hour, INCLUDING the end.
  "from 1 PM to 3 PM"        -> [13, 14, 15]
  "between 2 PM and 4 PM"    -> [14, 15, 16]
  "from 14:00 until 16:00"   -> [14, 15, 16]
  "during hour 9"            -> [9]"""

_VAGUE_RULE = """Vague periods map to these default hour blocks:
  early morning   -> [5, 6, 7]
  morning         -> [6, 7, 8, 9, 10, 11]
  midday / noon   -> [11, 12, 13]
  early afternoon -> [12, 13, 14]
  afternoon       -> [12, 13, 14, 15, 16]
  late afternoon  -> [15, 16, 17]
  evening         -> [17, 18, 19, 20, 21]
  peak evening    -> [18, 19, 20]
  night           -> [22, 23, 0, 1, 2, 3, 4]
  overnight       -> [22, 23, 0, 1, 2, 3, 4]
If a note gives no time information at all but clearly constrains operation,
apply it to every hour 0-23."""

_BASE_SYSTEM_PROMPT = """You convert free-text campus energy operator notes into structured scheduling
directives. You are an interpreter only.

ABSOLUTE RULES
1. Use ONLY these directive_type values: solar_reduction,
   minimum_battery_reserve, no_charge_window, no_discharge_window,
   max_grid_window, no_op. Never invent a new type.
2. Produce EXACTLY ONE interpretation object per input note, in the same order
   as the input, with note_index equal to the note's zero-based position.
3. A note that has no effect on electricity scheduling MUST become no_op with
   "applies": false and "structured_adjustment": null. Never manufacture a
   constraint from an unrelated note.
4. Never make optimization decisions. Do not suggest when to charge, when to
   buy from the grid, or what a schedule should look like. Do not restate or
   modify the scenario data.
5. Every "hours" array contains integers in 0-23, sorted ascending, with no
   duplicates. Convert 12-hour clock times to 24-hour integers
   (1 PM -> 13, midnight -> 0, 12 PM noon -> 12).
6. Return JSON only. No prose, no markdown, no code fences.

DIRECTIVE DEFINITIONS
solar_reduction  {"hours": [int], "factor": float}
    Solar generation is degraded. effective_solar = original_solar * factor.
    factor is the fraction that REMAINS, between 0 and 1.
      "output will drop TO 20%"     -> factor 0.2
      "output will drop BY 80%"     -> factor 0.2
      "only one fifth of normal"    -> factor 0.2
      "solar will be halved"        -> factor 0.5
      "panels offline / no output"  -> factor 0.0
      "output reduced by a third"   -> factor 0.667
    Triggers: cloud cover, haze, fog, smog, rain, dust, soiling, panel
    maintenance, shading, inverter faults, derating.

minimum_battery_reserve  {"hours": [int], "minimum_energy_kwh": float}
    Stored battery energy must stay at or above this level during these hours.
    minimum_energy_kwh is always an ABSOLUTE kWh figure, never a percentage.
    When the note states a proportion, convert it using the battery capacity
    given in the scenario context below:
      "at least 50% of battery capacity", capacity 200  -> 100
      "keep the battery a quarter full", capacity 240   -> 60
      "hold at least 90 kWh"                            -> 90
    Triggers: reserve for emergencies, keep a buffer, hold backup charge,
    maintain at least N kWh, data-center or life-safety backup.

no_charge_window  {"hours": [int]}
    The battery must not charge during these hours.
    Triggers: do not charge, avoid charging, charging disabled/prohibited,
    keep the battery from charging, no top-ups.

no_discharge_window  {"hours": [int]}
    The battery must not discharge during these hours.
    Triggers: do not discharge, no draw from the battery, battery must rest,
    preserve stored energy (when phrased as a prohibition on discharging).

max_grid_window  {"hours": [int], "max_grid_kwh": float}
    Grid import must not exceed this many kWh in any of these hours.
    Triggers: demand cap, import limit, feeder/transformer limit, load
    shedding order, contractual ceiling, "draw no more than N kWh".

no_op  structured_adjustment must be null
    Anything with no scheduling effect: catering, staffing, cleaning
    schedules, academic calendar, social events, unrelated maintenance,
    weather comments with no stated solar impact, or notes too vague to map
    onto a supported directive.

AMBIGUITY
If a note is energy-related but does not map onto exactly one supported
directive, or the required numbers are not derivable, return no_op and say so
in the explanation. Do not guess a number that the note does not support.

OUTPUT
Return a JSON object with a single key "interpretations" holding the array of
interpretation objects. Each object has: note_index (int), applies (bool),
directive_type (string), structured_adjustment (object or null),
explanation (string, one short sentence)."""


def build_system_prompt(inclusive_end: bool = False) -> str:
    """Assemble the system prompt for the configured hour-range convention."""
    range_rule = _INCLUSIVE_RULE if inclusive_end else _HALF_OPEN_RULE
    return f"{_BASE_SYSTEM_PROMPT}\n\n{range_rule}\n\n{_VAGUE_RULE}\n\n{_worked_examples(inclusive_end)}"


def _worked_examples(inclusive_end: bool) -> str:
    """Few-shot examples, rendered under the active hour convention."""
    if inclusive_end:
        solar_hours = [13, 14, 15]
        charge_hours = [14, 15, 16]
    else:
        solar_hours = [13, 14]
        charge_hours = [14, 15]

    example = {
        "interpretations": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": solar_hours, "factor": 0.2},
                "explanation": "Solar availability falls to one fifth during these hours.",
            },
            {
                "note_index": 1,
                "applies": True,
                "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": charge_hours},
                "explanation": "Battery charging is prohibited during this window.",
            },
            {
                "note_index": 2,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "The note does not affect energy scheduling.",
            },
        ]
    }
    return (
        "WORKED EXAMPLE\n"
        "Input notes:\n"
        '  0: "Solar output will drop to about 20% from 1 PM to 3 PM."\n'
        '  1: "Do not charge the battery between 2 PM and 4 PM."\n'
        '  2: "The cafeteria menu changes tomorrow."\n'
        "Correct output:\n" + json.dumps(example, indent=2)
    )


def build_scenario_context(context: Optional[Dict[str, Any]]) -> str:
    """Render the numeric facts a note may refer to proportionally.

    Only battery ratings are exposed. Demand, solar and tariff series are
    deliberately withheld: the model classifies language, and giving it the
    cost data would invite it to reason about scheduling, which is the
    optimizer's job.
    """
    if not context:
        return ""
    lines = [f"  {key}: {value:g}" for key, value in context.items()]
    return (
        "SCENARIO CONTEXT (use only to convert proportional wording into "
        "absolute kWh; never to make scheduling decisions)\n" + "\n".join(lines) + "\n\n"
    )


def build_user_prompt(notes: Sequence[str], context: Optional[Dict[str, Any]] = None) -> str:
    """Render the notes to classify, one numbered line per note."""
    lines = [f"  {index}: {note!r}" for index, note in enumerate(notes)]
    return (
        build_scenario_context(context)
        + f"Interpret the following {len(notes)} operator note(s). "
        f"Return exactly {len(notes)} interpretation objects with note_index "
        f"0 through {len(notes) - 1}, in order.\n\n"
        "Notes:\n" + "\n".join(lines)
    )


def build_repair_prompt(
    problems: List[str],
    notes: Sequence[str],
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Ask the model to correct a payload that failed deterministic validation."""
    issues = "\n".join(f"  - {problem}" for problem in problems)
    return (
        "Your previous response was rejected by the validation layer for these "
        "reasons:\n"
        f"{issues}\n\n"
        "Re-read the rules and return a corrected JSON object. Keep any "
        "interpretation that was already correct exactly as it was, and fix "
        "only what is listed above. Return JSON only.\n\n"
        + build_user_prompt(notes, context)
    )


def build_response_schema(note_count: int) -> Dict[str, Any]:
    """JSON Schema for providers that support structured output."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["interpretations"],
        "properties": {
            "interpretations": {
                "type": "array",
                "minItems": note_count,
                "maxItems": note_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "note_index",
                        "applies",
                        "directive_type",
                        "structured_adjustment",
                        "explanation",
                    ],
                    "properties": {
                        "note_index": {"type": "integer", "minimum": 0},
                        "applies": {"type": "boolean"},
                        "directive_type": {
                            "type": "string",
                            "enum": [t.value for t in DirectiveType],
                        },
                        "structured_adjustment": {
                            "type": ["object", "null"],
                            "properties": {
                                "hours": {
                                    "type": "array",
                                    "items": {"type": "integer", "minimum": 0, "maximum": 23},
                                },
                                "factor": {"type": "number", "minimum": 0, "maximum": 1},
                                "minimum_energy_kwh": {"type": "number", "minimum": 0},
                                "max_grid_kwh": {"type": "number", "minimum": 0},
                            },
                            "additionalProperties": False,
                        },
                        "explanation": {"type": "string"},
                    },
                },
            }
        },
    }
