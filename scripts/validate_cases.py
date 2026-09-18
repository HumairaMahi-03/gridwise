#!/usr/bin/env python3
"""Validate the service against the ten official public sample cases.

Three modes:

  --offline   (default) Solve in-process using the case pack's ground-truth
              directives. Needs no API key and no running server. Measures the
              optimizer, validator and totals.

  --local     Solve in-process but interpret the notes with the real LLM.
              Needs LLM_API_KEY. Measures interpretation accuracy too.

  --url URL   POST each case to a running server (local or deployed) and check
              the responses. Needs the server to have LLM credentials.

Every mode replays the full GridWise constraint set against whatever plan comes
back, and compares cost with the pack's reference optimum.

    python scripts/validate_cases.py
    python scripts/validate_cases.py --local
    python scripts/validate_cases.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "public_cases.json"
)
TOLERANCE = 0.01

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def load_cases() -> List[Dict[str, Any]]:
    with open(CASES_PATH) as handle:
        return json.load(handle)["cases"]


# --------------------------------------------------------------- plan checking


def check_plan(case: Dict[str, Any], body: Dict[str, Any], directives: List[Dict[str, Any]]) -> List[str]:
    """Replay every GridWise rule against a returned plan. Returns violations."""
    problems: List[str] = []
    scenario = case["input"]
    battery = scenario["battery"]
    hours = {record["hour"]: record for record in scenario["hours"]}
    plan = body.get("hourly_plan", [])

    if len(plan) != 24 or [entry["hour"] for entry in plan] != list(range(24)):
        return ["hourly_plan must contain exactly hours 0-23 in order"]

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for directive in directives:
        grouped.setdefault(directive["directive_type"], []).append(directive)

    solar_factor: Dict[int, float] = {}
    for directive in grouped.get("solar_reduction", []):
        for hour in directive["hours"]:
            solar_factor[hour] = min(solar_factor.get(hour, 1.0), directive["factor"])

    previous = battery["initial_energy_kwh"]
    for entry in plan:
        hour = entry["hour"]
        record = hours[hour]
        action = entry["battery_action"]
        charge = entry["battery_kwh"] if action == "charge" else 0.0
        discharge = entry["battery_kwh"] if action == "discharge" else 0.0

        if action not in {"charge", "discharge", "idle"}:
            problems.append(f"hour {hour}: invalid battery_action {action!r}")
        if action == "idle" and entry["battery_kwh"] != 0:
            problems.append(f"hour {hour}: idle with battery_kwh {entry['battery_kwh']}")

        supply = entry["grid_kwh"] + entry["solar_used_kwh"] + discharge
        demand = record["demand_kwh"] + charge
        if abs(supply - demand) > TOLERANCE:
            problems.append(f"hour {hour}: energy balance off by {supply - demand:+.3f}")

        effective = record["solar_kwh"] * solar_factor.get(hour, 1.0)
        if entry["solar_used_kwh"] > effective + TOLERANCE:
            problems.append(f"hour {hour}: solar_used {entry['solar_used_kwh']} > effective {effective}")

        if entry["grid_kwh"] < -TOLERANCE:
            problems.append(f"hour {hour}: negative grid import")
        if charge > battery["max_charge_kwh_per_hour"] + TOLERANCE:
            problems.append(f"hour {hour}: charge exceeds the hourly rate limit")
        if discharge > battery["max_discharge_kwh_per_hour"] + TOLERANCE:
            problems.append(f"hour {hour}: discharge exceeds the hourly rate limit")

        expected = previous + charge - discharge
        if abs(entry["battery_energy_after_kwh"] - expected) > TOLERANCE:
            problems.append(f"hour {hour}: state of charge does not follow from the previous hour")
        if entry["battery_energy_after_kwh"] < battery["minimum_energy_kwh"] - TOLERANCE:
            problems.append(f"hour {hour}: stored energy below the battery minimum")
        if entry["battery_energy_after_kwh"] > battery["capacity_kwh"] + TOLERANCE:
            problems.append(f"hour {hour}: stored energy above capacity")
        previous = entry["battery_energy_after_kwh"]

    if abs(plan[23]["battery_energy_after_kwh"] - battery["initial_energy_kwh"]) > TOLERANCE:
        problems.append("day-end battery energy does not match the initial level")

    by_hour = {entry["hour"]: entry for entry in plan}
    for directive in grouped.get("no_charge_window", []):
        for hour in directive["hours"]:
            if by_hour[hour]["battery_action"] == "charge":
                problems.append(f"hour {hour}: charging inside a no_charge_window")
    for directive in grouped.get("no_discharge_window", []):
        for hour in directive["hours"]:
            if by_hour[hour]["battery_action"] == "discharge":
                problems.append(f"hour {hour}: discharging inside a no_discharge_window")
    for directive in grouped.get("minimum_battery_reserve", []):
        for hour in directive["hours"]:
            if by_hour[hour]["battery_energy_after_kwh"] < directive["minimum_energy_kwh"] - TOLERANCE:
                problems.append(f"hour {hour}: below the required reserve")
    for directive in grouped.get("max_grid_window", []):
        for hour in directive["hours"]:
            if by_hour[hour]["grid_kwh"] > directive["max_grid_kwh"] + TOLERANCE:
                problems.append(f"hour {hour}: grid import above the cap")

    tariffs = {record["hour"]: record["tariff_bdt_per_kwh"] for record in scenario["hours"]}
    grid = sum(entry["grid_kwh"] for entry in plan)
    cost = sum(entry["grid_kwh"] * tariffs[entry["hour"]] for entry in plan)
    peak = max(entry["grid_kwh"] for entry in plan)
    if abs(body["total_grid_kwh"] - grid) > TOLERANCE:
        problems.append("total_grid_kwh does not match the plan")
    if abs(body["total_cost_bdt"] - cost) > TOLERANCE:
        problems.append("total_cost_bdt does not match the plan")
    if abs(body["peak_grid_kwh"] - peak) > TOLERANCE:
        problems.append("peak_grid_kwh does not match the plan")

    return problems


def compare_directives(case: Dict[str, Any], body: Dict[str, Any]) -> List[str]:
    """Compare returned interpretations with the pack's ground truth."""
    problems: List[str] = []
    expected = case["expected_directives"]
    actual = body.get("directive_interpretation", [])

    if len(actual) != len(expected):
        return [f"expected {len(expected)} interpretation(s), got {len(actual)}"]

    for index, (want, got) in enumerate(zip(expected, actual)):
        if got.get("note_index") != index:
            problems.append(f"note {index}: note_index is {got.get('note_index')}")
        if got.get("directive_type") != want["directive_type"]:
            problems.append(
                f"note {index}: expected {want['directive_type']}, got {got.get('directive_type')}"
            )
            continue
        if want["directive_type"] == "no_op":
            if got.get("applies") is not False or got.get("structured_adjustment") is not None:
                problems.append(f"note {index}: no_op must have applies=false and a null adjustment")
            continue
        if got.get("applies") is not True:
            problems.append(f"note {index}: an active directive must have applies=true")
        adjustment = got.get("structured_adjustment") or {}
        for key, value in want.items():
            if key == "directive_type":
                continue
            actual_value = adjustment.get(key)
            if key == "hours":
                if list(actual_value or []) != value:
                    problems.append(f"note {index}: hours {actual_value} != {value}")
            elif actual_value is None or abs(float(actual_value) - float(value)) > TOLERANCE:
                problems.append(f"note {index}: {key} {actual_value} != {value}")
    return problems


# ------------------------------------------------------------------- runners


def solve_in_process(case: Dict[str, Any], live_llm: bool) -> Dict[str, Any]:
    from app.models.request import OptimizationRequest
    from app.services.energy_service import EnergyService

    request = OptimizationRequest(**case["input"])

    if live_llm:
        from app.llm.interpreter import build_interpreter
        interpreter = build_interpreter()
    else:
        from tests.test_samples import payload_from
        from tests.helpers import MockNoteInterpreter
        interpreter = MockNoteInterpreter(payload_from(case["expected_directives"]))

    response = EnergyService(interpreter).optimize(request)
    return json.loads(response.model_dump_json()) if hasattr(response, "model_dump_json") \
        else json.loads(json.dumps(response.model_dump(mode="json")))


def solve_over_http(case: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    import httpx

    response = httpx.post(f"{base_url.rstrip('/')}/optimize-energy", json=case["input"], timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
    return response.json()


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--offline", action="store_true",
                       help="solve in-process with ground-truth directives (default)")
    group.add_argument("--local", action="store_true",
                       help="solve in-process, interpreting notes with the real LLM")
    group.add_argument("--url", metavar="URL", help="POST each case to a running server")
    parser.add_argument("--case", help="run a single case, e.g. SAMPLE-03")
    parser.add_argument("--verbose", "-v", action="store_true", help="print each returned plan")
    args = parser.parse_args()

    cases = load_cases()
    if args.case:
        cases = [case for case in cases if case["id"].lower() == args.case.lower()]
        if not cases:
            print(f"no such case: {args.case}")
            return 2

    live_llm = bool(args.local or args.url)
    mode = "live server" if args.url else ("in-process + live LLM" if args.local else "offline")
    print(f"GridWise case validation — mode: {mode}\n")
    print(f"{'case':<11} {'directives':>11} {'plan':>7} {'ref cost':>10} {'ours':>10} {'delta':>9}")
    print("-" * 62)

    failures: List[Tuple[str, List[str]]] = []
    for case in cases:
        try:
            body = solve_over_http(case, args.url) if args.url else solve_in_process(case, args.local)
        except Exception as exc:  # noqa: BLE001 — a failed case must not stop the run
            print(f"{case['id']:<11} {RED}ERROR{RESET} {type(exc).__name__}: {exc}")
            failures.append((case["id"], [str(exc)]))
            continue

        directive_problems = compare_directives(case, body) if live_llm else []
        plan_problems = check_plan(case, body, case["expected_directives"])

        reference = case["reference_totals"]["total_cost_bdt"]
        ours = body["total_cost_bdt"]
        delta = ours - reference

        directive_mark = (f"{GREEN}ok{RESET}" if not directive_problems else f"{RED}FAIL{RESET}") \
            if live_llm else f"{DIM}given{RESET}"
        plan_mark = f"{GREEN}ok{RESET}" if not plan_problems else f"{RED}FAIL{RESET}"
        if delta <= TOLERANCE:
            delta_text = f"{GREEN}{delta:+9.2f}{RESET}"
        else:
            delta_text = f"{RED}{delta:+9.2f}{RESET}"

        print(f"{case['id']:<11} {directive_mark:>20} {plan_mark:>16} "
              f"{reference:>10.2f} {ours:>10.2f} {delta_text}")

        problems = directive_problems + plan_problems
        if delta > TOLERANCE:
            problems.append(f"cost {ours} exceeds the reference optimum {reference}")
        if problems:
            failures.append((case["id"], problems))
        if args.verbose:
            print(json.dumps(body["hourly_plan"], indent=2))

    print()
    if failures:
        print(f"{RED}{len(failures)} case(s) failed{RESET}\n")
        for case_id, problems in failures:
            print(f"  {case_id}")
            for problem in problems[:12]:
                print(f"    - {problem}")
        return 1

    print(f"{GREEN}all {len(cases)} case(s) passed{RESET} — plans valid and cost at the reference optimum")
    return 0


if __name__ == "__main__":
    sys.exit(main())
