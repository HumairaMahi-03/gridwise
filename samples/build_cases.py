"""Generates samples/public_cases.json from the official public case pack.

Only the inputs, the expected directive semantics and the reference totals are
stored. Full reference schedules are deliberately not kept: the case pack states
that equivalent optimal schedules are accepted, so the meaningful checks are
(a) do we produce the same directives, and (b) is our cost at least as low.
"""

import json
import os

C = []


def case(cid, label, notes, demand, solar, tariff, battery, directives, totals):
    assert len(demand) == len(solar) == len(tariff) == 24, cid
    C.append({
        "id": cid,
        "label": label,
        "input": {
            "scenario_id": cid,
            "operator_notes": notes,
            "hours": [
                {"hour": h, "demand_kwh": demand[h], "solar_kwh": solar[h],
                 "tariff_bdt_per_kwh": tariff[h]}
                for h in range(24)
            ],
            "battery": dict(zip(
                ["capacity_kwh", "initial_energy_kwh", "minimum_energy_kwh",
                 "max_charge_kwh_per_hour", "max_discharge_kwh_per_hour"], battery)),
        },
        "expected_directives": directives,
        "reference_totals": dict(zip(["total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"], totals)),
    })


def sr(hours, factor):
    return {"directive_type": "solar_reduction", "hours": hours, "factor": factor}


def nc(hours):
    return {"directive_type": "no_charge_window", "hours": hours}


def nd(hours):
    return {"directive_type": "no_discharge_window", "hours": hours}


def mr(hours, kwh):
    return {"directive_type": "minimum_battery_reserve", "hours": hours, "minimum_energy_kwh": kwh}


def mg(hours, kwh):
    return {"directive_type": "max_grid_window", "hours": hours, "max_grid_kwh": kwh}


NOOP = {"directive_type": "no_op"}

# Several cases reuse the same load/solar/tariff profile.
D_A = [90, 85, 80, 80, 85, 95, 110, 130, 150, 165, 175, 180, 185, 180, 170, 165, 170, 185, 205, 215, 205, 175, 135, 105]
S_A = [0, 0, 0, 0, 0, 0, 5, 20, 50, 90, 130, 160, 180, 170, 140, 90, 45, 10, 0, 0, 0, 0, 0, 0]
T_A = [6, 6, 5, 5, 5, 6, 8, 10, 12, 14, 16, 16, 15, 14, 13, 14, 18, 22, 28, 30, 26, 18, 10, 7]

case("SAMPLE-01", "Solar cleaning + distractor",
     ["Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, "
      "usable solar should be treated as roughly 25% of the forecast.",
      "The sports office moved next month's registration deadline."],
     D_A, S_A, T_A, [220, 110, 40, 50, 50],
     [sr([12, 13], 0.25), NOOP], [2692.5, 38365, 175])

case("SAMPLE-02", "Battery charging maintenance",
     ["The battery charger will be isolated from 2 AM until 5 AM for electrical maintenance."],
     [100, 95, 90, 90, 95, 105, 120, 135, 145, 155, 165, 175, 180, 175, 165, 160, 170, 190, 210, 220, 210, 180, 145, 115],
     [0, 0, 0, 0, 0, 0, 0, 10, 30, 55, 80, 100, 110, 105, 85, 60, 30, 10, 0, 0, 0, 0, 0, 0],
     [6, 5, 4, 4, 4, 5, 7, 9, 11, 13, 15, 16, 16, 15, 14, 15, 19, 24, 31, 33, 29, 20, 11, 7],
     [200, 70, 30, 55, 55], [nc([2, 3, 4])], [2915, 42885, 180])

case("SAMPLE-03", "Emergency reserve as percentage",
     ["Keep at least 50% of the battery capacity stored in the battery from 6 PM until 9 PM "
      "for emergency operations."],
     D_A, S_A, T_A, [200, 120, 40, 50, 50],
     [mr([18, 19, 20], 100)], [2430, 35480, 205])

case("SAMPLE-04", "No-discharge protection test",
     ["For protection testing, the battery must not discharge from 6 PM until 8 PM."],
     [95, 90, 85, 85, 90, 100, 115, 130, 145, 155, 165, 175, 180, 175, 170, 165, 175, 195, 215, 225, 215, 185, 150, 120],
     [0, 0, 0, 0, 0, 0, 5, 15, 40, 75, 110, 145, 165, 155, 125, 80, 35, 5, 0, 0, 0, 0, 0, 0],
     [7, 6, 6, 5, 5, 6, 8, 10, 12, 14, 16, 16, 15, 14, 13, 14, 17, 21, 29, 32, 30, 20, 11, 8],
     [230, 130, 40, 55, 55], [nd([18, 19])], [2645, 40495, 225])

case("SAMPLE-05", "Temporary feeder grid cap",
     ["From 6 PM until 9 PM, campus grid import must not exceed 155 kWh in any hour because "
      "the feeder is operating under a temporary limit."],
     D_A, S_A, T_A, [240, 120, 30, 60, 60],
     [mg([18, 19, 20], 155)], [2430, 33950, 175])

case("SAMPLE-06", "Multiple notes with distractor",
     ["Cloud cover during panel inspection will leave about half of the forecast solar output "
      "from 10 AM until noon.",
      "The charging circuit will be unavailable from 2 PM until 4 PM.",
      "The library is extending book-return hours next week."],
     [85, 80, 80, 80, 85, 95, 110, 125, 140, 155, 165, 175, 180, 175, 170, 165, 175, 190, 205, 215, 205, 175, 140, 110],
     [0, 0, 0, 0, 0, 0, 5, 20, 55, 100, 150, 190, 210, 200, 160, 100, 50, 15, 0, 0, 0, 0, 0, 0],
     [5, 5, 5, 6, 6, 7, 8, 9, 11, 13, 15, 16, 16, 15, 14, 15, 18, 22, 27, 29, 27, 18, 10, 7],
     [220, 100, 35, 50, 50], [sr([10, 11], 0.5), nc([14, 15]), NOOP], [2395, 34090, 175])

case("SAMPLE-07", "Reserve plus transformer cap",
     ["Keep at least 90 kWh in the battery from 6 PM until 10 PM for emergency services.",
      "The evening transformer limit is 180 kWh of grid import from 7 PM until 9 PM."],
     [100, 95, 90, 90, 95, 105, 120, 135, 150, 165, 175, 185, 190, 185, 175, 170, 180, 195, 210, 225, 215, 185, 145, 115],
     [0, 0, 0, 0, 0, 0, 5, 20, 50, 90, 135, 170, 190, 180, 145, 95, 45, 10, 0, 0, 0, 0, 0, 0],
     [6, 6, 5, 5, 5, 6, 8, 10, 12, 14, 15, 16, 16, 15, 14, 15, 18, 23, 29, 32, 30, 21, 11, 7],
     [250, 150, 40, 60, 60], [mr([18, 19, 20, 21], 90), mg([19, 20], 180)], [2560, 38550, 185])

case("SAMPLE-08", "Separate charge/discharge outages",
     ["Battery charging is disabled from 11 AM until 1 PM while technicians inspect the charger.",
      "Do not discharge the battery from 5 PM until 7 PM during relay testing."],
     [90, 85, 80, 80, 85, 95, 110, 125, 140, 155, 165, 175, 180, 175, 165, 160, 170, 190, 210, 220, 210, 180, 145, 115],
     [0, 0, 0, 0, 0, 0, 0, 15, 40, 80, 120, 155, 175, 165, 130, 85, 40, 10, 0, 0, 0, 0, 0, 0],
     [6, 6, 5, 5, 5, 6, 8, 10, 12, 14, 15, 16, 15, 14, 13, 14, 18, 24, 30, 31, 28, 19, 10, 7],
     [210, 105, 35, 50, 50], [nc([11, 12]), nd([17, 18])], [2490, 37665, 210])

case("SAMPLE-09", "Reduction wording normalization",
     ["Expect an 80% reduction in rooftop solar between 11 AM and 2 PM because of inverter work.",
      "The student affairs office will publish club notices tomorrow."],
     [90, 85, 80, 80, 85, 95, 105, 120, 135, 150, 165, 175, 180, 175, 165, 160, 170, 185, 200, 210, 200, 170, 135, 105],
     [0, 0, 0, 0, 0, 0, 5, 25, 65, 120, 180, 230, 260, 240, 190, 120, 55, 10, 0, 0, 0, 0, 0, 0],
     [6, 6, 5, 5, 5, 6, 8, 10, 12, 13, 14, 15, 15, 14, 13, 14, 18, 22, 27, 29, 26, 18, 10, 7],
     [240, 120, 40, 60, 60], [sr([11, 12, 13], 0.2), NOOP], [2504, 34873, 170])

case("SAMPLE-10", "Multi-constraint evening operation",
     ["The data center requires at least 80 kWh to remain in the battery from 6 PM until 10 PM.",
      "Grid intake must stay at or below 190 kWh from 7 PM until 10 PM while the substation is constrained.",
      "A seminar room booking was moved to next week."],
     [105, 100, 95, 95, 100, 110, 125, 140, 155, 170, 180, 190, 195, 190, 180, 175, 185, 200, 215, 230, 220, 190, 150, 120],
     [0, 0, 0, 0, 0, 0, 5, 20, 50, 90, 130, 165, 185, 175, 140, 90, 40, 10, 0, 0, 0, 0, 0, 0],
     [7, 6, 6, 5, 5, 6, 8, 10, 12, 14, 16, 17, 16, 15, 14, 15, 19, 24, 30, 34, 31, 21, 11, 8],
     [260, 140, 40, 65, 65],
     [mr([18, 19, 20, 21], 80), mg([19, 20, 21], 190), NOOP], [2715, 41620, 190])

if __name__ == "__main__":
    out = {
        "_meta": {
            "source": "GridWise Public LLM-Assisted Sample Case Pack v2.0",
            "note": "Inputs and expected directive semantics only. Reference totals are the "
                    "pack's optimal values; equivalent-cost alternative schedules are accepted.",
            "hour_convention": "start-inclusive, end-exclusive (1 PM to 3 PM -> [13, 14])",
            "case_count": len(C),
        },
        "cases": C,
    }
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public_cases.json")
    with open(target, "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"wrote {target}: {len(C)} cases")
