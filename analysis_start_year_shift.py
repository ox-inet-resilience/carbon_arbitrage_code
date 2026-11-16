"""
PV benefits, PV costs and NPV of the coal phase-out (NZ2050 vs Current
Policies) for several end years, comparing a phase-out that starts in 2024
with one that starts in 2028.

For a later start year:
- No renewable capacity is built and no emissions are avoided before the
  start year (the world follows BAU). Opportunity costs and benefits before
  the start year are set to 0.
- From the start year onward, the BAU - NZ2050 difference in coal production
  and emissions is the same as in the original run; the NGFS projections are
  not altered.
- The renewable investment cost model is rerun, so all capacity needed in the
  start year is built in that year, Wright's law learning starts later, and
  the residual benefit reflects the new build-out.
- Costs are discounted to `discount_base_year` instead of 2022. Benefits are
  not discounted in the model (emissions avoided times a constant social cost
  of carbon).

Run from the repo root: python analysis_start_year_shift.py
"""

import json
import os

import analysis_main as am
import processed_revenue
import util

END_YEARS = [2030, 2040, 2050, 2070, 2100]
# (start year, discount base year)
CASES = [(2024, 2022), (2028, 2026), (2028, 2027)]

_START_YEAR = 2024
_original_calculate_investment_cost = am.InvestmentCostNewMethod.calculate_investment_cost


def _calculate_investment_cost_delayed(self, DeltaP, year, discount):
    # No renewable build-out before the phase-out start year.
    if year < _START_YEAR and not isinstance(DeltaP, float):
        DeltaP = DeltaP * 0.0
    return _original_calculate_investment_cost(self, DeltaP, year, discount)


am.InvestmentCostNewMethod.calculate_investment_cost = (
    _calculate_investment_cost_delayed
)


def run_yearly(start_year):
    """Return yearly costs (discounted to 2022) and benefits for each end year."""
    global _START_YEAR
    _START_YEAR = start_year
    f = am.floatify_array_of_mixed_objs
    results = {}
    for last_year in END_YEARS:
        # The model truncates at MID_YEAR; 2100 is always computed.
        am.MID_YEAR = last_year if last_year != 2100 else 2050
        summary = am.run_cost1(x=1, to_csv=False, do_round=False)
        yearly = am.run_cost1(x=1, to_csv=False, do_round=False, return_yearly=True)
        cond = f"2022-{last_year} 2DII + Net Zero 2050 Scenario"
        y = yearly[cond]
        years = list(range(2022, last_year + 1))
        oc = f(y["opportunity_cost"])
        ic = f(y["investment_cost"])
        benefit = list(y["benefit_non_discounted"])
        assert len(years) == len(oc) == len(ic) == len(benefit)
        assert all(ic[i] == 0 for i, yr in enumerate(years) if yr < start_year)
        results[last_year] = {
            "residual_benefit": summary[
                "Carbon arbitrage residual benefit (in trillion dollars)"
            ][cond],
            "yearly": {
                yr: {
                    "opportunity_cost": o if yr >= start_year else 0.0,
                    "investment_cost": i,
                    "benefit": b if yr >= start_year else 0.0,
                }
                for yr, o, i, b in zip(years, oc, ic, benefit)
            },
        }
    return results


def summarize(results, rho, discount_base_year):
    # Costs are discounted to 2022 in the model; re-base them.
    rebase = (1 + rho) ** (discount_base_year - 2022)
    out = {}
    for last_year, r in results.items():
        v = r["yearly"].values()
        benefit = sum(x["benefit"] for x in v)
        cost = rebase * sum(x["opportunity_cost"] + x["investment_cost"] for x in v)
        residual = r["residual_benefit"]
        out[last_year] = {
            "PV benefits": benefit,
            "PV benefits incl. residual": benefit + residual,
            "PV costs": cost,
            "NPV": benefit - cost,
            "NPV incl. residual": benefit + residual - cost,
        }
    return out


if __name__ == "__main__":
    rho = util.calculate_rho(processed_revenue.beta, rho_mode=am.RHO_MODE)
    print(f"rho = {rho:.4f}, SCC = {am.social_cost_of_carbon}")
    yearly_by_start = {}
    output = {}
    for start_year, base_year in CASES:
        if start_year not in yearly_by_start:
            yearly_by_start[start_year] = run_yearly(start_year)
        table = summarize(yearly_by_start[start_year], rho, base_year)
        label = f"start {start_year}, discounted to {base_year}"
        output[label] = table
        print(f"\n### Start {start_year}, costs discounted to {base_year}")
        cols = list(next(iter(table.values())).keys())
        print("| End | " + " | ".join(cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        for last_year, row in table.items():
            print(f"| {last_year} | " + " | ".join(f"{row[c]:.2f}" for c in cols) + " |")

    os.makedirs("output", exist_ok=True)
    with open("output/start_year_shift.json", "w") as fp:
        json.dump(
            {"rho": rho, "summary": output, "yearly": yearly_by_start}, fp, indent=1
        )
