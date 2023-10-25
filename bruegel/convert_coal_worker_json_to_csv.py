# This file converts data in
# https://greatcarbonarbitrage.com/opportunity_costs_map.html to a format that
# can be analyzed in MSFT excel.
import pathlib
import sys
from collections import defaultdict

import pandas as pd

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import util

data = util.read_json("./plots/coal_worker_sensitivity_analysis.json")
df = pd.concat({k: pd.DataFrame(v) for k, v in data.items()})
df.to_csv(
    "plots/bruegel_compensation_coal_worker_and_retraining_cost_billions_usd.csv",
    index_label=("discount rate", "country alpha2"),
)

# Part 2
phase_out = util.read_json("cache/website_sensitivity_opportunity_costs_phase_out.json")

discount_rate_map = {
    "0%": 0.0,
    "2.8% (WACC)": 0.02795381840850683,
    "3.6% (WACC, average risk-premium 100 years)": 0.036227985389412014,
    "5%": 0.05,
    "8%": 0.08,
}


def calculate_discounted_sum(arr, discount_rate):
    # Discounted where year 2022 is the start
    return round(sum(e * ((1 + discount_rate) ** -i) for i, e in enumerate(arr)), 5)


out = defaultdict(dict)
for discount_year, v in phase_out.items():
    discount_text, year_text = discount_year.split("_")
    discount_rate = discount_rate_map[discount_text]
    out[discount_text][int(year_text)] = {
        kk: calculate_discounted_sum(vv, discount_rate) for kk, vv in v.items()
    }
df = pd.concat({k: pd.DataFrame(v) for k, v in out.items()})
# Reorder column
df = df.loc[:, [2030, 2050, 2070, 2100]]
df.to_csv(
    "plots/bruegel_compensation_missed_cash_flow_billions_usd.csv",
    index_label=("discount rate", "country alpha2"),
)
