# This file converts data in
# https://greatcarbonarbitrage.com/opportunity_costs_map.html to a format that
# can be analyzed in MSFT excel.
import pathlib
import sys

import pandas as pd

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import util


def do_round(x):
    return round(x, 4)


# We use only the default 2.8%
discount_rate_map = {
    "0%": 0.0,
    "2.8% (WACC)": 0.02795381840850683,
    "3.6% (WACC, average risk-premium 100 years)": 0.036227985389412014,
    "5%": 0.05,
    "8%": 0.08,
}
discount_rate_text = "2.8% (WACC)"
discount_rate = discount_rate_map[discount_rate_text]

data = util.read_json("./plots/coal_worker_sensitivity_analysis.json")
data = data[discount_rate_text]
df1 = pd.DataFrame(data)

# Part 2
phase_out = util.read_json("cache/website_sensitivity_opportunity_costs_phase_out.json")


def calculate_discounted_sum(arr, discount_rate):
    # Discounted where year 2022 is the start
    return do_round(sum(e * ((1 + discount_rate) ** -i) for i, e in enumerate(arr)))


out = {}
years = [2030, 2050, 2100]
for year in years:
    key = f"{discount_rate_text}_{year}"
    v = phase_out[key]
    out[year] = {
        kk: calculate_discounted_sum(vv, discount_rate) for kk, vv in v.items()
    }
df2 = pd.DataFrame(out)
# Reorder column
df2 = df2.loc[:, years]

df = df1.merge(df2, left_index=True, right_index=True)
df = df.sort_index()
df = df.rename(
    columns={year: f"opportunity costs missed cash flow {year}" for year in years}
)
df = df.rename(
    columns={
        "compensation workers for lost wages": "opportunity costs compensation workers for lost wages",
        "retraining costs": "opportunity costs retraining costs",
    }
)
for year in years:
    df[f"opportunity costs total {year}"] = do_round(
        df["opportunity costs compensation workers for lost wages"]
        + df["opportunity costs retraining costs"]
        + df[f"opportunity costs missed cash flow {year}"],
    )

# Further rounding
for col in [
    "opportunity costs compensation workers for lost wages",
    "opportunity costs retraining costs",
]:
    df[col] = do_round(df[col])
df.to_csv("plots/oc_coal_worker_billion_usd.csv", index_label="alpha2")
