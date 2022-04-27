import json

import pandas as pd
from scipy.interpolate import interp1d

import util

# This df can be read from the xlsx file using pd.read_excel("data/TRISK-Data.xlsx", sheet_name="NGFS Scenario Data 2021 Power")
# But we save time using the csv version of it.
df = pd.read_csv("data_preparation/LearningSolarWind.csv.gz")
df = df[df.Model == util.NGFS_MODEL]
df = df[df.Region == "World"]
df = df[df.Scenario == "Net Zero 2050"]

df_solar = df[df.Variable == "Capacity|Electricity|Solar"].iloc[0]
df_wind_offshore = df[df.Variable == "Capacity|Electricity|Wind|Offshore"].iloc[0]
df_wind_onshore = df[df.Variable == "Capacity|Electricity|Wind|Onshore"].iloc[0]

df = pd.read_csv("data_preparation/Solar-Wind-Capacity.csv")
df = df.fillna(0.0, axis=1)
df_ca = {}
for tech in ["Solar", "Wind|Offshore", "Wind|Onshore"]:
    df_ca[tech] = df[
        df.Variable == f"Capacity Additions|Electricity|{tech}"
    ].iloc[0]

years_interpolated = list(range(2005, 2101))
ngfs_years = list(range(2005, 2105, 5))


def get_capacity_addition(tech):
    # We start at 2023, because the 2022 capacity is taken from IRENA 2020 data.
    return {y: df_ca[tech][str(y)] for y in range(2023, 2101)}


def get_interpolated(_df):
    ngfs_production_across_years = [_df[str(year)] for year in ngfs_years]
    f = interp1d(ngfs_years, ngfs_production_across_years)
    ngfs_production_across_years_interpolated = f(years_interpolated)
    ngfs_interpolated_dict = {
        years_interpolated[i]: ngfs_production_across_years_interpolated[i]
        for i in range(len(years_interpolated))
    }
    return ngfs_interpolated_dict


out = {}
out["solar"] = get_interpolated(df_solar)
out["offshore_wind"] = get_interpolated(df_wind_offshore)
out["onshore_wind"] = get_interpolated(df_wind_onshore)

# Not needed for now
# The unit of the production is GW!
# with open(f"data/NGFS_renewable_production_{util.NGFS_MODEL}.json", "w") as f:
#     json.dump(out, f)

out = {}
out["solar"] = get_capacity_addition("Solar")
out["offshore_wind"] = get_capacity_addition("Wind|Offshore")
out["onshore_wind"] = get_capacity_addition("Wind|Onshore")
# The unit of the production is GW/year!
with open(f"data/NGFS_renewable_additional_capacity_{util.NGFS_MODEL}.json", "w") as f:
    json.dump(out, f)
