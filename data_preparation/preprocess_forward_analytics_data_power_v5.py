import pathlib
import sys

import pandas as pd

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import util  # noqa

df = pd.read_csv("./data_private/FA_asset_power_v5.csv.gz", compression="gzip")
df = df[df["Status"] == "active"]
df_crosswalk = pd.read_csv("./data_private/power_oil_gas_fuel_crosswalk.csv")
df_crosswalk = df_crosswalk.drop("Technology", axis=1)

cols = "FA_Unit_ID Asset_Name Sector Technology Country_ISO3 Capacity Capacity_Unit Capacity_Factor Activity Activity_Unit Emissions_Factor_Scope_1 Emissions_Factor_Scope_2 Emissions_Factor_Unit Emissions_Unit".split()

df = df[cols]
# Merge df and df_crosswalk
df = pd.merge(df, df_crosswalk, on="FA_Unit_ID", how="left")

assert len(df) == len(df[df.Activity.notna()])
assert len(df) == len(df[df.Sector == "Energy"])

df = df[df.Technology.isin(["Coal", "Oil & Gas"])]

# Renames
iso3166_df = util.read_iso3166()
alpha3_to_alpha2 = iso3166_df.set_index("alpha-3")["alpha-2"].to_dict()
alpha3_to_alpha2["XKX"] = "XK"
df["asset_country"] = df["Country_ISO3"].apply(lambda a3: alpha3_to_alpha2.get(a3, a3))
df = df.drop("Country_ISO3", axis=1)

df.Sector = "Power"
df = df.rename(
    columns={
        "Activity": "activity",
        "Sector": "sector",
        "Activity_Unit": "activity_unit",
        # "FA_Unit_ID": "uniqueforwardassetid",
        "Asset_Name": "asset_name",
        # "Technology": "subsector",
        "Capacity": "capacity",
        "Capacity_Unit": "capacity_unit",
        "Capacity_Factor": "capacity_factor",
        "Emissions_Unit": "emissions_unit",
    }
)

df.loc[df["Technology"] == "Coal", "Fuel_Classification"] = "Coal"
df.loc[df["Fuel_Classification"].isna(), "Fuel_Classification"] = "Multi fuel"
df = df[df["Fuel_Classification"].notna()]
df["Fuel_Classification"] = df["Fuel_Classification"].replace(
    {"Oil only": "Oil", "Gas only": "Gas"}
)
# Split "Multi fuel" fuel classification into half oil, half gas
condition = df["Fuel_Classification"] == "Multi fuel"
rows_to_clone = df[condition].copy()
clone1 = rows_to_clone.copy()
clone1["activity"] /= 2
clone1["Fuel_Classification"] = "Oil"
clone2 = rows_to_clone.copy()
clone2["activity"] /= 2
clone2["Fuel_Classification"] = "Gas"
# Combine the original DataFrame (excluding cloned rows) with the new clones
df = pd.concat([df[~condition], clone1, clone2], ignore_index=True)

df = df.reset_index()
df = df.rename(
    columns={"index": "uniqueforwardassetid", "Fuel_Classification": "subsector"}
)

# Unit is MtCO2
df["annualco2tyear"] = (
    (df["Emissions_Factor_Scope_1"] + df["Emissions_Factor_Scope_2"])
    * df.activity
    / 1e6
)
# Not needed since they are all "tonnes of CO2e (100-year)"
assert len(set(df.emissions_unit)) == 1
df = df.drop(
    [
        "emissions_unit",
        "Emissions_Factor_Unit",
        "Emissions_Factor_Scope_1",
        "Emissions_Factor_Scope_2",
        "Technology",
    ],
    axis=1,
)

df.to_csv(
    "./data_private/FA_asset_power_v5_preprocessed.csv.gz",
    compression="gzip",
    index=False,
)
