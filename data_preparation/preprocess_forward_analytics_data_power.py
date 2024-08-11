import pandas as pd

import util

df = pd.read_csv(
    "./data_private/v3_power_Forward_Analytics2024.csv.zip", compression="zip"
)
df = df[df["status"] == "operating"]
print(len(df))
df = df[df.activity.notna()]

# Move Coal in "Power / Energy" to "Power" sector
df.loc[df.subsector == "Coal", "sector"] = "Power"
df = df[df.sector == "Power"]
print(set(df.sector))
print(set(df.subsector))
print(len(df))
print(df.columns)


def more_robust_a3_to_a2(a3):
    a2 = alpha3_to_alpha2.get(a3, a3)
    if a2 == a3:
        if a3 == "TZ1":
            return "TZ"
        elif a3 == "KOS":
            return "XK"
    return a2


iso3166_df = util.read_iso3166()
alpha3_to_alpha2 = iso3166_df.set_index("alpha-3")["alpha-2"].to_dict()
df["asset_country"] = df["countryiso3"].apply(more_robust_a3_to_a2)
df = df.drop("countryiso3", axis=1)

df.to_csv(
    "./data_private/FA2024_power_only_preprocessed.csv.gz",
    compression="gzip",
    index=False,
)
