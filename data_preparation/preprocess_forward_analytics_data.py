import pandas as pd
import numpy as np

import util

ngfs_peg_year = 2022
df = pd.read_csv("./data_private/forward_analytics_coal.csv.gz", compression="gzip")
df = df[df["Status"] == "Operating"]
# df = df[df["Start Year"].isin(["2022", "2023", "2024", "-", "various", "TBD"])]
df["Activity"] = df["Activity"].replace("*", np.nan).astype(float)
df["Capacity"] = df["Capacity"].apply(lambda c: c if isinstance(c, float) or "-" not in c else c.split("-")[0])
df["Capacity"] = df["Capacity"].replace("*", np.nan).astype(float)

cols = [
    "Forward Company ID",
    "Unique Forward Asset ID",
    "Sector",
    "Sub-sector",
    "Technology Type",
    "Technology Sub-Type",
    "Country (Iso-3)",
    "Latitude",
    "Longitude",
    "Activity",
    "Activity unit",
    "Activity year",
    "Capacity",
    "Capacity Unit",
    "Start Year",
    "Emissions (CO2e 20 years)",
    "Emissions (CO2e 100 years)",
]
df = df[cols]
df["Activity"] = df.apply(lambda row: row.Activity if not np.isnan(row.Activity) else row.Capacity, axis=1)
df = df[df.Activity.notna()]

mapper = {k: "first" for k in cols}
df = df.groupby(["Unique Forward Asset ID", "Forward Company ID"]).agg(
    {**mapper, "Activity": "first"}
)
# df = df[~df["Activity year"].isin([np.nan])]

iso3166_df = util.read_iso3166()
alpha3_to_alpha2 = iso3166_df.set_index("alpha-3")["alpha-2"].to_dict()
df["asset_country"] = df["Country (Iso-3)"].apply(
    lambda a3: alpha3_to_alpha2.get(a3, a3)
)
df = df.drop("Country (Iso-3)", axis=1)
df = df.rename(
    columns={"Activity": "_2022", "Sector": "sector", "Activity unit": "unit"}
)

df.to_csv(
    "./data_private/forward_analytics_coal_only_preprocessed_1.0.csv.gz",
    compression="gzip",
    index=False,
)
print(df)
# g = df.groupby(["Unique Forward Asset ID", "Activity year"])
# df = df.pivot(index="Unique Forward Asset ID", columns=["Activity year", "Forward Company ID"], values="Activity")
exit()
gs = list(g.groups)
for i in range(1000):
    ith = g.get_group(gs[i])
    if len(ith) > 1:
        print(i, len(ith))
for i, row in g.get_group(gs[563]).iterrows():
    print(i, row.Activity, row)
