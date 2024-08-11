import json

import pandas as pd

import util

year = 2023
iso3166_df = util.read_iso3166()
iso3166_df = iso3166_df.set_index("alpha-3")
alpha3_to_2 = iso3166_df["alpha-2"].to_dict()

mode = "percapita"
# mode = "marketcap"
print("MODE", mode)

if mode == "percapita":
    # Data taken from
    # https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
    # "Last Updated Date","2024-06-28",
    fname = "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_2788515.csv"
    indicator = "GDP per capita (current US$)"
    outname = "all_countries_gdp_per_capita_2023.json"
else:
    # Data taken from
    # https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
    # "Last Updated Date","2024-06-28",
    fname = "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_2788787.csv"
    indicator = "GDP (current US$)"
    outname = "all_countries_gdp_marketcap_2023.json"

df = pd.read_csv(
    f"data_preparation/gdp/{fname}"
)
df = df[df["Indicator Name"] == indicator]
df = df.set_index("Country Code")
gdp_dict = df[str(year)].to_dict()

iso_keys = set(alpha3_to_2.keys())
worldbank_keys = set(gdp_dict.keys())
print("ISO", len(iso_keys))
print("worldbank.org", len(worldbank_keys))
print("ISO - worldbank", len(iso_keys - worldbank_keys))
print("worldbank - ISO", len(worldbank_keys - iso_keys))
print("intersection", len(iso_keys.intersection(worldbank_keys)))

out = {
    alpha3_to_2[a3]: gdp_pc
    for a3, gdp_pc in gdp_dict.items()
    if a3 in alpha3_to_2
}

with open(f"data/{outname}", "w") as f:
    json.dump(out, f)
