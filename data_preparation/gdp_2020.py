import json

import pandas as pd

year = 2022
iso3166_df = pd.read_csv("data/country_ISO-3166_with_region.csv")
iso3166_df = iso3166_df.set_index("alpha-3")
alpha3_to_2 = iso3166_df["alpha-2"].to_dict()

mode = "percapita"
mode = "marketcap"
print("MODE", mode)

if mode == "percapita":
    # Data taken from
    # https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
    # "Last Updated Date","2021-12-16",
    fname = "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_3603754.csv"
    indicator = "GDP per capita (current US$)"
    outname = "all_countries_gdp_per_capita_2020.json"
else:
    # Data taken from
    # https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
    if year == 2022:
        # "Last Updated Date","2023-12-18",
        fname = "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6546136.csv"
        indicator = "GDP (current US$)"
        outname = "all_countries_gdp_marketcap_2022.json"
    else:
        # "Last Updated Date","2021-12-16",
        fname = "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3607744.csv"
        indicator = "GDP (current US$)"
        outname = "all_countries_gdp_marketcap_2020.json"

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
