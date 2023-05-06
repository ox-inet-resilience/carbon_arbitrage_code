import json
import subprocess
import time

import util

# These are countries that are found in Masterdata, but for nonpower coal only.
# As such, the number of countries is fewer than if it were sampled from the
# whole Masterdata.
countries = [
    "XK",
    "MM",
    "TZ",
    "GB",
    "HU",
    "RO",
    "TH",
    "CO",
    "US",
    "ZM",
    "MN",
    "PL",
    "KG",
    "ZW",
    "DE",
    "VN",
    "TJ",
    "RU",
    "CA",
    "TR",
    "LA",
    "BR",
    "NG",
    "IR",
    "MK",
    "RS",
    "UA",
    "MW",
    "ID",
    "PK",
    "KZ",
    "ME",
    "AU",
    "MX",
    "BG",
    "SK",
    "MZ",
    "JP",
    "CN",
    "ES",
    "MG",
    "ZA",
    "ET",
    "CL",
    "GR",
    "CD",
    "NE",
    "NO",
    "PE",
    "SI",
    "UZ",
    "IN",
    "CZ",
    "PH",
    "VE",
    "AR",
    "BA",
    "BD",
    "BW",
    "GE",
    "NZ",
]
countries.remove("XK")

import util
import pandas as pd
with open("cache/unilateral_emissions_GtCO2.json", "r") as f:
    data = json.load(f)
country_to_region, iso2_to_country_name = util.get_country_to_region()
for k, v in data.items():
    print(iso2_to_country_name[k], v)

iso3166_df = pd.read_csv("data/country_ISO-3166_with_region.csv")
africa_countries = list(iso3166_df[iso3166_df.region == "Africa"]["alpha-2"])
print()
for k, v in data.items():
    if k not in africa_countries:
        continue
    print(iso2_to_country_name[k], v)
exit()

def fn():
    tic = time.time()
    benefits = {}
    for country in countries:
        out = (
            subprocess.check_output(
                ["python", "analysis_main.py", country]
            )
            .decode("utf-8")
            .strip()
        )
        out = out.split("\n")[-1]
        out = out.replace("OUTPUT ", "")
        benefits[country] = float(out)

    assert len(benefits) == 60
    with open("cache/unilateral_emissions_GtCO2.json", "w") as f:
        json.dump(benefits, f)
    print("Elapsed", time.time() - tic)

fn()
