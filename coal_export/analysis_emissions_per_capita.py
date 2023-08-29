import json

import util
from coal_export.common import nonpower_coal, get_export_fraction, get_import_fraction

# Emissions in 2020
# In tonnes of coal
nonpower_coal["emissions_2020"] = nonpower_coal._2020 * nonpower_coal.emissions_factor
emissions = nonpower_coal.groupby("asset_country").emissions_2020.sum()
gdp_marketcap_dict = util.read_json("data/all_countries_gdp_marketcap_2020.json")
gdp_per_capita_dict = util.read_json("data/all_countries_gdp_per_capita_2020.json")


emissions_per_capita = {}
for k, v in emissions.to_dict().items():
    if k not in gdp_marketcap_dict:
        continue
    population = gdp_marketcap_dict[k] / gdp_per_capita_dict[k]
    # tonnes of coal
    emissions_per_capita[k] = v / population
# Sort descending
emissions_per_capita = dict(
    sorted(emissions_per_capita.items(), key=lambda item: item[1], reverse=True)
)

(
    _,
    iso3166_df_alpha2,
    _,
    _,
    _,
) = util.prepare_from_climate_financing_data()
alpha2_to_full_name = iso3166_df_alpha2["name"].to_dict()

# Full name
emissions_per_capita = {
    alpha2_to_full_name[k]: v for k, v in emissions_per_capita.items()
}
print(json.dumps(emissions_per_capita, indent=2))


# 2. Modified by export
emissions_modified_by_export = {}

for country, e in emissions.to_dict().items():
    modified_e = e * (1 - get_export_fraction(country)) + sum(
        v * get_import_fraction(k, country)
        for k, v in emissions.items()
        if k != country  # avoid self
    )
    emissions_modified_by_export[country] = modified_e

emission_per_capita_modified_by_export = {}
for k, v in emissions_modified_by_export.items():
    if k not in gdp_marketcap_dict:
        continue
    population = gdp_marketcap_dict[k] / gdp_per_capita_dict[k]
    # tonnes of coal
    emission_per_capita_modified_by_export[k] = v / population
# Sort descending
emission_per_capita_modified_by_export = dict(
    sorted(emission_per_capita_modified_by_export.items(), key=lambda item: item[1], reverse=True)
)
# Full name
emission_per_capita_modified_by_export = {
    alpha2_to_full_name[k]: v for k, v in emission_per_capita_modified_by_export.items()
}
print(json.dumps(emission_per_capita_modified_by_export, indent=2))
