import csv
import json

import util

avoided_emissions = {
    "CN": 324.31864198167125,
    "AU": 210.88392255442193,
    "US": 169.76007984197312,
    "IN": 151.62685162688658,
    "RU": 134.41053934436312,
    "ID": 91.08973829998874,
    "ZA": 88.31023529194344,
    "CA": 26.35656351707315,
    "PL": 23.138517157332764,
    "KZ": 22.11787208149155,
    "CO": 20.71410638221643,
    "DE": 17.716209938697503,
    "MZ": 17.69996535412617,
    "MN": 17.173148952818888,
    "UA": 11.191927795456502,
    "TR": 10.194717861180987,
    "VN": 10.152055694546538,
    "BW": 8.604314544061468,
    "GR": 7.573094423320425,
    "BR": 6.217290295422924,
    "CZ": 4.358417431326265,
    "BG": 4.1167546903089915,
    "RO": 3.8021364308579666,
    "TH": 3.5351996529100034,
    "RS": 3.502196239391694,
    "GB": 2.9816379175094783,
    "UZ": 2.9765779338010225,
    "PH": 2.9422508136490686,
    "ZW": 2.841921792053882,
    "NZ": 2.8324118700906484,
    "MX": 2.7321817378678874,
    "BD": 2.351176776282867,
    "BA": 1.8081467019809279,
    "LA": 1.7020376971081606,
    "IR": 1.607164480354883,
    "CL": 1.4227111251730238,
    "ES": 1.0544365328652607,
    "PK": 0.9966186815117757,
    "VE": 0.9130698572881613,
    "TZ": 0.8708146176678915,
    "HU": 0.8350768544942265,
    "ME": 0.6350296331043865,
    "SK": 0.4920182078251383,
    "ZM": 0.4834945171203027,
    "SI": 0.4809886062409861,
    "MG": 0.4205952819616051,
    "TJ": 0.4063008471678118,
    "MK": 0.30028583347475696,
    "GE": 0.299913882220746,
    "AR": 0.2783001174679414,
    "MM": 0.20621830932783047,
    "JP": 0.20024838735900657,
    "KG": 0.1855434374567874,
    "MW": 0.08492276037340747,
    "NG": 0.07525189031299664,
    "NE": 0.06315729833376413,
    "PE": 0.05601648436133825,
    "NO": 0.039792934993789636,
    "ET": 0.007854922444545226,
    "CD": 0.0008262877393210473,
}

_, nonpower_coal, _ = util.read_masterdata()
coal_export_content = util.read_json("coal_export/aggregated/combined_summed.json")
production_2019 = nonpower_coal.groupby("asset_country")._2019.sum()


def get_export_fraction(country):
    if country not in coal_export_content["E"]:
        fraction = 0
    else:
        # Exclude self export
        export = sum(
            v for k, v in coal_export_content["E"][country].items() if k != country
        )
        export /= 1e3  # Convert kg to tonnes of coal
        production = production_2019[country]
        fraction = export / production if production > 0 else 0
        if country == "PE" and fraction > 1:
            fraction = 1
    assert 0 <= fraction <= 1, (country, fraction)
    return fraction


def get_import_fraction(e, i):
    if e not in coal_export_content["E"]:
        fraction = 0
    else:
        _import = coal_export_content["E"][e].get(i, 0.0)
        _import /= 1e3  # Convert kg to tonnes of coal
        production = production_2019[e]
        fraction = _import / production if production > 0 else 0
    assert 0 <= fraction <= 1
    return fraction


emissions_modified_by_export = {}

for country, emissions in avoided_emissions.items():
    modified_e = emissions * (1 - get_export_fraction(country)) + sum(
        v * get_import_fraction(k, country)
        for k, v in avoided_emissions.items()
        if k != country  # avoid self
    )
    emissions_modified_by_export[country] = modified_e

# Sort from largest value to smallest
emissions_modified_by_export = dict(
    sorted(emissions_modified_by_export.items(), key=lambda item: item[1], reverse=True)
)

(
    _,
    iso3166_df_alpha2,
    _,
    _,
    _,
) = util.prepare_from_climate_financing_data()
alpha2_to_full_name = iso3166_df_alpha2["name"].to_dict()
emissions_full_name_version = {
    alpha2_to_full_name[k]: v for k, v in emissions_modified_by_export.items()
}
print("sanity check should add up to ~1424", sum(emissions_full_name_version.values()))
with open("plots/avoided_emissions_modified_by_coal_export.csv", "w") as f:
    for k, v in emissions_full_name_version.items():
        f.write(f"{k},{v}\n")
# print(json.dumps(emissions_full_name_version, indent=2))
avoided_emissions_full_name = {
    alpha2_to_full_name[k]: v for k, v in avoided_emissions.items()
}
with open("plots/avoided_emissions_nonadjusted.csv", "w") as f:
    for k, v in avoided_emissions_full_name.items():
        f.write(f"{k},{v}\n")
