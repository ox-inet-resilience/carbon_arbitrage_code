import util

_, nonpower_coal, _ = util.read_masterdata()
coal_export_content = util.read_json("coal_export/aggregated/combined_summed.json")
production_2019 = nonpower_coal.groupby("asset_country")._2019.sum()


def get_export_fraction(country):
    if country not in coal_export_content["export"]:
        fraction = 0
    else:
        # Exclude self export
        export = sum(
            v for k, v in coal_export_content["export"][country].items() if k != country
        )
        export /= 1e3  # Convert kg to tonnes of coal
        production = production_2019[country]
        fraction = export / production if production > 0 else 0
        if country == "PE" and fraction > 1:
            fraction = 1
    assert 0 <= fraction <= 1, (country, fraction)
    return fraction


def get_import_fraction(e, i):
    if e not in coal_export_content["export"]:
        fraction = 0
    else:
        _import = coal_export_content["export"][e].get(i, 0.0)
        _import /= 1e3  # Convert kg to tonnes of coal
        production = production_2019[e]
        fraction = _import / production if production > 0 else 0
    assert 0 <= fraction <= 1
    return fraction
