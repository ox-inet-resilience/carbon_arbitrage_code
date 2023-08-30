import util

_, nonpower_coal, _ = util.read_masterdata()
coal_export_content = util.read_json("coal_export/aggregated/combined_summed.json")
production_2019 = nonpower_coal.groupby("asset_country")._2019.sum()
masterdata_countries = set(
    "AR AU BA BD BG BR BW CA CD CL CN CO CZ DE ES ET GB GE GR HU ID IN IR JP KG KZ LA ME MG MK MM MN MW MX MZ NE NG NO NZ PE PH PK PL RO RS RU SI SK TH TJ TR TZ UA US UZ VE VN ZA ZM ZW".split()
)


def get_export_fraction(country):
    if country not in coal_export_content["export"]:
        fraction = 0
    else:
        # Exclude self export
        export = sum(
            v
            for k, v in coal_export_content["export"][country].items()
            if k != country and k in masterdata_countries
        )
        export /= 1e3  # Convert kg to tonnes of coal
        production = production_2019[country]
        fraction = export / production if production > 0 else 0
        if country in ["PE", "MZ"] and fraction > 1:
            fraction = 1
    assert 0 <= fraction <= 1, (country, fraction, export, production)
    return fraction


exports_absolute = {
    # Convert kg to tonnes of coal
    country: sum(exs.values()) / 1e3
    for country, exs in coal_export_content["export"].items()
}


def get_import_fraction(e, i):
    if e not in coal_export_content["export"]:
        fraction = 0
    else:
        _import = coal_export_content["export"][e].get(i, 0.0)
        _import /= 1e3  # Convert kg to tonnes of coal
        production = production_2019[e]
        if e in ["PE", "MZ"]:
            # If Masterdata value is smaller than WITS value, then we assume
            # 100% of the production goes to export, and so the fraction is
            # basically a renormalized version of exports_absolute.
            production = exports_absolute[e]
        elif e in ["MG", "NG"]:
            assert production == 0
        else:
            assert production > exports_absolute[e], (
                production,
                exports_absolute[e],
                e,
            )
        fraction = _import / production if production > 0 else 0
    assert 0 <= fraction <= 1, (fraction, i, e, production, _import)
    return fraction
