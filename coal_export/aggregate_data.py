import json
from collections import defaultdict

import pandas as pd

from coal_export.common_data import (
    masterdata_alpha2,
    non_masterdata_alpha2,
)


products = {
    270111: "Coal; anthracite, whether or not pulverised, but not agglomerated",
    270112: "Coal; bituminous, whether or not pulverised, but not agglomerated",
    270119: "Coal; (other than anthracite and bituminous), whether or not pulverised but not agglomerated",
    270120: "Briquettes, ovoids and similar solid fuels; manufactured from coal",
}

import util


(
    iso3166_df,
    _,
    _,
    _,
    _,
) = util.prepare_from_climate_financing_data()
a3_to_a2 = iso3166_df.set_index("alpha-3")["alpha-2"].to_dict()
a3_to_a2["NAM"] = "NA"
a2_to_a3 = iso3166_df.set_index("alpha-2")["alpha-3"].to_dict()
a2_to_a3["NA"] = "NAM"

masterdata_alpha2.append("WLD")

alpha2s = masterdata_alpha2 + non_masterdata_alpha2

with open("./coal_export/worldbank_country_name_map.json") as f:
    a3_to_name = json.load(f)
name_to_a3 = {v: k for k, v in a3_to_name.items()}

wld = defaultdict(dict)
for trade_flow in ["I", "E"]:
    for k, v in products.items():
        content = defaultdict(dict)
        wld[trade_flow][k] = {}
        for i, alpha2 in enumerate(alpha2s):
            filename = f"./coal_export/download_cache/{trade_flow}_{alpha2}_{k}.csv"
            alpha3 = a2_to_a3[alpha2] if alpha2 != "WLD" else "WLD"
            try:
                df = pd.read_csv(filename)
            except pd.errors.EmptyDataError:
                continue
            for idx, row in df.iterrows():
                reporter = name_to_a3[row.Reporter]
                # We include countries not in masterdata
                # if reporter not in masterdata_alpha3:
                #     continue
                # Skip
                if reporter in ["EUN", "OAS"]:
                    continue
                reporter = a3_to_a2[reporter]
                if alpha3 != "WLD":
                    partner = name_to_a3[row.Partner]
                    # We include countries not in masterdata
                    # if partner not in masterdata_alpha3:
                    #     continue
                    partner = a3_to_a2[partner]
                    assert partner == alpha2
                    # If I, reporter imports from partner
                    # If E, reporter exports to partner
                    content[reporter][partner] = row.Quantity
                else:
                    wld[trade_flow][k][reporter] = row.Quantity
        ie = "export" if trade_flow == "E" else "import"
        with open(f"coal_export/aggregated/{ie}_{k}.json", "w") as f:
            json.dump(content, f)
with open("coal_export/aggregated/world.json", "w") as f:
    json.dump(wld, f)
