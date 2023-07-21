import json
from collections import defaultdict

import pandas as pd


products = {
    270111: "Coal; anthracite, whether or not pulverised, but not agglomerated",
    270112: "Coal; bituminous, whether or not pulverised, but not agglomerated",
    270119: "Coal; (other than anthracite and bituminous), whether or not pulverised but not agglomerated",
    270120: "Briquettes, ovoids and similar solid fuels; manufactured from coal",
}

# Excluding XK
masterdata_alpha2 = "AR AU BA BD BG BR BW CA CD CL CN CO CZ DE ES ET GB GE GR HU ID IN IR JP KG KZ LA ME MG MK MM MN MW MX MZ NE NG NO NZ PE PH PK PL RO RS RU SI SK TH TJ TR TZ UA US UZ VE VN ZA ZM ZW".split()
# Excluding XK
masterdata_alpha3 = "ARG AUS BIH BGD BGR BRA BWA CAN COD CHL CHN COL CZE DEU ESP ETH GBR GEO GRC HUN IDN IND IRN JPN KGZ KAZ LAO MNE MDG MKD MMR MNG MWI MEX MOZ NER NGA NOR NZL PER PHL PAK POL ROU SRB RUS SVN SVK THA TJK TUR TZA UKR USA UZB VEN VNM ZAF ZMB ZWE".split()
a3_to_a2 = {}
for i, a2 in enumerate(masterdata_alpha2):
    a3_to_a2[masterdata_alpha3[i]] = a2
masterdata_alpha2.append("WLD")

with open("./worldbank_country_name_map.json") as f:
    a3_to_name = json.load(f)
name_to_a3 = {v: k for k,v in a3_to_name.items()}

wld = defaultdict(dict)
for trade_flow in ["I", "E"]:
    for k, v in products.items():
        content = defaultdict(dict)
        wld[trade_flow][k] = {}
        for i, alpha2 in enumerate(masterdata_alpha2):
            filename = f"./download_cache/{trade_flow}_{alpha2}_{k}.csv"
            alpha3 = masterdata_alpha3[i] if alpha2 != "WLD" else "WLD"
            try:
                df = pd.read_csv(filename)
            except pd.errors.EmptyDataError:
                continue
            for idx, row in df.iterrows():
                reporter = name_to_a3[row.Reporter]
                if reporter not in masterdata_alpha3:
                    continue
                reporter = a3_to_a2[reporter]
                if alpha3 != "WLD":
                    partner = name_to_a3[row.Partner]
                    if partner not in masterdata_alpha3:
                        continue
                    partner = a3_to_a2[partner]
                    content[reporter][partner] = row.Quantity
                else:
                    wld[trade_flow][k][reporter] = row.Quantity
        with open(f"aggregated/{trade_flow}_{k}.json", "w") as f:
            json.dump(content, f)
with open("aggregated/world.json", "w") as f:
    json.dump(wld, f)
