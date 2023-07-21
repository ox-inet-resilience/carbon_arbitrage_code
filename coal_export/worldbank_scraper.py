import os
import subprocess
import time

import pandas as pd

# Solved by https://stackoverflow.com/questions/71603314/ssl-error-unsafe-legacy-renegotiation-disabled
#pd.read_excel("./import_2019_270111.xlsx")
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
masterdata_alpha3.append("WLD")
for i, country in enumerate(masterdata_alpha3):
    for trade_flow in ["I", "E"]:
        for k, v in products.items():
            a2 = masterdata_alpha2[i] if country != "WLD" else "WLD"
            out_filename = f"./download_cache/{trade_flow}_{a2}_{k}.csv"
            if os.path.isfile(out_filename):
                print("Skipping", out_filename)
                continue

            cmd = f"/run/current-system/sw/bin/wget https://wits.worldbank.org/Download.aspx?Reporter=ALL&Year=2019&Tradeflow={trade_flow}&Partner={country}&product={k}&Type=HS6Productdata&Lang=en -O download_cache/current.xlsx"
            subprocess.check_output(cmd.split(" "), env={"OPENSSL_CONF": "openssl.conf"})
            time.sleep(5)
            df = pd.read_excel("./download_cache/current.xlsx", sheet_name="By-HS6Product")
            if len(df) == 0:
                print("Empty", out_filename)
                subprocess.check_output(["touch", out_filename])
                continue
            trade_flow_str = "Import" if trade_flow == "I" else "Export"
            assert list(set(df.TradeFlow))[0] == trade_flow_str
            units = list(set(df["Quantity Unit"]))
            if "Kg" not in units:
                print("PROBLEM WITH", country, trade_flow, k, units)
                subprocess.check_output(["touch", out_filename])
                continue
            assert len(units) <= 2
            assert int(list(set(df.Year))[0]) == 2019
            assert list(set(df["Product Description"]))[0] == v
            df.fillna(0, inplace=True)
            subset = df[["Reporter", "Partner", "Quantity"]]
            subset.to_csv(out_filename, index=False)
