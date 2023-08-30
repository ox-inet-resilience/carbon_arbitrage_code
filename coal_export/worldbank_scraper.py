import os
import subprocess
import time

import pandas as pd

from coal_export.common_data import (
    masterdata_alpha2,
    masterdata_alpha3,
    non_masterdata_alpha2,
    non_masterdata_alpha3,
)


# Solved by https://stackoverflow.com/questions/71603314/ssl-error-unsafe-legacy-renegotiation-disabled
# pd.read_excel("./import_2019_270111.xlsx")
products = {
    270111: "Coal; anthracite, whether or not pulverised, but not agglomerated",
    270112: "Coal; bituminous, whether or not pulverised, but not agglomerated",
    270119: "Coal; (other than anthracite and bituminous), whether or not pulverised but not agglomerated",
    270120: "Briquettes, ovoids and similar solid fuels; manufactured from coal",
}

masterdata_alpha3.append("WLD")

alpha3s = non_masterdata_alpha3
alpha2s = non_masterdata_alpha2
if 0:
    alpha3s = masterdata_alpha3
    alpha2s = masterdata_alpha2

for i, country in enumerate(alpha3s):
    for trade_flow in ["I", "E"]:
        for k, v in products.items():
            a2 = alpha2s[i] if country != "WLD" else "WLD"
            out_filename = f"./coal_export/download_cache/{trade_flow}_{a2}_{k}.csv"
            if os.path.isfile(out_filename):
                print("Skipping", out_filename)
                continue

            cache_fname = f"current_{trade_flow}_{country}_{k}.xlsx"
            cmd = f"/run/current-system/sw/bin/wget https://wits.worldbank.org/Download.aspx?Reporter=ALL&Year=2019&Tradeflow={trade_flow}&Partner={country}&product={k}&Type=HS6Productdata&Lang=en -O ./coal_export/download_cache/{cache_fname}"
            subprocess.check_output(
                cmd.split(" "), env={"OPENSSL_CONF": "./coal_export/openssl.conf"}
            )
            time.sleep(2.5)
            df = pd.read_excel(
                f"./coal_export/download_cache/{cache_fname}",
                sheet_name="By-HS6Product",
            )
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
