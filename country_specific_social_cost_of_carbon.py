import json

import pandas as pd
import matplotlib.pyplot as plt

import util

# The data source is public, taken from
# https://www.nature.com/articles/s41558-018-0282-y
df = pd.read_csv(
    "data/41558_2018_282_MOESM2_ESM.csv.gz",
    compression="gzip",
    converters={"prtp": str, "eta": str, "dr": str},
)


def constrain(full_df, model, prtp, ssp, rcp):
    _df = full_df.copy()
    # We constrain to one particular model, e.g. bhm_lr
    _df = _df[_df.run == model]
    # Sampling method
    sampling_method = "estimates"
    climate = "expected"
    if model == "djo":
        climate = "uncertain"
    _df = _df[_df.dmgfuncpar == sampling_method]
    # climate
    _df = _df[_df.climate == climate]
    # shared social economic pathway
    _df = _df[_df.SSP == ssp]
    # Radiative forcing
    _df = _df[_df.RCP == rcp]
    # Pure rate of time preference. NA means fix discounting.
    _df = _df[_df.prtp == prtp]
    eta = "NA"
    dr = "3"
    if prtp == "2":
        eta = "1p5"
        dr = "NA"
    _df = _df[_df.eta == eta]
    _df = _df[_df.dr == dr]
    # world_only = _df[_df.ISO3 == "WLD"].iloc[0]["50%"]
    # Exclude world
    _df = _df[_df.ISO3 != "WLD"]
    assert len(_df) > 0, (model, ssp, rcp)
    assert len(_df) == len(set(_df.ISO3)), (len(_df), len(set(_df.ISO3)))
    return _df


baseline = constrain(df, "bhm_lr", "2", "SSP2", "rcp85")
# Sort based on this df
baseline = baseline.sort_values(by="50%", ascending=False)
fifty_percent_baseline = baseline["50%"].tolist()


def get_scc_fifty(_df, country):
    return _df[_df.ISO3 == country].iloc[0]["50%"]


for model in ["bhm_lr", "bhm_sr"]:
    plt.figure()
    for ssp in ["SSP2", "SSP3", "SSP5"]:
        for rcp in ["rcp85", "rcp60"]:
            try:
                val = constrain(df, model, "2", ssp, rcp)
            except AssertionError:
                print("Skipping", model, ssp, rcp)
                continue
            fifty_percent = [get_scc_fifty(val, country) for country in baseline.ISO3]
            wld = sum(fifty_percent)
            fifty_percent = [i * 100 / wld for i in fifty_percent]

            plt.plot(fifty_percent, label=f"{ssp}, {rcp.upper()}")
    plt.legend(title="Business-as-usual scenario")
    plt.xlabel("Country")
    plt.ylabel("Country-specific SCC relative to global SCC (%)")
    plt.savefig(f"plots/country_specific_plot_scc_{model}.png")


iso3166_df = util.read_iso3166()
alpha3_to_2 = iso3166_df.set_index("alpha-3")["alpha-2"].to_dict()
if 1:
    alpha2_to_full_name = iso3166_df.set_index("alpha-2")["name"].to_dict()
    baseline["ISO2"] = baseline.ISO3.apply(lambda x: alpha3_to_2[x])
    total_scc = sum(fifty_percent_baseline)
    baseline["share_of_scc"] = baseline["50%"] / total_scc * 100
    baseline["country_name"] = baseline.ISO2.apply(lambda x: alpha2_to_full_name[x])
    baseline[["country_name", "ISO2", "share_of_scc"]].to_csv("plots/country_specific_share_scc.csv")


# Save the data
if 1:
    baseline["ISO2"] = baseline.ISO3.apply(lambda x: alpha3_to_2[x])
    mapper = baseline.set_index("ISO2")["50%"].to_dict()
    with open("plots/country_specific_scc.json", "w") as f:
        json.dump(mapper, f)
