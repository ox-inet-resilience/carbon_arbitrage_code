import pandas as pd
import matplotlib.pyplot as plt

# The data source is public, taken from
# https://www.nature.com/articles/s41558-018-0282-y
df = pd.read_csv(
    "data/41558_2018_282_MOESM2_ESM.csv.gz",
    compression="gzip",
    converters={"prtp": str, "eta": str, "dr": str},
)


def constrain(full_df, model, prtp):
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
    _df = _df[_df.SSP == "SSP2"]
    # Radiative forcing
    _df = _df[_df.RCP == "rcp85"]
    # Pure rate of time preference. NA means fix discounting.
    _df = _df[_df.prtp == prtp]
    eta = "NA"
    dr = "3"
    if prtp == "2":
        eta = "1p5"
        dr = "NA"
    _df = _df[_df.eta == eta]
    _df = _df[_df.dr == dr]
    world_only = _df[_df.ISO3 == "WLD"].iloc[0]["50%"]
    # Exclude world
    _df = _df[_df.ISO3 != "WLD"]
    assert len(_df) > 0
    assert len(_df) == len(set(_df.ISO3)), (len(_df), len(set(_df.ISO3)))
    return _df, world_only


bhm_lr_prtp_2, wld_lr_prtp_2 = constrain(df, "bhm_lr", "2")
# Sort based on this df
bhm_lr_prtp_2 = bhm_lr_prtp_2.sort_values(by="50%", ascending=False)
bhm_sr_prtp_2, wld_sr_prtp_2 = constrain(df, "bhm_sr", "2")
# djo = constrain(df, "djo")


def get_scc_fifty(_df, country):
    return _df[_df.ISO3 == country].iloc[0]["50%"]


# fifty_percent_djo = [get_scc_fifty(djo, country) for country in bhm_lr.ISO3]
fifty_percent_lr_prtp_2 = bhm_lr_prtp_2["50%"].tolist()
fifty_percent_sr_prtp_2 = [
    get_scc_fifty(bhm_sr_prtp_2, country) for country in bhm_lr_prtp_2.ISO3
]

print("Total lr 2", sum(fifty_percent_lr_prtp_2), "WLD", wld_lr_prtp_2)
print("Total sr 2", sum(fifty_percent_sr_prtp_2), "WLD", wld_sr_prtp_2)
plt.figure()
# xticks = list(range(len(fifty_percent_lr)))
# plt.bar(xticks, fifty_percent_lr)
plt.plot(fifty_percent_lr_prtp_2, label="Long run PRTP 2")
plt.plot(fifty_percent_sr_prtp_2, label="Short run PRTP 2")
# plt.plot(fifty_percent_djo, label="DJO, uncertain")
plt.legend()
plt.title("50%")
plt.savefig("plots/country_specific_scc.png")
