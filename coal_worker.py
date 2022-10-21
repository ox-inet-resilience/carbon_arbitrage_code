import pandas as pd

# g = nonpower_coal.groupby("asset_country")["_2022"].sum().sort_values(ascending=False).to_frame()
# g["country_full_name"] = g.apply(lambda x: alpha2_to_full_name.get(x.name, x.name), axis=1)
# g["num_coal_workers"] = 0
# g["coal_wage"] = 0
# g.to_csv("plots/coal_producer_2022.csv")

iso3166_df = pd.read_csv("data/country_ISO-3166_with_region.csv")
alpha3_to_alpha2 = iso3166_df.set_index("alpha-3")["alpha-2"].to_dict()
alpha2_to_full_name = iso3166_df.set_index("alpha-2")["name"].to_dict()
# delete this
# df_currency = pd.read_csv("data/oecd_currency_exchange_rate_2021.csv")
# df_currency["alpha-2"] = df_currency.LOCATION.apply(lambda x: alpha3_to_alpha2.get(x, x))
# cur_exchange = df_currency.set_index("alpha-2")["Value"].to_dict()
# end of delete this

df = pd.read_csv("data/coal_producer_2022.csv")
# Sanity check -- no zero salary
assert len(df[df.coal_wage_local_currency == 0]) == 0

# Source: google.com as of oct 21 2022
cur_exchange = dict(
    CN=0.14,
    US=1,
    IN=0.012,
    AU=1.60,
    RU=0.016,
    ID=0.000064,
    ZA=0.054,
    DE=0.98,  # euro
    KZ=0.0021,
    PL=0.20,
    TR=0.054,
    CO=0.00021,
    GR=0.98,  # euro
    CA=0.72,
    UA=0.027,
    VN=0.000040,
    MN=0.000295089,
    BG=0.50,
    CZ=0.040,
    RS=0.0083,
    RO=0.20,
    TH=0.026,
    BR=0.19,
    PH=0.017,
    MZ=0.016,
    BA=0.50,
    LA=0.000058,
    UZ=0.000089,
    MX=0.050,
    ZW=0.003106,
    GB=1.11,
    XK=0.98,  # euro
    HU=0.0024,
    IR=0.000024,
    MK=0.016,
    PK=0.0045,
    ES=0.98,  # euro
    SI=0.98,  # euro
    SK=0.98,  # euro
    NZ=0.56,
    ME=0.98,  # euro
    BW=0.074,
    TJ=0.098,
    CL=0.0010,
    VE=0.119596,
    MM=0.00047,
    AR=0.0065,
    TZ=0.00043,
    ZM=0.062,
    GE=0.36,
    JP=0.0066,
    KG=0.012,
    BD=0.0098,
    MW=0.00097,
    NE=0.0015,
    NO=0.094,
    PE=0.25,
    ET=0.019,
    CD=0.00049,
    NG=0.0023,
    MG=0.00023,
)

# Wage
df["coal_wage_usd"] = df.apply(lambda row: row.coal_wage_local_currency * cur_exchange[row.asset_country], axis=1)
df = df.sort_values(by="coal_wage_usd", ascending=False)
df[["asset_country", "coal_wage_usd"]].to_csv("plots/coal_wage_usd.csv")
exit()

has_data = df[df.num_coal_workers_source == "worldbank"]
no_data = df[df.num_coal_workers_source == "TODO"]

P_total = df._2022.sum()
P_covered = has_data._2022.sum()
num_workers_has_data = has_data.num_coal_workers.sum()

def f(row):
    if row.num_coal_workers_source == "worldbank":
        return row.num_coal_workers
    return int((4.7e6 - num_workers_has_data) * row._2022 / (P_total - P_covered))
df["num_coal_workers"] = df.apply(f, axis=1)
df["num_coal_workers_source"] = df.num_coal_workers_source.apply(lambda x: x.replace("TODO", "estimated"))
df.to_csv("plots/coal_producer_2022_estimated.csv")
