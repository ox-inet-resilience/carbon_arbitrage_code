import math

import pandas as pd
import matplotlib.pyplot as plt

import processed_revenue
import util
from util import years_masterdata

# g = nonpower_coal.groupby("asset_country")["_2022"].sum().sort_values(ascending=False).to_frame()
# g["country_full_name"] = g.apply(lambda x: alpha2_to_full_name.get(x.name, x.name), axis=1)
# g["num_coal_workers"] = 0
# g["coal_wage"] = 0
# g.to_csv("plots/coal_producer_2022.csv")

iso3166_df = pd.read_csv("data/country_ISO-3166_with_region.csv")
alpha3_to_alpha2 = iso3166_df.set_index("alpha-3")["alpha-2"].to_dict()
alpha2_to_full_name = iso3166_df.set_index("alpha-2")["name"].to_dict()

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
df["coal_wage_usd"] = df.apply(
    lambda row: row.coal_wage_local_currency * cur_exchange[row.asset_country], axis=1
)
# df = df.sort_values(by="coal_wage_usd", ascending=False)
# df[["asset_country", "coal_wage_usd"]].to_csv("plots/coal_wage_usd.csv")
wage_usd_dict = df.set_index("asset_country")["coal_wage_usd"].to_dict()


def prepare_num_workers_dict(df):
    # Number of workers
    is_covered = df[df.num_coal_workers_source == "worldbank"]
    P_total = df._2022.sum()
    P_covered = is_covered._2022.sum()
    num_workers_is_covered = is_covered.num_coal_workers.sum()

    def f(row):
        if row.num_coal_workers_source == "worldbank":
            return row.num_coal_workers
        return int((4.7e6 - num_workers_is_covered) * row._2022 / (P_total - P_covered))

    df["num_coal_workers"] = df.apply(f, axis=1)
    df["num_coal_workers_source"] = df.num_coal_workers_source.apply(
        lambda x: x.replace("TODO", "estimated")
    )
    # df.to_csv("plots/coal_producer_2022_estimated.csv")
    num_coal_workers_dict = df.set_index("asset_country")["num_coal_workers"].to_dict()
    return num_coal_workers_dict


num_coal_workers_dict = prepare_num_workers_dict(df)

# Data analysis part
RHO_MODE = "default"
NGFS_PEG_YEAR = 2023
scenario = "Net Zero 2050"
_, nonpower_coal, _ = util.read_masterdata()
ngfss = util.read_ngfs_coal_and_power()
rho = util.calculate_rho(processed_revenue.beta, rho_mode=RHO_MODE)
countries = list(set(nonpower_coal.asset_country))

grouped = nonpower_coal.groupby("asset_country")
total_production_by_year_masterdata = []
for year in years_masterdata:
    tonnes_coal = grouped[f"_{year}"].sum()
    production = tonnes_coal / 1e9  # convert to giga tonnes of coal
    total_production_by_year_masterdata.append(production)

total_production_peg_year = total_production_by_year_masterdata[NGFS_PEG_YEAR - 2022]
years_masterdata_up_to_peg = list(range(2022, NGFS_PEG_YEAR + 1))
fraction_increase_after_peg_year = util.calculate_ngfs_fractional_increase(
    ngfss, "Coal", scenario, start_year=NGFS_PEG_YEAR
)
array_of_coal_production = total_production_by_year_masterdata[
    : len(years_masterdata_up_to_peg)
] + [
    total_production_peg_year * v_np
    for v_np in fraction_increase_after_peg_year.values()
]


def get_j_num_workers_lost_job(country, t):
    num_workers_2022 = num_coal_workers_dict[country]
    production_2022 = array_of_coal_production[0][country]
    production_t = array_of_coal_production[t - 2022][country]
    production_t_minus_1 = array_of_coal_production[t - 1 - 2022][country]
    if math.isclose(production_2022, 0):
        return 0
    return num_workers_2022 * (production_t_minus_1 - production_t) / production_2022


opportunity_cost_series = []
years = range(2023, 2101)
for t in years:
    oc = 0.0
    for country in countries:
        j_lost_job = get_j_num_workers_lost_job(country, t)
        wage = wage_usd_dict[country]
        oc += j_lost_job * 5 * wage
    # Division by 1e9 converts dollars to billion dollars
    opportunity_cost_series.append(oc / 1e9)

# i + 1, because we start from 2023
pv_opportunity_cost = sum(
    oc * util.calculate_discount(rho, i + 1)
    for i, oc in enumerate(opportunity_cost_series)
)
print("PV opportunity cost", pv_opportunity_cost, "billion dollars")

plt.figure()
plt.plot(years, opportunity_cost_series)
plt.xlabel("Time")
plt.ylabel("Compensation for lost wage (billion dollars)")
plt.savefig("plots/coal_worker_compensation.png")
