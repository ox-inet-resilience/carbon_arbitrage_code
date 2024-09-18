import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import util

iso3166_df = util.read_iso3166()
alpha3_to_alpha2 = iso3166_df.set_index("alpha-3")["alpha-2"].to_dict()
alpha2_to_full_name = iso3166_df.set_index("alpha-2")["name"].to_dict()
alpha2_to_alpha3 = iso3166_df.set_index("alpha-2")["alpha-3"].to_dict()

df_coal_worker = pd.read_csv(
    "./data_private/v0_main_powerplant_salaries.csv",
    na_filter=False,
).set_index("asset_location")


def reduce_precision(dictionary):
    # Reduce precision to save space of the JSON output
    return {k: float(f"{v:.8f}") for k, v in dictionary.items()}



# Website sensitivity params
rho_mode_map = {
    "0%": "0%",
    "2.8% (WACC)": "default",
    "3.6% (WACC, average risk-premium 100 years)": "100year",
    "5%": "5%",
    "8%": "8%",
}
# End of website sensitivity params

# Data analysis part
NGFS_PEG_YEAR = 2024
LAST_YEAR = 2050
SECTOR_INCLUDED = "Power"
scenario = "Net Zero 2050"
ic_usa = 7231

years = range(NGFS_PEG_YEAR + 1, LAST_YEAR + 1)
df, df_sector = util.read_forward_analytics_data(SECTOR_INCLUDED)
ngfs_df = util.read_ngfs()

countries = list(set(df_sector.asset_country))

# Giga tonnes of coal
total_production_fa = util.get_production_by_country(df_sector, SECTOR_INCLUDED)
# Giga tonnes of coal
production_with_ngfs_projection, _, _ = util.calculate_ngfs_projection(
    "production",
    total_production_fa,
    ngfs_df,
    SECTOR_INCLUDED,
    scenario,
    NGFS_PEG_YEAR,
    LAST_YEAR,
    alpha2_to_alpha3,
)
production_with_ngfs_projection_CPS, _, _ = util.calculate_ngfs_projection(
    "production",
    total_production_fa,
    ngfs_df,
    SECTOR_INCLUDED,
    "Current Policies",
    NGFS_PEG_YEAR,
    LAST_YEAR,
    alpha2_to_alpha3,
)

DeltaP = util.subtract_array(
    production_with_ngfs_projection_CPS, production_with_ngfs_projection
)


def calculate(rho_mode, subsector_column, do_plot=False, full_version=False):
    rho = util.calculate_rho(util.beta, rho_mode=rho_mode)
    # Wage
    wage_usd_dict = (
        df_coal_worker[f"Average Annual Salary {subsector_column} USD"]
        .replace(r"^\s*$", np.nan, regex=True)
        .astype(float)
        .to_dict()
    )
    num_coal_workers_dict = df_coal_worker[f"{subsector_column}_workers"].to_dict()

    def get_j_num_workers_lost_job(country, t):
        try:
            num_workers_peg_year = num_coal_workers_dict[country]
        except KeyError:
            # print("Missing num workers", country)
            num_workers_peg_year = 0
        production_peg_year = DeltaP[0][country]
        production_t = DeltaP[t - NGFS_PEG_YEAR][country]
        production_t_minus_1 = DeltaP[t - 1 - NGFS_PEG_YEAR][country]
        if math.isclose(production_peg_year, 0):
            return 0
        return (
            num_workers_peg_year
            * (production_t_minus_1 - production_t)
            / production_peg_year
        )

    wage_lost_series = []
    wage_lost_series_by_country = []
    for t in years:
        wl = 0.0
        wl_dict = {}
        for country in countries:
            j_lost_job = get_j_num_workers_lost_job(country, t)
            wage = wage_usd_dict.get(country, 0)
            val = j_lost_job * wage
            wl += val
            wl_dict[country] = val
        # Division by 1e9 converts dollars to billion dollars
        wage_lost_series.append(wl / 1e9)
        wage_lost_series_by_country.append({k: v / 1e9 for k, v in wl_dict.items()})
    wage_lost_series = np.array(wage_lost_series)

    # i + 1, because we start from 2023
    pv_wage_lost = sum(
        wl * util.calculate_discount(rho, i + 1)
        for i, wl in enumerate(wage_lost_series)
    )

    opportunity_cost_by_country = {}
    retraining_cost_by_country = {}
    rhos = [util.calculate_discount(rho, i + 1) for i in range(len(wage_lost_series))]
    for country in countries:
        opportunity_cost_by_country[country] = sum(
            wage_lost_series_by_country[i][country] * 5 * _r
            for i, _r in enumerate(rhos)
        )
        retraining_cost_by_country[country] = sum(
            wage_lost_series_by_country[i][country] * ic_usa / wage_usd_dict["US"] * _r
            for i, _r in enumerate(rhos)
        )

    retraining_cost = pv_wage_lost * ic_usa / wage_usd_dict["US"]
    pv_opportunity_cost = pv_wage_lost * 5  # billion dollars

    # Sanity check
    assert math.isclose(
        retraining_cost,
        sum(retraining_cost_by_country.values()),
    )
    assert math.isclose(
        pv_opportunity_cost,
        sum(opportunity_cost_by_country.values()),
    )

    opportunity_cost_series = wage_lost_series * 5
    if do_plot:
        print("PV opportunity cost", pv_opportunity_cost, "billion dollars")
        print(
            "IC retraining USA",
            ic_usa,
            "Retraining cost",
            retraining_cost,
            "billion dollars",
        )

        plt.figure()
        plt.plot(years, opportunity_cost_series)
        plt.xlabel("Time")
        plt.ylabel("Compensation for lost wage (billion dollars)")
        plt.savefig("plots/coal_worker_compensation.png")

    out = {
        "compensation workers for lost wages": reduce_precision(
            opportunity_cost_by_country  # billion dollars
        ),
        "retraining costs": reduce_precision(
            retraining_cost_by_country
        ),  # billion dollars
    }
    # This is used in the battery branch for yearly climate financing.
    if full_version:
        out["wage_lost_series"] = wage_lost_series
        out["opportunity_cost_series"] = opportunity_cost_series
    return out


if __name__ == "__main__":
    subsector_column = "CFPP"
    calculate("default", subsector_column, do_plot=True)
    exit()

    # To be used in greatcarbonarbitrage.com
    out = {}
    for key, rho_mode in rho_mode_map.items():
        out[key] = calculate(rho_mode)
    util.write_small_json(
        out, f"plots/coal_worker_sensitivity_analysis_{SECTOR_INCLUDED}.json"
    )
