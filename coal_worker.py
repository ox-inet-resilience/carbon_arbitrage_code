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
SECTOR_INCLUDED = "Power"
ic_usa = 7231

df, df_sector = util.read_forward_analytics_data(SECTOR_INCLUDED)
ngfs_df = util.read_ngfs()

countries = list(set(df_sector.asset_country))

# Giga tonnes of coal
total_production_fa = util.get_production_by_country(df_sector, SECTOR_INCLUDED)

subsector_column_map = {
    "Coal": "CFPP",
    "Oil": "OFPP",
    "Gas": "GFPP",
}


def calculate(
    rho_mode,
    subsector,
    last_year,
    do_plot=False,
    full_version=False,
    included_countries=None,
    return_summed=False,
    scenario="Net Zero 2050",
):
    years = range(NGFS_PEG_YEAR + 1, last_year + 1)
    rho = util.calculate_rho(util.beta, rho_mode=rho_mode)
    subsector_column = subsector_column_map[subsector]
    if included_countries is None:
        included_countries = countries

    # Giga tonnes of coal
    P_s2, _, _ = util.calculate_ngfs_projection(
        "production",
        total_production_fa,
        ngfs_df,
        SECTOR_INCLUDED,
        scenario,
        NGFS_PEG_YEAR,
        last_year,
        alpha2_to_alpha3,
        filter_subsector=subsector,
    )

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
        production_peg_year = P_s2[0].get(country, 0)
        if math.isclose(production_peg_year, 0):
            return 0
        production_t = P_s2[t - NGFS_PEG_YEAR][country]
        production_t_minus_1 = P_s2[t - 1 - NGFS_PEG_YEAR][country]
        # max(Pt-1 - Pt, 0) is so that the term is never negative.
        return (
            num_workers_peg_year
            * max(production_t_minus_1 - production_t, 0)
            / production_peg_year
        )

    wage_lost_series = []
    wage_lost_series_by_country = []
    for t in years:
        wl = 0.0
        wl_dict = {}
        for country in included_countries:
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
    for country in included_countries:
        opportunity_cost_by_country[country] = sum(
            wage_lost_series_by_country[i][country] * 5 * _r
            for i, _r in enumerate(rhos)
        )
        retraining_cost_by_country[country] = sum(
            wage_lost_series_by_country[i][country] * ic_usa / wage_usd_dict["US"] * _r
            for i, _r in enumerate(rhos)
        )

    retraining_cost = pv_wage_lost * ic_usa / wage_usd_dict["US"]
    pv_compensation = pv_wage_lost * 5  # billion dollars

    # Sanity check
    assert math.isclose(
        retraining_cost,
        sum(retraining_cost_by_country.values()),
    )
    assert math.isclose(
        pv_compensation,
        sum(opportunity_cost_by_country.values()),
    )

    compensation_series = wage_lost_series * 5
    # retraining_cost_series = wage_lost_series * ic_usa / wage_usd_dict["US"]

    if return_summed:
        return pv_compensation, retraining_cost

    if do_plot:
        print("PV opportunity cost", pv_compensation, "billion dollars")
        print(
            "IC retraining USA",
            ic_usa,
            "Retraining cost",
            retraining_cost,
            "billion dollars",
        )

        plt.figure()
        plt.plot(years, compensation_series)
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
        out["compensation_series"] = compensation_series
    return out


if __name__ == "__main__":
    for last_year in [2035, 2050]:
        for subsector in subsector_column_map:
            print("Subsector", subsector, last_year)
            calculate("default", subsector, last_year, do_plot=True)
    exit()

    # To be used in greatcarbonarbitrage.com
    out = {}
    for key, rho_mode in rho_mode_map.items():
        out[key] = calculate(rho_mode)
    util.write_small_json(
        out, f"plots/coal_worker_sensitivity_analysis_{SECTOR_INCLUDED}.json"
    )
