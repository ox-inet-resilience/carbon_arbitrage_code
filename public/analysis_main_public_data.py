import pandas as pd
from scipy.interpolate import interp1d

import with_learning

# Constants and global variables
# Unleveraged beta. Calculated using misc/aggregate_beta.py.
beta = 0.9132710997126332
social_cost_of_carbon = 80
# Taken from
# https://www.iea.org/reports/co2-emissions-in-2022
coal_emissions_2022_iea = 15.5

# solar 50% + wind 50% LCOE
global_lcoe_average = 59.25  # $/MWh

# Global CO2 emissions according to NGFS, in million tonnes of CO2
# Global coal production according to NGFS, in EJ/yr
# From https://data.ece.iiasa.ac.at/ar6, with scenario "NGFS2_Current Policies" and "NGFS2_Net-Zero 2050"
df_ngfs = pd.read_csv("./data/ar6_snapshot_1700882949.csv")
ngfs_first_year = 2010
scenario_cps = "NGFS2_Current Policies"
scenario_nz2050 = "NGFS2_Net-Zero 2050"

# Alternatively, you may use either NGFS phase 4 or NGFS phase 3:
# # From https://data.ene.iiasa.ac.at/ngfs NGFS phase 4 GCAM 6.0
# df_ngfs = pd.read_csv("./data/ngfs_snapshot_1700884404.csv")
# ngfs_first_year = 2020
# scenario_cps = "Current Policies"
# scenario_nz2050 = "Net Zero 2050"
# # From https://data.ece.iiasa.ac.at/ngfs-phase-3 NGFS phase 3 GCAM 5.3
# df_ngfs = pd.read_csv("./data/ngfs-phase-3_snapshot_1700884745.csv")
# ngfs_first_year = 2005
# scenario_cps = "Current Policies"
# scenario_nz2050 = "Net Zero 2050"

ngfs_years = list(range(ngfs_first_year, 2101, 5))
full_years = range(2023, 2101)

owid_coal_production = pd.read_csv("./data/coal-production-by-country.csv.zip")
owid_coal_production = owid_coal_production[owid_coal_production.Code.str.len() == 3]
owid_coal_production = owid_coal_production[owid_coal_production.Year == 2022]
# This series sums up to 47825 TWh.
# If you compare with OWID_WRL ("World") in 2022, its values is 48490 TWh.
coal_production_by_country_2022 = (
    owid_coal_production.set_index("Code")["Coal production (TWh)"]
    * 1e6
    / with_learning.GJ2MWh(1)
)


def calculate_rho(beta):
    rho_f = 0.0208
    carp = 0.0299
    # Always subtract 1%
    carp -= 0.01
    # This average leverage weighted with equity lambda in the paper
    _lambda = 0.5175273490449868
    # This is chi in the paper
    tax_rate = 0.15
    # This is 0.02795381840850683
    rho = _lambda * rho_f * (1 - tax_rate) + (1 - _lambda) * (rho_f + beta * carp)
    return rho


def coal2GJ(x):
    # tonnes of coal to GJ
    # See https://en.wikipedia.org/wiki/Ton#Tonne_of_coal_equivalent
    # 1 tce is 29.3076 GJ
    # 1 tce is 8.141 MWh
    return x * 29.3076


def calculate_emissions_and_production(ngfs_scenario, rho):
    ngfs_scenario = df_ngfs[df_ngfs.Scenario == ngfs_scenario]

    emissions_row = ngfs_scenario[ngfs_scenario.Variable == "Emissions|CO2"].iloc[0]
    # Convert to GtCO2
    emissions_values = [emissions_row[str(year)] / 1e3 for year in ngfs_years]

    # Interpolate to be yearly, because the data is periodic for every 5 years.
    f_e = interp1d(ngfs_years, emissions_values)
    total_emissions = sum(f_e(y) for y in full_years)
    # We rescale to the emissions data from IEA in 2022,
    # because we want to take into account of coal only.
    total_emissions *= coal_emissions_2022_iea / f_e(2022)

    production_row = ngfs_scenario[
        ngfs_scenario.Variable == "Primary Energy|Coal"
    ].iloc[0]
    production_values = [production_row[str(year)] for year in ngfs_years]
    f_p = interp1d(ngfs_years, production_values)
    production_timeseries = [f_p(y) for y in full_years]
    production_discounted = sum(
        f_p(y) * with_learning.calculate_discount(rho, y - 2022) for y in full_years
    )

    return {
        "emissions": total_emissions,
        "production_2022": with_learning.EJ2Mcoal(f_p(2022)),
        "production_discounted": production_discounted,
        "production_timeseries": production_timeseries,
    }


def calculate_ic_wrights_law(p_timeseries_cps, p_timeseries_nz2050, rho):
    cost_obj = with_learning.InvestmentCostWithLearning()
    for year in full_years:
        delta_e_fractional_increase = (
            p_timeseries_cps[year - 2023] / p_timeseries_cps[0]
            - p_timeseries_nz2050[year - 2023] / p_timeseries_nz2050[0]
        )
        DeltaP = delta_e_fractional_increase * coal_production_by_country_2022
        discount = with_learning.calculate_discount(rho, year - 2022)
        cost_obj.calculate_investment_cost(DeltaP, year, discount)
    cost_discounted = sum([sum(e.values()) for e in cost_obj.cost_discounted])
    return cost_discounted


def calculate_ic_lcoe(ep_cps, ep_nz2050):
    # in EJ
    discounted_production_increase = (
        ep_cps["production_discounted"] - ep_nz2050["production_discounted"]
    )
    discounted_production_increase_mwh = with_learning.GJ2MWh(discounted_production_increase * 1e9)
    return global_lcoe_average * discounted_production_increase_mwh


def calculate_cost_and_benefit():
    rho = calculate_rho(beta)
    ep_cps = calculate_emissions_and_production(scenario_cps, rho)
    ep_nz2050 = calculate_emissions_and_production(scenario_nz2050, rho)

    avoided_emissions = ep_cps["emissions"] - ep_nz2050["emissions"]

    # Here we simplify by calculating it only as investment cost.
    # Here, you may choose to calculate using LCOE, or
    # cost = calculate_ic_lcoe(ep_cps, ep_nz2050)
    cost = calculate_ic_wrights_law(
        ep_cps["production_timeseries"], ep_nz2050["production_timeseries"], rho
    )
    cost /= 1e12  # trillion dollars
    benefit = avoided_emissions * social_cost_of_carbon / 1e3  # trillion dollars
    return avoided_emissions, cost, benefit, ep_cps["production_2022"]


avoided_emissions, cost, benefit, coal_production_2022 = calculate_cost_and_benefit()
print(
    f"Global coal production in 2022 {coal_production_2022:.2f} million tonnes of coal"
)
print(f"Cost {cost:.2f} trillion dollars")
print(f"Total emissions prevented {avoided_emissions:.2f} GtCO2")
print(f"Benefit {benefit:.2f} trillion dollars")
print(f"Carbon arbitrage opportunity {benefit - cost:.2f} trillion dollars")
