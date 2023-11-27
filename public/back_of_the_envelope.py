# This self-contained code calculates the carbon arbitrage opportunity using
# the following formulas:
# carbon_arbitrage_opportunity = benefit - cost
# where
# benefit = social_cost_of_carbon * (cumulative_emissions_current_policies - cumulative_emissions_Net_Zero_2050)
# The cumulative emissions are summed from 2023 to 2100.
# cost = renewable_LCOE * (discounted_cumulative_production_current_policies - discounted_cumulative_production_Net_Zero_2050)
# The data sources are taken from IEA and NGFS.
# Differences from the full version:
# - Uses public data
# - Calculates investment cost using renewables LCOE. For the version that uses Wright's law learning (used in the full version), see public/analysis_main_public_data.py
# - Doesn't calculate the residual benefit, which is about 21 trillion dollars in the paper
# - Doesn't calculate the opportunity cost, which is about 50 billion dollars in the paper
# The result (in trillion dollars):
# - cost: 72.88
# - benefit: 96.36
# - carbon arbitrate opportunity: 23.48
import pandas as pd
from scipy.interpolate import interp1d

# Constants and global variables
social_cost_of_carbon = 80
# Unleveraged beta. Calculated using misc/aggregate_beta.py.
beta = 0.9132710997126332
# Taken from
# https://www.iea.org/data-and-statistics/charts/global-coal-production-2000-2025
# in 2022
coal_production_2022 = 8318  # million tonnes of coal
# Compare this with the aggregate of the AR data (private)
coal_production_2022 = 6552.1595  # AR data
# Taken from
# https://www.iea.org/reports/co2-emissions-in-2022
coal_emissions_2022_iea = 15.5

# solar 50% + wind 50% LCOE
global_lcoe_average = 59.25  # $/MWh

# Global CO2 emissions according to NGFS, in million tonnes of CO2
# Global coal production according to NGFS, in EJ/yr
# From https://data.ece.iiasa.ac.at/ar6, with scenario "NGFS2_Current Policies" and "NGFS2_Net-Zero 2050"
df_ngfs = pd.read_csv("./data/ar6_snapshot_1700882949.csv")

ngfs_years = list(range(2010, 2101, 5))
full_years = range(2023, 2101)


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


def calculate_discount(rho, deltat):
    return (1 + rho) ** -deltat


def EJ2MWh(x):
    # EJ to J
    joule = x * 1e18
    # J to Wh
    wh = joule / 3600
    # Wh to MWh
    return wh / 1e6


def EJ2Mcoal(x):
    coal = x * 1e9 / 29.3076
    return coal / 1e6  # million tonnes of coal


def calculate_emissions_and_production(ngfs_scenario):
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
    rho = calculate_rho(beta)
    production_discounted = sum(
        f_p(y) * calculate_discount(rho, y - 2022) for y in full_years
    )

    return {
        "emissions": total_emissions,
        "production_2022": EJ2Mcoal(f_p(2022)),  # from EJ to million tonnes of coal
        "production_discounted": production_discounted,
    }


def calculate_cost_and_benefit():
    ep_cps = calculate_emissions_and_production("NGFS2_Current Policies")
    ep_nz2050 = calculate_emissions_and_production("NGFS2_Net-Zero 2050")

    avoided_emissions = ep_cps["emissions"] - ep_nz2050["emissions"]

    # in EJ
    discounted_production_increase = (
        ep_cps["production_discounted"] - ep_nz2050["production_discounted"]
    )
    discounted_production_increase_mwh = EJ2MWh(discounted_production_increase)

    # Here we simplify by calculating it only as investment cost.
    cost = global_lcoe_average * discounted_production_increase_mwh
    cost /= 1e12  # trillion dollars
    benefit = avoided_emissions * social_cost_of_carbon / 1e3  # trillion dollars
    return avoided_emissions, cost, benefit, ep_cps["production_2022"]


avoided_emissions, cost, benefit, coal_production_2022 = calculate_cost_and_benefit()
print(
    f"""
Global coal production in 2022 {coal_production_2022:.2f} million tonnes of coal
Cost {cost:.2f} trillion dollars
Total emissions prevented {avoided_emissions:.2f} GtCO2
Benefit {benefit:.2f} trillion dollars
Carbon arbitrage opportunity {benefit - cost:.2f} trillion dollars"""
)
