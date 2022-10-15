import copy
import json
import math
import os
from collections import defaultdict
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker

import util
from util import years_masterdata
import processed_revenue
from util import (
    social_cost_of_carbon,
    world_gdp_2020,
)


# Ensure that plots directory exists
os.makedirs("plots", exist_ok=True)

# Params that can be modified
lcoe_mode = "solar+wind"
# lcoe_mode="solar+wind+gas"
ENABLE_NEW_METHOD = 1
ENABLE_RENEWABLE_GRADUAL_DEGRADATION = 1
ENABLE_RENEWABLE_30Y_LIFESPAN = 1
ENABLE_WRIGHTS_LAW = 1
ENABLE_WRIGHT_USE_NGFS_DATA = 0
ENABLE_RESIDUAL_BENEFIT = 1
MID_YEAR = 2050
# The year where the NGFS value is pegged/rescaled to be the same as Masterdata
# global production value.
NGFS_PEG_YEAR = 2023
# Assert the peg year to be at most the last year of masterdata.
assert NGFS_PEG_YEAR <= 2026
NGFS_PEG_YEAR_ORIGINAL = NGFS_PEG_YEAR
# Lifespan of the renewable energy
RENEWABLE_LIFESPAN = 30  # years
NGFS_RENEWABLE_WEIGHT = "static_50%"
SECTOR_INCLUDED = "nonpower"
RENEWABLE_WEIGHTS = {
    "solar": 0.5,
    "onshore_wind": 0.25,
    "offshore_wind": 0.25,
}
# Possible values: "default", "100year", "5%", "8%", "0%"
RHO_MODE = "default"
# Whether to reverse-discount the gross benefit at 1% per
# year.
ENABLE_BENEFIT_NET_GROWTH = 0
BENEFIT_NET_GROWTH_RATE = 0.01
# None means disabled
WEIGHT_GAS = None
# WEIGHT_GAS = 0.1
# WEIGHT_GAS = 0.33

print("Use new method:", ENABLE_NEW_METHOD)
print("Renewable degradation:", ENABLE_RENEWABLE_GRADUAL_DEGRADATION)
print("30 year lifespan:", ENABLE_RENEWABLE_30Y_LIFESPAN)
print("Wright's law", ENABLE_WRIGHTS_LAW)
print("Weight mode", NGFS_RENEWABLE_WEIGHT)
print("Residual benefit", ENABLE_RESIDUAL_BENEFIT)
print("Sector included", SECTOR_INCLUDED)
print("BENEFIT NET GROWTH", ENABLE_BENEFIT_NET_GROWTH)
print("WEIGHT_GAS", WEIGHT_GAS)

if ENABLE_NEW_METHOD:
    assert ENABLE_RENEWABLE_GRADUAL_DEGRADATION or ENABLE_RENEWABLE_30Y_LIFESPAN

assert NGFS_RENEWABLE_WEIGHT in ["static_50%", "static_NGFS", "dynamic_NGFS"]

assert SECTOR_INCLUDED in ["power", "nonpower", "both"]


def round2(x):
    return round(x, 2)


def maybe_round2(do_it, x):
    return round2(x) if do_it else x


def pandas_divide_or_zero(num, dem):
    return (num / dem).replace([np.inf, -np.inf], 0)


def set_matplotlib_tick_spacing(tick_spacing):
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))


ngfss = util.read_ngfs_coal_and_power()

df, nonpower_coal, power_coal = util.read_masterdata()

if WEIGHT_GAS is None:
    weighted_emissions_factor_gas = 0.0
else:
    weighted_emissions_factor_gas = util.calculate_weighted_emissions_factor_gas(df)

# Reduce df to have companies only that do coal (both power and nonpower).
# Note that these companies may have non-coal rows.
# NOTE we don't do this anymore, because we use the same value for average unit
# profit across companies anyway.
# df = df[df.company_id.isin(processed_revenue.revenue_data_companies)]

# Prepare lcoe
# Unit is $/MWh
global_lcoe_average = util.get_lcoe_info(lcoe_mode)

processed_revenue.prepare_average_unit_profit(df)

irena = util.read_json("data/irena.json")

# To generate this data, make sure to generate
# NGFS_renewable_additional_capacity_MODEL.json by running
# data_preparation/prepare_world_wright_learning.py.
# And then run misc/NGFS_renewable_additional_capacity.py
if NGFS_RENEWABLE_WEIGHT == "dynamic_NGFS":
    NGFS_dynamic_weight = util.read_json(
        f"data/NGFS_renewable_dynamic_weight_{util.NGFS_MODEL}.json"
    )
else:
    NGFS_dynamic_weight = None

# We re-generate nonpower_coal, power_coal again now that df has "energy_type_specific_average_unit_profit".
_, nonpower_coal, power_coal = util.read_masterdata(df)

if SECTOR_INCLUDED == "power":
    nonpower_coal = nonpower_coal.drop(nonpower_coal.index)
elif SECTOR_INCLUDED == "nonpower":
    power_coal = power_coal.drop(power_coal.index)


def sum_array_of_mixed_objs(x):
    out = 0.0
    for e in x:
        if isinstance(e, float):
            out += e
        elif isinstance(e, dict):
            out += sum(e.values())
        else:
            out += e.sum()
    return out


def divide_array_of_mixed_objs(arr, divider):
    out = []
    for e in arr:
        if isinstance(e, dict):
            out.append({k: v / divider for k, v in e.items()})
        else:
            # float or Pandas Series
            out.append(e / divider)
    return out


def add_array_of_mixed_objs(x, y):
    out = []
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        if isinstance(xi, pd.Series):
            xi = xi.to_dict()
        if isinstance(yi, pd.Series):
            yi = yi.to_dict()

        if isinstance(xi, dict):
            z = {k: v + yi[k] for k, v in xi.items()}
            out.append(z)
        else:
            # float
            out.append(xi + yi)
    return out


def calculate_cost1_info(
    do_round,
    data_set,
    time_period,
    total_production,
    array_of_total_emissions_non_discounted,
    total_emissions_discounted,
    array_of_cost_non_discounted_revenue,
    array_of_cost_discounted_revenue,  # opportunity cost
    array_of_cost_non_discounted_investment,
    array_of_cost_discounted_investment,
    current_policies=None,
    never_discount_the_cost=False,
    residual_emissions=0.0,
    residual_production=0.0,
    cost_gas_investment=0.0,
):
    out_yearly_info = {}
    cost_non_discounted_revenue = sum_array_of_mixed_objs(
        array_of_cost_non_discounted_revenue
    )
    cost_discounted_revenue = sum_array_of_mixed_objs(array_of_cost_discounted_revenue)
    cost_non_discounted_investment = sum_array_of_mixed_objs(
        array_of_cost_non_discounted_investment
    )
    cost_discounted_investment = sum_array_of_mixed_objs(
        array_of_cost_discounted_investment
    )
    if current_policies is None:
        assert data_set == "2DII" or "Current Policies" in data_set, data_set
        saved_non_discounted = sum(array_of_total_emissions_non_discounted)
        saved_discounted = total_emissions_discounted
        total_production_avoided = total_production
        out_yearly_info["benefit_non_discounted"] = np.array(
            array_of_total_emissions_non_discounted
        )
    else:
        assert not (data_set == "2DII" or "Current Policies" in data_set)
        saved_non_discounted = sum(current_policies["emissions_non_discounted"]) - sum(
            array_of_total_emissions_non_discounted
        )
        saved_discounted = (
            current_policies["emissions_discounted"] - total_emissions_discounted
        )
        total_production_avoided = (
            current_policies["total_production"] - total_production
        )
        out_yearly_info["benefit_non_discounted"] = np.array(
            current_policies["emissions_non_discounted"]
        ) - np.array(array_of_total_emissions_non_discounted)
    # We multiply by 1e9 to go from GtCO2 to tCO2
    # We divide by 1e12 to get trilllion USD
    out_yearly_info["benefit_non_discounted"] *= 1e9 / 1e12 * social_cost_of_carbon
    # Whether to reverse-discount the gross benefit at 1% per
    # year.
    if ENABLE_BENEFIT_NET_GROWTH:
        # Note: the reverse discount starts from 2022!
        reverse_discount = [
            util.calculate_discount(BENEFIT_NET_GROWTH_RATE, -deltat)
            for deltat in range(len(out_yearly_info["benefit_non_discounted"]))
        ]
        out_yearly_info["benefit_non_discounted"] *= reverse_discount
    # Summed benefit
    benefit_non_discounted = sum(out_yearly_info["benefit_non_discounted"])

    # Convert to trillion USD
    cost_non_discounted_revenue /= 1e12
    cost_discounted_revenue /= 1e12
    cost_non_discounted_investment /= 1e12
    cost_discounted_investment /= 1e12
    array_of_cost_non_discounted_revenue_trillions = divide_array_of_mixed_objs(
        array_of_cost_non_discounted_revenue, 1e12
    )
    array_of_cost_discounted_revenue_trillions = divide_array_of_mixed_objs(
        array_of_cost_discounted_revenue, 1e12
    )
    array_of_cost_non_discounted_investment_trillions = divide_array_of_mixed_objs(
        array_of_cost_non_discounted_investment, 1e12
    )
    array_of_cost_discounted_investment_trillions = divide_array_of_mixed_objs(
        array_of_cost_discounted_investment, 1e12
    )
    residual_benefit = residual_emissions * social_cost_of_carbon / 1e12

    # Convert to Gigatonnes of coal
    residual_production /= 1e9
    # Convert to Gigatonnes of CO2
    residual_emissions /= 1e9

    if never_discount_the_cost:
        # This is a quick hack so that we can have the result without the
        # discount.
        cost_discounted_revenue = cost_non_discounted_revenue
        cost_discounted_investment = cost_non_discounted_investment
        out_yearly_info[
            "opportunity_cost"
        ] = array_of_cost_non_discounted_revenue_trillions
        out_yearly_info[
            "investment_cost"
        ] = array_of_cost_non_discounted_investment_trillions
    else:
        out_yearly_info["opportunity_cost"] = array_of_cost_discounted_revenue_trillions
        out_yearly_info[
            "investment_cost"
        ] = array_of_cost_discounted_investment_trillions
    out_yearly_info["cost"] = add_array_of_mixed_objs(
        out_yearly_info["opportunity_cost"], out_yearly_info["investment_cost"]
    )

    # Costs of avoiding coal emissions
    assert cost_discounted_investment >= 0
    if WEIGHT_GAS is None:
        cost_discounted = cost_discounted_revenue + cost_discounted_investment
    else:
        # Take into account of gas, when enabled.
        # The unit is tCO2/tce * (Gtce)
        # We multiply by 1e9 to go from Gtce to tce
        emissions_gas = weighted_emissions_factor_gas * total_production_avoided * 1e9
        # We divide by 1e12 to get trilllion USD
        benefit_non_discounted -= (
            WEIGHT_GAS * social_cost_of_carbon * emissions_gas / 1e12
        )
        weight_non_gas = 1 - WEIGHT_GAS
        cost_discounted = (
            cost_discounted_revenue
            + weight_non_gas * cost_discounted_investment
            + cost_gas_investment
        )

    # Equation 1 in the paper
    net_benefit = benefit_non_discounted - cost_discounted

    last_year = int(time_period.split("-")[1])
    # We use NGFS_PEG_YEAR_ORIGINAL instead of NGFS_PEG_YEAR because the
    # arbitrage period starts in the same year for either scenario, even though
    # in CPS, the s1 is the same as masterdata up till 2026.
    arbitrage_period = 1 + (last_year - (NGFS_PEG_YEAR_ORIGINAL + 1))

    data = {
        "Using production projections of data set": data_set,
        "Time Period of Carbon Arbitrage": time_period,
        "Total coal production avoided (Giga tonnes)": maybe_round2(
            do_round, total_production_avoided
        ),
        "Total coal production avoided including residual (Giga tonnes)": maybe_round2(
            do_round, total_production_avoided + residual_production
        ),
        "Total emissions avoided (GtCO2)": maybe_round2(do_round, saved_non_discounted),
        "Total emissions avoided including residual (GtCO2)": maybe_round2(
            do_round, saved_non_discounted + residual_emissions
        ),
        "Benefits of avoiding coal emissions (in trillion dollars)": maybe_round2(
            do_round, benefit_non_discounted
        ),
        "Costs of avoiding coal emissions (in trillion dollars)": maybe_round2(
            do_round, cost_discounted
        ),
        "Opportunity costs represented by missed coal revenues (in trillion dollars)": maybe_round2(
            do_round, cost_discounted_revenue
        ),
        "Investment costs in renewable energy (in trillion dollars)": maybe_round2(
            do_round, cost_discounted_investment
        ),
        "Carbon arbitrage opportunity (in trillion dollars)": maybe_round2(
            do_round, net_benefit
        ),
        "Carbon arbitrage opportunity relative to world GDP (%)": maybe_round2(
            do_round, net_benefit * 100 / (world_gdp_2020 * arbitrage_period)
        ),
        "Carbon arbitrage residual benefit (in trillion dollars)": maybe_round2(
            do_round, residual_benefit
        ),
        "Carbon arbitrage including residual benefit (in trillion dollars)": maybe_round2(
            do_round, net_benefit + residual_benefit
        ),
        "Carbon arbitrage including residual benefit relative to world GDP (%)": maybe_round2(
            do_round,
            (net_benefit + residual_benefit)
            * 100
            / (world_gdp_2020 * arbitrage_period),
        ),
        "Benefits of avoiding coal emissions including residual benefit (in trillion dollars)": maybe_round2(
            do_round, benefit_non_discounted + residual_benefit
        ),
    }
    return data, out_yearly_info


ngfs_renewable_additional_capacity = None
if ENABLE_WRIGHT_USE_NGFS_DATA:
    ngfs_renewable_additional_capacity = util.read_json(
        f"data/NGFS_renewable_additional_capacity_{util.NGFS_MODEL}.json"
    )


class InvestmentCostNewMethod:
    techs = ["solar", "onshore_wind", "offshore_wind"]
    weights_static_NGFS = {
        "solar": 55.98399919148438 / 100,
        "onshore_wind": 42.000406987439874 / 100,
        "offshore_wind": 2.0155938210757327 / 100,
    }
    # Mentioned in the carbon arbitrage paper page 21, which is from Staffell
    # and Green 2014.
    degradation_rate = {
        "solar": 0.5 / 100,
        "onshore_wind": 0.48 / 100,
        "offshore_wind": 0.48 / 100,
    }
    # Wright's law learning rate
    # See equation 15 in the carbon arbitrage paper on how these numbers are
    # calculated.
    gammas = {"solar": 0.32, "onshore_wind": 0.07, "offshore_wind": 0.04}

    def __init__(self):
        self.capacity_factors = {}
        self.installed_costs = {}
        self.global_installed_capacities_kW_2020 = {}
        self.alphas = {}
        for tech in self.techs:
            # The [-1] is needed to get the value in 2020.
            self.capacity_factors[tech] = (
                irena[f"capacity_factor_{tech}_2010_2020_percent"][-1] / 100
            )
            # Same as investment cost
            installed_cost = irena[f"installed_cost_{tech}_2010_2020_$/kW"][-1]
            self.installed_costs[tech] = installed_cost
            global_installed_capacity_kW = (
                irena[f"{tech}_MW_world_cumulative_total_installed_capacity_2011_2020"][
                    -1
                ]
                * 1e3
            )
            self.global_installed_capacities_kW_2020[
                tech
            ] = global_installed_capacity_kW
            alpha = installed_cost / (
                global_installed_capacity_kW ** -self.gammas[tech]
            )
            self.alphas[tech] = alpha
        self.stocks_kW = {tech: {} for tech in self.techs}

        # To be used in the full cost1 table calculation
        self.cost_non_discounted = []
        self.cost_discounted = []

        self.cached_wrights_law_investment_costs = {
            "solar": {},
            "onshore_wind": {},
            "offshore_wind": {},
        }

    def get_static_weight(self, tech):
        return RENEWABLE_WEIGHTS[tech]

    def GJ2kW(self, x):
        # MW
        mw = util.GJ2MW(x)
        # kW
        return mw * 1e3

    def kW2GJ(self, x):
        # MW
        mw = x / 1e3
        return util.MW2GJ(mw)

    def calculate_total_R(self, country_name, year):
        total_R = 0.0
        for tech in self.techs:
            S = self.get_stock(country_name, tech, year)
            R = self.kW2GJ(S) * self.capacity_factors[tech]
            total_R += R
        return total_R

    def get_cumulative_G_from_NGFS(self, tech, year):
        irena_2020 = self.global_installed_capacities_kW_2020[tech]
        cumulative_G = irena_2020
        for y, additional_capacity in ngfs_renewable_additional_capacity[tech].items():
            if int(y) >= year:
                break
            # Multiplication by 1e6 converts from GW to kW
            cumulative_G += irena_2020 + (additional_capacity * 1e6)
        return cumulative_G

    def _calculate_wrights_law(self, tech, year, cumulative_G):
        return self.alphas[tech] * (cumulative_G ** -self.gammas[tech])

    def calculate_wrights_law_investment_cost(self, tech, year):
        if year in self.cached_wrights_law_investment_costs[tech]:
            return self.cached_wrights_law_investment_costs[tech][year]
        if ENABLE_WRIGHT_USE_NGFS_DATA:
            cumulative_G = self.get_cumulative_G_from_NGFS(tech, year)
        else:
            cumulative_G = self.global_installed_capacities_kW_2020[
                tech
            ] + self.get_stock_without_degradation(tech, year)
        ic = self._calculate_wrights_law(tech, year, cumulative_G)
        self.cached_wrights_law_investment_costs[tech][year] = ic
        return ic

    def calculate_investment_cost_one_country(
        self, country_name, DeltaP, year, discount
    ):
        # in GJ
        total_R = self.calculate_total_R(country_name, year)
        # in GJ
        D = max(0, DeltaP - total_R)
        if math.isclose(D, 0):
            self.cost_non_discounted[-1][country_name] = 0.0
            self.cost_discounted[-1][country_name] = 0.0
            for tech in self.techs:
                if year in self.stocks_kW[tech]:
                    self.stocks_kW[tech][year][country_name] = 0.0
                else:
                    self.stocks_kW[tech][year] = {country_name: 0.0}
            return
        # in kW because installed_costs is in $/kW
        D_kW = self.GJ2kW(D)

        investment_cost = 0.0
        for tech in self.techs:
            # in kW
            if NGFS_RENEWABLE_WEIGHT == "static_50%":
                weight = self.get_static_weight(tech)
            elif NGFS_RENEWABLE_WEIGHT == "static_NGFS":
                weight = self.weights_static_NGFS[tech]
            else:
                # Needs division by 100, because the unit is still in percent.
                weight = NGFS_dynamic_weight[tech][str(year)] / 100
            G = weight * D_kW / self.capacity_factors[tech]
            installed_cost = self.installed_costs[tech]
            if ENABLE_WRIGHTS_LAW:
                installed_cost = self.calculate_wrights_law_investment_cost(tech, year)
            investment_cost += G * installed_cost
            if year in self.stocks_kW[tech]:
                self.stocks_kW[tech][year][country_name] = G
            else:
                self.stocks_kW[tech][year] = {country_name: G}
        self.cost_non_discounted[-1][country_name] = investment_cost
        self.cost_discounted[-1][country_name] = investment_cost * discount

    def calculate_investment_cost(self, DeltaP, year, discount):
        if isinstance(DeltaP, float):
            assert math.isclose(DeltaP, 0)
            self.cost_non_discounted.append(0.0)
            self.cost_discounted.append(0.0)
            return
        self.cost_non_discounted.append({})
        self.cost_discounted.append({})
        for country_name, dp in DeltaP.items():
            self.calculate_investment_cost_one_country(country_name, dp, year, discount)

    def get_stock(self, country_name, tech, year):
        out = 0.0
        if len(self.stocks_kW[tech]) == 0:
            return out
        for stock_year, stock_amount in self.stocks_kW[tech].items():
            if stock_year >= year:
                break
            age = year - stock_year
            s = stock_amount[country_name]
            if ENABLE_RENEWABLE_GRADUAL_DEGRADATION:
                s *= (1 - self.degradation_rate[tech]) ** age

            if ENABLE_RENEWABLE_30Y_LIFESPAN:
                if age <= RENEWABLE_LIFESPAN:
                    out += s
            else:
                # No lifespan checking is needed.
                out += s
        return out

    def get_stock_without_degradation(self, tech, year):
        out = 0.0
        if len(self.stocks_kW[tech]) == 0:
            return out
        for stock_year, stock_amount in self.stocks_kW[tech].items():
            if stock_year >= year:
                break
            out += sum(stock_amount.values())
        return out

    def zero_out_costs_and_stocks(self, peg_year):
        raise Exception("Don't use this!")
        # We zero out the values up to peg_year.
        peg_year_index = peg_year - 2022
        for i in range(peg_year_index + 1):
            self.cost_non_discounted[i] = 0.0
            self.cost_discounted[i] = 0.0
        for tech in self.techs:
            # Zero out all the stocks up to peg_year.
            self.stocks_kW[tech] = {
                y: (0.0 if y <= peg_year else v)
                for y, v in self.stocks_kW[tech].items()
            }

    def calculate_residual_one_year(self, year, weighted_emissions_factor_by_country):
        equivalent_emissions = 0.0
        equivalent_production = 0.0
        for (
            country_name,
            emissions_factor,
        ) in weighted_emissions_factor_by_country.items():
            # in GJ
            total_R = self.calculate_total_R(country_name, year)
            tonnes_of_coal_equivalent = util.GJ2coal(total_R)
            equivalent_emissions += tonnes_of_coal_equivalent * emissions_factor
            equivalent_production += tonnes_of_coal_equivalent
        return equivalent_emissions, equivalent_production

    def calculate_residual(
        self, year_start, year_end, weighted_emissions_factor_by_country
    ):
        if not ENABLE_RESIDUAL_BENEFIT:
            return 0.0, 0.0
        residual_emissions = 0.0
        residual_production = 0.0
        for year in range(year_start, year_end + 1):
            growth = (
                util.calculate_discount(BENEFIT_NET_GROWTH_RATE, -(year - 2022))
                if ENABLE_BENEFIT_NET_GROWTH
                else 1
            )
            (
                equivalent_emissions,
                equivalent_production,
            ) = self.calculate_residual_one_year(
                year, weighted_emissions_factor_by_country
            )
            residual_emissions += growth * equivalent_emissions
            residual_production += growth * equivalent_production
        return residual_emissions, residual_production


def calculate_weighted_emissions_factor_by_country_2020(_df_nonpower, _df_power):
    if not ENABLE_RESIDUAL_BENEFIT:
        return None
    colname = "_2020"
    # In tonnes of CO2
    _df_nonpower["emissions_2020"] = (
        _df_nonpower[colname] * _df_nonpower.emissions_factor
    )
    _df_power["emissions_2020"] = (
        _df_power[colname] * _df_power.emissions_factor * util.hours_in_1year
    )

    grouped_np = _df_nonpower.groupby("asset_country")
    grouped_p = _df_power.groupby("asset_country")

    # In tce
    production_2020_np = grouped_np[colname].sum()
    production_2020_p = util.MW2Gigatonnes_of_coal(grouped_p[colname].sum()) * 1e9
    production_2020 = production_2020_np.add(production_2020_p, fill_value=0.0)

    # In tonnes of CO2
    emissions_2020_np = grouped_np["emissions_2020"].sum()
    emissions_2020_p = grouped_p["emissions_2020"].sum()
    emissions_2020 = emissions_2020_np.add(emissions_2020_p, fill_value=0.0)
    ef = emissions_2020 / production_2020
    # There are some countries with 0 production, and so it is division by
    # zero. We set them to 0.0 for now
    ef = ef.fillna(0.0)
    return ef


def generate_cost1_output(
    rho,
    do_round,
    total_production_by_year,
    total_emissions_masterdata_by_year_non_discounted,
    total_emissions_by_year_discounted,
    _df_nonpower,
    _df_power,
    years_masterdata,
    use_new_method=False,
):
    # Sanity check, assert the year range is 2022-2026 inclusive.
    assert len(total_production_by_year["Coal"]) == 5
    assert len(total_emissions_masterdata_by_year_non_discounted["Coal"]) == 5
    assert len(total_emissions_by_year_discounted["Coal"]) == 5

    global NGFS_PEG_YEAR

    # We need to save the peg year value, because NGFS_PEG_YEAR will change to
    # 2026 for current policies scenario.
    original_ngfs_peg_year = NGFS_PEG_YEAR

    out = {}
    out_yearly = {}

    current_policies = {
        MID_YEAR: None,
        2100: None,
    }

    weighted_emissions_factor_by_country_2020 = (
        calculate_weighted_emissions_factor_by_country_2020(_df_nonpower, _df_power)
    )

    def get_emissions_after_peg_year_discounted(
        last_year, fraction_increase_after_peg_year, sector
    ):
        # In GtCO2
        _df = _df_nonpower if sector == "Coal" else _df_power
        emissions_peg_year = _df[f"_{NGFS_PEG_YEAR}"] * _df.emissions_factor / 1e9
        emissions_peg_year_summed = emissions_peg_year.sum()
        # For power sector, _df[f"_{NGFS_PEG_YEAR}"]'s unit is MW,
        # while the emissions_factor unit is "tonnes of CO2 per MWh".
        if sector == "Power":
            emissions_peg_year_summed *= util.hours_in_1year

        out = 0.0
        for y, v in fraction_increase_after_peg_year[sector].items():
            if (last_year == MID_YEAR) and y > MID_YEAR:
                break
            discount = util.calculate_discount(rho, y - 2022)
            # v and discount are scalar
            out += v * discount
        out *= emissions_peg_year_summed
        return out

    def get_cost_including_ngfs_both(
        scenario,
        last_year,
        fraction_increase_after_peg_year,
        non_discounted_2022_to_pegyear,
        discounted_2022_to_pegyear,
        rev_ren,  # the value is either "revenue" or "renewable"
    ):
        if scenario == "Net Zero 2050":
            # If the scenario is NZ2050, the 2022-NGFS_PEG_YEAR values are
            # force-set to 0, because the diff of s2-s1 in
            # 2022-NGFS_PEG_YEAR is 0.
            out_non_discounted = [0.0] * len(non_discounted_2022_to_pegyear)
            out_discounted = [0.0] * len(discounted_2022_to_pegyear)
        else:
            assert scenario == "Current Policies ", scenario
            out_non_discounted = non_discounted_2022_to_pegyear.copy()
            out_discounted = discounted_2022_to_pegyear.copy()
            original_ngfs_peg_year_index = original_ngfs_peg_year - 2022
            # We specify such that if scenario is current policies, only
            # set the cost to nonzero only after the peg year.
            for i in range(original_ngfs_peg_year_index + 1):
                out_non_discounted[i] = 0.0
                out_discounted[i] = 0.0

        def calculate_cost(sector, year):
            if sector == "Coal":
                _df = _df_nonpower
                _convert2GJ = util.coal2GJ
            else:
                _df = _df_power
                _convert2GJ = util.MW2GJ

            grouped = _df.groupby("asset_country")
            coal_production_by_country = grouped[f"_{year}"].sum()
            in_gj = _convert2GJ(coal_production_by_country)

            if rev_ren == "revenue":
                # In this case, the energy-type-specific is coal
                # aup is the same across the df rows anyway
                if len(_df) > 0:
                    aup = _df.energy_type_specific_average_unit_profit.iloc[0]
                else:
                    aup = 0.0
                cost = aup * in_gj
            else:
                assert rev_ren == "renewable"
                cost = util.GJ2MWh(in_gj) * global_lcoe_average
            return cost

        _c_sum_nonpower = calculate_cost("Coal", NGFS_PEG_YEAR)
        _c_sum_power = calculate_cost("Power", NGFS_PEG_YEAR)
        for y, fraction_increase_np in fraction_increase_after_peg_year["Coal"].items():
            if (last_year == MID_YEAR) and y > MID_YEAR:
                break
            # The following commented out code is for sensitivity analysis.
            # if rev_ren == "revenue":
            #     discount = util.calculate_discount(rho + 0.005, y - 2022)
            # else:
            #     assert rev_ren == "renewable"
            #     discount = util.calculate_discount(rho, y - 2022)
            discount = util.calculate_discount(rho, y - 2022)
            if scenario == "Net Zero 2050":
                if y <= 2026:
                    # If the year is <= 2026, we use masterdata for CPS later.
                    v_np = -fraction_increase_np
                    v_p = -fraction_increase_after_peg_year["Power"][y]
                else:
                    v_np = (
                        fraction_increase_after_peg_year_CPS["Coal"][y]
                        - fraction_increase_np
                    )
                    v_p = (
                        fraction_increase_after_peg_year_CPS["Power"][y]
                        - fraction_increase_after_peg_year["Power"][y]
                    )
            else:
                assert scenario == "Current Policies ", scenario
                v_np = fraction_increase_np
                v_p = fraction_increase_after_peg_year["Power"][y]
            # discount and v are scalar
            _c_sum_v = (_c_sum_nonpower * v_np).add(_c_sum_power * v_p, fill_value=0.0)
            if (scenario == "Net Zero 2050") and y <= 2026:
                # Sanity check, because we will use masterdata for CPS
                # here.
                assert v_np <= 0.0
                assert v_p <= 0.0
                # It's slightly more complicated to calculate DeltaP in
                # this case, because we have to use the masterdata value
                # (obtained via calculate_cost) instead of the NGFS
                # fractional increase from NGFS_PEG_YEAR..
                _c_sum_nonpower_y = calculate_cost("Coal", y)
                _c_sum_power_y = calculate_cost("Power", y)
                _c_sum_v = _c_sum_v.add(_c_sum_nonpower_y, fill_value=0.0).add(
                    _c_sum_power_y, fill_value=0.0
                )

            out_non_discounted.append(_c_sum_v)
            out_discounted.append(_c_sum_v * discount)

        out_non_discounted = list(out_non_discounted)
        out_discounted = list(out_discounted)
        return (
            out_non_discounted,
            out_discounted,
        )

    def get_cost_including_ngfs_both_renewable_new_method(
        scenario,
        last_year,
        fraction_increase_after_peg_year,
        non_discounted_2022_to_pegyear,
        discounted_2022_to_pegyear,
        rev_ren,  # the value is either "revenue" or "renewable"
        _cost_new_method,
    ):
        # We copy _cost_new_method because it is going to be reused for
        # different scenario and year range.
        temp_cost_new_method = copy.deepcopy(_cost_new_method)

        def calculate_gj(sector, year):
            if sector == "Coal":
                _df = _df_nonpower
                _convert2GJ = util.coal2GJ
            else:
                _df = _df_power
                _convert2GJ = util.MW2GJ

            grouped = _df.groupby("asset_country")
            coal_production_by_country = grouped[f"_{year}"].sum()
            in_gj = _convert2GJ(coal_production_by_country)
            return in_gj

        _gj_sum_nonpower = calculate_gj("Coal", NGFS_PEG_YEAR)
        _gj_sum_power = calculate_gj("Power", NGFS_PEG_YEAR)
        for y, fraction_increase_np in fraction_increase_after_peg_year["Coal"].items():
            if (last_year == MID_YEAR) and y > MID_YEAR:
                break
            discount = util.calculate_discount(rho, y - 2022)
            if scenario == "Net Zero 2050":
                if y <= 2026:
                    # If the year is <= 2026, we use masterdata for CPS later.
                    v_np = -fraction_increase_np
                    v_p = -fraction_increase_after_peg_year["Power"][y]
                else:
                    v_np = (
                        fraction_increase_after_peg_year_CPS["Coal"][y]
                        - fraction_increase_np
                    )
                    v_p = (
                        fraction_increase_after_peg_year_CPS["Power"][y]
                        - fraction_increase_after_peg_year["Power"][y]
                    )
            else:
                assert scenario == "Current Policies ", scenario
                v_np = fraction_increase_np
                v_p = fraction_increase_after_peg_year["Power"][y]
            DeltaP = (_gj_sum_nonpower * v_np).add(_gj_sum_power * v_p, fill_value=0.0)
            if (scenario == "Net Zero 2050") and y <= 2026:
                # Sanity check, because we will use masterdata for CPS
                # here.
                assert v_np <= 0.0
                assert v_p <= 0.0
                # It's slightly more complicated to calculate DeltaP in
                # this case, because we have to use the masterdata value
                # (obtained via calculate_gj_and_c) instead of the NGFS
                # fractional increase from NGFS_PEG_YEAR..
                _gj_sum_nonpower_y = calculate_gj("Coal", y)
                _gj_sum_power_y = calculate_gj("Power", y)
                DeltaP = DeltaP.add(_gj_sum_nonpower_y, fill_value=0.0).add(
                    _gj_sum_power_y, fill_value=0.0
                )

            temp_cost_new_method.calculate_investment_cost(DeltaP, y, discount)
        out_non_discounted = list(temp_cost_new_method.cost_non_discounted)
        out_discounted = list(temp_cost_new_method.cost_discounted)
        residual_benefits_years_offset = RENEWABLE_LIFESPAN
        (
            residual_emissions,
            residual_production,
        ) = temp_cost_new_method.calculate_residual(
            last_year + 1,
            last_year + residual_benefits_years_offset,
            weighted_emissions_factor_by_country_2020,
        )
        return (
            out_non_discounted,
            out_discounted,
            residual_emissions,
            residual_production,
        )

    fraction_increase_after_peg_year_CPS = None
    for scenario in ["Current Policies ", "Net Zero 2050"]:
        if scenario == "Current Policies ":
            NGFS_PEG_YEAR = 2026
        else:
            assert scenario == "Net Zero 2050"
            NGFS_PEG_YEAR = original_ngfs_peg_year

        years_masterdata_up_to_peg = list(range(2022, NGFS_PEG_YEAR + 1))

        # 2022-NGFS_PEG_YEAR
        cost_2022_to_pegyear_non_discounted_revenue = []
        cost_2022_to_pegyear_discounted_revenue = []
        cost_2022_to_pegyear_non_discounted_renewable = []
        cost_2022_to_pegyear_discounted_renewable = []
        if use_new_method:
            cost_new_method = InvestmentCostNewMethod()
        else:
            cost_new_method = None
        for year in years_masterdata_up_to_peg:
            if year <= original_ngfs_peg_year:
                # From 2022 up to original peg year, the companies follow
                # the original unchanged trajectory, and so there is no new
                # investment yet.
                cost_2022_to_pegyear_non_discounted_revenue.append(0.0)
                cost_2022_to_pegyear_discounted_revenue.append(0.0)
                cost_2022_to_pegyear_non_discounted_renewable.append(0.0)
                cost_2022_to_pegyear_discounted_renewable.append(0.0)
                if use_new_method:
                    deltaP = 0.0
                    discount = 0.0
                    cost_new_method.calculate_investment_cost(deltaP, year, discount)
                continue

            def get_c(sector):
                if sector == "Coal":
                    _df = _df_nonpower
                    _convert2GJ = util.coal2GJ
                else:
                    _df = _df_power
                    _convert2GJ = util.MW2GJ
                coal_production = _df[f"_{year}"]
                _gj = _convert2GJ(coal_production)
                # In this case, the energy-type-specific is coal
                _df[f"_{year}_cost"] = (
                    _df.energy_type_specific_average_unit_profit * _gj
                )
                _df[f"_{year}_cost_renewable"] = util.GJ2MWh(_gj) * global_lcoe_average
                _df[f"_{year}_GJ"] = _gj
                grouped = _df.groupby("asset_country")
                _c = grouped[f"_{year}_cost"].sum()
                _c_renewable = grouped[f"_{year}_cost_renewable"].sum()
                if use_new_method:
                    deltaP = grouped[f"_{year}_GJ"].sum()
                else:
                    deltaP = None
                return _c, _c_renewable, deltaP

            _c_np, _c_renewable_np, deltaP_np = get_c("Coal")
            _c_p, _c_renewable_p, deltaP_p = get_c("Power")
            _c = _c_np.add(_c_p, fill_value=0.0)
            _c_renewable = _c_renewable_np.add(_c_renewable_p, fill_value=0.0)
            discount = util.calculate_discount(rho, year - 2022)
            cost_2022_to_pegyear_non_discounted_revenue.append(_c)
            cost_2022_to_pegyear_discounted_revenue.append(discount * _c)
            cost_2022_to_pegyear_non_discounted_renewable.append(_c_renewable)
            cost_2022_to_pegyear_discounted_renewable.append(discount * _c_renewable)

            if use_new_method:
                cost_new_method.calculate_investment_cost(
                    deltaP_np.add(deltaP_p), year, discount
                )

        total_production_peg_year = {
            "Coal": total_production_by_year["Coal"][NGFS_PEG_YEAR - 2022],
            "Power": total_production_by_year["Power"][NGFS_PEG_YEAR - 2022],
        }

        total_emissions_peg_year_non_discounted = {
            "Coal": total_emissions_masterdata_by_year_non_discounted["Coal"][
                NGFS_PEG_YEAR - 2022
            ],
            "Power": total_emissions_masterdata_by_year_non_discounted["Power"][
                NGFS_PEG_YEAR - 2022
            ],
        }

        fraction_increase_after_peg_year = {
            sector: util.calculate_ngfs_fractional_increase(
                ngfss, sector, scenario, start_year=NGFS_PEG_YEAR
            )
            for sector in ["Coal", "Power"]
        }
        total_production_masterdata = sum(
            sum(total_production_by_year[sector][: len(years_masterdata_up_to_peg)])
            for sector in ["Coal", "Power"]
        )
        total_emissions_masterdata_discounted = sum(
            sum(
                total_emissions_by_year_discounted[sector][
                    : len(years_masterdata_up_to_peg)
                ]
            )
            for sector in ["Coal", "Power"]
        )

        array_of_total_emissions_masterdata_non_discounted = list_elementwise_sum(
            total_emissions_masterdata_by_year_non_discounted["Coal"][
                : len(years_masterdata_up_to_peg)
            ],
            total_emissions_masterdata_by_year_non_discounted["Power"][
                : len(years_masterdata_up_to_peg)
            ],
        )

        if scenario == "Current Policies ":
            # To prepare for the s2-s1 for NZ2050
            fraction_increase_after_peg_year_CPS = {
                sector: util.calculate_ngfs_fractional_increase(
                    ngfss, sector, scenario, start_year=original_ngfs_peg_year
                )
                for sector in ["Coal", "Power"]
            }

        # Remove weird character
        scenario = scenario.replace("Â", "")
        scenario_formatted = f"2DII + {scenario} Scenario"

        for last_year in [MID_YEAR, 2100]:
            # 2022-last_year
            sum_frac_increase_non_discounted = {
                sector: sum(
                    v
                    for k, v in fraction_increase_after_peg_year[sector].items()
                    if k <= last_year
                )
                for sector in ["Coal", "Power"]
            }

            gigatonnes_coal_production = (
                total_production_masterdata
                + total_production_peg_year["Coal"]
                * sum_frac_increase_non_discounted["Coal"]
                + total_production_peg_year["Power"]
                * sum_frac_increase_non_discounted["Power"]
            )
            array_of_total_emissions_non_discounted = (
                array_of_total_emissions_masterdata_non_discounted
                + [
                    total_emissions_peg_year_non_discounted["Coal"] * v_np
                    + total_emissions_peg_year_non_discounted["Power"]
                    * fraction_increase_after_peg_year["Power"][k]
                    for k, v_np in fraction_increase_after_peg_year["Coal"].items()
                    if k <= last_year
                ]
            )
            total_emissions_discounted = (
                total_emissions_masterdata_discounted
                + get_emissions_after_peg_year_discounted(
                    last_year, fraction_increase_after_peg_year, "Coal"
                )
                + get_emissions_after_peg_year_discounted(
                    last_year, fraction_increase_after_peg_year, "Power"
                )
            )
            (
                cost_non_discounted_revenue,
                cost_discounted_revenue,
            ) = get_cost_including_ngfs_both(
                scenario,
                last_year,
                fraction_increase_after_peg_year,
                cost_2022_to_pegyear_non_discounted_revenue,
                cost_2022_to_pegyear_discounted_revenue,
                "revenue",
            )
            if use_new_method:
                (
                    cost_non_discounted_investment,
                    cost_discounted_investment,
                    residual_emissions,
                    residual_production,
                ) = get_cost_including_ngfs_both_renewable_new_method(
                    scenario,
                    last_year,
                    fraction_increase_after_peg_year,
                    cost_2022_to_pegyear_non_discounted_renewable,
                    cost_2022_to_pegyear_discounted_renewable,
                    "renewable",
                    cost_new_method,
                )
            else:
                (
                    cost_non_discounted_investment,
                    cost_discounted_investment,
                ) = get_cost_including_ngfs_both(
                    scenario,
                    last_year,
                    fraction_increase_after_peg_year,
                    cost_2022_to_pegyear_non_discounted_renewable,
                    cost_2022_to_pegyear_discounted_renewable,
                    "renewable",
                )
                residual_emissions = 0.0
                residual_production = 0.0

            # For gas sensitivty analysis
            if WEIGHT_GAS is None:
                cost_gas_investment = 0.0
            else:
                (
                    cost_non_discounted_investment,
                    cost_discounted_investment,
                ) = get_cost_including_ngfs_both(
                    scenario,
                    last_year,
                    fraction_increase_after_peg_year,
                    cost_2022_to_pegyear_non_discounted_renewable,
                    cost_2022_to_pegyear_discounted_renewable,
                    "renewable",
                )
                discounted_production = (
                    sum_array_of_mixed_objs(cost_discounted_investment)
                    / global_lcoe_average
                )

                lcoe_gas = 69.8  # $/MWh
                cost_gas_investment = (
                    lcoe_gas * WEIGHT_GAS * discounted_production / 1e12
                )

            for never_discount in [False, True]:
                never_discount_text = " NON-DISCOUNTED" if never_discount else ""
                text = f"2022-{last_year} {scenario_formatted}{never_discount_text}"
                cost1_info, yearly_info = calculate_cost1_info(
                    do_round,
                    scenario_formatted,
                    f"2022-{last_year}",
                    gigatonnes_coal_production,
                    array_of_total_emissions_non_discounted,
                    total_emissions_discounted,
                    cost_non_discounted_revenue,
                    cost_discounted_revenue,
                    cost_non_discounted_investment,
                    cost_discounted_investment,
                    current_policies=current_policies[last_year],
                    never_discount_the_cost=never_discount,
                    residual_emissions=residual_emissions,
                    residual_production=residual_production,
                    cost_gas_investment=cost_gas_investment,
                )
                out[text] = cost1_info
                out_yearly[text] = yearly_info
            if scenario == "Current Policies ":
                current_policies[last_year] = {
                    "emissions_non_discounted": array_of_total_emissions_non_discounted,
                    "emissions_discounted": total_emissions_discounted,
                    "total_production": gigatonnes_coal_production,
                }

    out = pd.DataFrame(out)
    return out, out_yearly


def floatify_array_of_mixed_objs(x):
    out = []
    for e in x:
        if isinstance(e, float):
            out.append(e)
        elif isinstance(e, dict):
            out.append(sum(e.values()))
        else:
            out.append(e.sum())
    return out


def do_plot_yearly_cost1(yearly_both):
    full_years_2100 = range(2022, 2100 + 1)
    full_years_midyear = range(2022, MID_YEAR + 1)
    for condition, value in yearly_both.items():
        print(condition)
        plt.figure()
        x = full_years_2100 if "2022-2100" in condition else full_years_midyear
        plt.plot(x, floatify_array_of_mixed_objs(value["cost"]), label="Costs")
        plt.plot(
            x,
            floatify_array_of_mixed_objs(value["investment_cost"]),
            label="Investment costs",
        )
        plt.plot(
            x,
            floatify_array_of_mixed_objs(value["opportunity_cost"]),
            label="Opportunity costs",
        )
        plt.plot(
            x,
            floatify_array_of_mixed_objs(value["benefit_non_discounted"]),
            label="Benefit",
        )
        plt.xlabel("Time")
        plt.ylabel("Trillion dollars")
        plt.legend()
        plt.title(condition)
        plt.axvline(NGFS_PEG_YEAR)
        util.savefig(f"yearly_{condition}")
        plt.close()


def list_elementwise_sum(x, y):
    return [x[i] + y[i] for i in range(len(x))]


def run_cost1(
    x,
    to_csv=False,
    do_round=False,
    plot_yearly=False,
    return_yearly=False,
):
    use_new_method = ENABLE_NEW_METHOD
    # print("# exp cost1")
    # Non-Power cost

    # Production
    total_production_by_year_nonpower = (
        util.get_coal_nonpower_global_generation_across_years(
            nonpower_coal, years_masterdata
        )
    )
    total_production_by_year_power = util.get_coal_power_global_generation_across_years(
        power_coal, years_masterdata
    )
    total_production_by_year = {
        "Coal": total_production_by_year_nonpower,
        "Power": total_production_by_year_power,
    }

    # Emissions non-discounted
    total_emissions_by_year_non_discounted_nonpower = (
        util.get_coal_nonpower_global_emissions_across_years(
            nonpower_coal, years_masterdata
        )
    )
    total_emissions_by_year_non_discounted_power = (
        util.get_coal_power_global_emissions_across_years(power_coal, years_masterdata)
    )
    total_emissions_by_year_non_discounted = {
        "Coal": total_emissions_by_year_non_discounted_nonpower,
        "Power": total_emissions_by_year_non_discounted_power,
    }

    # rho is the same everywhere
    rho = util.calculate_rho(processed_revenue.beta, rho_mode=RHO_MODE)

    # Emissions discounted
    total_emissions_by_year_discounted_nonpower = (
        util.get_coal_nonpower_global_emissions_across_years(
            nonpower_coal,
            years_masterdata,
            discounted=True,
            rho=rho,
        )
    )
    total_emissions_by_year_discounted_power = (
        util.get_coal_power_global_emissions_across_years(
            power_coal,
            years_masterdata,
            discounted=True,
            rho=rho,
        )
    )
    total_emissions_by_year_discounted = {
        "Coal": total_emissions_by_year_discounted_nonpower,
        "Power": total_emissions_by_year_discounted_power,
    }

    out, yearly = generate_cost1_output(
        rho,
        do_round,
        total_production_by_year,
        total_emissions_by_year_non_discounted,
        total_emissions_by_year_discounted,
        nonpower_coal,
        power_coal,
        years_masterdata,
        use_new_method=use_new_method,
    )
    # if to_csv:
    #     uid = util.get_unique_id(include_date=False)
    #     out_nonpower.to_csv(f"plots/cost1_nonpower_{uid}.csv")
    # And then converto markdown table using
    # https://csvtomd.com.

    out_dict = out.T.to_dict()
    both_dict = {}
    for key in out_dict.keys():
        if key in [
            "Using production projections of data set",
            "Time Period of Carbon Arbitrage",
        ]:
            both_dict[key] = out_dict[key]
        else:
            both_dict[key] = maybe_round2(
                do_round,
                pd.Series(out_dict[key]),
            ).to_dict()
    if to_csv:
        uid = util.get_unique_id(include_date=False)
        ext = ""
        if ENABLE_RENEWABLE_GRADUAL_DEGRADATION:
            ext += "_degrade"
        if ENABLE_RENEWABLE_30Y_LIFESPAN:
            ext += "_30Y"
        if ENABLE_WRIGHTS_LAW:
            ext += "_wright"
        if not ENABLE_NEW_METHOD:
            ext = ""
        fname = f"plots/cost1_both_{uid}{ext}_{social_cost_of_carbon}.csv"
        pd.DataFrame(both_dict).T.to_csv(fname)

    if plot_yearly:
        do_plot_yearly_cost1(yearly)

    if return_yearly:
        return yearly

    return both_dict


def calculate_carbon_adjusted_earnings():
    nonpower_grouped = nonpower_coal.groupby("company_id")
    year = 2021

    # nonpower emissions
    def process(g):
        tonnes_coal = g[f"_{year}"]
        # emissions_factor unit tonnes of CO2 per tonnes of coal
        emissions_of_g = (tonnes_coal * g.emissions_factor).sum()  # in tCO2
        return emissions_of_g

    nonpower_emissions = nonpower_grouped.apply(process)

    # nonpower profits
    def process(g):
        tonnes_coal = g[f"_{year}"]
        gj = util.coal2GJ(tonnes_coal)
        return (gj * g.energy_type_specific_average_unit_profit).sum()

    nonpower_profits = nonpower_grouped.apply(process)

    power_grouped = power_coal.groupby("company_id")

    # power emissions
    def process(g):
        mw_coal = g[f"_{year}"]
        # the emissions_factor unit is "tonnes of CO2 per MWh"
        emissions_of_g = (
            mw_coal * util.hours_in_1year * g.emissions_factor
        ).sum()  # in tCO2
        return emissions_of_g

    power_emissions = power_grouped.apply(process)

    def process(g):
        # In MW
        mw_coal = g[f"_{year}"]
        gj = util.MW2GJ(mw_coal)
        return (gj * g.energy_type_specific_average_unit_profit).sum()

    power_profits = power_grouped.apply(process)

    emissions = nonpower_emissions.add(power_emissions, fill_value=0)
    profits = nonpower_profits.add(power_profits, fill_value=0)
    sorted_profit_index = profits.sort_values(ascending=False).index

    plt.figure()
    sorted_profits = profits.loc[sorted_profit_index]
    number_of_nonzeros = len(sorted_profits[sorted_profits > 0])
    cutoff = number_of_nonzeros
    xticks = range(cutoff)
    plt.fill_between(xticks, list(sorted_profits / 1e9)[:cutoff])
    plt.yscale("log")
    plt.xlabel("Coal companies")
    plt.ylabel("Earnings (billion dollars)")
    plt.savefig("plots/distribution_earnings.png")

    plt.figure(dpi=600)
    carbon_adjusted_profits = (
        profits.loc[sorted_profit_index]
        - emissions.loc[sorted_profit_index] * social_cost_of_carbon
    )
    carbon_adjusted_profits_with_cutoff = list(carbon_adjusted_profits / 1e9)[:cutoff]
    plt.bar(xticks, carbon_adjusted_profits_with_cutoff)
    plt.xlabel("Coal companies")
    plt.ylabel("Earnings (billion dollars)")
    plt.savefig("plots/distribution_earnings_carbon_adjusted.png")
    print("Total number of coal companies", len(profits))
    print(
        "Number of positive carbon adjusted earnings",
        len(carbon_adjusted_profits[carbon_adjusted_profits > 0]),
    )
    print("Out of", cutoff)

    # Statistics
    print("Mean", np.mean(carbon_adjusted_profits_with_cutoff))
    print("Std", np.std(carbon_adjusted_profits_with_cutoff, ddof=1))
    print("Median", np.median(carbon_adjusted_profits_with_cutoff))
    print("Min", np.min(carbon_adjusted_profits_with_cutoff))
    print("Max", np.max(carbon_adjusted_profits_with_cutoff))


def prepare_per_company_emissions_and_profit(
    scenario, nonpower_coal, power_coal, summation_start_year
):
    emissions_filename = "plots/per_company_emissions.csv.gz"
    profit_filename = "plots/per_company_profit.csv.gz"
    if os.path.isfile(emissions_filename):
        print("Processed per company data found. Reading...")
        per_company_emissions = pd.read_csv(emissions_filename, compression="gzip")
        per_company_profit = pd.read_csv(profit_filename, compression="gzip")
        print("Done")
        return per_company_emissions, per_company_profit

    peg_year = 2026

    assert SECTOR_INCLUDED == "nonpower"

    nonpower_fraction_increase_after_2026 = util.calculate_ngfs_fractional_increase(
        ngfss, "Coal", scenario, start_year=peg_year
    )
    # power_fraction_increase_after_2026 = util.calculate_ngfs_fractional_increase(
    #     ngfss, "Power", scenario, start_year=peg_year
    # )

    nonpower_per_company_emissions = (
        util.get_coal_nonpower_per_company_NON_discounted_emissions_summed_over_years(
            peg_year,
            nonpower_coal,
            years_masterdata,
            nonpower_fraction_increase_after_2026,
            summation_start_year=summation_start_year,
        )
    )
    # power_per_company_emissions = (
    #     util.get_coal_power_per_company_NON_discounted_emissions_summed_over_years(
    #         peg_year,
    #         power_coal,
    #         years_masterdata,
    #         power_fraction_increase_after_2026,
    #         summation_start_year=summation_start_year,
    #     )
    # )

    # per_company_emissions = nonpower_per_company_emissions.add(
    #     power_per_company_emissions, fill_value=0
    # )
    per_company_emissions = nonpower_per_company_emissions

    nonpower_per_company_profit = (
        util.get_coal_nonpower_per_company_discounted_PROFIT_summed_over_years(
            peg_year,
            nonpower_coal,
            years_masterdata,
            nonpower_fraction_increase_after_2026,
            processed_revenue.beta,
            summation_start_year=summation_start_year,
        )
    )
    # power_per_company_profit = (
    #     util.get_coal_power_per_company_discounted_PROFIT_summed_over_years(
    #         peg_year,
    #         power_coal,
    #         years_masterdata,
    #         power_fraction_increase_after_2026,
    #         processed_revenue.beta,
    #         summation_start_year=summation_start_year,
    #     )
    # )
    # per_company_profit = nonpower_per_company_profit.add(
    #     power_per_company_profit, fill_value=0
    # )
    per_company_profit = nonpower_per_company_profit

    per_company_emissions.to_csv(emissions_filename)
    per_company_profit.to_csv(profit_filename)
    return per_company_emissions, per_company_profit


def calculate_social_value_of_stranded_assets(year_2020_only=False):
    scenario = "Current Policies "
    summation_start_year = NGFS_PEG_YEAR_ORIGINAL + 1
    print("Summation start year", summation_start_year)
    if year_2020_only:
        np_g = nonpower_coal.groupby("company_id")
        p_g = power_coal.groupby("company_id")
        np_e = np_g.apply(lambda g: (g._2020 * g.emissions_factor).sum())
        p_e = p_g.apply(
            lambda g: (g._2020 * g.emissions_factor).sum() * util.hours_in_1year
        )
        per_company_emissions = np_e.add(p_e, fill_value=0.0)
        # Convert to df to make it consistent with the summed-over-years version.
        per_company_emissions = pd.DataFrame(
            {"0": per_company_emissions.values}, index=per_company_emissions.index
        )
        np_p = np_g.apply(
            lambda g: (
                util.coal2GJ(g._2020) * g.energy_type_specific_average_unit_profit
            ).sum()
        )
        p_p = p_g.apply(
            lambda g: (
                util.MW2GJ(g._2020) * g.energy_type_specific_average_unit_profit
            ).sum()
        )
        per_company_profit = np_p.add(p_p, fill_value=0.0)
        # Convert to df to make it consistent with the summed-over-years version.
        per_company_profit = pd.DataFrame(
            {"0": per_company_profit.values}, index=per_company_profit.index
        )
    else:
        (
            per_company_emissions,
            per_company_profit,
        ) = prepare_per_company_emissions_and_profit(
            scenario, nonpower_coal, power_coal, summation_start_year
        )

    sorted_profit_index = per_company_profit.sort_values(by="0", ascending=False).index

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(axs[0])
    sorted_profits = per_company_profit.loc[sorted_profit_index]["0"]
    number_of_nonzeros = len(sorted_profits[sorted_profits > 0])
    print(
        "Number of zero profits",
        len(sorted_profits) - number_of_nonzeros,
        "out of",
        len(sorted_profits),
    )
    cutoff = number_of_nonzeros
    xticks = range(cutoff)
    if year_2020_only:
        ylabel = "2020 earnings (million dollars)"
        ylabel_social = "2020 carbon-adjusted earnings (million dollars)"
        scale_mul = 1e3
    else:
        ylabel = "Stranded asset value (billion dollars)"
        ylabel_social = "Social stranded asset value\n(billion dollars)"
        scale_mul = 1
    plt.fill_between(xticks, list(sorted_profits / 1e9 * scale_mul)[:cutoff])
    plt.yscale("log")
    plt.xlabel("Coal companies")
    plt.ylabel(ylabel)
    ext = "_2020_only" if year_2020_only else ""

    plt.sca(axs[1])
    social = (
        per_company_profit.loc[sorted_profit_index]["0"]
        - per_company_emissions.loc[sorted_profit_index]["0"] * social_cost_of_carbon
    )
    strange_outlier = social[social < -10_000 * 1e9]
    social = social[social > -10_000 * 1e9]
    social_with_cutoff = list(social / 1e9 * scale_mul)[:cutoff]
    plt.fill_between(xticks, social_with_cutoff)
    plt.xlabel("Coal companies")
    plt.ylabel(ylabel_social)
    plt.tight_layout()
    plt.savefig(f"plots/distribution_stranded{ext}.png")
    print("Total number of coal companies", len(per_company_profit))
    print("Number of positive social stranded asset", len(social[social > 0]))
    print("Out of", cutoff)
    print("Positive social", social[social > 0])

    # Statistics
    print("Mean", np.mean(social_with_cutoff))
    print("Std", np.std(social_with_cutoff, ddof=1))
    print("Median", np.median(social_with_cutoff))
    print("Min", np.min(social_with_cutoff))
    print("Max", np.max(social_with_cutoff))


def make_carbon_arbitrage_opportunity_plot(relative_to_world_gdp=False):
    from collections import defaultdict

    global social_cost_of_carbon
    social_costs = np.linspace(0, 200, 3)
    ydict = defaultdict(list)
    chosen_scenario = "2022-2100 2DII + Net Zero 2050 Scenario"
    for social_cost in social_costs:
        util.social_cost_of_carbon = social_cost
        social_cost_of_carbon = social_cost
        out = run_cost1(x=1, to_csv=False, do_round=False)
        carbon_arbitrage_opportunity = out[
            "Carbon arbitrage including residual benefit (in trillion dollars)"
        ]
        for scenario, value in carbon_arbitrage_opportunity.items():
            if scenario != chosen_scenario:
                continue
            if relative_to_world_gdp:
                value = value / world_gdp_2020 * 100
            ydict[scenario].append(value)
    mapper = {
        f"2022-{MID_YEAR} 2DII + Current Policies  Scenario": f"s2=0, T={MID_YEAR}",
        "2022-2100 2DII + Current Policies  Scenario": "s2=0, T=2100",
        f"2022-{MID_YEAR} 2DII + Net Zero 2050 Scenario": f"s2=Net Zero 2050, T={MID_YEAR}",
        "2022-2100 2DII + Net Zero 2050 Scenario": "s2=Net Zero 2050, T=2100",
    }

    # Find the intersect with the x axis
    from scipy.stats import linregress

    social_cost_zeros = {}
    for scenario, values in ydict.items():
        slope, intercept, r_value, p_value, std_err = linregress(social_costs, values)
        sc_zero = -intercept / slope
        print("Social cost when zero of", mapper[scenario], sc_zero)
        print("  r value", r_value)
        social_cost_zeros[scenario] = sc_zero

    plt.figure()
    for scenario, values in ydict.items():
        plt.plot(social_costs, values, label=mapper[scenario])
    plt.axhline(0, color="gray", linewidth=1)  # So that we can see the zeros

    # Vertical lines
    from matplotlib.pyplot import text

    vertical_lines = {
        51: "Biden administration, 51 $/tC02",
        61.4: "Lower estimate, Rennert et al. (2021), 61.4 $/tC02",
        80: "Pindyck (2019), 80 $/tCO2",
        114.9: "Mid estimate, Rennert et al. (2021), 114.9 $/tC02",
        168.4: "Upper estimate, Rennert et al. (2021), 168.4 $/tC02",
    }
    ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    # y_mid = (y_min + y_max) / 2
    for x, text_content in vertical_lines.items():
        plt.axvline(x, color="gray", linestyle="dashed")
        text(
            x - 7,
            0.98 * y_max,
            text_content,
            color="gray",
            rotation=90,
            verticalalignment="top",
            fontdict={"size": 10},
        )

    # Marker at intersection with zero
    x_intersect = social_cost_zeros[chosen_scenario]
    plt.plot(x_intersect, 0, "o", color="tab:blue")
    text(
        x_intersect,
        32,
        f"{x_intersect:.1f} $/tC02",
        color="gray",
        verticalalignment="center",
        horizontalalignment="center",
        fontdict={"size": 10},
    )
    # plt.legend()
    plt.xlabel("Social cost of carbon (dollars/tCO2)")
    if relative_to_world_gdp:
        print("Relative to 2020 world GDP")
        plt.ylabel("Carbon Arbitrage relative to 2020 World GDP (%)")
    else:
        plt.ylabel("Carbon Arbitrage (trillion dollars)")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    suffix = "_relative" if relative_to_world_gdp else ""
    plt.savefig(f"plots/carbon_arbitrage_opportunity{suffix}.png")


def prepare_regions_for_climate_financing(iso3166_df):
    asia_countries = list(iso3166_df[iso3166_df.region == "Asia"]["alpha-2"])
    africa_countries = list(iso3166_df[iso3166_df.region == "Africa"]["alpha-2"])
    north_america_countries = list(
        iso3166_df[iso3166_df["sub-region"] == "Northern America"]["alpha-2"]
    )
    lac_countries = list(
        iso3166_df[iso3166_df["sub-region"] == "Latin America and the Caribbean"][
            "alpha-2"
        ]
    )
    europe_countries = list(iso3166_df[iso3166_df.region == "Europe"]["alpha-2"])
    au_and_nz = list(
        iso3166_df[iso3166_df["sub-region"] == "Australia and New Zealand"]["alpha-2"]
    )

    region_countries_map = {
        "Asia": asia_countries,
        "Africa": africa_countries,
        "North America": north_america_countries,
        "Latin America & the Carribean": lac_countries,
        "Europe": europe_countries,
        "Australia & New Zealand": au_and_nz,
    }
    # Just to make sure that the order is deterministic.
    # Unlikely, but just to be sure.
    regions = [
        "Asia",
        "Africa",
        "North America",
        "Latin America & the Carribean",
        "Europe",
        "Australia & New Zealand",
    ]
    return region_countries_map, regions


def calculate_each_countries_cost_with_cache(
    chosen_s2_scenario, cache_json_path, ignore_cache=False
):
    use_cache = not ignore_cache
    if use_cache and os.path.isfile(cache_json_path):
        print("Cached climate financing json found. Reading...")
        costs_dict = util.read_json(cache_json_path)
        print("Done")
    else:
        costs_dict = {}
        out = run_cost1(x=1, to_csv=False, do_round=False, return_yearly=True)
        yearly_cost_for_avoiding = out[chosen_s2_scenario]["cost"]
        country_names = list(yearly_cost_for_avoiding[-1].keys())
        for country_name in country_names:
            country_level_cost = 0.0
            for e in yearly_cost_for_avoiding:
                if isinstance(e, float):
                    country_level_cost += e
                elif isinstance(e, dict):
                    country_level_cost += e[country_name]
                else:
                    # Pandas series
                    country_level_cost += e.loc[country_name]
            costs_dict[country_name] = country_level_cost
        if use_cache:
            with open(cache_json_path, "w") as f:
                json.dump(costs_dict, f)
    return costs_dict


def make_climate_financing_plot_by_developed_countries(plotname_suffix=""):
    # This plot is probably unused, and can be removed.
    raise Exception("Not used")
    # plt.figure(figsize=(7, 5))
    # domestic = [developed_total_cost] + [
    #     costs_dict.get(country_shortname, 0.0)
    #     for country_shortname in developed_country_shortnames
    # ]
    # foreign = [developing_cost_for_avoiding] + [
    #     foreign_costs_dict[country_shortname]
    #     for country_shortname in developed_country_shortnames
    # ]
    # developed_country_longnames = [
    #     iso3166_df_alpha2.loc[sn]["name"] if sn != "GB" else "United Kingdom"
    #     for sn in developed_country_shortnames
    # ]
    # xticks = ["Developed"] + developed_country_longnames
    # util.plot_stacked_bar(
    #     xticks,
    #     [
    #         ("Domestic climate\nfinance (subsidy)", domestic),
    #         ("Foreign aid climate\nfinance (subsidy)", foreign),
    #     ],
    # )
    # plt.xticks(xticks, rotation=90)
    # plt.ylabel("Requisite climate financing provision\n(trillion dollars)")
    # plt.legend(loc="upper right")
    # plt.tight_layout()
    # util.savefig(f"climate_financing{plotname_suffix}")

    # # Now we do the same but scaled by GDP
    # plt.figure(figsize=(7, 5))
    # # Multiplication by 1e12 converts the cost from trillion dollars back to dollars
    # # Multiplication of GDP by 1e6 converts it from million dollars back to dollars
    # domestic = [developed_total_cost * 1e12 / (total_developed_gdp * 1e6) * 100] + [
    #     costs_dict.get(country_shortname, 0.0)
    #     * 1e12
    #     / (get_gdp(country_shortname) * 1e6)
    #     * 100
    #     for country_shortname in developed_country_shortnames
    # ]
    # foreign = [
    #     developing_cost_for_avoiding * 1e12 / (total_developed_gdp * 1e6) * 100
    # ] + [
    #     foreign_costs_dict[country_shortname]
    #     * 1e12
    #     / (get_gdp(country_shortname) * 1e6)
    #     * 100
    #     for country_shortname in developed_country_shortnames
    # ]
    # util.plot_stacked_bar(
    #     xticks,
    #     [
    #         ("Domestic climate\nfinance (subsidy)", domestic),
    #         ("Foreign aid climate\nfinance (subsidy)", foreign),
    #     ],
    # )
    # plt.xticks(xticks, rotation=90)
    # plt.ylabel("Requisite climate financing provision\nrelative to GDP (%)")
    # plt.legend(loc="upper right")
    # plt.tight_layout()
    # util.savefig(f"climate_financing_relative{plotname_suffix}")
    # plt.close()


def do_climate_financing_sanity_check(
    git_branch,
    plotname_suffix,
    chosen_s2_scenario,
    ignore_cache,
    developed_country_shortnames,
    developING_country_shortnames,
    emerging_country_shortnames,
    colname_for_gdp,
    developed_gdp,
    _world_sum,
    _developed_sum,
    _developing_sum,
    _emerging_sum,
    _region_sum,
    regions,
    region_countries_map,
):
    # This function is not used for now.
    # But may be used from time to time.
    cache_json_path = f"plots/climate_financing_{git_branch}{plotname_suffix}.json"
    costs_dict = calculate_each_countries_cost_with_cache(
        chosen_s2_scenario, cache_json_path, ignore_cache=ignore_cache
    )

    def _get_total_cost(shortnames):
        return sum(costs_dict.get(n, 0.0) for n in shortnames)

    developed_total_cost = _get_total_cost(developed_country_shortnames)

    # Calculating the climate change cost for developing countries
    developing_cost_for_avoiding = _get_total_cost(developING_country_shortnames)

    # Calculating for emerging countries
    emerging_cost_for_avoiding = _get_total_cost(emerging_country_shortnames)

    def get_gdp(country_shortname):
        return developed_gdp.loc[country_shortname][colname_for_gdp]

    # Calculating foreign costs
    developed_gdp = developed_gdp.set_index("country_shortcode")
    total_developed_gdp = developed_gdp[colname_for_gdp].sum()
    foreign_costs_dict = {}
    for country_shortname in developed_country_shortnames:
        gdp_fraction = get_gdp(country_shortname) / total_developed_gdp
        foreign_costs_dict[country_shortname] = (
            gdp_fraction * developing_cost_for_avoiding
        )

    # Now we group by regions
    def get_avoiding_cost_by_region(included_country_names):
        return _get_total_cost(included_country_names)

    # Calculating the cost for the whole world
    world_cost_for_avoiding = sum(costs_dict.values())

    # Sanity checks
    assert math.isclose(_world_sum, world_cost_for_avoiding), (
        _world_sum,
        world_cost_for_avoiding,
    )
    assert math.isclose(_developed_sum, developed_total_cost)
    assert math.isclose(_developing_sum, developing_cost_for_avoiding)
    assert math.isclose(_emerging_sum, emerging_cost_for_avoiding)
    for region in regions:
        assert math.isclose(
            _region_sum[region],
            get_avoiding_cost_by_region(region_countries_map[region]),
        )


def make_climate_financing_plot(
    plotname_suffix="",
    plot_name=None,
    svg=False,
    ignore_cache=False,
    chosen_s2_scenario=None,
):
    global nonpower_coal, power_coal

    if chosen_s2_scenario is None:
        chosen_s2_scenario = "2022-2100 2DII + Net Zero 2050 Scenario"

    (
        iso3166_df,
        iso3166_df_alpha2,
        developed_gdp,
        colname_for_gdp,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()

    # print("Developed total GDP", developed_gdp[colname_for_gdp].sum())

    git_branch = util.get_git_branch()

    developING_country_shortnames = util.get_developing_countries()
    emerging_country_shortnames = util.get_emerging_countries()

    region_countries_map, regions = prepare_regions_for_climate_financing(iso3166_df)

    cache_json_path = (
        f"plots/climate_financing_yearly_discounted_{git_branch}{plotname_suffix}.json"
    )
    yearly_costs_dict = None
    if ignore_cache:
        yearly_costs_dict = calculate_yearly_costs_dict(chosen_s2_scenario)
    elif os.path.isfile(cache_json_path):
        yearly_costs_dict = util.read_json(cache_json_path)
    else:
        yearly_costs_dict = calculate_yearly_costs_dict(chosen_s2_scenario)
        with open(cache_json_path, "w") as f:
            json.dump(yearly_costs_dict, f)

    def _get_year_range_cost(year_start, year_end, included_countries=None):
        out = 0.0
        for c, e in yearly_costs_dict.items():
            if included_countries is not None and c not in included_countries:
                continue
            out += sum(e[year_start - 2022 : year_end + 1 - 2022])
        return out

    plot_data = []
    # Used for sanity check.
    _world_sum = 0.0
    _developed_sum = 0.0
    _developing_sum = 0.0
    _emerging_sum = 0.0
    _region_sum = defaultdict(float)
    for year_start, year_end in [(NGFS_PEG_YEAR + 1, 2050), (2051, 2070), (2071, 2100)]:
        _world = _get_year_range_cost(year_start, year_end)
        _world_sum += _world
        _developed = _get_year_range_cost(
            year_start, year_end, developed_country_shortnames
        )
        _developed_sum += _developed
        _developing = _get_year_range_cost(
            year_start, year_end, developING_country_shortnames
        )
        _developing_sum += _developing
        _emerging = _get_year_range_cost(
            year_start, year_end, emerging_country_shortnames
        )
        _emerging_sum += _emerging
        region_costs = [
            _world,
            _developed,
            _developing,
            _emerging,
        ]
        for region in regions:
            _region_cost = _get_year_range_cost(
                year_start, year_end, region_countries_map[region]
            )
            _region_sum[region] += _region_cost
            region_costs.append(_region_cost)
        plot_data.append((f"{year_start}-{year_end}", region_costs))

    xticks = [
        "World",
        "Developed Countries",
        "Developing Countries",
        "Emerging Market Countries",
    ] + regions
    plt.figure(figsize=(6, 6))
    util.plot_stacked_bar(xticks, plot_data)

    # Add separator between 3 types of grouping
    # Right after World
    plt.axvline(0.5, color="gray", linestyle="dashed")
    # Right after Emerging Market Countries
    plt.axvline((3 + 4) / 2, color="gray", linestyle="dashed")
    # Add explanatory text
    plt.text(
        1,
        25,
        "By level of\ndevelopment",
        color="gray",
        verticalalignment="top",
        # fontdict={"size": 10},
    )
    plt.text(
        5,
        25,
        "By region",
        color="gray",
        verticalalignment="top",
        # fontdict={"size": 10},
    )

    plt.legend()
    plt.xticks(xticks, rotation=45, ha="right")
    plt.ylabel("PV climate financing (trillion dollars)")
    plt.tight_layout()
    util.savefig(
        plot_name if plot_name else f"climate_financing_by_region{plotname_suffix}",
        svg=svg,
    )
    plt.close()

    if ignore_cache:
        return

    # This is for comparing the by-region and by-world of NGFS fractional
    # increase.
    climate_financing_dict = defaultdict(float)
    for i in range(len(xticks)):
        for d in plot_data:
            climate_financing_dict[xticks[i]] += d[1][i]
    with open(f"plots/for_comparison_pv_climate_financing_{git_branch}.json", "w") as f:
        json.dump(climate_financing_dict, f)


def annotate(xs, ys, labels, filter_labels=None, no_zero_x=False, fontsize=None):
    for x, y, label in zip(xs, ys, labels):
        if (filter_labels is not None) and (label not in filter_labels):
            continue
        if no_zero_x and math.isclose(x, 0):
            continue
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",  # how to position the text
            xytext=(0, 5),  # distance from text to points (x,y)
            ha="center",  # horizontal alignment
            fontsize=fontsize,
        )


def make_climate_financing_SCATTER_plot():
    gdp_per_capita_dict = util.read_json("data/all_countries_gdp_per_capita_2020.json")
    # Taiwan in 2020
    # https://knoema.com/atlas/Taiwan-Province-of-China/GDP-per-capita
    # gdp_per_capita_dict["TW"] = 28306
    # Kosovo in 2020
    # https://data.worldbank.org/indicator/NY.GDP.PCAP.CD?locations=XK
    # gdp_per_capita_dict["XK"] = 4346.6

    gdp_marketcap_dict = util.read_json("data/all_countries_gdp_marketcap_2020.json")
    # Taiwan in 2020
    # https://www.statista.com/statistics/727589/gross-domestic-product-gdp-in-taiwan/
    # gdp_marketcap_dict["TW"] = 668.16 * 1e9  # billion USD
    # Kosovo in 2020
    # https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?locations=XK
    # gdp_marketcap_dict["XK"] = 7716925.36

    worldbank_set = set(gdp_per_capita_dict.keys())
    masterdata_coal_set = set(nonpower_coal.asset_country).union(
        set(power_coal.asset_country)
    )
    divide_by_marketcap = True

    print("MODE divide by marketcap", divide_by_marketcap)
    # Data checking
    print("worldbank.org", len(worldbank_set))
    print("masterdata", len(masterdata_coal_set))
    print("intersection", len(worldbank_set.intersection(masterdata_coal_set)))
    print("masterdata - worldbank", masterdata_coal_set - worldbank_set)
    # Only in masterdata: {nan, 'TW', 'XK'}

    # country_shortnames = list(masterdata_coal_set - {np.nan, "TW", "XK"})
    chosen_s2_scenario = "2022-2100 2DII + Net Zero 2050 Scenario"
    cache_json_path = "plots/climate_financing.json"

    costs_dict = calculate_each_countries_cost_with_cache(
        chosen_s2_scenario, cache_json_path
    )

    developing_shortnames = util.get_developing_countries()
    emerging_shortnames = util.get_emerging_countries()
    developed_gdp = pd.read_csv("data/GDP-Developed-World.csv", thousands=",")
    colname_for_gdp = "2020 GDP (million dollars)"
    developed_country_shortnames = list(
        developed_gdp.sort_values(by=colname_for_gdp, ascending=False).country_shortcode
    )
    # multiplier_mode = "trillion"
    # If the multiplier mode is billion, we restrict the ylim to up to 1
    # trillion.
    multiplier_mode = "billion"

    def plot_scatter(shortnames, label):
        x = []
        y = []
        if divide_by_marketcap:
            # one trillion
            # We convert the cost to just dollars.
            mul = 1e12
        else:
            mul = 1 if multiplier_mode == "trillion" else 1e3

        arbitrage_period = 1 + (2100 - (NGFS_PEG_YEAR + 1))
        print("Peg year", NGFS_PEG_YEAR, "arbitrage period", arbitrage_period)
        plot_labels = []
        filter_labels = []
        for country_shortname in shortnames:
            if country_shortname not in costs_dict:
                # print(f"{country_shortname} is missing")
                continue
            if country_shortname in ["TW", "XK"]:
                print("Intentionally skipping", country_shortname)
                continue
            x_val = gdp_per_capita_dict[country_shortname]
            x.append(x_val)
            plot_labels.append(country_shortname)
            if divide_by_marketcap:
                mul_marketcap = (
                    1 / (gdp_marketcap_dict[country_shortname] * arbitrage_period) * 100
                )
            else:
                mul_marketcap = 1
            cost = costs_dict[country_shortname]
            val = cost * mul * mul_marketcap
            # Sanity check
            if cost > 4.0:
                print(
                    "Beyond 4 trillion dollars:",
                    country_shortname,
                    f"cost {cost:.2f} trillion dollars",
                    "GDP",
                    gdp_marketcap_dict[country_shortname],
                )
            if divide_by_marketcap and val > 200:
                print(
                    f"Beyond 200% of GDP: {country_shortname} {val:.2f}%",
                )
            # End of sanity check
            y.append(val)
            if not ((x_val <= 20_000) and (val <= 5)):
                filter_labels.append(country_shortname)
        plt.plot(x, y, label=label, linewidth=0, marker="o", fillstyle="none")
        annotate(x, y, plot_labels, fontsize=10, filter_labels=filter_labels)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # By level of development
    # Sanity check
    by_development = set(
        developing_shortnames + emerging_shortnames + developed_country_shortnames
    )
    print("Not in level of development", masterdata_coal_set - by_development)
    # End of sanity check
    plt.sca(axs[0])
    plot_scatter(developing_shortnames, "Developing country")
    plot_scatter(emerging_shortnames, "Emerging country")
    plot_scatter(developed_country_shortnames, "Developed country")
    plt.xlabel("GDP per capita (dollars)")
    if divide_by_marketcap:
        ylabel = "PV Climate financing / country GDP (%)"
    else:
        ylabel = f"PV Climate financing ({multiplier_mode} dollars)"
    plt.ylabel(ylabel)
    if not divide_by_marketcap and multiplier_mode == "billion":
        plt.ylim(0, 1e3)
    plt.legend(loc="upper right")

    # By region
    iso3166_df = pd.read_csv("data/country_ISO-3166_with_region.csv")
    region_countries_map, regions = prepare_regions_for_climate_financing(iso3166_df)
    # Sanity check
    by_region = []
    for country_names in region_countries_map.values():
        by_region += country_names
    by_region = set(by_region)
    print("Not in by region", masterdata_coal_set - by_region)
    # End of sanity check
    plt.sca(axs[1])
    for region_name, country_names in region_countries_map.items():
        if region_name == "Latin America & the Carribean":
            region_name = "Latin America &\nthe Carribean"
        elif region_name == "Australia & New Zealand":
            region_name = "Australia &\nNew Zealand"
        plot_scatter(country_names, region_name)
    plt.xlabel("GDP per capita (dollars)")
    plt.ylabel(ylabel)
    if not divide_by_marketcap and multiplier_mode == "billion":
        plt.ylim(0, 1e3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    util.savefig("climate_financing_scatter")


def calculate_yearly_costs_dict(chosen_s2_scenario):
    yearly_costs_dict = {}
    out = run_cost1(x=1, to_csv=False, do_round=False, return_yearly=True)
    yearly_cost_for_avoiding = out[chosen_s2_scenario]["cost"]
    country_names = list(yearly_cost_for_avoiding[-1].keys())
    for country_name in country_names:
        country_level_cost = []
        for e in yearly_cost_for_avoiding:
            if isinstance(e, float):
                country_level_cost.append(e)
            elif isinstance(e, dict):
                country_level_cost.append(e[country_name])
            else:
                # Pandas series
                country_level_cost.append(e.loc[country_name])
        yearly_costs_dict[country_name] = country_level_cost
    return yearly_costs_dict


def make_yearly_climate_financing_plot():
    global nonpower_coal, power_coal

    chosen_s2_scenario = "2022-2100 2DII + Net Zero 2050 Scenario"
    chosen_s2_scenario += " NON-DISCOUNTED"

    (
        iso3166_df,
        iso3166_df_alpha2,
        developed_gdp,
        colname_for_gdp,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()

    git_branch = util.get_git_branch()
    # The cache is used only for each of the developed countries.
    cache_json_path = f"plots/climate_financing_yearly_{git_branch}.json"
    if os.path.isfile(cache_json_path):
        print("Cached climate YEARLY financing json found. Reading...")
        yearly_costs_dict = util.read_json(cache_json_path)
        print("Done")
    else:
        yearly_costs_dict = calculate_yearly_costs_dict(chosen_s2_scenario)
        with open(cache_json_path, "w") as f:
            json.dump(yearly_costs_dict, f)
    whole_years = range(2022, 2100 + 1)

    def _get_yearly_cost(shortnames):
        out = np.zeros(len(whole_years))
        for n in shortnames:
            if n in yearly_costs_dict:
                out += np.array(yearly_costs_dict[n])
        return out

    yearly_developed_cost = _get_yearly_cost(developed_country_shortnames)

    # Calculating the cost for the whole world
    yearly_world_cost = np.zeros(len(whole_years))
    for v in yearly_costs_dict.values():
        yearly_world_cost += np.array(v)

    # Calculating the climate change cost for developing countries
    developING_country_shortnames = util.get_developing_countries()
    yearly_developing_cost = _get_yearly_cost(developING_country_shortnames)

    # Calculating for emerging countries
    emerging_country_shortnames = util.get_emerging_countries()
    yearly_emerging_cost = _get_yearly_cost(emerging_country_shortnames)

    # Sanity check
    # The world's cost must be equal to sum of its parts.
    sum_individuals = (
        sum(yearly_developed_cost)
        + sum(yearly_developing_cost)
        + sum(yearly_emerging_cost)
    )
    assert math.isclose(sum(yearly_world_cost), sum_individuals)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(axs[0])
    plt.plot(whole_years, yearly_world_cost, label="World")
    plt.plot(whole_years, yearly_developed_cost, label="Developed countries")
    plt.plot(whole_years, yearly_developing_cost, label="Developing countries")
    plt.plot(whole_years, yearly_emerging_cost, label="Emerging-market countries")

    plt.xlabel("Time")
    plt.ylabel("Annual climate financing\n(trillion dollars)")
    set_matplotlib_tick_spacing(10)
    plt.xticks(rotation=45, ha="right")
    plt.legend()

    for_comparison = {
        "by_development": {
            "World": list(yearly_world_cost),
            "Developed countries": list(yearly_developed_cost),
            "Developing countries": list(yearly_developing_cost),
            "Emerging-market countries": list(yearly_emerging_cost),
        },
        "by_region": {},
    }

    # Part 2. By regions
    plt.sca(axs[1])
    plt.plot(whole_years, yearly_world_cost, label="World")
    region_countries_map, regions = prepare_regions_for_climate_financing(iso3166_df)
    for region in regions:
        included_country_names = region_countries_map[region]
        yearly_cost = _get_yearly_cost(included_country_names)
        plt.plot(whole_years, yearly_cost, label=region)
        for_comparison["by_region"][region] = list(yearly_cost)

    plt.xlabel("Time")
    plt.ylabel("Annual climate financing\n(trillion dollars)")
    set_matplotlib_tick_spacing(10)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    util.savefig("climate_financing_yearly")
    plt.close()

    # This is used for by-region vs by-world of NGFS fractional increase
    # method.
    with open(f"plots/for_comparison_yearly_{git_branch}.json", "w") as f:
        json.dump(for_comparison, f)

    # Part 3. Relative to 2020 developed GDP.
    # million dollars.
    total_developed_gdp = developed_gdp[colname_for_gdp].sum()
    # Convert to trillion dollars.
    total_developed_gdp /= 1e6
    plt.figure()

    def do_plot(y, label, linestyle=None):
        plt.plot(
            whole_years,
            np.array(y) * 100 / total_developed_gdp,
            label=label,
            linestyle=linestyle,
        )

    do_plot(yearly_world_cost, "World", linestyle="dashed")
    do_plot(
        np.array(yearly_developing_cost) + np.array(yearly_emerging_cost),
        "Developing & emerging\nworld",
        linestyle="dashed",
    )
    do_plot(yearly_emerging_cost, "Emerging world")
    do_plot(yearly_developed_cost, "Developed world")
    do_plot(yearly_developing_cost, "Developing world")

    plt.xlabel("Time")
    plt.ylabel("Annual climate financing / developed world GDP (%)")
    plt.legend(title="Annual climate financing:")
    util.savefig("climate_financing_yearly_relative")


def make_yearly_climate_financing_plot_SENSITIVITY_ANALYSIS():
    chosen_s2_scenario_discounted = "2022-2100 2DII + Net Zero 2050 Scenario"
    chosen_s2_scenario_non_discounted = (
        chosen_s2_scenario_discounted + " NON-DISCOUNTED"
    )

    whole_years = range(2022, 2100 + 1)

    def calculate_yearly_world_cost(s2_scenario):
        yearly_costs_dict = calculate_yearly_costs_dict(s2_scenario)
        # Calculating the cost for the whole world
        yearly_world_cost = np.zeros(len(whole_years))
        for v in yearly_costs_dict.values():
            yearly_world_cost += np.array(v)
        return yearly_world_cost

    def _get_year_range_cost(year_start, year_end, yearly_world_cost):
        return sum(yearly_world_cost[year_start - 2022 : year_end + 1 - 2022])

    label_map = {
        "30Y": "30Y lifetime, D, E",
        "30Y_noE": "30Y lifetime, D, no E",
        "50Y": "50Y lifetime, D, E",
        "200Y": "Lifetime dictated\nby D, E",
        "LCOE": "LCOE proxy\ninvestment costs",
    }

    def reset():
        global ENABLE_NEW_METHOD, ENABLE_WRIGHTS_LAW, RENEWABLE_LIFESPAN
        ENABLE_NEW_METHOD = 1
        ENABLE_WRIGHTS_LAW = 1
        RENEWABLE_LIFESPAN = 30

    data_for_barchart = {
        (NGFS_PEG_YEAR + 1, 2050): {},
        (2051, 2070): {},
        (2071, 2100): {},
    }
    global ENABLE_NEW_METHOD, ENABLE_WRIGHTS_LAW, RENEWABLE_LIFESPAN
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(axs[0])
    for key, label in label_map.items():
        reset()
        if key == "LCOE":
            ENABLE_NEW_METHOD = 0
        elif key.endswith("Y"):
            RENEWABLE_LIFESPAN = int(key[:-1])
        else:
            assert key == "30Y_noE"
            ENABLE_WRIGHTS_LAW = 0
        linestyle = "-" if key == "30Y" else "dotted"
        yearly = calculate_yearly_world_cost(chosen_s2_scenario_non_discounted)
        plt.plot(
            whole_years,
            yearly,
            label=label,
            linestyle=linestyle,
            linewidth=2.5,
        )

        yearly_discounted = calculate_yearly_world_cost(chosen_s2_scenario_discounted)
        for year_start, year_end in [
            (NGFS_PEG_YEAR + 1, 2050),
            (2051, 2070),
            (2071, 2100),
        ]:
            aggregate = _get_year_range_cost(year_start, year_end, yearly_discounted)
            data_for_barchart[(year_start, year_end)][label] = aggregate
    plt.xlabel("Time")
    plt.ylabel("Global annual climate financing\n(trillion dollars)")
    plt.legend(
        bbox_to_anchor=(0.5, -0.2),
        loc="upper center",
        ncol=2,
    )

    # Bar chart
    plt.sca(axs[1])
    xticks = None
    stacked_bar_data = []
    for year_pair, data in data_for_barchart.items():
        xticks = list(data.keys())
        stacked_bar_data.append((f"{year_pair[0]}-{year_pair[1]}", list(data.values())))
    util.plot_stacked_bar(
        xticks,
        stacked_bar_data,
    )
    plt.xticks(xticks, rotation=45, ha="right")
    plt.ylabel("PV global climate financing\n(trillion dollars)")
    plt.legend(
        loc="upper left",
    )
    plt.tight_layout()
    util.savefig("climate_financing_sensitivity", tight=True)


def calculate_capacity_investment_gamma():
    from scipy import stats

    print("Gamma calculation")
    # We pick the years to be the overlap of the IRENA data
    # (2011-2020 and 2010-2020).

    irena = util.read_json("data/irena.json")
    for tech in ["solar", "onshore_wind", "offshore_wind"]:
        cumulative_capacity = irena[
            f"{tech}_MW_world_cumulative_total_installed_capacity_2011_2020"
        ]
        log_cumulative_capacity = np.log(cumulative_capacity)
        # Limit to 2011-2020
        log_investment_cost = np.log(irena[f"installed_cost_{tech}_2010_2020_$/kW"][1:])

        slope, intercept, r_value, *_ = stats.linregress(
            log_cumulative_capacity, log_investment_cost
        )
        print(tech, "r_value", r_value, "slope", slope, "alpha", np.exp(intercept))

        plt.figure()
        plt.scatter(log_cumulative_capacity, log_investment_cost)
        plt.plot(log_cumulative_capacity, intercept + slope * log_cumulative_capacity)
        plt.xlabel("log(Cumulative installed capacity in MW)")
        plt.ylabel("log(Investment costs in $/kW)")
        util.savefig(f"gamma_{tech}")


# For website sensitivity analysis
def nested_dict(n, _type):
    if n == 1:
        return defaultdict(_type)
    else:
        return defaultdict(lambda: nested_dict(n - 1, _type))


def initialize_website_sensitivity_analysis_params():
    measure_map = {
        "cao": "Carbon arbitrage including residual benefit (in trillion dollars)",
        "cao_relative": "Carbon arbitrage including residual benefit relative to world GDP (%)",
        "cost": "Costs of avoiding coal emissions (in trillion dollars)",
        "benefit": "Benefits of avoiding coal emissions including residual benefit (in trillion dollars)",
        "production_avoided": "Total coal production avoided including residual (Giga tonnes)",
        "emissions_avoided": "Total emissions avoided including residual (GtCO2)",
        "opportunity_cost": "Opportunity costs represented by missed coal revenues (in trillion dollars)",
        "investment_cost": "Investment costs in renewable energy (in trillion dollars)",
    }

    social_costs = list(range(25, 400 + 1, 25))
    time_horizons = [2050, 2070, 2100]
    # For these dicts, the keys are usually the string value as shown in the
    # demonstration website
    # (https://ox-inet-resilience.github.io/carbon_arbitrage/).
    coal_replacements = {
        "50% solar, 25% wind onshore, 25% wind offshore": (0.5, 0.25, 0.25),
        "100% solar, 0% wind": (1.0, 0.0, 0.0),
        "56% solar, 42% wind onshore, 2% wind offshore": (0.56, 0.42, 0.02),
        "0% solar, 100% wind onshore, 0% wind offshore": (0.0, 1.0, 0.0),
        "0% solar, 0% wind onshore, 100% wind offshore": (0.0, 0.0, 1.0),
    }
    lifetimes = [30, 50]
    learning_curve_map = {
        "Learning (investment cost drop because of learning)": True,
        "No learning (no investment cost drop)": False,
    }
    rho_mode_map = {
        "0%": "0%",
        "2.8% (WACC)": "default",
        "3.6% (WACC, average risk-premium 100 years)": "100year",
        "5%": "5%",
        "8%": "8%",
    }

    return (
        measure_map,
        social_costs,
        time_horizons,
        coal_replacements,
        lifetimes,
        learning_curve_map,
        rho_mode_map,
    )


# End of for website sensitivity analysis


def do_website_sensitivity_analysis():
    out_dict = nested_dict(5, dict)
    (
        measure_map,
        social_costs,
        time_horizons,
        coal_replacements,
        lifetimes,
        learning_curve_map,
        rho_mode_map,
    ) = initialize_website_sensitivity_analysis_params()
    for learning_curve in learning_curve_map:
        for lifetime in lifetimes:
            for coal_replacement in coal_replacements:
                for last_year in time_horizons:
                    for rho_mode in rho_mode_map:
                        all_scs_output = mp.Manager().dict()

                        def fn(sc):
                            global ENABLE_WRIGHTS_LAW, RENEWABLE_LIFESPAN, social_cost_of_carbon, MID_YEAR, RENEWABLE_WEIGHTS, RHO_MODE
                            if last_year == 2070:
                                MID_YEAR = 2070
                            else:
                                MID_YEAR = 2050
                            conditions = {
                                "Net Zero 2050 (NGFS global scenario)": f"2022-{last_year} 2DII + Net Zero 2050 Scenario",
                                "Halt to coal production": f"2022-{last_year} 2DII + Current Policies  Scenario",
                            }

                            ENABLE_WRIGHTS_LAW = learning_curve_map[learning_curve]
                            RENEWABLE_LIFESPAN = lifetime
                            weights = coal_replacements[coal_replacement]
                            RENEWABLE_WEIGHTS = {
                                "solar": weights[0],
                                "onshore_wind": weights[1],
                                "offshore_wind": weights[2],
                            }
                            RHO_MODE = rho_mode_map[rho_mode]
                            util.social_cost_of_carbon = sc
                            social_cost_of_carbon = sc  # noqa: F811
                            out = run_cost1(
                                x=1, to_csv=False, do_round=True, plot_yearly=False
                            )
                            fn_output = defaultdict(dict)
                            for k, v in measure_map.items():
                                for (
                                    condition_key,
                                    condition_value,
                                ) in conditions.items():
                                    fn_output[k][condition_key] = out[v][
                                        condition_value
                                    ]
                            all_scs_output[str(sc)] = fn_output

                        util.run_parallel(fn, social_costs, ())
                        out_dict[learning_curve][str(lifetime)][coal_replacement][
                            str(last_year)
                        ][rho_mode] = dict(all_scs_output)
    with open("sensitivity_analysis_result.json", "w") as f:
        json.dump(out_dict, f, separators=(",", ":"))


def common_set_website_sensitiviy_analysis_params(
    param, learning_curve_map, coal_replacements
):
    global ENABLE_WRIGHTS_LAW, RENEWABLE_LIFESPAN, MID_YEAR, RENEWABLE_WEIGHTS
    learning_curve = param["learning_curve"]
    lifetime = param["lifetime"]
    coal_replacement = param["coal_replacement"]
    last_year = param.get("last_year", 2100)
    s2_scenario = param["s2_scenario"]

    if last_year == 2070:
        MID_YEAR = 2070
    else:
        MID_YEAR = 2050

    discount = " NON-DISCOUNTED" if "NON-DISCOUNTED" in s2_scenario else ""
    if "Net Zero 2050 (NGFS global scenario)" in s2_scenario:
        chosen_s2_scenario = (
            f"2022-{last_year} 2DII + Net Zero 2050 Scenario" + discount
        )
    else:
        assert "Halt to coal production" in s2_scenario
        chosen_s2_scenario = (
            f"2022-{last_year} 2DII + Current Policies  Scenario" + discount
        )

    ENABLE_WRIGHTS_LAW = learning_curve_map[learning_curve]
    RENEWABLE_LIFESPAN = lifetime
    weights = coal_replacements[coal_replacement]
    RENEWABLE_WEIGHTS = {
        "solar": weights[0],
        "onshore_wind": weights[1],
        "offshore_wind": weights[2],
    }
    return chosen_s2_scenario


def do_website_sensitivity_analysis_climate_financing():
    global ENABLE_RESIDUAL_BENEFIT
    ENABLE_RESIDUAL_BENEFIT = 0
    os.makedirs("plots/climate_financing", exist_ok=True)

    (
        measure_map,
        social_costs,
        time_horizons,
        coal_replacements,
        lifetimes,
        learning_curve_map,
        rho_mode_map,
    ) = initialize_website_sensitivity_analysis_params()

    s2_scenarios = ["Net Zero 2050 (NGFS global scenario)", "Halt to coal production"]

    params_flat = []
    for learning_curve in learning_curve_map:
        for lifetime in lifetimes:
            for coal_replacement in coal_replacements:
                for s2_scenario in s2_scenarios:
                    params_flat.append(
                        {
                            "learning_curve": learning_curve,
                            "lifetime": lifetime,
                            "coal_replacement": coal_replacement,
                            # Important: must be non-discounted
                            "s2_scenario": s2_scenario + " NON-DISCOUNTED",
                        }
                    )

    print("Total number of params", len(params_flat))

    output = mp.Manager().dict()

    def fn(param):
        chosen_s2_scenario = common_set_website_sensitiviy_analysis_params(
            param, learning_curve_map, coal_replacements
        )
        param_key = "_".join(str(v) for v in param.values())

        yearly_costs_dict = calculate_yearly_costs_dict(chosen_s2_scenario)
        # Reduce the floating precision to save space
        yearly_costs_dict = {
            k: [float(f"{i:.8f}") for i in v] for k, v in yearly_costs_dict.items()
        }
        output[param_key] = yearly_costs_dict

    util.run_parallel_ncpus(8, fn, params_flat, ())

    with open("plots/website_sensitivity_climate_financing.json", "w") as f:
        json.dump(dict(output), f, separators=(",", ":"))

    # Reenable residual benefit again
    ENABLE_RESIDUAL_BENEFIT = 1


if __name__ == "__main__":
    if 0:
        print("# exp cost5")
        both_coal = nonpower_coal.append(power_coal).reset_index()
        all_coal_company_ids = set(both_coal.company_id)
        print("Number of coal companies (nonpower & power)", len(all_coal_company_ids))
        non_coal_oilgas_power_company_ids = set()
        non_coal_oilgas_power_sectors = set()
        for company_id in all_coal_company_ids:
            rows_for_one_id = df[df.company_id == company_id]
            for idx, row in rows_for_one_id.iterrows():
                if row.sector not in ["Coal", "Oil&Gas", "Power"]:
                    non_coal_oilgas_power_company_ids.add(row.company_id)
                    non_coal_oilgas_power_sectors.add(row.sector)
        print(
            "Number of coal companies with sectors outside of Coal/Oil&Gas/Power:",
            len(non_coal_oilgas_power_company_ids),
        )
        print("The sectors:", non_coal_oilgas_power_sectors)
        print("Units of power sector", set(df[df.sector == "Power"].unit))
        exit()

    if 0:
        print("# exp cost6")
        # Figuring out all of the coal companies.
        nonpower_coal_unique = nonpower_coal.drop_duplicates(subset=["company_id"])
        power_coal_unique = power_coal.drop_duplicates(subset=["company_id"])
        nonpower_set = set(nonpower_coal_unique.company_id)
        power_set = set(power_coal_unique.company_id)
        revenue_set = processed_revenue.revenue_data_companies
        print("Revenue data", len(revenue_set))
        print("Non-Power", len(nonpower_coal_unique))
        print("Power", len(power_coal_unique))
        print(
            "Non-Power intersect with Power", len(nonpower_set.intersection(power_set))
        )
        print(
            "Revenue data intersect with Non-Power",
            len(revenue_set.intersection(nonpower_set)),
        )
        print(
            "Revenue data intersect with Power",
            len(revenue_set.intersection(power_set)),
        )
        print(
            "Revenue data intersect with Non-Power intersect with Power",
            len(revenue_set.intersection(nonpower_set).intersection(power_set)),
        )
        # both_coal_unique = nonpower_coal_unique.append(power_coal_unique).reset_index().drop_duplicates(subset=["company_id"])
        # both_coal_unique.set_index("company_id")["company_name"].to_csv("plots/both_coal.csv")
        exit()

    if 0:
        # Calculating the top 5 coal producers in the US
        both_coal = nonpower_coal.append(power_coal).reset_index()
        coal_prod_em = {}
        for idx, row in both_coal.iterrows():
            if row.asset_country != "US":
                continue
            if row.sector == "Coal":
                tonnes_coal = row._2021
                gj = util.coal2GJ(tonnes_coal)
                # In tCO2
                em = tonnes_coal * row.emissions_factor
            else:
                mw_coal = row._2021
                gj = util.MW2GJ(mw_coal)
                # In tCO2
                em = mw_coal * util.hours_in_1year * row.emissions_factor
            if row.company_id not in coal_prod_em:
                coal_prod_em[row.company_id] = {
                    "prod": gj,
                    "emissions": em,
                    "name": row.company_name,
                    "country": row.asset_country,
                }
            else:
                coal_prod_em[row.company_id]["prod"] += gj
                coal_prod_em[row.company_id]["emissions"] += em
        print("Unit for prod: GJ; Unit for emissions: tCO2")
        for measure in ["prod", "emissions"]:
            print("Measure:", measure)
            by_x_list = sorted(
                coal_prod_em,
                key=lambda x: coal_prod_em[x][measure],
                reverse=True,
            )
            for i in range(5):
                print(coal_prod_em[by_x_list[i]])
        exit()

    # calculate_capacity_investment_gamma()
    if 0:
        run_cost1(x=1, to_csv=True, do_round=True, plot_yearly=False)
        exit()
    if 0:
        do_website_sensitivity_analysis()
        # do_website_sensitivity_analysis_climate_financing()
        exit()
    if 0:
        # Run for 3 levels of social cost of carbon.
        global social_cost_of_carbon
        mode = "cao"
        # mode = "cao_relative"
        # mode = "cost"
        # mode = "benefit"
        if mode == "cao":
            cao_name = "Carbon arbitrage opportunity (in trillion dollars)"
            cao_name_with_residual = (
                "Carbon arbitrage including residual benefit (in trillion dollars)"
            )
        elif mode == "cao_relative":
            cao_name = "Carbon arbitrage opportunity relative to world GDP (%)"
            cao_name_with_residual = (
                "Carbon arbitrage including residual benefit relative to world GDP (%)"
            )
        elif mode == "cost":
            cao_name = "Costs of avoiding coal emissions (in trillion dollars)"
            cao_name_with_residual = cao_name
        elif mode == "benefit":
            cao_name = "Benefits of avoiding coal emissions (in trillion dollars)"
            cao_name_with_residual = cao_name
        print(cao_name)
        for last_year in [2050, 2070, 2100]:
            if last_year == 2070:
                MID_YEAR = 2070
            else:
                MID_YEAR = 2050
            caos = []
            caos_with_residual = []
            condition = f"2022-{last_year} 2DII + Net Zero 2050 Scenario"
            # condition = f"2022-{last_year} 2DII + Current Policies  Scenario"
            scs = [
                util.social_cost_of_carbon_lower_bound,
                util.social_cost_of_carbon_imf,
                util.social_cost_of_carbon_upper_bound,
            ]
            for sc in scs:
                util.social_cost_of_carbon = sc
                social_cost_of_carbon = sc  # noqa: F811
                out = run_cost1(x=1, to_csv=True, do_round=True, plot_yearly=False)
                cao = out[cao_name][condition]
                caos.append(f"{cao:.2f}")
                cao_with_residual = out[cao_name_with_residual][condition]
                caos_with_residual.append(f"{cao_with_residual:.2f}")
            if "Net Zero 2050" in condition:
                info = "NZ2050"
            elif "Current Policies" in condition:
                info = "CPS"
            else:
                raise Exception(f"condition not expected: {condition}")
            print(last_year, info, "with residual:")
            # print(" & ".join(caos))
            if ENABLE_RESIDUAL_BENEFIT and (cao_name != cao_name_with_residual):
                # print("With residual:")
                print(" & ".join(caos_with_residual))
        exit()
    # plot_break_even_carbon_tax()
    # calculate_carbon_adjusted_earnings()
    # calculate_social_value_of_stranded_assets()
    # calculate_social_value_of_stranded_assets(year_2020_only=True)
    # make_carbon_arbitrage_opportunity_plot()
    # exit()
    # make_carbon_arbitrage_opportunity_plot(relative_to_world_gdp=True)
    # It is faster not to calculate residual benefit for climate financing.
    ENABLE_RESIDUAL_BENEFIT = 0
    # make_climate_financing_plot()
    make_climate_financing_SCATTER_plot()
    exit()
    # make_yearly_climate_financing_plot()
    make_yearly_climate_financing_plot_SENSITIVITY_ANALYSIS()
    # Reenable residual benefit again
    ENABLE_RESIDUAL_BENEFIT = 1
