import copy
import functools
import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler

import util
from util import (
    social_cost_of_carbon,
    world_gdp_2023,
)
import with_learning

sns.set_theme(style="ticks")
# TODO these globals could be removed.
global_cost_with_learning = None
MEASURE_GLOBAL_VARS = False
# Only change this to current policies if you want to see the result
# for halt to fossil fuel production scenario
MEASURE_GLOBAL_VARS_SCENARIO = "Net Zero 2050"


# Ensure that plots directory exists
os.makedirs("plots", exist_ok=True)

# Params that can be modified
lcoe_mode = "solar+wind"
# lcoe_mode="solar+wind+gas"
ENABLE_COAL_EXPORT = 0
ENABLE_WORKER = 1
LAST_YEAR = 2050
# The year where the NGFS value is pegged/rescaled to be the same as Masterdata
# global production value.
NGFS_PEG_YEAR = 2024
# SECTOR_INCLUDED = "Coal"
SECTOR_INCLUDED = "Power"
# Possible values: "default", "100year", "5%", "8%", "0%"
RHO_MODE = "default"
# This is used in Bruegel analysis. Might be deleted later.
INVESTMENT_COST_DIVIDER = 1

print("Renewable degradation:", with_learning.ENABLE_RENEWABLE_GRADUAL_DEGRADATION)
print("30 year lifespan:", with_learning.ENABLE_RENEWABLE_30Y_LIFESPAN)
print("Wright's law", with_learning.ENABLE_WRIGHTS_LAW)
print("Residual benefit", with_learning.ENABLE_RESIDUAL_BENEFIT)
print("Sector included", SECTOR_INCLUDED)
print("BATTERY_SHORT", with_learning.ENABLE_BATTERY_SHORT)
print("BATTERY_LONG", with_learning.ENABLE_BATTERY_LONG)


assert SECTOR_INCLUDED in ["Power", "Coal"]


def maybe_round6(do_it, x):
    return round(x, 6) if do_it else x


def pandas_divide_or_zero(num, dem):
    return (num / dem).replace([np.inf, -np.inf], 0)


ngfs_df = util.read_ngfs()
iso3166_df = util.read_iso3166()
unit_profit_df = pd.read_csv(
    "data_private/v5_final_country_region_profit_analysis.csv.zip",
    compression="zip",
)
alpha2_to_alpha3 = iso3166_df.set_index("alpha-2")["alpha-3"].to_dict()

_, df_sector = util.read_forward_analytics_data(SECTOR_INCLUDED)
country_sccs = pd.Series(util.read_country_specific_scc_filtered())


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
    assert len(x) == len(y)
    out = []
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        if isinstance(xi, pd.Series):
            xi = xi.to_dict()
        if isinstance(yi, pd.Series):
            yi = yi.to_dict()

        if isinstance(xi, dict):
            if isinstance(yi, float) and yi == 0.0:
                out.append(xi.copy())
                continue
            z = {}
            # We need to include keys from both xi and yi, because recently in
            # the coal export, there are 100% importer countries that are not
            # part of masterdata.
            for key in set(xi) | set(yi):
                z[key] = xi.get(key, 0) + yi.get(key, 0)
            out.append(z)
        else:
            # float
            out.append(xi + yi)
    return out


def calculate_table1_info(
    do_round,
    scenario,
    data_set,
    time_period,
    total_production,
    array_of_total_emissions_non_discounted,
    array_of_cost_non_discounted_owner_by_subsector,
    array_of_cost_discounted_owner_by_subsector,  # opportunity cost
    array_of_cost_non_discounted_investment,
    array_of_cost_discounted_investment,
    current_policies=None,
    residual_emissions_series=0.0,
    residual_production_series=0.0,
    final_cost_with_learning=None,
    included_countries=None,
):
    if INVESTMENT_COST_DIVIDER > 1:
        array_of_cost_discounted_investment = divide_array_of_mixed_objs(
            array_of_cost_discounted_investment, INVESTMENT_COST_DIVIDER
        )
        array_of_cost_non_discounted_investment = divide_array_of_mixed_objs(
            array_of_cost_non_discounted_investment, INVESTMENT_COST_DIVIDER
        )
    out_yearly_info = {}

    # Workers
    subsectors = util.SUBSECTORS
    cost_discounted_worker = 0
    worker_compensation = None
    worker_retraining_cost = None
    if ENABLE_WORKER:
        import coal_worker

        worker_by_subsector = {
            subsector: coal_worker.calculate(
                "default",
                subsector,
                LAST_YEAR,
                included_countries=included_countries,
                return_summed=True,
                scenario=scenario,
            )
            for subsector in subsectors
        }
        worker_compensation = {
            subsector: worker_by_subsector[subsector][0] for subsector in subsectors
        }
        worker_compensation_sum_trillion = sum(worker_compensation.values()) / 1e3
        worker_retraining_cost = {
            subsector: worker_by_subsector[subsector][1] for subsector in subsectors
        }
        worker_rc_sum_trillion = sum(worker_retraining_cost.values()) / 1e3
        cost_discounted_worker = (
            worker_compensation_sum_trillion + worker_rc_sum_trillion
        )
    # End of workers

    # Sum across subsectors
    array_of_cost_non_discounted_owner = functools.reduce(
        util.add_array, array_of_cost_non_discounted_owner_by_subsector.values()
    )
    array_of_cost_discounted_owner = functools.reduce(
        util.add_array, array_of_cost_discounted_owner_by_subsector.values()
    )

    cost_non_discounted_owner = sum_array_of_mixed_objs(
        array_of_cost_non_discounted_owner
    )
    # nansum is needed because PG has nan value for gas
    cost_discounted_owner = np.nansum(array_of_cost_discounted_owner)
    cost_non_discounted_investment = sum_array_of_mixed_objs(
        array_of_cost_non_discounted_investment
    )
    cost_discounted_investment = sum_array_of_mixed_objs(
        array_of_cost_discounted_investment
    )
    # In GtCO2
    residual_emissions = residual_emissions_series.sum() / 1e9
    if current_policies is None:
        assert data_set == "FA" or "Current Policies" in data_set, data_set
        # The groupby-sum aggregates across subsectors
        avoided_emissions_by_country: pd.Series = (
            sum(array_of_total_emissions_non_discounted).groupby(level=0).sum()
        )
        avoided_emissions_non_discounted: float = avoided_emissions_by_country.sum()
        avoided_emissions_by_subsector = (
            sum(array_of_total_emissions_non_discounted).groupby(level=1).sum()
        )
        total_production_avoided = total_production
        out_yearly_info["benefit_non_discounted"] = list(
            array_of_total_emissions_non_discounted
        )
    else:
        assert not (data_set == "FA" or "Current Policies" in data_set)
        # The groupby-sum aggregates across subsectors
        avoided_emissions_by_country: pd.Series = (
            (
                sum(current_policies["emissions_non_discounted"])
                - sum(array_of_total_emissions_non_discounted)
            )
            .groupby(level=0)
            .sum()
        )
        avoided_emissions_non_discounted: float = avoided_emissions_by_country.sum()
        avoided_emissions_by_subsector: pd.Series = (
            (
                sum(current_policies["emissions_non_discounted"])
                - sum(array_of_total_emissions_non_discounted)
            )
            .groupby(level=1)
            .sum()
        )

        total_production_avoided = (
            current_policies["total_production"] - total_production
        )
        out_yearly_info["benefit_non_discounted"] = util.subtract_array(
            current_policies["emissions_non_discounted"],
            array_of_total_emissions_non_discounted,
        )
    # Rescale to include residual emissions
    ae_by_subsector_with_residual = (
        avoided_emissions_by_subsector
        * (1 + residual_emissions / avoided_emissions_by_subsector.sum())
    ).to_dict()

    # We multiply by 1e9 to go from GtCO2 to tCO2
    # We divide by 1e12 to get trilllion USD
    for i in range(len(out_yearly_info["benefit_non_discounted"])):
        out_yearly_info["benefit_non_discounted"][i] *= (
            1e9 / 1e12 * social_cost_of_carbon
        )

    # Division of residual emissions dict by 1e9 converts to GtCO2
    out_yearly_info["avoided_emissions_including_residual_emissions"] = (
        avoided_emissions_by_country + residual_emissions_series / 1e9
    )
    # Sanity check
    expected = avoided_emissions_non_discounted + residual_emissions
    assert math.isclose(
        out_yearly_info["avoided_emissions_including_residual_emissions"].sum(),
        expected,
    )
    # Division by 1e3 converts to trillion dollars, because the ae is in GtCO2
    out_yearly_info["country_benefit_country_reduction"] = (
        (
            out_yearly_info["avoided_emissions_including_residual_emissions"]
            * country_sccs
            / country_sccs.sum()
            * social_cost_of_carbon
            / 1e3
        )
        .dropna()
        .to_dict()
    )
    out_yearly_info["global_benefit_country_reduction"] = (
        out_yearly_info["avoided_emissions_including_residual_emissions"]
        * social_cost_of_carbon
        / 1e3
    ).to_dict()
    out_yearly_info["avoided_emissions_including_residual_emissions"] = out_yearly_info[
        "avoided_emissions_including_residual_emissions"
    ].to_dict()

    # Summed benefit
    benefit_non_discounted = sum_array_of_mixed_objs(
        out_yearly_info["benefit_non_discounted"]
    )

    # Convert to trillion USD
    cost_non_discounted_owner /= 1e12
    cost_discounted_owner /= 1e12
    cost_non_discounted_investment /= 1e12
    cost_discounted_investment /= 1e12
    array_of_cost_non_discounted_owner_trillions = divide_array_of_mixed_objs(
        array_of_cost_non_discounted_owner, 1e12
    )
    array_of_cost_discounted_owner_trillions = divide_array_of_mixed_objs(
        array_of_cost_discounted_owner, 1e12
    )
    array_of_cost_non_discounted_investment_trillions = divide_array_of_mixed_objs(
        array_of_cost_non_discounted_investment, 1e12
    )
    array_of_cost_discounted_investment_trillions = divide_array_of_mixed_objs(
        array_of_cost_discounted_investment, 1e12
    )
    # In trillion dollars
    residual_benefit = residual_emissions * social_cost_of_carbon / 1e3

    # Convert to Gigatonnes of coal
    residual_production = residual_production_series.sum() / 1e9

    # rho is the same everywhere
    rho = util.calculate_rho(util.beta, rho_mode=RHO_MODE)

    def discount_the_array(arr):
        out = []
        for i, e in enumerate(arr):
            discount = util.calculate_discount(rho, i)
            if len(e) == 0:
                out.append(0.0)
            else:
                out.append({k: v * discount for k, v in e.items()})
        return out

    out_yearly_info["opportunity_cost_non_discounted"] = (
        array_of_cost_non_discounted_owner_trillions
    )
    out_yearly_info["investment_cost_non_discounted"] = (
        array_of_cost_non_discounted_investment_trillions
    )
    # Division by 1e12 converts to trillion
    out_yearly_info["cost_battery_short_non_discounted"] = divide_array_of_mixed_objs(
        final_cost_with_learning.cost_non_discounted_battery_short_by_country, 1e12
    )
    out_yearly_info["cost_battery_long_non_discounted"] = divide_array_of_mixed_objs(
        final_cost_with_learning.cost_non_discounted_battery_long_by_country, 1e12
    )
    out_yearly_info["cost_battery_pe_non_discounted"] = divide_array_of_mixed_objs(
        final_cost_with_learning.cost_non_discounted_battery_pe_by_country, 1e12
    )
    out_yearly_info["cost_battery_grid_non_discounted"] = divide_array_of_mixed_objs(
        final_cost_with_learning.cost_non_discounted_battery_grid_by_country, 1e12
    )
    out_yearly_info["opportunity_cost_owner"] = array_of_cost_discounted_owner_trillions
    out_yearly_info["investment_cost"] = array_of_cost_discounted_investment_trillions
    out_yearly_info["cost_battery_short"] = discount_the_array(
        divide_array_of_mixed_objs(
            final_cost_with_learning.cost_non_discounted_battery_short_by_country,
            1e12,
        )
    )
    out_yearly_info["cost_battery_long"] = discount_the_array(
        divide_array_of_mixed_objs(
            final_cost_with_learning.cost_non_discounted_battery_long_by_country,
            1e12,
        )
    )
    out_yearly_info["cost_battery_pe"] = discount_the_array(
        divide_array_of_mixed_objs(
            final_cost_with_learning.cost_non_discounted_battery_pe_by_country, 1e12
        )
    )
    out_yearly_info["cost_battery_grid"] = discount_the_array(
        divide_array_of_mixed_objs(
            final_cost_with_learning.cost_non_discounted_battery_grid_by_country,
            1e12,
        )
    )
    out_yearly_info["cost"] = add_array_of_mixed_objs(
        out_yearly_info["opportunity_cost_owner"], out_yearly_info["investment_cost"]
    )
    out_yearly_info["residual_benefit"] = {
        k: v * social_cost_of_carbon / 1e12
        for k, v in residual_emissions_series.items()
    }

    # Costs of avoiding coal emissions
    assert cost_discounted_investment >= 0
    cost_discounted = (
        cost_discounted_owner + cost_discounted_investment + cost_discounted_worker
    )

    # Equation 1 in the paper
    net_benefit = benefit_non_discounted - cost_discounted

    last_year = int(time_period.split("-")[1])
    arbitrage_period = last_year - NGFS_PEG_YEAR

    owner_dict = {
        # nansum is needed because PG has nan value for gas
        f"OC owner {subsector}": np.nansum(
            array_of_cost_discounted_owner_by_subsector[subsector]
        )
        / 1e12
        for subsector in subsectors
    }
    worker_dict = {}
    if ENABLE_WORKER:
        for subsector in subsectors:
            # Trillion
            worker_dict[f"OC workers lost wages {subsector}"] = (
                worker_compensation[subsector] / 1e3
            )
            worker_dict[f"OC workers retraining cost {subsector}"] = (
                worker_retraining_cost[subsector] / 1e3
            )

    ic_battery_short = sum_array_of_mixed_objs(out_yearly_info["cost_battery_short"])
    ic_battery_long = sum_array_of_mixed_objs(out_yearly_info["cost_battery_long"])
    ic_battery_pe = sum_array_of_mixed_objs(out_yearly_info["cost_battery_pe"])
    ic_battery_grid = sum_array_of_mixed_objs(out_yearly_info["cost_battery_grid"])
    ic_battery = ic_battery_short + ic_battery_long + ic_battery_pe + ic_battery_grid
    data = {
        "Using production projections of data set": data_set,
        "Time Period of Carbon Arbitrage": time_period,
        "Total coal production avoided (Giga tonnes)": total_production_avoided,
        "Total coal production avoided including residual (Giga tonnes)": total_production_avoided
        + residual_production,
        "Electricity generation avoided including residual (PWh)": (
            # Multiplication by 1e9 converts from Giga tonnes to tonnes
            # Division by seconds in 1 hr converts from GJ to GWh
            # Division by 1e6 converts from GWh to PWh
            util.coal2GJ((total_production_avoided + residual_production) * 1e9)
            / util.seconds_in_1hour
            / 1e6
        ),
        "Total emissions avoided (GtCO2)": avoided_emissions_non_discounted,
        "Total emissions avoided including residual (GtCO2)": avoided_emissions_non_discounted
        + residual_emissions,
        **{f"AE+residual {k}": v for k, v in ae_by_subsector_with_residual.items()},
        "Costs of avoiding coal emissions (in trillion dollars)": cost_discounted,
        "Opportunity costs (in trillion dollars)": cost_discounted_owner
        + cost_discounted_worker,
        "OC owner (in trillion dollars)": cost_discounted_owner,
        **owner_dict,
        **worker_dict,
        "investment_cost_battery_short_trillion": ic_battery_short,
        "investment_cost_battery_long_trillion": ic_battery_long,
        "investment_cost_battery_pe_trillion": ic_battery_pe,
        "investment_cost_battery_grid_trillion": ic_battery_grid,
        "Investment costs in renewable energy": cost_discounted_investment - ic_battery,
        "Investment costs (in trillion dollars)": cost_discounted_investment,
        "Carbon arbitrage opportunity (in trillion dollars)": net_benefit,
        "Carbon arbitrage opportunity relative to world GDP (%)": net_benefit
        * 100
        / (world_gdp_2023 * arbitrage_period),
        "Carbon arbitrage residual benefit (in trillion dollars)": residual_benefit,
        "Carbon arbitrage including residual benefit (in trillion dollars)": net_benefit
        + residual_benefit,
        "Carbon arbitrage including residual benefit relative to world GDP (%)": (
            net_benefit + residual_benefit
        )
        * 100
        / (world_gdp_2023 * arbitrage_period),
        "Benefits of avoiding coal emissions (in trillion dollars)": benefit_non_discounted,
        "Benefits of avoiding coal emissions including residual benefit (in trillion dollars)": benefit_non_discounted
        + residual_benefit,
        "country_benefit_country_reduction": sum(
            out_yearly_info["country_benefit_country_reduction"].values()
        ),
    }

    for k, v in data.items():
        if k in [
            "Using production projections of data set",
            "Time Period of Carbon Arbitrage",
        ]:
            continue
        data[k] = maybe_round6(do_round, v)
    # TODO included worker
    # out_yearly_info = None
    return data, out_yearly_info


def calculate_weighted_emissions_factor_by_country_peg_year(_df_nonpower):
    if not with_learning.ENABLE_RESIDUAL_BENEFIT:
        return None
    colname = "activity"

    grouped_np = _df_nonpower.groupby("asset_country")

    # In tce
    production_pegyear_np = grouped_np[colname].sum()
    production_pegyear = production_pegyear_np

    # Convert million tonnes of CO2 to tCO2
    emissions_pegyear_np = grouped_np[util.EMISSIONS_COLNAME].sum() * 1e6
    emissions_pegyear = emissions_pegyear_np
    ef = emissions_pegyear / production_pegyear
    # There are some countries with 0 production, and so it is division by
    # zero. We set them to 0.0 for now
    ef = ef.fillna(0.0)
    return ef


def get_opportunity_cost_owner(
    rho,
    delta_profit,
    scenario,
    last_year,
):
    # Calculate cost
    out_non_discounted = delta_profit
    out_discounted = {
        subsector: util.discount_array(out_non_discounted[subsector], rho)
        for subsector in util.SUBSECTORS
    }
    return (
        out_non_discounted,
        out_discounted,
    )


def get_cost_including_ngfs_renewable(
    _df_nonpower,
    rho,
    DeltaP,
    weighted_emissions_factor_by_country_peg_year,
    scenario,
    last_year,
    _cost_new_method,
):
    # We copy _cost_new_method because it is going to be reused for
    # different scenario and year range.
    temp_cost_with_learning = copy.deepcopy(_cost_new_method)

    for i, dp in enumerate(DeltaP):
        discount = util.calculate_discount(rho, i)
        temp_cost_with_learning.calculate_investment_cost(
            dp, NGFS_PEG_YEAR + i, discount
        )

    out_non_discounted = list(temp_cost_with_learning.cost_non_discounted)
    out_discounted = list(temp_cost_with_learning.cost_discounted)
    residual_benefits_years_offset = with_learning.RENEWABLE_LIFESPAN
    (
        residual_emissions,
        residual_production,
    ) = temp_cost_with_learning.calculate_residual(
        last_year + 1,
        last_year + residual_benefits_years_offset,
        weighted_emissions_factor_by_country_peg_year,
    )
    if MEASURE_GLOBAL_VARS and scenario == MEASURE_GLOBAL_VARS_SCENARIO:
        global global_cost_with_learning
        global_cost_with_learning = temp_cost_with_learning
    return (
        out_non_discounted,
        out_discounted,
        residual_emissions,
        residual_production,
        temp_cost_with_learning,
    )


def generate_table1_output(
    rho,
    do_round,
    _df_nonpower,
    included_countries=None,
):
    out = {}
    out_yearly = {}

    # Production
    # Giga tonnes of coal
    total_production_fa = util.get_production_by_country(df_sector, SECTOR_INCLUDED)

    # Emissions
    emissions_fa = util.get_emissions_by_country(df_sector)

    current_policies = None
    production_2019 = (
        _df_nonpower.groupby("asset_country")._2019.sum()
        if ENABLE_COAL_EXPORT
        else None
    )

    weighted_emissions_factor_by_country_peg_year = (
        calculate_weighted_emissions_factor_by_country_peg_year(_df_nonpower)
    )

    production_with_ngfs_projection_CPS = None
    profit_ngfs_projection_CPS = None
    for scenario in ["Current Policies", "Net Zero 2050"]:
        # NGFS_PEG_YEAR
        cost_new_method = with_learning.InvestmentCostWithLearning()

        last_year = LAST_YEAR
        # Giga tonnes of coal
        (
            production_with_ngfs_projection,
            gigatonnes_coal_production,
            profit_ngfs_projection,
        ) = util.calculate_ngfs_projection(
            "production",
            total_production_fa,
            ngfs_df,
            SECTOR_INCLUDED,
            scenario,
            NGFS_PEG_YEAR,
            last_year,
            alpha2_to_alpha3,
            unit_profit_df=unit_profit_df,
        )
        emissions_with_ngfs_projection, _, _ = util.calculate_ngfs_projection(
            "emissions",
            emissions_fa,
            ngfs_df,
            SECTOR_INCLUDED,
            scenario,
            NGFS_PEG_YEAR,
            last_year,
            alpha2_to_alpha3,
        )

        if scenario == "Current Policies":
            # To prepare for the s2-s1 for NZ2050
            production_with_ngfs_projection_CPS = production_with_ngfs_projection.copy()
            profit_ngfs_projection_CPS = profit_ngfs_projection.copy()

        scenario_formatted = f"FA + {scenario} Scenario"

        # NGFS_PEG_YEAR-last_year
        array_of_total_emissions_non_discounted = emissions_with_ngfs_projection

        if scenario == "Net Zero 2050":
            DeltaP = util.subtract_array(
                production_with_ngfs_projection_CPS, production_with_ngfs_projection
            )
            delta_profit = {
                subsector: util.subtract_array(
                    profit_ngfs_projection_CPS[subsector],
                    profit_ngfs_projection[subsector],
                )
                for subsector in util.SUBSECTORS
            }
        else:
            DeltaP = production_with_ngfs_projection_CPS
            delta_profit = profit_ngfs_projection_CPS
        # Convert Giga tonnes of coal to GJ
        DeltaP = util.coal2GJ([dp * 1e9 for dp in DeltaP])

        (
            cost_non_discounted_owner,
            cost_discounted_owner,
        ) = get_opportunity_cost_owner(
            rho,
            delta_profit,
            scenario,
            last_year,
        )
        (
            cost_non_discounted_investment,
            cost_discounted_investment,
            residual_emissions,
            residual_production,
            final_cost_with_learning,
        ) = get_cost_including_ngfs_renewable(
            _df_nonpower,
            rho,
            DeltaP,
            weighted_emissions_factor_by_country_peg_year,
            scenario,
            last_year,
            cost_new_method,
        )

        if ENABLE_COAL_EXPORT:
            from coal_export.common import modify_array_based_on_coal_export

            modify_array_based_on_coal_export(
                cost_non_discounted_investment, production_2019
            )
            modify_array_based_on_coal_export(
                cost_discounted_investment, production_2019
            )

        if included_countries is not None:

            def _filter(e, is_multiindex=False):
                if isinstance(e, dict):
                    e = pd.Series(e)
                if is_multiindex:
                    return e[e.index.get_level_values(0).isin(included_countries)]
                return e[e.index.isin(included_countries)]

            def _filter_arr(arr, is_multiindex=False):
                return [_filter(e, is_multiindex) for e in arr]

            gigatonnes_coal_production = _filter(
                sum(production_with_ngfs_projection)
            ).sum()

            array_of_total_emissions_non_discounted = _filter_arr(
                array_of_total_emissions_non_discounted, is_multiindex=True
            )
            cost_non_discounted_owner = {
                subsector: _filter_arr(cost_non_discounted_owner[subsector])
                for subsector in util.SUBSECTORS
            }
            cost_discounted_owner = {
                subsector: _filter_arr(cost_discounted_owner[subsector])
                for subsector in util.SUBSECTORS
            }
            cost_non_discounted_investment = _filter_arr(cost_non_discounted_investment)
            cost_discounted_investment = _filter_arr(cost_discounted_investment)
            residual_emissions = _filter(residual_emissions)
            residual_production = _filter(residual_production)
            final_cost_with_learning.cost_non_discounted_battery_short_by_country = _filter_arr(
                final_cost_with_learning.cost_non_discounted_battery_short_by_country
            )
            final_cost_with_learning.cost_non_discounted_battery_long_by_country = (
                _filter_arr(
                    final_cost_with_learning.cost_non_discounted_battery_long_by_country
                )
            )
            final_cost_with_learning.cost_non_discounted_battery_pe_by_country = (
                _filter_arr(
                    final_cost_with_learning.cost_non_discounted_battery_pe_by_country
                )
            )
            final_cost_with_learning.cost_non_discounted_battery_grid_by_country = (
                _filter_arr(
                    final_cost_with_learning.cost_non_discounted_battery_grid_by_country
                )
            )

        text = f"{NGFS_PEG_YEAR}-{last_year} {scenario_formatted}"
        table1_info, yearly_info = calculate_table1_info(
            do_round,
            scenario,
            scenario_formatted,
            f"{NGFS_PEG_YEAR}-{last_year}",
            gigatonnes_coal_production,
            copy.deepcopy(array_of_total_emissions_non_discounted),
            cost_non_discounted_owner,
            cost_discounted_owner,
            cost_non_discounted_investment,
            cost_discounted_investment,
            current_policies=current_policies,
            residual_emissions_series=residual_emissions,
            residual_production_series=residual_production,
            final_cost_with_learning=final_cost_with_learning,
            included_countries=included_countries,
        )
        out[text] = table1_info
        out_yearly[text] = yearly_info
        if scenario == "Current Policies":
            current_policies = {
                "emissions_non_discounted": copy.deepcopy(
                    array_of_total_emissions_non_discounted
                ),
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


def do_plot_yearly_table1(yearly_both):
    full_years_2100 = range(NGFS_PEG_YEAR, 2100 + 1)
    full_years_midyear = range(NGFS_PEG_YEAR, LAST_YEAR + 1)
    for condition, value in yearly_both.items():
        print(condition)
        plt.figure()
        x = (
            full_years_2100
            if f"{NGFS_PEG_YEAR}-2100" in condition
            else full_years_midyear
        )
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


def run_table1(
    to_csv=False,
    do_round=False,
    plot_yearly=False,
    return_yearly=False,
    included_countries=None,
):
    # rho is the same everywhere
    rho = util.calculate_rho(util.beta, rho_mode=RHO_MODE)

    out, yearly = generate_table1_output(
        rho,
        do_round,
        df_sector,
        included_countries=included_countries,
    )

    out_dict = out.T.to_dict()
    both_dict = {}
    for key in out_dict.keys():
        if key in [
            "Using production projections of data set",
            "Time Period of Carbon Arbitrage",
        ]:
            both_dict[key] = out_dict[key]
        else:
            both_dict[key] = maybe_round6(
                do_round,
                pd.Series(out_dict[key]),
            ).to_dict()
    if to_csv:
        uid = util.get_unique_id(include_date=False)
        ext = ""
        if with_learning.ENABLE_RENEWABLE_GRADUAL_DEGRADATION:
            ext += "_degrade"
        if with_learning.ENABLE_RENEWABLE_30Y_LIFESPAN:
            ext += "_30Y"
        if with_learning.ENABLE_WRIGHTS_LAW:
            ext += "_wright"
        fname = f"plots/table1_{uid}{ext}_{social_cost_of_carbon}_{LAST_YEAR}.csv"
        pd.DataFrame(both_dict).T.to_csv(fname)

    if plot_yearly:
        do_plot_yearly_table1(yearly)

    if return_yearly:
        return yearly

    return both_dict


def run_table2(name="", included_countries=None):
    global social_cost_of_carbon, LAST_YEAR
    result = {}
    sccs = [
        util.social_cost_of_carbon_imf,
        util.scc_biden_administration,
        util.scc_bilal,
    ]
    scc_default = util.social_cost_of_carbon_imf
    last_years = [2035, 2050]
    for last_year in last_years:
        util.social_cost_of_carbon = scc_default
        social_cost_of_carbon = scc_default
        LAST_YEAR = last_year
        result[last_year] = run_table1(included_countries=included_countries)
    gc_benefit_old_name = "Benefits of avoiding coal emissions including residual benefit (in trillion dollars)"
    subsectors = ["Coal", "Oil", "Gas"]
    mapper_worker = {}
    for subsector in subsectors:
        val = f"OC owner {subsector}"
        mapper_worker[val] = val
        val = f"OC workers lost wages {subsector}"
        mapper_worker[val] = val
        val = f"OC workers retraining cost {subsector}"
        mapper_worker[val] = val

    # Rename the key
    mapper = {
        "Time Period": "Time Period of Carbon Arbitrage",
        "Avoided fossil fuel electricity generation (PWh)": "Electricity generation avoided including residual (PWh)",
        "Avoided emissions (GtCO2e)": "Total emissions avoided including residual (GtCO2)",
        **{f"Avoided emissions {s}": f"AE+residual {s}" for s in util.SUBSECTORS},
        "Costs of power sector decarbonization (in trillion dollars)": "Costs of avoiding coal emissions (in trillion dollars)",
        "Opportunity costs (in trillion dollars)": "Opportunity costs (in trillion dollars)",
        "OC owner (in trillion dollars)": "OC owner (in trillion dollars)",
        **mapper_worker,
        "Investment costs (in trillion dollars)": "Investment costs (in trillion dollars)",
        "Investment costs in renewable energy": "Investment costs in renewable energy",
        "Investment costs short-term storage": "investment_cost_battery_short_trillion",
        "Investment costs long-term storage": "investment_cost_battery_long_trillion",
        "Investment costs renewables to power electrolyzers": "investment_cost_battery_pe_trillion",
        "Investment costs grid extension": "investment_cost_battery_grid_trillion",
    }

    def _s(y):
        return f"2024-{y} FA + Net Zero 2050 Scenario"

    table = defaultdict(list)
    for k, v in mapper.items():
        for y in last_years:
            try:
                table[k].append(result[y][v][_s(y)])
            except KeyError:
                print("Missing either of", k, y, v)
                continue
        if len(table[k]) == 0:
            # Some countries may have missing at least one of coal/oil/gas
            # e.g. Viet Nam has 0 oil
            del table[k]

    gdp_2023 = world_gdp_2023
    if included_countries is not None:
        gdp_marketcap_dict = util.read_json(util.gdp_marketcap_path)
        # actual = np.nansum(list(gdp_marketcap_dict.values()))
        # actual's value is 103.74, different from worldbank's own data of 105.44.
        # Both come from worldbank.
        # Convert to trillion dollars
        gdp_2023 = (
            np.nansum(
                [
                    gdp
                    for country, gdp in gdp_marketcap_dict.items()
                    if country in included_countries
                ]
            )
            / 1e12
        )
    for i, y in enumerate(last_years):
        table["Costs per avoided tCO2e ($/tCO2e)"].append(
            table["Costs of power sector decarbonization (in trillion dollars)"][i]
            * 1e12
            / (table["Avoided emissions (GtCO2e)"][i] * 1e9)
        )
        arbitrage_period = y - NGFS_PEG_YEAR
        for scc in sccs:
            scc_scale = scc / scc_default
            table[f"scc {scc} GC benefit (in trillion dollars)"].append(
                result[y][gc_benefit_old_name][_s(y)] * scc_scale
            )
            table[f"scc {scc} CC benefit (in trillion dollars)"].append(
                result[y]["country_benefit_country_reduction"][_s(y)] * scc_scale
            )

            table[f"scc {scc} GC benefit per avoided tCO2e ($/tCO2e)"].append(
                result[y][gc_benefit_old_name][_s(y)]
                * scc_scale
                * 1e12
                / (table["Avoided emissions (GtCO2e)"][i] * 1e9)
            )
            table[f"scc {scc} CC benefit per avoided tCO2e ($/tCO2e)"].append(
                result[y]["country_benefit_country_reduction"][_s(y)]
                * scc_scale
                * 1e12
                / (table["Avoided emissions (GtCO2e)"][i] * 1e9)
            )

            table[f"scc {scc} GC net benefit (in trillion dollars)"].append(
                table[f"scc {scc} GC benefit (in trillion dollars)"][i]
                - table["Costs of power sector decarbonization (in trillion dollars)"][
                    i
                ]
            )
            table[f"scc {scc} CC net benefit (in trillion dollars)"].append(
                table[f"scc {scc} CC benefit (in trillion dollars)"][i]
                - table["Costs of power sector decarbonization (in trillion dollars)"][
                    i
                ]
            )

            table[f"scc {scc} CC Net benefit relative to GDP (%)"].append(
                table[f"scc {scc} CC net benefit (in trillion dollars)"][i]
                * 100
                / (gdp_2023 * arbitrage_period)
            )

            table[f"scc {scc} GC Net benefit per avoided tCO2e ($/tCO2e)"].append(
                table[f"scc {scc} GC net benefit (in trillion dollars)"][i]
                * 1e12
                / (table["Avoided emissions (GtCO2e)"][i] * 1e9)
            )
            table[f"scc {scc} CC Net benefit per avoided tCO2e ($/tCO2e)"].append(
                table[f"scc {scc} CC net benefit (in trillion dollars)"][i]
                * 1e12
                / (table["Avoided emissions (GtCO2e)"][i] * 1e9)
            )
        table["GDP over time period (in trillion dollars)"].append(
            gdp_2023 * arbitrage_period
        )
    scc_share_percent = 100
    if included_countries is not None:
        scc_share_percent = (
            100
            * sum(country_sccs.get(c, 0) for c in included_countries)
            / country_sccs.sum()
        )

    table["scc_share (%)"] = scc_share_percent
    uid = util.get_unique_id(include_date=False)
    df = pd.DataFrame(table).round(6).T
    df.to_csv(f"plots/table2_{name}_{uid}.csv")
    return df


def calculate_each_countries_with_cache(
    chosen_s2_scenario,
    cache_json_path,
    ignore_cache=False,
    info_name="cost",
    last_year=None,
    scc=None,
):
    # IMPORTANT: the chosen s2 scenario indicates whether the yearly cost for
    # avoiding is discounted or not.
    if last_year is not None:
        global LAST_YEAR
        LAST_YEAR = last_year
    use_cache = not ignore_cache
    if use_cache and os.path.isfile(cache_json_path):
        info_dict = util.read_json(cache_json_path)
    else:
        print("Cached json not found. Calculating from scratch...")
        info_dict = {}
        if scc is not None:
            global social_cost_of_carbon
            social_cost_of_carbon = scc
            util.social_cost_of_carbon = scc
        out = run_table1(to_csv=False, do_round=False, return_yearly=True)
        for key, yearly in out[chosen_s2_scenario].items():
            if key in [
                "residual_benefit",
                "avoided_emissions_including_residual_emissions",
                "country_benefit_country_reduction",
                "global_benefit_country_reduction",
            ]:
                # Is already in the format of dict[str, float]
                # of country, value.
                value = yearly
                info_dict[key] = value
                continue
            # Collect all country names
            country_names = set()
            for e in yearly:
                if isinstance(e, float):
                    continue
                elif isinstance(e, dict):
                    country_names = country_names.union(e.keys())
                elif isinstance(e, pd.Series):
                    country_names = country_names.union(e.index)
                else:
                    print(e)
                    raise Exception("Should not happen")

            each_key_dict = {}
            for country_name in country_names:
                country_level_cost = 0.0
                for e in yearly:
                    if isinstance(e, float):
                        assert math.isclose(e, 0.0)
                    elif isinstance(e, dict):
                        # We use .get instead of [], for battery's case
                        country_level_cost += e.get(country_name, 0.0)
                    else:
                        # pandas series
                        country_level_cost += e.loc[country_name]
                each_key_dict[country_name] = country_level_cost
            info_dict[key] = each_key_dict.copy()
        if use_cache:
            with open(cache_json_path, "w") as f:
                json.dump(info_dict, f)
    return info_dict[info_name]


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
    gdp_per_capita_dict = util.read_json(util.gdp_per_capita_path)
    # Taiwan in 2024
    # https://www.imf.org/external/datamapper/NGDPDPC@WEO/ADVEC/WEOWORLD/TWN/CHN
    # gdp_per_capita_dict["TW"] = 34430
    # Kosovo in 2023
    # https://data.worldbank.org/indicator/NY.GDP.PCAP.CD?locations=XK
    # gdp_per_capita_dict["XK"] = 5943.1

    gdp_marketcap_dict = util.read_json(util.gdp_marketcap_path)
    # Taiwan in 2023
    # https://www.statista.com/statistics/727589/gross-domestic-product-gdp-in-taiwan/
    # gdp_marketcap_dict["TW"] = 756.59 * 1e9  # billion USD
    # Kosovo in 2023
    # https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?locations=XK
    # gdp_marketcap_dict["XK"] = 10.44 * 1e9  # to billion uSD

    worldbank_set = set(gdp_per_capita_dict.keys())
    masterdata_coal_set = set(df_sector.asset_country)
    divide_by_marketcap = True

    print("MODE divide by marketcap", divide_by_marketcap)
    # Data checking
    print("worldbank.org", len(worldbank_set))
    print("masterdata", len(masterdata_coal_set))
    print("intersection", len(worldbank_set.intersection(masterdata_coal_set)))
    print("masterdata - worldbank", masterdata_coal_set - worldbank_set)
    # Only in masterdata: {nan, 'TW', 'XK'}

    # country_shortnames = list(masterdata_coal_set - {np.nan, "TW", "XK"})
    chosen_s2_scenario = f"{NGFS_PEG_YEAR}-2100 FA + Net Zero 2050 Scenario"
    cache_json_path = "plots/climate_financing.json"

    costs_dict = calculate_each_countries_cost_with_cache(
        chosen_s2_scenario, cache_json_path
    )

    developing_shortnames = util.get_developing_countries()
    emerging_shortnames = util.get_emerging_countries()
    raise Exception("developed_gdp uses outdated GDP here")
    developed_gdp = pd.read_csv("data/GDP-Developed-World.csv", thousands=",")
    colname_for_gdp = "2023 GDP (million dollars)"
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
    iso3166_df = util.read_iso3166()
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


def calculate_yearly_info_dict(chosen_s2_scenario, info_name="cost"):
    yearly_costs_dict = {}
    out = run_table1(to_csv=False, do_round=False, return_yearly=True)
    yearly_cost_for_avoiding = out[chosen_s2_scenario][info_name]
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


def do_cf_battery_yearly():
    chosen_s2_scenario = f"{NGFS_PEG_YEAR}-2100 FA + Net Zero 2050 Scenario"

    whole_years = range(NGFS_PEG_YEAR, 2100 + 1)

    def calculate_yearly_world_cost(s2_scenario):
        yearly_costs_dict = calculate_yearly_info_dict(s2_scenario)
        # Calculating the cost for the whole world
        yearly_world_cost = np.zeros(len(whole_years))
        for v in yearly_costs_dict.values():
            yearly_world_cost += np.array(v)
        return yearly_world_cost

    def _get_year_range_cost(year_start, year_end, yearly_world_cost):
        return sum(
            yearly_world_cost[year_start - NGFS_PEG_YEAR : year_end + 1 - NGFS_PEG_YEAR]
        )

    labels = [
        "30Y, D, E",
        "30Y, D, E, S",
        "30Y, D, E, L",
        "30Y, D, E, O",
        "30Y, D, E, S+L",
        "30Y, D, E, S+L+O",
        "30Y, D, no E, S+L+O",
    ]
    import coal_worker

    cw_out = coal_worker.calculate("default", full_version=True)
    retraining_series = (
        cw_out["wage_lost_series"]
        * coal_worker.ic_usa
        / coal_worker.wage_usd_dict["US"]
    )
    # Set NGFS_PEG_YEAR value to 0
    retraining_series = np.insert(retraining_series, 0, 0)
    opportunity_cost_series = cw_out["opportunity_cost_series"]
    opportunity_cost_series = np.insert(opportunity_cost_series, 0, 0)

    # Just for bar chart and original
    data_for_barchart = {
        (NGFS_PEG_YEAR + 1, 2050): {},
        (2051, 2070): {},
        (2071, 2100): {},
    }
    label_map_original = {
        "30Y": "30Y, D, E",
        "30Y_noE": "30Y, D, no E",
        "LCOE": "LCOE proxy",
        "50Y": "50Y, D, E",
        "200Y": "Lifetime by D, E",
    }

    def reset():
        global ENABLE_NEW_METHOD
        ENABLE_NEW_METHOD = 1
        with_learning.ENABLE_WRIGHTS_LAW = 1
        with_learning.RENEWABLE_LIFESPAN = 30
        with_learning.ENABLE_BATTERY_SHORT = False
        with_learning.ENABLE_BATTERY_LONG = False
        with_learning.ENABLE_BATTERY_GRID = False

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(axs[0])
    global ENABLE_NEW_METHOD
    for key, label in label_map_original.items():
        reset()
        if key == "LCOE":
            ENABLE_NEW_METHOD = 0
        elif key.endswith("Y"):
            with_learning.RENEWABLE_LIFESPAN = int(key[:-1])
        else:
            assert key == "30Y_noE"
            with_learning.ENABLE_WRIGHTS_LAW = 0

        yearly = calculate_yearly_world_cost(chosen_s2_scenario, discounted=False)
        linestyle = "-" if label == "30Y, D, E" else "dotted"
        plt.plot(
            whole_years,
            yearly,
            label=label,
            linestyle=linestyle,
            linewidth=2.5,
        )

        yearly_discounted = calculate_yearly_world_cost(chosen_s2_scenario)
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
    reset()
    # End just for bar chart and original

    # Battery only
    plt.sca(axs[1])
    fname = "plots/cf_battery_cache.json"
    if os.path.isfile(fname):
        yearly_all, extra_data_for_barchart = util.read_json(fname)
    else:
        extra_data_for_barchart = {
            f"{NGFS_PEG_YEAR + 1}-2050": {},
            "2051-2070": {},
            "2071-2100": {},
        }
        yearly_all = {}
        for label in labels:
            with_learning.ENABLE_WRIGHTS_LAW = "no E" not in label
            with_learning.ENABLE_BATTERY_SHORT = "S" in label
            with_learning.ENABLE_BATTERY_LONG = "L" in label

            yearly = calculate_yearly_world_cost(chosen_s2_scenario, discounted=False)
            if "O" in label:
                # Division by 1e3 converts to trillion dollars
                yearly += opportunity_cost_series / 1e3
                yearly += retraining_series / 1e3
            yearly_all[label] = list(yearly)

            yearly_discounted = calculate_yearly_world_cost(chosen_s2_scenario)
            for year_start, year_end in [
                (NGFS_PEG_YEAR + 1, 2050),
                (2051, 2070),
                (2071, 2100),
            ]:
                aggregate = _get_year_range_cost(
                    year_start, year_end, yearly_discounted
                )
                extra_data_for_barchart[f"{year_start}-{year_end}"][label] = aggregate
        with open(fname, "w") as f:
            json.dump([yearly_all, extra_data_for_barchart], f)
    for label in labels:
        linestyle = "-" if label == "30Y, D, E" else "dotted"
        if label != "30Y, D, no E, S+L+O":
            plt.plot(
                whole_years,
                yearly_all[label],
                label=label,
                linestyle=linestyle,
                linewidth=2.5,
            )

    plt.xlabel("Time")
    plt.ylabel("Global annual climate financing\n(trillion dollars)")
    plt.legend(
        bbox_to_anchor=(0.5, -0.2),
        loc="upper center",
        ncol=2,
    )
    plt.tight_layout()

    util.savefig("cf_battery_yearly", tight=True)

    # Bar plot
    plt.figure()
    # Merge the barchart data
    for k in data_for_barchart:
        k_str = f"{k[0]}-{k[1]}"
        for _k, _v in extra_data_for_barchart[k_str].items():
            data_for_barchart[k][_k] = _v
    xticks = None
    stacked_bar_data = []
    for year_pair, data in data_for_barchart.items():
        xticks = list(data.keys())
        stacked_bar_data.append((f"{year_pair[0]}-{year_pair[1]}", list(data.values())))
    util.plot_stacked_bar(
        xticks,
        stacked_bar_data,
    )
    # For separating the baseline
    plt.axvline(0.5, color="gray", linestyle="dashed")
    # For separating the battery ones
    plt.axvline((4 + 5) / 2, color="gray", linestyle="dashed")
    plt.xticks(xticks, rotation=90, ha="center")
    plt.ylabel("PV global climate financing\n(trillion dollars)")
    plt.legend(loc="upper center")
    plt.tight_layout()
    util.savefig("cf_battery_pv")


def run_3_level_scc():
    # Run for 3 levels of social cost of carbon.
    global social_cost_of_carbon, LAST_YEAR
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
    print(cao_name)
    for last_year in [2050, 2070, 2100]:
        LAST_YEAR = last_year
        caos = []
        caos_with_residual = []
        condition = f"{NGFS_PEG_YEAR}-{last_year} FA + Net Zero 2050 Scenario"
        # condition = f"{NGFS_PEG_YEAR}-{last_year} FA + Current Policies  Scenario"
        scs = [
            util.social_cost_of_carbon_lower_bound,
            util.social_cost_of_carbon_imf,
            util.social_cost_of_carbon_upper_bound,
        ]
        for sc in scs:
            util.social_cost_of_carbon = sc
            social_cost_of_carbon = sc  # noqa: F811
            out = run_table1(to_csv=False, do_round=True, plot_yearly=False)
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
        print(" & ".join(caos_with_residual))


def get_yearly_by_country():
    # Ensure plot output dir exists
    os.makedirs("plots/bruegel", exist_ok=True)

    global ENABLE_COAL_EXPORT
    for enable in [False, True]:
        ENABLE_COAL_EXPORT = enable
        out = run_table1(to_csv=False, do_round=True, return_yearly=True)
        nz2050 = out[f"{NGFS_PEG_YEAR}-2100 FA + Net Zero 2050 Scenario"]
        series_ics = []
        series_ocs = []
        for i in range(2, 2100 - NGFS_PEG_YEAR + 1):
            # Trillions
            series_ocs.append(
                nz2050["opportunity_cost_non_discounted"][i].rename(NGFS_PEG_YEAR + i)
            )
            series_ics.append(
                pd.Series(
                    nz2050["investment_cost_non_discounted"][i],
                    name=(NGFS_PEG_YEAR + i),
                )
            )
        git_branch = util.get_git_branch()
        a2_to_full_name = util.prepare_alpha2_to_full_name_concise()

        df = pd.concat(series_ocs, axis=1)
        df.index = df.index.to_series().apply(lambda a2: a2_to_full_name[a2])
        suffix = f"{git_branch}_coalexport_{enable}"
        df.to_csv(
            f"plots/bruegel/yearly_by_country_opportunity_cost_NONDISCOUNTED_{suffix}.csv"
        )

        df = pd.concat(series_ics, axis=1)
        df.index = df.index.to_series().apply(lambda a2: a2_to_full_name[a2])
        df.to_csv(
            f"plots/bruegel/yearly_by_country_investment_cost_NONDISCOUNTED_{suffix}.csv"
        )

        yearly_ae = util.read_json(
            "./cache/unilateral_benefit_yearly_avoided_emissions_GtCO2_2100.json"
        )
        if ENABLE_COAL_EXPORT:
            from coal_export.common import modify_avoided_emissions_based_on_coal_export

            yearly_ae = modify_avoided_emissions_based_on_coal_export(yearly_ae)
        df = pd.DataFrame(
            yearly_ae, index=list(range(NGFS_PEG_YEAR, 2100 + 1))
        ).transpose()
        df.index = df.index.to_series().apply(lambda a2: a2_to_full_name[a2])
        df.to_csv(f"plots/bruegel/yearly_by_country_avoided_emissions_{suffix}.csv")


def get_yearly_by_country_power():
    out = run_table1(to_csv=False, do_round=True, return_yearly=True)
    nz2050 = out[f"{NGFS_PEG_YEAR}-{LAST_YEAR} FA + Net Zero 2050 Scenario"]
    series = defaultdict(list)
    ignore = [
        "avoided_emissions_including_residual_emissions",
        "country_benefit_country_reduction",
        "global_benefit_country_reduction",
        "residual_benefit",
    ]
    for i in range(LAST_YEAR - NGFS_PEG_YEAR + 1):
        # Trillions
        for key, value in nz2050.items():
            if key in ignore:
                continue
            if isinstance(value[i], pd.Series):
                element = value[i].rename(NGFS_PEG_YEAR + i)
            else:
                element = pd.Series(
                    value[i],
                    name=(NGFS_PEG_YEAR + i),
                )
            series[key].append(element)
    git_branch = util.get_git_branch()
    for key, value in series.items():
        df = pd.concat(value, axis=1)
        df.to_csv(f"plots/bruegel/yearly_by_country_{key}_{git_branch}.csv")


def make_battery_unit_ic_plot(scenario, countries_included):
    # Ensure plot output dir exists
    os.makedirs("plots/phase_in", exist_ok=True)

    global MEASURE_GLOBAL_VARS
    global MEASURE_GLOBAL_VARS_SCENARIO
    MEASURE_GLOBAL_VARS_SCENARIO = scenario
    MEASURE_GLOBAL_VARS = True
    with_learning.VERBOSE_ANALYSIS = True
    util.CARBON_BUDGET_CONSISTENT = "15-50"
    years = range(2024, 2050 + 1)

    def per_GJ2per_kWh(arr):
        # Convert from $/GJ to $/kWh
        return [i / (util.GJ2MWh(1) * 1e3) for i in arr]

    def kW2GW(arr):
        return [i / 1e6 for i in arr]

    def GJ2TW(arr):
        return [util.GJ2MW(i) / 1e6 for i in arr]

    name_labels = {
        "solar": "Solar",
        "onshore_wind": "Wind onshore",
        "offshore_wind": "Wind Offshore",
        "geothermal": "Geothermal",
        "hydropower": "Hydropower",
    }

    a2_to_full_name = util.prepare_alpha2_to_full_name_concise()
    # for country in "WORLD EMDE IN ID DE US TR VN PL KZ".split():
    for country in countries_included:
        with_learning.VERBOSE_ANALYSIS_COUNTRY = country
        title = (
            a2_to_full_name[country]
            if country not in ["WORLD", "EMDE", "Developed_UNFCCC", "Developing_UNFCCC"]
            else country
        )
        print(title)
        try:
            run_table1(to_csv=False, do_round=False, plot_yearly=False)
        except Exception as e:
            print("FA doesn't have this country's data:", e)
            continue
        fig, axs = plt.subplots(1, 2, figsize=(8, 5))

        plt.sca(axs[0])
        for name, label in name_labels.items():
            plt.plot(
                years,
                global_cost_with_learning.cached_investment_costs[name].values(),
                label=label,
            )
        # Need to convert $/GJ to $/kWh, then to $/kW, by multiplying by number of seconds in 1 day
        # This is because the short term battery is fully discharged by each day.
        plt.plot(
            years,
            np.array(
                per_GJ2per_kWh(
                    global_cost_with_learning.battery_unit_ic["short"].values()
                )
            )
            * 24
            * 3600,
            label="Short",
        )
        plt.plot(
            years,
            global_cost_with_learning.battery_unit_ic["long"].values(),
            label="Battery long",
        )
        plt.xlabel("Time")
        plt.ylabel("Unit investment cost ($/kW)")

        plt.sca(axs[1])
        # Define a custom color and marker cycle
        colors = plt.cm.tab10.colors[:9]
        markers = ["^", "v", ">", "<", "o", "s", "D", "P", "*"]
        markersize = 2
        axs[1].set_prop_cycle(cycler(color=colors) + cycler(marker=markers))
        country_name = with_learning.VERBOSE_ANALYSIS_COUNTRY
        plt.title(title)
        for tech, label in {
            **name_labels,
            "short": "Battery short",
            "long": "Battery long",
        }.items():
            y = np.cumsum(
                kW2GW(
                    list(
                        global_cost_with_learning.cached_stock_without_degradation[
                            tech
                        ].values()
                    )
                )
            )
            plt.plot(
                years,
                y,
                label=label,
                markersize=markersize,
                linewidth=0.8,
            )

        # Reset color cycler
        axs[1].set_prop_cycle(cycler(color=colors) + cycler(marker=markers))
        energy_produced_1country = [
            {
                tech: e[country_name][tech] if country_name in e else 0
                for tech in with_learning.TECHS
            }
            for e in global_cost_with_learning.green_energy_produced_by_country
        ]
        for tech, label in name_labels.items():
            plt.plot(
                years,
                util.GJ2MW(np.array([e[tech] for e in energy_produced_1country])) / 1e3,
                linestyle="dotted",
                label=tech,
                markersize=markersize,
            )

        plt.xlabel("Time")
        plt.ylabel("Capacity (GW)")

        # Deduplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncol=5,
        )
        plt.tight_layout()
        plt.savefig(
            f"plots/phase_in/battery_unit_ic_{MEASURE_GLOBAL_VARS_SCENARIO}_{country}.png",
            bbox_inches="tight",
        )
        plt.close()

        # 2nd file
        fig = plt.figure()
        plt.title(title)
        for tech, label in {
            **name_labels,
            "short": "Battery short",
            "long": "Battery long",
        }.items():
            y = kW2GW(
                list(
                    global_cost_with_learning.cached_stock_without_degradation[
                        tech
                    ].values()
                )
            )
            plt.plot(
                years,
                y,
                label=label,
                markersize=markersize,
                linewidth=0.8,
            )
        plt.xlabel("Time")
        plt.ylabel("Annual installed capacity (GW)")
        # Deduplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncol=5,
        )
        plt.tight_layout()
        plt.savefig(
            f"plots/phase_in/battery_yearly_installed_capacity_{MEASURE_GLOBAL_VARS_SCENARIO}_{country}.png",
            bbox_inches="tight",
        )
        plt.close()
        util.write_small_json(
            dict(global_cost_with_learning.cached_stock_without_degradation),
            f"plots/phase_in/battery_yearly_installed_capacity_{MEASURE_GLOBAL_VARS_SCENARIO}_{country}.json",
        )
        util.write_small_json(
            dict(global_cost_with_learning.cached_stock),
            f"plots/phase_in/battery_yearly_available_capacity_{MEASURE_GLOBAL_VARS_SCENARIO}_{country}.json",
        )

        # 3rd file
        fig = plt.figure()
        plt.title(title)
        y = sum(
            np.array(list(d.values()))
            for d in global_cost_with_learning.cached_stock_without_degradation.values()
        )
        y_cumsum = np.cumsum(y)

        plt.plot(years, kW2GW(y), label="Annual")
        plt.plot(years, kW2GW(y_cumsum), label="Cumulative")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Annual installed capacity (GW)")
        plt.savefig(
            f"plots/phase_in/battery_yearly_installed_capacity_{MEASURE_GLOBAL_VARS_SCENARIO}_{country}_summed.png"
        )
        plt.close()

    MEASURE_GLOBAL_VARS = False
    MEASURE_GLOBAL_VARS_SCENARIO = "Net Zero 2050"
    with_learning.VERBOSE_ANALYSIS = False
    util.CARBON_BUDGET_CONSISTENT = False


if __name__ == "__main__":
    if 0:
        get_yearly_by_country_power()
        # get_yearly_by_country()
        exit()

    if 0:
        run_table2()
        exit()

    if 0:
        sccs = [
            util.social_cost_of_carbon_imf,
            util.scc_biden_administration,
            util.scc_bilal,
        ]
        for scc in sccs:
            util.social_cost_of_carbon = scc
            social_cost_of_carbon = scc  # noqa: F811
            out = run_table1(to_csv=True, do_round=True, plot_yearly=False)
        exit()
    if 1:
        # Battery yearly
        # do_cf_battery_yearly()
        # make_battery_plot()
        countries = with_learning.DEVELOPING_UNFCCC
        countries = (
            "WORLD Developed_UNFCCC Developing_UNFCCC EG IN ID ZA MX VN IR TH".split()
        )
        countries = ["CA"]
        # countries = sorted(list(set(df_sector.asset_country.tolist())))
        make_battery_unit_ic_plot("Net Zero 2050", countries)
        # Halt to coal production
        # make_battery_unit_ic_plot("Current Policies", countries)
        exit()
    if 0:
        run_3_level_scc()
        exit()
