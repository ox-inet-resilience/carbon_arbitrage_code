import copy
import functools
import math

import numpy as np
import pandas as pd

import util
from util import world_gdp_2023
from gca.lib.array_ops import (
    maybe_round6,
    sum_array_of_mixed_objs,
    divide_array_of_mixed_objs,
    add_array_of_mixed_objs,
)
import gca.parameters as parameters

import with_learning


# This is used in Bruegel analysis. Might be deleted later.
INVESTMENT_COST_DIVIDER = 1

ngfs_df = util.read_ngfs()
_, df_sector = util.read_forward_analytics_data(parameters.SECTOR_INCLUDED)
unit_profit_df = pd.read_csv(
    "data_private/v5_final_country_region_profit_analysis.csv.zip",
    compression="zip",
)
country_sccs = pd.Series(util.read_country_specific_scc_filtered())
iso3166_df = util.read_iso3166()
alpha2_to_alpha3 = iso3166_df.set_index("alpha-2")["alpha-3"].to_dict()


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
            dp, parameters.NGFS_PEG_YEAR + i, discount
        )
    if with_learning.VERBOSE_ANALYSIS:
        # This is to calculate available capacity beyond parameters.LAST_YEAR
        for year in range(
            parameters.LAST_YEAR + 1, parameters.LAST_YEAR + 1 + with_learning.RENEWABLE_LIFESPAN
        ):
            temp_cost_with_learning.initialize_verbose_analysis(
                {c: 0 for c in DeltaP[0].keys()}, year
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
    if parameters.MEASURE_GLOBAL_VARS and scenario == parameters.MEASURE_GLOBAL_VARS_SCENARIO:
        parameters.global_cost_with_learning = temp_cost_with_learning
    return (
        out_non_discounted,
        out_discounted,
        residual_emissions,
        residual_production,
        temp_cost_with_learning,
    )


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
    if parameters.ENABLE_WORKER:
        import coal_worker

        worker_by_subsector = {
            subsector: coal_worker.calculate(
                "default",
                subsector,
                parameters.LAST_YEAR,
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
            1e9 / 1e12 * util.social_cost_of_carbon
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
            * util.social_cost_of_carbon
            / 1e3
        )
        .dropna()
        .to_dict()
    )
    out_yearly_info["global_benefit_country_reduction"] = (
        out_yearly_info["avoided_emissions_including_residual_emissions"]
        * util.social_cost_of_carbon
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
    residual_benefit = residual_emissions * util.social_cost_of_carbon / 1e3

    # Convert to Gigatonnes of coal
    residual_production = residual_production_series.sum() / 1e9

    # rho is the same everywhere
    rho = util.calculate_rho(util.beta)

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
        k: v * util.social_cost_of_carbon / 1e12
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
    arbitrage_period = last_year - parameters.NGFS_PEG_YEAR

    owner_dict = {
        # nansum is needed because PG has nan value for gas
        f"OC owner {subsector}": np.nansum(
            array_of_cost_discounted_owner_by_subsector[subsector]
        )
        / 1e12
        for subsector in subsectors
    }
    worker_dict = {}
    if parameters.ENABLE_WORKER:
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


def _prepare_table1_base_data(_df_nonpower):
    """Prepare base production and emissions data for table1 analysis."""
    total_production_fa = util.get_production_by_country(df_sector, parameters.SECTOR_INCLUDED)
    emissions_fa = util.get_emissions_by_country(df_sector)

    production_2019 = (
        _df_nonpower.groupby("asset_country")._2019.sum()
        if parameters.ENABLE_COAL_EXPORT
        else None
    )

    weighted_emissions_factor_by_country_peg_year = (
        calculate_weighted_emissions_factor_by_country_peg_year(_df_nonpower)
    )

    return (
        total_production_fa,
        emissions_fa,
        production_2019,
        weighted_emissions_factor_by_country_peg_year,
    )


def _process_table1_scenario(
    scenario,
    total_production_fa,
    emissions_fa,
    production_with_ngfs_projection_CPS,
    profit_ngfs_projection_CPS,
):
    """Process a single scenario for table1 analysis."""
    last_year = parameters.LAST_YEAR

    # Calculate NGFS projections
    (
        production_with_ngfs_projection,
        gigatonnes_coal_production,
        profit_ngfs_projection,
    ) = util.calculate_ngfs_projection(
        "production",
        total_production_fa,
        ngfs_df,
        parameters.SECTOR_INCLUDED,
        scenario,
        parameters.NGFS_PEG_YEAR,
        last_year,
        alpha2_to_alpha3,
        unit_profit_df=unit_profit_df,
    )

    emissions_with_ngfs_projection, _, _ = util.calculate_ngfs_projection(
        "emissions",
        emissions_fa,
        ngfs_df,
        parameters.SECTOR_INCLUDED,
        scenario,
        parameters.NGFS_PEG_YEAR,
        last_year,
        alpha2_to_alpha3,
    )

    # Store Current Policies data for Net Zero 2050 comparison
    if scenario == "Current Policies":
        production_with_ngfs_projection_CPS = production_with_ngfs_projection.copy()
        profit_ngfs_projection_CPS = profit_ngfs_projection.copy()

    scenario_formatted = f"FA + {scenario} Scenario"
    array_of_total_emissions_non_discounted = emissions_with_ngfs_projection

    # Calculate delta production and profit
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

    scenario_results = (
        gigatonnes_coal_production,
        array_of_total_emissions_non_discounted,
        DeltaP,
        scenario_formatted,
        delta_profit,
        production_with_ngfs_projection,
    )

    return (
        scenario_results,
        production_with_ngfs_projection_CPS,
        profit_ngfs_projection_CPS,
    )


def _apply_table1_country_filter(
    included_countries,
    production_with_ngfs_projection,
    gigatonnes_coal_production,
    array_of_total_emissions_non_discounted,
    cost_non_discounted_owner,
    cost_discounted_owner,
    cost_non_discounted_investment,
    cost_discounted_investment,
    residual_emissions,
    residual_production,
    final_cost_with_learning,
):
    """Apply country filtering to table1 data."""

    def _filter(e, is_multiindex=False):
        if isinstance(e, dict):
            e = pd.Series(e)
        if is_multiindex:
            return e[e.index.get_level_values(0).isin(included_countries)]
        return e[e.index.isin(included_countries)]

    def _filter_arr(arr, is_multiindex=False):
        return [_filter(e, is_multiindex) for e in arr]

    # Filter production data
    gigatonnes_coal_production = _filter(sum(production_with_ngfs_projection)).sum()

    # Filter emissions data
    array_of_total_emissions_non_discounted = _filter_arr(
        array_of_total_emissions_non_discounted, is_multiindex=True
    )

    # Filter cost data
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

    # Filter residual data
    residual_emissions = _filter(residual_emissions)
    residual_production = _filter(residual_production)

    # Filter battery cost data
    final_cost_with_learning.cost_non_discounted_battery_short_by_country = _filter_arr(
        final_cost_with_learning.cost_non_discounted_battery_short_by_country
    )
    final_cost_with_learning.cost_non_discounted_battery_long_by_country = _filter_arr(
        final_cost_with_learning.cost_non_discounted_battery_long_by_country
    )
    final_cost_with_learning.cost_non_discounted_battery_pe_by_country = _filter_arr(
        final_cost_with_learning.cost_non_discounted_battery_pe_by_country
    )
    final_cost_with_learning.cost_non_discounted_battery_grid_by_country = _filter_arr(
        final_cost_with_learning.cost_non_discounted_battery_grid_by_country
    )

    return (
        gigatonnes_coal_production,
        array_of_total_emissions_non_discounted,
        cost_non_discounted_owner,
        cost_discounted_owner,
        cost_non_discounted_investment,
        cost_discounted_investment,
        residual_emissions,
        residual_production,
        final_cost_with_learning,
    )


def generate_table1_output(
    rho,
    do_round,
    _df_nonpower,
    included_countries=None,
):
    """Generate table 1 output for carbon arbitrage analysis.

    This function calculates the costs and benefits of carbon arbitrage under
    two scenarios: Current Policies and Net Zero 2050. It processes production,
    emissions, and cost data for both scenarios and returns comprehensive results.

    Args:
        rho: Discount rate for present value calculations
        do_round: Whether to round numerical results
        _df_nonpower: Non-power sector dataframe with asset data
        included_countries: Optional list of country codes to include in analysis

    Returns:
        Tuple of (results_dataframe, yearly_info_dict):
            - results_dataframe: Summary results by scenario
            - yearly_info_dict: Detailed yearly breakdown by scenario
    """
    out = {}
    out_yearly = {}

    # Prepare base production and emissions data
    (
        total_production_fa,
        emissions_fa,
        production_2019,
        weighted_emissions_factor_by_country_peg_year,
    ) = _prepare_table1_base_data(_df_nonpower)

    # Initialize scenario tracking variables
    current_policies = None
    production_with_ngfs_projection_CPS = None
    profit_ngfs_projection_CPS = None

    # Process both scenarios: Current Policies provides baseline, Net Zero 2050 shows intervention
    for scenario in ["Current Policies", "Net Zero 2050"]:
        (
            scenario_results,
            production_with_ngfs_projection_CPS,
            profit_ngfs_projection_CPS,
        ) = _process_table1_scenario(
            scenario,
            total_production_fa,
            emissions_fa,
            production_with_ngfs_projection_CPS,
            profit_ngfs_projection_CPS,
        )

        (
            gigatonnes_coal_production,
            array_of_total_emissions_non_discounted,
            DeltaP,
            scenario_formatted,
            delta_profit,
            production_with_ngfs_projection,
        ) = scenario_results

        # Calculate opportunity costs
        (
            cost_non_discounted_owner,
            cost_discounted_owner,
        ) = get_opportunity_cost_owner(
            rho,
            delta_profit,
            scenario,
            parameters.LAST_YEAR,
        )

        # Calculate investment costs
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
            parameters.LAST_YEAR,
            with_learning.InvestmentCostWithLearning(),
        )

        if parameters.ENABLE_COAL_EXPORT:
            from coal_export.common import modify_array_based_on_coal_export

            modify_array_based_on_coal_export(
                cost_non_discounted_investment, production_2019
            )
            modify_array_based_on_coal_export(
                cost_discounted_investment, production_2019
            )

        if included_countries is not None:
            (
                gigatonnes_coal_production,
                array_of_total_emissions_non_discounted,
                cost_non_discounted_owner,
                cost_discounted_owner,
                cost_non_discounted_investment,
                cost_discounted_investment,
                residual_emissions,
                residual_production,
                final_cost_with_learning,
            ) = _apply_table1_country_filter(
                included_countries,
                production_with_ngfs_projection,
                gigatonnes_coal_production,
                array_of_total_emissions_non_discounted,
                cost_non_discounted_owner,
                cost_discounted_owner,
                cost_non_discounted_investment,
                cost_discounted_investment,
                residual_emissions,
                residual_production,
                final_cost_with_learning,
            )

        # Generate scenario identifier and calculate table info
        text = f"{parameters.NGFS_PEG_YEAR}-{parameters.LAST_YEAR} {scenario_formatted}"
        table1_info, yearly_info = calculate_table1_info(
            do_round,
            scenario,
            scenario_formatted,
            f"{parameters.NGFS_PEG_YEAR}-{parameters.LAST_YEAR}",
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

        # Store results
        out[text] = table1_info
        out_yearly[text] = yearly_info

        # Store Current Policies baseline for Net Zero 2050 comparison
        if scenario == "Current Policies":
            current_policies = {
                "emissions_non_discounted": copy.deepcopy(
                    array_of_total_emissions_non_discounted
                ),
                "total_production": gigatonnes_coal_production,
            }

    # Convert results to DataFrame and return
    out = pd.DataFrame(out)
    return out, out_yearly


def run_table1(
    to_csv=False,
    do_round=False,
    return_yearly=False,
    included_countries=None,
):
    # rho is the same everywhere
    rho = util.calculate_rho(util.beta)

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
        fname = f"plots/table1_{uid}{ext}_{util.social_cost_of_carbon}_{parameters.LAST_YEAR}.csv"
        pd.DataFrame(both_dict).T.to_csv(fname)

    if return_yearly:
        return yearly

    return both_dict
