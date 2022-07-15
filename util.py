import csv
import json
import subprocess
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import seaborn as sns

# We set our own color pallete
sns.set_palette("muted")

# Commented out because we prefer the default font for now.
# Preparation for the font
from matplotlib import font_manager

font_path = os.path.dirname(__file__) + "/data_private/Arial.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = prop.get_name()
plt.rc("font", size=13)
plt.rc("legend", fontsize=11, title_fontsize=11)
# End font

# Possible models are:
# {'REMIND-MAgPIE 2.1-4.2', 'REMIND-MAgPIE 2.1-4.2 IntegratedPhysicalDamages
# (95th)', 'GCAM5.3_NGFS',
# 'REMIND-MAgPIE 2.1-4.2 IntegratedPhysicalDamages (median)',
# 'MESSAGEix-GLOBIOM 1.1'}
NGFS_MODEL = "GCAM5.3_NGFS"
# NGFS_MODEL = "MESSAGEix-GLOBIOM 1.1"
# NGFS_MODEL = "REMIND-MAgPIE 2.1-4.2"
# CPS stands for current policies scenario
NGFS_MODEL_FOR_CPS = "GCAM5.3_NGFS"

hours_in_1year = 24 * 365.25
seconds_in_1hour = 3600  # seconds
# The years in NGFS data
ngfs_years = list(range(2005, 2105, 5))
# The years in masterdata.
years_masterdata = range(2022, 2027)

# Constants
# We obtain 114.9 by averaging 61.4 and 168.4.
# For the source of the 2 numbers, see:
# https://mitsloan.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=f56f202e-adc4-48c6-8b1d-adc8013d54d9
# and https://resilience.zulipchat.com/#narrow/stream/285747-TRISK/topic/to.20do/near/265246287
# The unit is dollars per tCO2
social_cost_of_carbon_lower_bound = 61.4  # min
social_cost_of_carbon_upper_bound = 168.4  # max
social_cost_of_carbon_mid = 114.9
social_cost_of_carbon_imf = 75
# social_cost_of_carbon = social_cost_of_carbon_lower_bound
social_cost_of_carbon = social_cost_of_carbon_imf
# social_cost_of_carbon = social_cost_of_carbon_mid
# social_cost_of_carbon = social_cost_of_carbon_upper_bound
world_gdp_2020 = 84.705  # trillion dolars


def read_json(filename):
    with open(filename) as f:
        obj = json.load(f)
    return obj


def get_unique_id(include_date=True):
    git_rev_hash = (
        subprocess.check_output("git rev-parse HEAD".split()).decode("utf-8").strip()
    )
    # if include_date:
    #     now = datetime.datetime.now(pytz.utc).strftime("%Y-%m-%d-%H-%M-%S")
    #     return f"{git_rev_hash}_{now}"
    return git_rev_hash


def get_git_branch():
    branches = subprocess.check_output(["git", "branch"]).decode().splitlines()
    for branch in branches:
        if "*" in branch:
            return branch[2:]
    return None


def run_parallel(_func, arr, args):
    # Execute the function `_func` for each element in the array `arr` in
    # parallel (in different processes).
    procs = []
    for x in arr:
        proc = mp.Process(target=_func, args=(x,) + args)
        proc.start()
        procs.append(proc)
    for p in procs:
        p.join()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def run_parallel_ncpus(numcpu, _func, arr, args):
    """run_parallel but ensure only numcpu processes are running at a time."""
    clusters = list(chunks(arr, numcpu))
    for cluster in clusters:
        run_parallel(_func, cluster, args)


def savefig(sname, tight=False, svg=False):
    ext = "svg" if svg else "png"
    plotname = f"{sname}_{get_unique_id()}.{ext}"
    print(plotname)
    fpath = "plots/" + plotname
    if tight:
        plt.savefig(fpath, bbox_inches="tight")
    else:
        plt.savefig(fpath)


def fill_nan_with_mean(_df, _col):
    # Replace the nan values of a column with the mean of the non-nan values.
    mean = _df[_col].mean()
    _df[_col] = _df[_col].fillna(mean)


def get_in_between_year(year):
    # E.g. if year is 2023, return (2020, 2025).
    rounded = 5 * round(year / 5)
    if rounded > year:
        return rounded - 5, rounded
    return rounded, rounded + 5


def get_country_to_region():
    with open("data/2DII-country-name-shortcuts.csv") as f:
        country_to_region = {}
        iso2_to_country_name = {}
        for row in csv.DictReader(f):
            iso2 = row["iso2"]
            region = row["region_name"]
            country_to_region[iso2] = region if len(region) > 0 else "Other"
            iso2_to_country_name[iso2] = row["country_name"].replace(
                "United States of America", "USA"
            )
    return country_to_region, iso2_to_country_name


def get_lcoe_info(lcoe_mode):
    if lcoe_mode == "solar+wind":
        # Multiplication by 1e3 converts from $/kW to $/MW
        # No need to do extra operation to go from $/MW to $/MWh
        global_lcoe_solar = 0.057 * 1e3
        global_lcoe_offshore_wind = 0.084 * 1e3
        global_lcoe_onshore_wind = 0.039 * 1e3
        global_lcoe_wind = (global_lcoe_offshore_wind + global_lcoe_onshore_wind) / 2
        global_lcoe_average = (global_lcoe_solar + global_lcoe_wind) / 2
        # TODO: we can refactor so that lcoe_dict is not needed, so that we
        # only use scalar instead of costly extra DF column.
    else:
        assert lcoe_mode == "solar+wind+gas"
        global_lcoe_average = None
        raise Exception("Not yet implemented")
    return global_lcoe_average


def read_iea():
    iea = pd.read_csv("data_private/IEA-Scenarios.csv")
    # Constrain further
    iea = iea[(iea.scenario_geography == "Global") & (iea.technology == "CoalCap")]
    iea_sds = iea[iea.scenario == "SDS"]
    iea_sps = iea[iea.scenario == "SPS"]
    return iea_sds, iea_sps


def set_integer_xaxis():
    ax = plt.gca()
    ax.get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))


def coal2GJ(x):
    # tonnes of coal to GJ
    # See https://en.wikipedia.org/wiki/Ton#Tonne_of_coal_equivalent
    # 1 tce is 29.3076 GJ
    # 1 tce is 8.141 MWh
    return x * 29.3076


def GJ2coal(x):
    return x / 29.3076


def MW2GJ(x):
    # MW to GW
    gw = x / 1e3
    # GW to GJ
    return gw * hours_in_1year * seconds_in_1hour


def GJ2MW(x):
    # GJ to MJ
    mj = x * 1e3
    # MJ to MW
    return mj / (hours_in_1year * seconds_in_1hour)


def GJ2MWh(x):
    # GJ to J
    joule = x * 1e9
    # J to Wh
    wh = joule / 3600
    # Wh to MWh
    return wh / 1e6


def MWh2GJ(x):
    return x * 3.6


def MW2Gigatonnes_of_coal(x):
    gj = MW2GJ(x)
    # In tonnes of coal
    tc = GJ2coal(gj)
    # In Giga tonnes of coal
    return tc / 1e9


def maybe_load_masterdata(pre_existing_df=None, use_pams=False):
    if use_pams:
        print("Reading from PAMS")
        filename = "pams_total.csv.gz"
        encoding = None
    else:
        print("Reading from Masterdata")
        filename = "masterdata_ownership_PROCESSED_capacity_factor.csv.gz"
        encoding = "latin1"
    if pre_existing_df is None:
        df = pd.read_csv(
            f"data_private/{filename}",
            compression="gzip",
            encoding=encoding,
            dtype={
                "company_name": "string[pyarrow]",
                "company_id": int,
                "sector": "string[pyarrow]",
                "technology": "string[pyarrow]",
                "technology_type": "string[pyarrow]",
                "asset_country": "string[pyarrow]",
                "emissions_factor_unit": "string[pyarrow]",
                "unit": "string[pyarrow]",
            },
        )
    else:
        df = pre_existing_df
    return df


def replace_countries(_df):
    # Merge Guadeloupe, Martinique, and New Caledonia into France.
    gmn = ["GP", "MQ", "NC"]
    _df.loc[_df.asset_country.isin(gmn), "asset_country"] = "FR"
    # PAMS data only:
    # Merge Bermuda, Cayman Islands, Virgin Islands (British) into GB
    bcv = ["BM", "KY", "VG"]
    _df.loc[_df.asset_country.isin(bcv), "asset_country"] = "GB"
    # PAMS data only:
    # Merge Virgin Islands (US) into US
    vi = ["VI"]
    _df.loc[_df.asset_country.isin(vi), "asset_country"] = "US"


def read_masterdata(pre_existing_df=None, use_pams=False):
    df = maybe_load_masterdata(pre_existing_df, use_pams)
    replace_countries(df)
    # Remove rows without proper asset_country.
    # df = df[~pd.isna(df.asset_country)]
    # All sectors are:
    # {'Coal', 'Oil&Gas', 'Aviation', 'Shipping', 'HDV', 'Steel', 'Power',
    # 'Automotive', 'Cement'}
    nonpower_coal = df[df.sector == "Coal"].copy()
    power_companies = df[df.sector == "Power"]
    power_coal = power_companies[power_companies.technology == "CoalCap"].copy()
    return df, nonpower_coal, power_coal


def calculate_weighted_emissions_factor_gas(df):
    # We weigh using the production in 2020 only

    # Uncomment for power sector only
    # - emissions factor unit is tonnes of CO2 per MWh per year
    # - production unit is MW
    power_companies = df[df.sector == "Power"]
    power_gas = power_companies[power_companies.technology == "GasCap"]
    # weighted_emissions_factor = (
    #     power_gas._2020 * power_gas.emissions_factor
    # ).sum() / power_gas._2020.sum()
    # print("Power emissions factor in tCO2 per MWh per year", weighted_emissions_factor)
    # # Convert tCO2/MWh tCO2/GJ (enable this only for power)
    # weighted_emissions_factor /= MWh2GJ(1)
    # # Convert tCO2/GJ to tCO2/tce
    # weighted_emissions_factor /= GJ2coal(1)

    # We use nonpower sector only
    # - emissions factor unit is tonnes of CO2 per GJ
    # - production unit is GJ
    nonpower_gas = df[df.sector == "Oil&Gas"]
    nonpower_gas = nonpower_gas[nonpower_gas.technology == "Gas"]
    # weighted_emissions_factor = (
    #     nonpower_gas._2020 * nonpower_gas.emissions_factor
    # ).sum() / nonpower_gas._2020.sum()
    # print("Nonpower emissions factor in tCO2 per GJ", weighted_emissions_factor)
    # # Convert tCO2/GJ to tCO2/tce
    # weighted_emissions_factor /= GJ2coal(1)

    # Both
    emissions_both = (
        power_gas._2020 * power_gas.emissions_factor
    ).sum() * hours_in_1year + (
        nonpower_gas._2020 * nonpower_gas.emissions_factor
    ).sum()
    # In tCO2/tce
    weighted_emissions_factor = emissions_both / (
        MW2Gigatonnes_of_coal(power_gas._2020.sum()) * 1e9
        + GJ2coal(nonpower_gas._2020.sum())
    )
    # print("Combined in tCO2/GJ", weighted_emissions_factor / coal2GJ(1))

    return weighted_emissions_factor


# Calculated using set(ngfs_global.Scenario)
scenarios = [
    "Current Policies ",
    "Delayed transition",
    "Below 2Â°C",
    "Net Zero 2050",
    "Nationally Determined Contributions (NDCs) ",
    "Divergent Net Zero",
]


def read_ngfs_coal_and_power():
    return {
        "Coal": pd.read_csv("data/ngfs_scenario_production_fossil.csv"),
        "Power": pd.read_csv(
            "data/NGFS-Power-Sector-Scenarios.csv.gz", compression="gzip"
        ),
    }


def calculate_ngfs_fractional_increase(ngfss, sector, scenario, start_year):
    if sector == "Coal":
        variable = "Primary Energy|Coal"
    else:
        variable = "Capacity|Electricity|Coal"
    ngfs = ngfss[sector]
    ngfs_global = ngfs[ngfs.Region == "World"]
    ngfs_global_coal = ngfs_global[ngfs_global.Variable == variable]
    ngfs_model = NGFS_MODEL_FOR_CPS if scenario == "Current Policies " else NGFS_MODEL
    ngfs_global_coal = ngfs_global_coal[ngfs_global_coal.Model == ngfs_model]
    ngfs_global_coal_scenario = ngfs_global_coal[
        ngfs_global_coal.Scenario == scenario
    ].iloc[0]
    years_interpolated = list(range(2005, 2101))
    ngfs_production_across_years = [
        ngfs_global_coal_scenario[str(year)] for year in ngfs_years
    ]
    f = interp1d(ngfs_years, ngfs_production_across_years)
    ngfs_production_across_years_interpolated = f(years_interpolated)
    ngfs_interpolated_dict = {
        years_interpolated[i]: ngfs_production_across_years_interpolated[i]
        for i in range(len(years_interpolated))
    }
    fraction_increase_after_start_year = {
        year: (ngfs_interpolated_dict[year] / ngfs_interpolated_dict[start_year])
        for year in range(start_year + 1, 2101)
    }
    return fraction_increase_after_start_year


def calculate_rho(beta, rho_mode="default"):
    # See the carbon arbitrage paper page 11, in the paragraph that starts with
    # "We discount expected free cash flows of ...".
    # This function is used on masterdata
    rho_f = 0.0208
    # How the CARP is calculated can be obtained in
    # https://resilience.zulipchat.com/#narrow/stream/285747-TRISK/topic/to.20do/near/271060780.
    # Updated in
    # https://resilience.zulipchat.com/#narrow/stream/285747-TRISK/topic/shiller/near/271375355.
    if rho_mode == "default":
        carp = 0.0299
    elif rho_mode == "100year":
        # To get this number, run misc/shiller.py
        carp = 0.04867810945273632
    elif rho_mode == "5%":
        return 0.05
    elif rho_mode == "8%":
        return 0.08
    elif rho_mode == "0%":
        return 0.0
    else:
        raise Exception("Unexpected rho_mode")
    # Always subtract 1%
    carp -= 0.01
    # This is lambda in the paper
    average_leverage_weighted_with_equity = 0.5175273490449868
    _lambda = average_leverage_weighted_with_equity
    # This is chi in the paper
    tax_rate = 0.15

    # This is equation 7 in the paper.
    rho = _lambda * rho_f * (1 - tax_rate) + (1 - _lambda) * (rho_f + beta * carp)
    # Truncate rho to be >= 0.
    # Future payoff are <= current payoff.
    rho = max(rho, 0)
    return rho


def calculate_discount(rho, deltat):
    # This is equation 5 in the paper.
    return (1 + rho) ** -deltat


def get_coal_nonpower_global_emissions_across_years(
    nonpower_coal, years, discounted=False, rho=None
):
    emissions_list = []
    if discounted:
        assert years[0] == 2022
        assert rho is not None
    for year in years:
        tonnes_coal = nonpower_coal[f"_{year}"]
        discount = 1
        if discounted:
            discount = calculate_discount(rho, year - 2022)
        # This is equation 3 in the carbon arbitrage paper.
        # emissions_factor unit is tonnes of CO2 per tonnes of coal.
        # The division by 1e9 converts to GtCO2.
        emissions = (
            tonnes_coal * nonpower_coal.emissions_factor * discount
        ).sum() / 1e9
        emissions_list.append(emissions)
    return emissions_list


def get_coal_nonpower_per_company_NON_discounted_emissions_summed_over_years(
    ngfs_peg_year,
    nonpower_coal,
    masterdata_years,
    fraction_increase_after_peg_year,
    x=1,
    multiply_by_country_tax_increase=False,
    summation_start_year=None,
):
    emissions_list = []
    assert ngfs_peg_year == 2026
    assert masterdata_years[0] == 2022
    assert summation_start_year is not None
    grouped = nonpower_coal.groupby("company_id")
    for year in masterdata_years:
        if year < summation_start_year:
            continue

        def process(g):
            if multiply_by_country_tax_increase:
                # The unit no longer make sense if we
                # multiply by g.country_tax_increase. But we
                # add a conditional branch here to minimize
                # code duplication.
                tonnes_coal = g[f"_{year}"] * g.country_tax_increase
            else:
                tonnes_coal = g[f"_{year}"]
            # emissions_factor unit tonnes of CO2 per tonnes of coal
            emissions_of_g = (tonnes_coal * g.emissions_factor).sum()  # in tCO2
            return emissions_of_g

        emissions = grouped.apply(process)
        emissions_list.append(emissions)
    # From NGFS

    def process(g):
        # This function is the same as the previous process() except for
        # tonnes_coal.
        if multiply_by_country_tax_increase:
            tonnes_coal = g[f"_{ngfs_peg_year}"] * g.country_tax_increase
        else:
            tonnes_coal = g[f"_{ngfs_peg_year}"]
        # emissions_factor unit tonnes of CO2 per tonnes of coal
        emissions_of_g = (tonnes_coal * g.emissions_factor).sum()  # in tCO2
        return emissions_of_g

    emissions_peg_year = grouped.apply(process)

    sum_frac_increase = sum(fraction_increase_after_peg_year.values())
    emissions_list.append(emissions_peg_year * sum_frac_increase)
    return sum(emissions_list)


def get_coal_nonpower_global_generation_across_years(nonpower_coal, years):
    production_list = []
    for year in years:
        tonnes_coal = nonpower_coal[f"_{year}"]
        production = tonnes_coal.sum() / 1e9  # convert to giga tonnes of coal
        production_list.append(production)
    return production_list


def get_coal_nonpower_per_company_discounted_PROFIT_summed_over_years(
    ngfs_peg_year,
    nonpower_coal,
    masterdata_years,
    fraction_increase_after_peg_year,
    beta,
    x=1,
    summation_start_year=None,
):
    profits_list = []
    assert ngfs_peg_year == 2026
    assert masterdata_years[0] == 2022
    assert summation_start_year is not None
    assert len(masterdata_years) == (ngfs_peg_year - 2022 + 1)
    assert beta is not None
    rho = calculate_rho(beta)
    grouped = nonpower_coal.groupby("company_id")
    for year in masterdata_years:
        if year < summation_start_year:
            continue

        discount = calculate_discount(rho, year - 2022)

        def process(g):
            tonnes_coal = g[f"_{year}"]
            gj = coal2GJ(tonnes_coal)
            return (gj * g.energy_type_specific_average_unit_profit * discount).sum()

        profits = grouped.apply(process)
        profits_list.append(profits)
    # From NGFS

    def process(g):
        # This function is the same as the previous process() except for
        # tonnes_coal.
        tonnes_coal = g[f"_{ngfs_peg_year}"]
        gj = coal2GJ(tonnes_coal)
        return (gj * g.energy_type_specific_average_unit_profit).sum()

    profits_peg_year = grouped.apply(process)

    sum_frac_increase = 0.0
    for year, v in fraction_increase_after_peg_year.items():
        discount = calculate_discount(rho, year - 2022)
        sum_frac_increase += discount * v

    profits_list.append(profits_peg_year * sum_frac_increase)
    return sum(profits_list)


def get_capacity_factor(iea_df, year):
    try:
        return iea_df[iea_df.year == year].iloc[0].capacity_factor
    except IndexError:
        if year < 2018:
            truncated_year = 2018
        else:
            # For year > 2040
            truncated_year = 2040
        return iea_df[iea_df.year == truncated_year].iloc[0].capacity_factor


def get_coal_power_global_emissions_across_years(
    power_coal, years, discounted=False, rho=None
):
    emissions_list = []
    if discounted:
        assert years[0] == 2022
        assert rho is not None
    for year in years:
        mw_coal = power_coal[f"_{year}"]
        discount = 1
        if discounted:
            discount = calculate_discount(rho, year - 2022)
        # the emissions_factor unit is "tonnes of CO2 per MWh"
        emissions = (
            mw_coal * hours_in_1year * power_coal.emissions_factor * discount
        ).sum()
        # Convert to GtCO2
        emissions /= 1e9
        emissions_list.append(emissions)
    return emissions_list


def get_coal_power_per_company_NON_discounted_emissions_summed_over_years(
    ngfs_peg_year,
    power_coal,
    masterdata_years,
    fraction_increase_after_peg_year,
    x=1,
    multiply_by_country_tax_increase=False,
    summation_start_year=None,
):
    emissions_list = []
    assert ngfs_peg_year == 2026
    assert masterdata_years[0] == 2022
    assert summation_start_year is not None
    assert len(masterdata_years) == (ngfs_peg_year - 2022 + 1)
    grouped = power_coal.groupby("company_id")
    for year in masterdata_years:
        if year < summation_start_year:
            continue

        def process(g):
            if multiply_by_country_tax_increase:
                # The unit no longer make sense if we
                # multiply by g.country_tax_increase. But we
                # add a conditional branch here to minimize
                # code duplication.
                mw_coal = g[f"_{year}"] * g.country_tax_increase
            else:
                mw_coal = g[f"_{year}"]
            # the emissions_factor unit is "tonnes of CO2 per MWh"
            emissions_of_g = (
                mw_coal * hours_in_1year * g.emissions_factor
            ).sum()  # in tCO2
            return emissions_of_g

        emissions = grouped.apply(process)
        emissions_list.append(emissions)
    # From NGFS

    def process(g):
        # This function is the same as the previous process(), except for
        # the mw_coal calculation.
        if multiply_by_country_tax_increase:
            mw_coal = g[f"_{ngfs_peg_year}"] * g.country_tax_increase
        else:
            mw_coal = g[f"_{ngfs_peg_year}"]
        # the emissions_factor unit is "tonnes of CO2 per MWh"
        emissions_of_g = (
            mw_coal * hours_in_1year * g.emissions_factor
        ).sum()  # in tCO2
        return emissions_of_g

    emissions_peg_year = grouped.apply(process)

    sum_frac_increase = sum(fraction_increase_after_peg_year.values())
    emissions_list.append(emissions_peg_year * sum_frac_increase)
    return sum(emissions_list)


def get_coal_power_global_generation_across_years(power_coal, years):
    generations = []
    for year in years:
        # In MW
        mw_coal = power_coal[f"_{year}"].sum()
        # In Giga tonnes of coal
        generation = MW2Gigatonnes_of_coal(mw_coal)
        generations.append(generation)
    return generations


def get_coal_power_per_company_discounted_PROFIT_summed_over_years(
    ngfs_peg_year,
    power_coal,
    masterdata_years,
    fraction_increase_after_peg_year,
    beta,
    x=1,
    summation_start_year=None,
):
    profits_list = []
    assert ngfs_peg_year == 2026
    assert masterdata_years[0] == 2022
    assert summation_start_year is not None
    assert beta is not None
    rho = calculate_rho(beta)
    grouped = power_coal.groupby("company_id")
    for year in masterdata_years:
        if year < summation_start_year:
            continue

        discount = calculate_discount(rho, year - 2022)

        def process(g):
            # In MW
            mw_coal = g[f"_{year}"]
            gj = MW2GJ(mw_coal)
            return (gj * g.energy_type_specific_average_unit_profit * discount).sum()

        profits = grouped.apply(process)
        profits_list.append(profits)
    # From NGFS

    def process(g):
        # This function is the same as the previous process() except for
        # mw_coal.
        # In MW
        mw_coal = g[f"_{ngfs_peg_year}"]
        gj = MW2GJ(mw_coal)
        return (gj * g.energy_type_specific_average_unit_profit).sum()

    profits_peg_year = grouped.apply(process)

    sum_frac_increase = 0.0
    for year, v in fraction_increase_after_peg_year.items():
        discount = calculate_discount(rho, year - 2022)
        sum_frac_increase += discount * v

    profits_list.append(profits_peg_year * sum_frac_increase)
    return sum(profits_list)


def plot_stacked_bar(x, data, width=0.8, color=None, bar_fn=None):
    if bar_fn is None:
        bar_fn = plt.bar
    # plot the first one
    label, y = data[0]
    if color:  # for the first one only
        _bar = bar_fn(x, y, width, color=color, label=label)
    else:
        _bar = bar_fn(x, y, width, label=label)
    bars = [_bar]
    new_bottom = np.array(y)
    # plot the rest
    for label, _y in data[1:]:
        _bar = bar_fn(x, _y, width, bottom=new_bottom, label=label)
        bars.append(_bar)
        new_bottom += _y
    return bars


def read_csv_1d_list(filename):
    return list(pd.read_csv(filename, header=None)[0])


def get_developing_countries():
    # From IMF data as of 2021. Table A page 76
    developing_shortnames = read_csv_1d_list("data/developing_shortnames.csv")
    assert len(developing_shortnames) == 58
    # These are added manually because they are not stated by the IMF data.
    # We manually decide that they are developing.
    developing_shortnames += [
        "BW",  # Botswana
        "KP",  # North Korea
        "PS",  # Palestine, State of
    ]
    return developing_shortnames


def get_emerging_countries():
    # From IMF data as of 2021. Table A page 76
    emerging_shortnames = read_csv_1d_list("data/emerging_shortnames.csv")
    assert len(emerging_shortnames) == 85
    emerging_shortnames += [
        "CU",  # Cuba
    ]
    return emerging_shortnames


def prepare_from_climate_financing_data():
    # Data source:
    # https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv
    iso3166_df = pd.read_csv("data/country_ISO-3166_with_region.csv")
    iso3166_df_alpha2 = iso3166_df.set_index("alpha-2")

    developed_gdp = pd.read_csv("data/GDP-Developed-World.csv", thousands=",")
    # Note Puerto Rico, Macao, and San Marino are developed.
    # Add Liechtenstein
    # Taken from https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?locations=LI
    # NOTE This is 2019 value!!
    li = {
        "Developed world": "Liechtenstein",
        "country_shortcode": "LI",
        "2020 GDP (million dollars)": 6684.44,
        "Unnamed: 3": "Euro Area",
    }
    # Add Taiwan
    # Data source https://countryeconomy.com/gdp/taiwan
    # This is 2020 value
    tw = {
        "Developed world": "Taiwan, Province of China",
        "country_shortcode": "TW",
        "2020 GDP (million dollars)": 668156,
        "Unnamed: 3": "",
    }
    developed_gdp = pd.concat(
        [developed_gdp, pd.DataFrame([li, tw])], ignore_index=True
    )

    colname_for_gdp = "2020 GDP (million dollars)"
    # Sort by largest GDP
    developed_country_shortnames = list(
        developed_gdp.sort_values(by=colname_for_gdp, ascending=False).country_shortcode
    )
    # These countries are overseas territories of either UK,
    # FR, or NL, but they are not in the coal companies data,
    # hence not considered.
    # GP Guadeloupe
    # FK Falkland Islands (Malvinas)
    # GF French Guiana
    # CW Curaçao
    # BQ Bonaire, Sint Eustatius and Saba
    # BV Bouvet Island
    # MQ Martinique
    # MS Montserrat
    # BL Saint Barthélemy
    # MF Saint Martin (French part)
    # SX Sint Maarten (Dutch part)
    # GS South Georgia and the South Sandwich Islands
    # TC Turks and Caicos Islands
    # VI Virgin Islands (U.S.)
    # AX Åland Islands
    # AD Andorra
    # FO Faroe Islands
    # GI Gibraltar
    # GG Guernsey
    # VA Holy See
    # IM Isle of Man
    # JE Jersey
    # MC Monaco
    # SJ Svalbard and Jan Mayen
    # CX Christmas Island
    # CC Cocos (Keeling) Islands
    # HM Heard Island and McDonald Islands
    # NF Norfolk Island
    # BM Bermuda
    # GL Greenland
    # PM Saint Pierre and Miquelon
    # AI Anguilla
    # YT Mayotte
    # RE Réunion
    # SH Saint Helena, Ascension and Tristan da Cunha
    # IO British Indian Ocean Territory
    # TF French Southern Territories
    return (
        iso3166_df,
        iso3166_df_alpha2,
        developed_gdp,
        colname_for_gdp,
        developed_country_shortnames,
    )
