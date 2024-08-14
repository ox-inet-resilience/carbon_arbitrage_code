import csv
import json
import subprocess
import multiprocessing as mp
import os
from functools import lru_cache

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

hours_in_1year = 24 * 365.25
seconds_in_1hour = 3600  # seconds
# The years in NGFS data
ngfs_years = list(range(2005, 2105, 5))

# Constants
# and https://resilience.zulipchat.com/#narrow/stream/285747-TRISK/topic/to.20do/near/265246287
# The unit is dollars per tCO2
social_cost_of_carbon_imf = 80
scc_biden_administration = 190
# Bilal, Adrien, and Diego R. Känzig. The Macroeconomic Impact of Climate Change: Global vs. Local Temperature. No. w32450. National Bureau of Economic Research, 2024.
scc_bilal = 1056
social_cost_of_carbon = social_cost_of_carbon_imf
# social_cost_of_carbon = social_cost_of_carbon_biden_administration
# social_cost_of_carbon = scc_bilal
world_gdp_2023 = 105.44  # trillion dolars
gdp_marketcap_path = "data/all_countries_gdp_marketcap_2023.json"
gdp_per_capita_path = "data/all_countries_gdp_per_capita_2023.json"
EMISSIONS_COLNAME = "Emissions (CO2e 20 years)"
# EMISSIONS_COLNAME = "Emissions (CO2e 100 years)"
EMISSIONS_COLNAME = "annualco2tyear"


def read_json(filename):
    with open(filename) as f:
        obj = json.load(f)
    return obj


def write_small_json(content, filename):
    with open(filename, "w") as f:
        json.dump(content, f, separators=(",", ":"))


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


def add_array(a, b):
    assert len(a) == len(b)
    return [a[i] + b[i] for i in range(len(a))]


def subtract_array(a, b):
    assert len(a) == len(b)

    return [a[i] - b[i] for i in range(len(a))]


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
    mul = 29.3076
    if isinstance(x, list):
        return [e * mul for e in x]
    return x * mul


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


def maybe_load_forward_analytics_data(pre_existing_df=None):
    print("Reading from Masterdata")
    filename = "v3_power_Forward_Analytics2024.csv.zip"
    filename = "FA2024_power_only_preprocessed.csv.gz"
    # filename = "masterdata_ownership_PROCESSED_capacity_factor.csv.gz"
    # encoding = "latin1"
    compression = "gzip" if filename.endswith(".gz") else "zip"
    if pre_existing_df is None:
        _str_type = "string[pyarrow]"
        # TODO We use str for now even though pyarrow is more performant. There
        # is a bug if you use pyarrow.
        _str_type = str
        df = pd.read_csv(
            f"data_private/{filename}",
            compression=compression,
            # encoding=encoding,
            dtype={
                # "company_name": _str_type,
                # "company_id": int,
                "sector": _str_type,
                # "technology": _str_type,
                # "technology_type": _str_type,
                "asset_country": _str_type,
                # "emissions_factor_unit": _str_type,
                # "unit": _str_type,
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


def read_forward_analytics_data(sector, pre_existing_df=None):
    df = maybe_load_forward_analytics_data(pre_existing_df)
    replace_countries(df)
    # Remove rows without proper asset_country.
    # df = df[~pd.isna(df.asset_country)]
    # All sectors are:
    # {'Coal', 'Oil&Gas', 'Aviation', 'Shipping', 'HDV', 'Steel', 'Power',
    # 'Automotive', 'Cement'}
    df_sector = df[df.sector == sector].copy()
    # power_companies = df[df.sector == "Power"]
    # power_coal = power_companies[power_companies.technology == "CoalCap"].copy()
    return df, df_sector


def read_ngfs():
    return {
        "production": pd.read_csv("data/2-GCAM6-filtered-prim-and-secon-energy.csv.gz"),
        "emissions": pd.read_csv("data/3-GCAM6-emissions.csv.gz"),
    }


def calculate_ngfs_projection(
    production_or_emissions,
    value_fa,
    ngfs_df,
    sector,
    scenario,
    start_year,
    last_year,
    alpha2_to_alpha3,
    filter_subsector=None,
):
    assert sector == "Power"
    ngfs = ngfs_df[production_or_emissions]
    ngfs = ngfs[ngfs.Scenario == scenario]
    if sector == "Coal":
        variable = "Primary Energy|Coal"
    subsectors = ["Coal", "Oil", "Gas"]
    if filter_subsector is not None:
        subsectors = [filter_subsector]
    years_interpolated = list(range(start_year, last_year + 1))
    # Use set to deduplicate countries list.
    countries = list(set(value_fa.index.get_level_values("asset_country").to_list()))
    out = None

    ngfs_country_wo_iea_stats = ngfs[
        ngfs.Region == "Downscaling|Countries without IEA statistics"
    ]

    def country_mapper(alpha2):
        if alpha2 == "XK":  # Kosovo
            return "XKX"
        return alpha2_to_alpha3[alpha2]

    for country in countries:
        ngfs_country = ngfs[ngfs.Region == country_mapper(country)]
        if len(ngfs_country) == 0:
            ngfs_country = ngfs_country_wo_iea_stats
        value_fa_country = value_fa[country]
        timeseries = None
        for subsector in subsectors:
            if subsector not in value_fa_country:
                continue
            variable = f"Secondary Energy|Electricity|{subsector}"
            ngfs_country_subsector = ngfs_country[
                ngfs_country.Variable == variable
            ].iloc[0]
            across_years = [
                ngfs_country_subsector[str(year)] for year in years_interpolated
            ]
            if across_years[0] == 0:
                # NGFS has countries that start from 0 value
                ngfs_country_subsector = ngfs_country_wo_iea_stats[
                    ngfs_country_wo_iea_stats.Variable == variable
                ].iloc[0]
                across_years = [
                    ngfs_country_subsector[str(year)] for year in years_interpolated
                ]
            # rescale NGFS part, so that at the first year, it is 1
            across_years = [
                value_fa[country][subsector] * e / across_years[0] for e in across_years
            ]
            if timeseries is None:
                timeseries = across_years.copy()
            else:
                timeseries = add_array(timeseries, across_years)
        if timeseries is None:
            continue
        if out is None:
            out = {country: timeseries}
        else:
            out[country] = timeseries
    if out is None:
        return pd.Series([]), 0
    summed = 0
    for value in out.values():
        summed += sum(value)
    final_out = []
    for i in range(len(years_interpolated)):
        final_out.append(
            pd.Series({country: value[i] for country, value in out.items()})
        )
    return final_out, summed


def calculate_ngfs_projection_by_subsector(
    production_or_emissions,
    value_fa,
    ngfs_df,
    sector,
    scenario,
    start_year,
    last_year,
    alpha2_to_alpha3,
    filter_subsector=None,
):
    out = {}
    for subsector in ["Coal", "Oil", "Gas"]:
        by_subsector, _ = calculate_ngfs_projection(
            production_or_emissions,
            value_fa,
            ngfs_df,
            sector,
            scenario,
            start_year,
            last_year,
            alpha2_to_alpha3,
            filter_subsector=subsector,
        )
        out[subsector] = by_subsector

    return out


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


def sum_discounted(array, rho):
    return sum(e * calculate_discount(rho, i) for i, e in enumerate(array))


def discount_array(array, rho):
    return [e * calculate_discount(rho, i) for i, e in enumerate(array)]


def get_emissions_by_country(nonpower_coal, discounted=False):
    # The division by 1e3 converts MtCO2 to GtCO2.
    emissions = (
        nonpower_coal.groupby(["asset_country", "subsector"])[EMISSIONS_COLNAME].sum()
        / 1e3
    )
    return emissions


def get_production_by_country(_df, sector):
    # convert to giga tonnes of coal
    if sector == "Extraction":
        mul = 1e-3  # From mega tonnes to giga tonnes
    elif sector == "Power":
        # From MWh to MJ to GJ to giga tonnes of coal
        mul = seconds_in_1hour / 1e3 * GJ2coal(1) / 1e9
    else:
        raise Exception("Should never happen")
    production = _df.groupby(["asset_country", "subsector"])["activity"].sum() * mul
    return production


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


def plot_stacked_bar(
    x, data, width=0.8, color=None, bar_fn=None, overlapping_insteadof_stacking=False
):
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
        bottom = new_bottom
        if overlapping_insteadof_stacking:
            bottom = None
        _bar = bar_fn(x, _y, width, bottom=bottom, label=label)
        bars.append(_bar)
        new_bottom += _y
    return bars


def read_csv_1d_list(filename):
    return list(pd.read_csv(filename, header=None, dtype=str, keep_default_na=False)[0])


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
    assert len(emerging_shortnames) == 89
    emerging_shortnames += [
        "CU",  # Cuba
    ]
    return emerging_shortnames


def read_iso3166():
    return pd.read_csv(
        "data/country_ISO-3166_with_region.csv",
        # Needs this because otherwise NA for Namibia is interpreted as NaN
        na_filter=False,
    )


def read_country_specific_scc_filtered():
    country_specific_scc = read_json("plots/country_specific_scc.json")
    # Remove these countries
    # Because they are in Oceania:
    # New Caledonia
    # Fiji
    # Solomon Islands
    # Vanuatu
    # They are not part of the 6 regions (Asia, Africa, NA,
    # LAC, Europe, AUS&NZ), nor are they part of the
    # developing, emerging, developed world.
    for country in ["NC", "FJ", "SB", "VU"]:
        del country_specific_scc[country]
    return country_specific_scc


def prepare_from_climate_financing_data():
    # Data source:
    # https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv
    iso3166_df = read_iso3166()
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


def prepare_alpha2_to_full_name_concise():
    iso3166_df = read_iso3166()
    iso3166_df_alpha2 = iso3166_df.set_index("alpha-2")
    alpha2_to_full_name = iso3166_df_alpha2["name"].to_dict()
    alpha2_to_full_name["GB"] = "Great Britain"
    alpha2_to_full_name["US"] = "USA"
    alpha2_to_full_name["RU"] = "Russia"
    alpha2_to_full_name["TW"] = "Taiwan"
    alpha2_to_full_name["KR"] = "South Korea"
    alpha2_to_full_name["LA"] = "Laos"
    alpha2_to_full_name["VE"] = "Venezuela"
    alpha2_to_full_name["CD"] = "Congo-Kinshasa"
    alpha2_to_full_name["IR"] = "Iran"
    alpha2_to_full_name["TZ"] = "Tanzania"
    alpha2_to_full_name["BA"] = "B&H"
    alpha2_to_full_name["XK"] = "Kosovo"
    return alpha2_to_full_name
