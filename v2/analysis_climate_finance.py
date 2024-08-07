import json
import math
import os
import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import analysis_main  # noqa
import util  # noqa
from util import world_gdp_2023  # noqa

# Common variables
git_branch = util.get_git_branch()
cost_batteries = [
    "cost_battery_short",
    "cost_battery_long",
    "cost_battery_pe",
    "cost_battery_grid",
]
investment_label_map = {
    "renewables": "Renewables",
    "cost_battery_short": "Battery (short)",
    "cost_battery_long": "Battery (long)",
    "cost_battery_pe": "Battery (PE)",
    "cost_battery_grid": "Battery (grid)",
}


def set_matplotlib_tick_spacing(tick_spacing):
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))


def get_info(info_name, last_year, included_countries=None):
    chosen_s2_scenario = (
        f"{analysis_main.NGFS_PEG_YEAR}-{last_year} FA + Net Zero 2050 Scenario"
    )

    if info_name == "benefit":
        info = analysis_main.calculate_each_countries_with_cache(
            chosen_s2_scenario,
            f"cache/country_specific_info_{last_year}_{git_branch}.json",
            ignore_cache=False,
            info_name="benefit_non_discounted",
            last_year=last_year,
        )
        info_residual_benefit = analysis_main.calculate_each_countries_with_cache(
            chosen_s2_scenario,
            f"cache/country_specific_info_{last_year}_{git_branch}.json",
            ignore_cache=False,
            info_name="residual_benefit",
            last_year=last_year,
        )
        # Add residual benefit to benefit
        for k, v in info_residual_benefit.items():
            info[k] += v

    else:
        info = analysis_main.calculate_each_countries_with_cache(
            chosen_s2_scenario,
            f"cache/country_specific_info_{last_year}_{git_branch}.json",
            ignore_cache=False,
            info_name=info_name,
            last_year=last_year,
        )
    value = 0.0
    for c, e in info.items():
        if included_countries is not None and c not in included_countries:
            continue
        value += e
    return value


def get_info_investment(info_name, last_year, included_countries=None):
    if info_name == "renewables":
        return get_info("investment_cost", last_year, included_countries) - sum(
            get_info(b, last_year, included_countries) for b in cost_batteries
        )
    return get_info(info_name, last_year, included_countries)


def make_climate_financing_plot(
    plot_name=None,
    svg=False,
    info_name="cost",
):
    (
        iso3166_df,
        _,
        _,
        _,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()

    developING_country_shortnames = util.get_developing_countries()
    emerging_country_shortnames = util.get_emerging_countries()

    region_countries_map, regions = analysis_main.prepare_regions_for_climate_financing(
        iso3166_df
    )

    def get_info_with_start_year(
        start_year=None, last_year=None, included_countries=None
    ):
        if start_year == analysis_main.NGFS_PEG_YEAR:
            return get_info(info_name, last_year, included_countries)
        return get_info(info_name, last_year, included_countries) - get_info(
            info_name, start_year - 1, included_countries
        )

    plot_data = []
    # Used for sanity check.
    _developed_sum = 0.0
    _developing_sum = 0.0
    _emerging_sum = 0.0
    _region_sum = defaultdict(float)
    for year_start, year_end in [(analysis_main.NGFS_PEG_YEAR, 2030), (2031, 2050)]:
        _world = get_info_with_start_year(year_start, year_end)
        _developed = get_info_with_start_year(
            year_start, year_end, developed_country_shortnames
        )
        _developed_sum += _developed
        _developing = get_info_with_start_year(
            year_start, year_end, developING_country_shortnames
        )
        _developing_sum += _developing
        _emerging = get_info_with_start_year(
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
            _region_cost = get_info_with_start_year(
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
    print(info_name, plot_data)
    text_height = max(plot_data[1][1])
    plt.text(
        1,
        text_height,
        "By level of\ndevelopment",
        color="gray",
        verticalalignment="top",
        # fontdict={"size": 10},
    )
    plt.text(
        5,
        text_height,
        "By region",
        color="gray",
        verticalalignment="top",
        # fontdict={"size": 10},
    )

    plt.legend()
    plt.xticks(xticks, rotation=45, ha="right")
    label = info_name
    if info_name == "cost":
        label = "PV climate financing"
    elif info_name == "benefit":
        label = "Benefit non-discounted"
    label += " (trillion dollars)"
    plt.ylabel(label)
    plt.tight_layout()
    util.savefig(
        plot_name if plot_name else f"climate_financing_by_region_{info_name}",
        svg=svg,
    )
    plt.close()


def make_cf_investment_cost_plot(last_year):
    (
        iso3166_df,
        _,
        _,
        _,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()

    developING_country_shortnames = util.get_developing_countries()
    emerging_country_shortnames = util.get_emerging_countries()

    region_countries_map, regions = prepare_regions_for_climate_financing(
        iso3166_df
    )

    plot_data = []
    # Used for sanity check.
    _developed_sum = 0.0
    _developing_sum = 0.0
    _emerging_sum = 0.0
    _region_sum = defaultdict(float)

    for info_name in ["renewables"] + cost_batteries:
        _world = get_info_investment(info_name, last_year)
        _developed = get_info_investment(
            info_name, last_year, developed_country_shortnames
        )
        _developed_sum += _developed
        _developing = get_info_investment(
            info_name, last_year, developING_country_shortnames
        )
        _developing_sum += _developing
        _emerging = get_info_investment(
            info_name, last_year, emerging_country_shortnames
        )
        _emerging_sum += _emerging
        region_costs = [
            _world,
            _developed,
            _developing,
            _emerging,
        ]
        for region in regions:
            _region_cost = get_info_investment(
                info_name, last_year, region_countries_map[region]
            )
            _region_sum[region] += _region_cost
            region_costs.append(_region_cost)
        plot_data.append((investment_label_map[info_name], region_costs))

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
    print(info_name, plot_data)
    text_height = None
    if last_year == 2030:
        text_height = 4
    elif last_year == 2050:
        text_height = 11
    plt.text(
        1,
        text_height,
        "By level of\ndevelopment",
        color="gray",
        verticalalignment="top",
        # fontdict={"size": 10},
    )
    plt.text(
        5,
        text_height,
        "By region",
        color="gray",
        verticalalignment="top",
        # fontdict={"size": 10},
    )

    plt.legend()
    plt.xticks(xticks, rotation=45, ha="right")
    label = "Investment costs (trillion dollars)"
    plt.ylabel(label)
    plt.tight_layout()
    util.savefig(f"cf_investment_costs_by_region_{last_year}")
    plt.close()


def make_climate_financing_top15_plot(last_year):
    top15_power = "CN IN US VN ID TR JP RU DE BD ZA KR PL GB SA".split()
    plot_data = []
    for info_name in ["renewables"] + cost_batteries:
        costs = [
            get_info_investment(info_name, last_year, [country_name])
            for country_name in top15_power
        ]
        plot_data.append((investment_label_map[info_name], costs))
    xticks = top15_power
    plt.figure(figsize=(6, 6))
    util.plot_stacked_bar(xticks, plot_data)
    plt.legend()
    plt.xticks(xticks, rotation=45, ha="right")
    label = "Investment costs (trillion dollars)"
    plt.ylabel(label)
    plt.tight_layout()
    util.savefig(f"cf_investment_costs_top15_{last_year}")
    plt.close()


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


def make_yearly_climate_financing_plot():
    global df_sector

    chosen_s2_scenario = f"{analysis_main.NGFS_PEG_YEAR}-{analysis_main.LAST_YEAR} FA + Net Zero 2050 Scenario"
    chosen_s2_scenario += " NON-DISCOUNTED"

    (
        iso3166_df,
        iso3166_df_alpha2,
        developed_gdp,
        colname_for_gdp,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()

    # The cache is used only for each of the developed countries.
    cache_json_path = f"plots/climate_financing_yearly_{git_branch}.json"
    if os.path.isfile(cache_json_path):
        print("Cached climate YEARLY financing json found. Reading...")
        yearly_costs_dict = util.read_json(cache_json_path)
    else:
        yearly_costs_dict = calculate_yearly_info_dict(chosen_s2_scenario)
        with open(cache_json_path, "w") as f:
            json.dump(yearly_costs_dict, f)
    whole_years = range(NGFS_PEG_YEAR, 2100 + 1)

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

    # Part 2. By regions
    plt.sca(axs[1])
    plt.plot(whole_years, yearly_world_cost, label="World")
    region_countries_map, regions = prepare_regions_for_climate_financing(iso3166_df)
    for region in regions:
        included_country_names = region_countries_map[region]
        yearly_cost = _get_yearly_cost(included_country_names)
        plt.plot(whole_years, yearly_cost, label=region)

    plt.xlabel("Time")
    plt.ylabel("Annual climate financing\n(trillion dollars)")
    set_matplotlib_tick_spacing(10)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    util.savefig("climate_financing_yearly")
    plt.close()

    # Part 3. Relative to 2023 developed GDP.
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

    plt.figure()
    benefit_relative_to_gdp = [
        (0.12707470412 - 0.05845436389) * (1 - math.exp(-0.01 * (t - 2016))) * 100
        for t in whole_years
    ]
    world_cost_relative = np.array(yearly_world_cost) * 100 / world_gdp_2023
    plt.plot(
        whole_years,
        benefit_relative_to_gdp,
        label="Global benefits (avoided climate damages)",
    )
    plt.plot(
        whole_years,
        world_cost_relative * 0.1,
        label="Global public costs (10% of climate finance need)",
    )
    # Reset color cycler
    plt.gca().set_prop_cycle(None)
    # Skip blue
    next(plt.gca()._get_lines.prop_cycler)
    plt.plot(
        whole_years,
        world_cost_relative,
        label="Global costs (100% of climate finance need)",
        linestyle="dashed",
    )
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("% of GDP")
    util.savefig("climate_financing_yearly_world")
    plt.close()

    # benefit and cost for NA and Europe
    gdp_marketcap_dict = util.read_json(util.gdp_marketcap_path)
    na = region_countries_map["North America"]
    europe = region_countries_map["Europe"]
    gdp_na = 0.0
    for c in na:
        _gdp = gdp_marketcap_dict.get(c, 0.0)
        if not np.isnan(_gdp):
            gdp_na += _gdp
    gdp_na /= 1e12
    yearly_cost_na = _get_yearly_cost(na) / gdp_na * 100
    gdp_europe = 0.0
    for c in europe:
        _gdp = gdp_marketcap_dict.get(c, 0.0)
        if not np.isnan(_gdp):
            gdp_europe += _gdp
    gdp_europe /= 1e12
    yearly_cost_europe = _get_yearly_cost(europe) / gdp_europe * 100
    benefit_world_relative = np.array(benefit_relative_to_gdp)
    country_specific_scc_dict = util.read_json("plots/country_specific_scc.json")
    scc_na_fraction = sum(country_specific_scc_dict.get(c, 0.0) for c in na) / sum(
        country_specific_scc_dict.values()
    )
    scc_europe_fraction = sum(
        country_specific_scc_dict.get(c, 0.0) for c in europe
    ) / sum(country_specific_scc_dict.values())
    benefit_na = benefit_world_relative * scc_na_fraction * world_gdp_2023 / gdp_na
    benefit_europe = (
        benefit_world_relative * scc_europe_fraction * world_gdp_2023 / gdp_europe
    )
    plt.figure()
    plt.plot(whole_years, yearly_cost_na * 0.1, label="NA public costs")
    plt.plot(whole_years, yearly_cost_na, label="NA costs")
    plt.plot(whole_years, benefit_na, label="NA benefits")
    plt.plot(whole_years, yearly_cost_europe * 0.1, label="Europe public costs")
    plt.plot(whole_years, yearly_cost_europe, label="Europe costs")
    plt.plot(whole_years, benefit_europe, label="Europe benefits")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("% of GDP")
    util.savefig("climate_financing_yearly_na_europe")
    plt.close()


if __name__ == "__main__":
    for info_name in [
        "cost",
        "benefit",
        "opportunity_cost",
        "investment_cost",
    ]:
        make_climate_financing_plot(info_name=info_name)
    for last_year in [2030, 2050]:
        make_cf_investment_cost_plot(last_year)
        make_climate_financing_top15_plot(last_year)
    # make_climate_financing_SCATTER_plot()
    exit()
    make_yearly_climate_financing_plot()
    exit()
    make_yearly_climate_financing_plot_SENSITIVITY_ANALYSIS()
