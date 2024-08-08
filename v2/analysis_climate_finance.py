import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import analysis_main  # noqa
import util  # noqa

git_branch = util.get_git_branch()


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
    _world_sum = 0.0
    _developed_sum = 0.0
    _developing_sum = 0.0
    _emerging_sum = 0.0
    _region_sum = defaultdict(float)
    for year_start, year_end in [(analysis_main.NGFS_PEG_YEAR, 2030), (2031, 2050)]:
        _world = get_info_with_start_year(year_start, year_end)
        _world_sum += _world
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

    region_countries_map, regions = analysis_main.prepare_regions_for_climate_financing(
        iso3166_df
    )

    cost_batteries = [
        "cost_battery_short",
        "cost_battery_long",
        "cost_battery_pe",
        "cost_battery_grid",
    ]

    def get_info_wrapped(info_name, included_countries=None):
        if info_name == "renewables":
            return get_info("investment_cost", last_year, included_countries) - sum(
                get_info(b, last_year, included_countries) for b in cost_batteries
            )
        return get_info(info_name, last_year, included_countries)

    plot_data = []
    # Used for sanity check.
    _world_sum = 0.0
    _developed_sum = 0.0
    _developing_sum = 0.0
    _emerging_sum = 0.0
    _region_sum = defaultdict(float)
    label_map = {
        "renewables": "Renewables",
        "cost_battery_short": "Battery (short)",
        "cost_battery_long": "Battery (long)",
        "cost_battery_pe": "Battery (PE)",
        "cost_battery_grid": "Battery (grid)",
    }
    for info_name in ["renewables"] + cost_batteries:
        _world = get_info_wrapped(info_name)
        _world_sum += _world
        _developed = get_info_wrapped(info_name, developed_country_shortnames)
        _developed_sum += _developed
        _developing = get_info_wrapped(info_name, developING_country_shortnames)
        _developing_sum += _developing
        _emerging = get_info_wrapped(info_name, emerging_country_shortnames)
        _emerging_sum += _emerging
        region_costs = [
            _world,
            _developed,
            _developing,
            _emerging,
        ]
        for region in regions:
            _region_cost = get_info_wrapped(info_name, region_countries_map[region])
            _region_sum[region] += _region_cost
            region_costs.append(_region_cost)
        plot_data.append((label_map[info_name], region_costs))

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


if __name__ == "__main__":
    for info_name in [
        "cost",
        "benefit",
        "opportunity_cost",
        "investment_cost",
    ]:
        make_climate_financing_plot(info_name=info_name)
    make_cf_investment_cost_plot(2030)
    make_cf_investment_cost_plot(2050)
    # make_climate_financing_SCATTER_plot()
    exit()
    make_yearly_climate_financing_plot()
    exit()
    make_yearly_climate_financing_plot_SENSITIVITY_ANALYSIS()
