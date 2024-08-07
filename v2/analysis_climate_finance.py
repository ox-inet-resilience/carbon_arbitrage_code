import json
import os
import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import analysis_main  # noqa
import util  # noqa


def make_climate_financing_plot(
    plotname_suffix="",
    plot_name=None,
    svg=False,
    info_name="cost",
):
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

    region_countries_map, regions = analysis_main.prepare_regions_for_climate_financing(
        iso3166_df
    )

    def get_info(info_name, last_year, included_countries=None):
        chosen_s2_scenario = (
            f"{analysis_main.NGFS_PEG_YEAR}-{last_year} FA + Net Zero 2050 Scenario"
        )

        info = analysis_main.calculate_each_countries_with_cache(
            chosen_s2_scenario,
            f"cache/country_specific_info_{last_year}_{git_branch}.json",
            ignore_cache=True,
            info_name=info_name,
            last_year=last_year,
        )
        value = 0.0
        for c, e in info.items():
            if included_countries is not None and c not in included_countries:
                continue
            value += e
        return value

    def get_info_with_start_year(
        start_year=None, last_year=None, included_countries=None
    ):
        if start_year is None:
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
    for year_start, year_end in [(None, 2030), (2031, 2050)]:
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
    plt.text(
        1,
        9,
        "By level of\ndevelopment",
        color="gray",
        verticalalignment="top",
        # fontdict={"size": 10},
    )
    plt.text(
        5,
        9,
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


if __name__ == "__main__":
    make_climate_financing_plot()
    # make_climate_financing_SCATTER_plot()
    exit()
    make_yearly_climate_financing_plot()
    exit()
    make_yearly_climate_financing_plot_SENSITIVITY_ANALYSIS()
