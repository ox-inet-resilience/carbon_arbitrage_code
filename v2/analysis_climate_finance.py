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
import with_learning


def make_climate_financing_plot(
    plotname_suffix="",
    plot_name=None,
    svg=False,
    ignore_cache=False,
    chosen_s2_scenario=None,
):
    global nonpower_coal

    if chosen_s2_scenario is None:
        chosen_s2_scenario = f"{analysis_main.NGFS_PEG_YEAR}-2050 FA + Net Zero 2050 Scenario"

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

    region_countries_map, regions = analysis_main.prepare_regions_for_climate_financing(iso3166_df)

    cache_json_path = (
        f"cache/climate_financing_yearly_discounted_{git_branch}{plotname_suffix}.json"
    )
    yearly_costs_dict = None
    if ignore_cache:
        yearly_costs_dict = analysis_main.calculate_yearly_costs_dict(chosen_s2_scenario)
    elif os.path.isfile(cache_json_path):
        yearly_costs_dict = util.read_json(cache_json_path)
    else:
        yearly_costs_dict = analysis_main.calculate_yearly_costs_dict(chosen_s2_scenario)
        with open(cache_json_path, "w") as f:
            json.dump(yearly_costs_dict, f)

    def _get_year_range_cost(year_start, year_end, included_countries=None):
        out = 0.0
        for c, e in yearly_costs_dict.items():
            if included_countries is not None and c not in included_countries:
                continue
            out += sum(e[year_start - analysis_main.NGFS_PEG_YEAR : year_end + 1 - analysis_main.NGFS_PEG_YEAR])
        return out

    plot_data = []
    # Used for sanity check.
    _world_sum = 0.0
    _developed_sum = 0.0
    _developing_sum = 0.0
    _emerging_sum = 0.0
    _region_sum = defaultdict(float)
    for year_start, year_end in [(analysis_main.NGFS_PEG_YEAR + 1, 2030), (2031, 2050)]:
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


if __name__ == "__main__":
    # It is faster not to calculate residual benefit for climate financing.
    with_learning.ENABLE_RESIDUAL_BENEFIT = 0
    make_climate_financing_plot()
    # make_climate_financing_SCATTER_plot()
    exit()
    make_yearly_climate_financing_plot()
    exit()
    make_yearly_climate_financing_plot_SENSITIVITY_ANALYSIS()
    # Reenable residual benefit again
    with_learning.ENABLE_RESIDUAL_BENEFIT = 1
