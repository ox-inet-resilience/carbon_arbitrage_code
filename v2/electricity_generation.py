import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pycountry

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import analysis_main  # noqa
import util  # noqa
import with_learning  # noqa


def plot_electricity_generation(ax, region_name, ylabel=False):
    analysis_main.MEASURE_GLOBAL_VARS_SCENARIO = "Net Zero 2050"
    analysis_main.MEASURE_GLOBAL_VARS = True
    util.CARBON_BUDGET_CONSISTENT = "15-50"
    years = range(2024, 2050 + 1)

    region_key = region_name
    match region_name:
        case "Global":
            region_key = "WORLD"
        case "Developed Countries":
            region_key = "Developed_UNFCCC"
        case "Developing Countries":
            region_key = "Developing_UNFCCC"
        case _:
            region_key = pycountry.countries.lookup(region_name).alpha_2

    def GtCoal2GJ(x):
        return util.coal2GJ(x * 1e9)

    # Production
    # Giga tonnes of coal
    total_production_fa = util.get_production_by_country(
        analysis_main.df_sector, analysis_main.SECTOR_INCLUDED
    )
    last_year = analysis_main.LAST_YEAR
    production_projection = {}
    scenarios = ["Current Policies", "Net Zero 2050"]
    for scenario in scenarios:
        # Giga tonnes of coal
        (
            production_with_ngfs_projection,
            gigatonnes_coal_production,
            profit_ngfs_projection,
        ) = util.calculate_ngfs_projection(
            "production",
            total_production_fa,
            analysis_main.ngfs_df,
            analysis_main.SECTOR_INCLUDED,
            scenario,
            analysis_main.NGFS_PEG_YEAR,
            last_year,
            analysis_main.alpha2_to_alpha3,
            unit_profit_df=analysis_main.unit_profit_df,
        )
        production_projection[scenario] = production_with_ngfs_projection
    DeltaP = util.subtract_array(
        production_projection["Current Policies"],
        production_projection["Net Zero 2050"],
    )

    # green energy production
    def get_arr(x):
        if region_key == "WORLD":
            return np.array([GtCoal2GJ(e.sum()) for e in x])
        elif region_key == "Developing_UNFCCC":
            return np.array(
                [
                    GtCoal2GJ(e.filter(items=with_learning.DEVELOPING_UNFCCC).sum())
                    for e in x
                ]
            )
        elif region_key == "Developed_UNFCCC":
            return np.array(
                [
                    GtCoal2GJ(e.filter(items=with_learning.DEVELOPED_UNFCCC).sum())
                    for e in x
                ]
            )
        else:
            return np.array([GtCoal2GJ(e.filter(items=[region_key]).sum()) for e in x])

    y_deltap = get_arr(DeltaP)
    with_learning.VERBOSE_ANALYSIS = True
    with_learning.VERBOSE_ANALYSIS_COUNTRY = region_key
    analysis_main.run_table1(to_csv=False, do_round=False, plot_yearly=False)

    if region_key == "WORLD":
        green_energy_produced_by_country_group = [
            {tech: sum(e[a2][tech] for a2 in e) for tech in with_learning.TECHS}
            for e in analysis_main.global_cost_with_learning.green_energy_produced_by_country
        ]
    elif region_key == "Developing_UNFCCC":
        green_energy_produced_by_country_group = [
            {
                tech: sum(
                    e[c][tech] if c in e else 0 for c in with_learning.DEVELOPING_UNFCCC
                )
                for tech in with_learning.TECHS
            }
            for e in analysis_main.global_cost_with_learning.green_energy_produced_by_country
        ]
    elif region_key == "Developed_UNFCCC":
        green_energy_produced_by_country_group = [
            {
                tech: sum(
                    e[c][tech] if c in e else 0 for c in with_learning.DEVELOPED_UNFCCC
                )
                for tech in with_learning.TECHS
            }
            for e in analysis_main.global_cost_with_learning.green_energy_produced_by_country
        ]
    else:
        green_energy_produced_by_country_group = [
            {
                tech: e[region_key][tech] if region_key in e else 0
                for tech in with_learning.TECHS
            }
            for e in analysis_main.global_cost_with_learning.green_energy_produced_by_country
        ]

    # now we group by tech
    # in GJ
    green_energy_produced_by_country_group = {
        tech: np.array([e[tech] for e in green_energy_produced_by_country_group])
        for tech in with_learning.TECHS
    }

    def gj2xwh(arr):
        return [util.GJ2MWh(e) / 1e6 for e in arr]

    plt.sca(ax)
    plt.title(region_name)
    ys = {scenario: get_arr(production_projection[scenario]) for scenario in scenarios}

    scenario = "Current Policies"
    plt.plot(years, gj2xwh(ys[scenario]), label=scenario, linestyle="dashdot")

    plt.plot(
        years,
        gj2xwh(y_deltap + ys["Net Zero 2050"]),
        label=r"Net Zero 2050 $1.5\degree 50\%$",
        linestyle="dotted",
    )

    plt.plot(years, gj2xwh(y_deltap), label="Green")  # this is actually DeltaQ

    scenario = "Net Zero 2050"
    plt.plot(years, gj2xwh(ys[scenario]), label="Brown", linestyle="dashdot")

    # for tech in with_learning.TECHS:
    #     plt.plot(years, energy_produced_by_country_group[tech], label=tech)
    y_green = np.array(sum(green_energy_produced_by_country_group.values()))
    plt.plot(years, gj2xwh(y_green), label="Green")
    plt.plot(years, gj2xwh(y_green + ys["Net Zero 2050"]), label="Green + Brown")
    if ylabel:
        plt.ylabel("Energy (TWh)")
    analysis_main.MEASURE_GLOBAL_VARS = False
    analysis_main.MEASURE_GLOBAL_VARS_SCENARIO = "Net Zero 2050"
    with_learning.VERBOSE_ANALYSIS = False
    util.CARBON_BUDGET_CONSISTENT = False


if 1:
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    plot_electricity_generation(axes[0], "Global", ylabel=True)
    plot_electricity_generation(axes[1], "Developed Countries")
    plot_electricity_generation(axes[2], "Developing Countries")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.2),
        fontsize=10,
        frameon=False,
    )
    plt.subplots_adjust(bottom=0.01, hspace=0.4, wspace=0.6)
    plt.savefig(
        "plots/phase_in/electricity_generation_combined.png", bbox_inches="tight"
    )
    plt.close()

    # plt.savefig("plots/electricity_generation.png")
    exit()

if 1:
    # Second graph: 8 countries
    fig = plt.figure(figsize=(12, 6))  # Adjusted figure size for grid layout
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])
    plot_electricity_generation(fig.add_subplot(gs[0, 0]), "India", ylabel=True)
    plot_electricity_generation(fig.add_subplot(gs[0, 1]), "Indonesia")
    plot_electricity_generation(fig.add_subplot(gs[0, 2]), "South Africa")
    plot_electricity_generation(fig.add_subplot(gs[0, 3]), "Mexico")
    plot_electricity_generation(fig.add_subplot(gs[1, 0]), "Viet Nam", ylabel=True)
    plot_electricity_generation(fig.add_subplot(gs[1, 1]), "Iran")
    plot_electricity_generation(fig.add_subplot(gs[1, 2]), "Thailand")
    plot_electricity_generation(fig.add_subplot(gs[1, 3]), "Egypt")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.2),
        fontsize=10,
        frameon=False,
    )
    plt.subplots_adjust(bottom=0.01, hspace=0.4, wspace=0.6)
    plt.savefig(
        "plots/phase_in/electricity_generation_8countries.png", bbox_inches="tight"
    )
    plt.close()
