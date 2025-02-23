import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import analysis_main  # noqa
import util  # noqa
import with_learning  # noqa


def plot_electricity_generation():
    analysis_main.MEASURE_GLOBAL_VARS_SCENARIO = "Net Zero 2050"
    analysis_main.MEASURE_GLOBAL_VARS = True
    util.CARBON_BUDGET_CONSISTENT = "15-50"
    years = range(2024, 2050 + 1)

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
    country_name = "WORLD"
    country_name = "IN"
    country_name = "PL"

    if country_name == "WORLD":
        y_deltap = np.array([GtCoal2GJ(e.sum()) for e in DeltaP])
    else:
        y_deltap = np.array(
            [GtCoal2GJ(e.filter(items=[country_name]).sum()) for e in DeltaP]
        )

    with_learning.VERBOSE_ANALYSIS = True
    with_learning.VERBOSE_ANALYSIS_COUNTRY = country_name
    analysis_main.run_table1(to_csv=False, do_round=False, plot_yearly=False)

    if country_name == "WORLD":
        green_energy_produced_by_country_group = [
            {tech: sum(e[a2][tech] for a2 in e) for tech in with_learning.TECHS}
            for e in analysis_main.global_cost_with_learning.green_energy_produced_by_country
        ]
    else:
        green_energy_produced_by_country_group = [
            {
                tech: e[country_name][tech] if country_name in e else 0
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

    plt.figure()
    plt.title(country_name)
    ys = {}
    for scenario in scenarios:
        # Sum across countries
        if country_name == "WORLD":
            y = [GtCoal2GJ(e.sum()) for e in production_projection[scenario]]
        else:
            y = [
                GtCoal2GJ(e.filter(items=[country_name]).sum())
                for e in production_projection[scenario]
            ]
        ys[scenario] = y
        label = scenario if scenario == "Current Policies" else "Brown"
        plt.plot(years, y, label=label)
    # for tech in with_learning.TECHS:
    #     plt.plot(years, energy_produced_by_country_group[tech], label=tech)
    y_green = np.array(sum(green_energy_produced_by_country_group.values()))
    plt.plot(years, y_green, label="Green")
    plt.plot(years, y_green + ys["Net Zero 2050"], label="Green + Brown")
    plt.plot(years, y_deltap, label="DeltaQ")
    plt.plot(years, y_deltap + ys["Net Zero 2050"], label="DeltaQ + Brown")
    plt.legend()
    plt.savefig("plots/electricity_generation.png")
    plt.close()
    analysis_main.MEASURE_GLOBAL_VARS = False
    analysis_main.MEASURE_GLOBAL_VARS_SCENARIO = "Net Zero 2050"
    with_learning.VERBOSE_ANALYSIS = False
    util.CARBON_BUDGET_CONSISTENT = False


if 1:
    plot_electricity_generation()
    exit()
