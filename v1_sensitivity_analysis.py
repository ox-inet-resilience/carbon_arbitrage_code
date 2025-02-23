import matplotlib.pyplot as plt
import numpy as np

import analysis_main
import util
import with_learning


def make_carbon_arbitrage_opportunity_plot(relative_to_world_gdp=False):
    from collections import defaultdict

    global social_cost_of_carbon
    social_costs = np.linspace(0, 200, 3)
    ydict = defaultdict(list)
    chosen_scenario = f"{analysis_main.NGFS_PEG_YEAR}-2100 FA + Net Zero 2050 Scenario"
    for social_cost in social_costs:
        util.social_cost_of_carbon = social_cost
        social_cost_of_carbon = social_cost
        out = analysis_main.run_table1(to_csv=False, do_round=False)
        carbon_arbitrage_opportunity = out[
            "Carbon arbitrage including residual benefit (in trillion dollars)"
        ]
        for scenario, value in carbon_arbitrage_opportunity.items():
            if scenario != chosen_scenario:
                continue
            if relative_to_world_gdp:
                value = value / util.world_gdp_2023 * 100
            ydict[scenario].append(value)
    mapper = {
        f"{analysis_main.NGFS_PEG_YEAR}-{analysis_main.LAST_YEAR} FA + Current Policies  Scenario": f"s2=0, T={analysis_main.LAST_YEAR}",
        f"{analysis_main.NGFS_PEG_YEAR}-2100 FA + Current Policies  Scenario": "s2=0, T=2100",
        f"{analysis_main.NGFS_PEG_YEAR}-{analysis_main.LAST_YEAR} FA + Net Zero 2050 Scenario": f"s2=Net Zero 2050, T={analysis_main.LAST_YEAR}",
        f"{analysis_main.NGFS_PEG_YEAR}-2100 FA + Net Zero 2050 Scenario": "s2=Net Zero 2050, T=2100",
    }

    # Find the intersect with the x axis
    from scipy.stats import linregress

    social_cost_zeros = {}
    for scenario, values in ydict.items():
        slope, intercept, r_value, p_value, std_err = linregress(social_costs, values)
        sc_zero = -intercept / slope
        print("Social cost when zero of", mapper[scenario], sc_zero)
        print("  r value", r_value)
        social_cost_zeros[scenario] = sc_zero

    plt.figure()
    for scenario, values in ydict.items():
        plt.plot(social_costs, values, label=mapper[scenario])
    plt.axhline(0, color="gray", linewidth=1)  # So that we can see the zeros

    # Vertical lines
    from matplotlib.pyplot import text

    vertical_lines = {
        51: "Biden administration, 51 $/tC02",
        61.4: "Lower estimate, Rennert et al. (2021), 61.4 $/tC02",
        80: "Pindyck (2019), 80 $/tCO2",
        114.9: "Mid estimate, Rennert et al. (2021), 114.9 $/tC02",
        168.4: "Upper estimate, Rennert et al. (2021), 168.4 $/tC02",
    }
    ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    # y_mid = (y_min + y_max) / 2
    for x, text_content in vertical_lines.items():
        plt.axvline(x, color="gray", linestyle="dashed")
        text(
            x - 7,
            0.98 * y_max,
            text_content,
            color="gray",
            rotation=90,
            verticalalignment="top",
            fontdict={"size": 10},
        )

    # Marker at intersection with zero
    x_intersect = social_cost_zeros[chosen_scenario]
    plt.plot(x_intersect, 0, "o", color="tab:blue")
    text(
        x_intersect,
        32,
        f"{x_intersect:.1f} $/tC02",
        color="gray",
        verticalalignment="center",
        horizontalalignment="center",
        fontdict={"size": 10},
    )
    # plt.legend()
    plt.xlabel("Social cost of carbon (dollars/tCO2)")
    if relative_to_world_gdp:
        print("Relative to 2023 world GDP")
        plt.ylabel("Carbon Arbitrage relative to 2023 World GDP (%)")
    else:
        plt.ylabel("Carbon Arbitrage (trillion dollars)")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    suffix = "_relative" if relative_to_world_gdp else ""
    plt.savefig(f"plots/carbon_arbitrage_opportunity{suffix}.png")


def make_yearly_climate_financing_plot_SENSITIVITY_ANALYSIS():
    chosen_s2_scenario = (
        f"{analysis_main.NGFS_PEG_YEAR}-2100 FA + Net Zero 2050 Scenario"
    )

    whole_years = range(analysis_main.NGFS_PEG_YEAR, 2100 + 1)

    def calculate_yearly_world_cost(s2_scenario):
        yearly_costs_dict = analysis_main.calculate_yearly_info_dict(s2_scenario)
        # Calculating the cost for the whole world
        yearly_world_cost = np.zeros(len(whole_years))
        for v in yearly_costs_dict.values():
            yearly_world_cost += np.array(v)
        return yearly_world_cost

    def _get_year_range_cost(year_start, year_end, yearly_world_cost):
        return sum(
            yearly_world_cost[
                year_start - analysis_main.NGFS_PEG_YEAR : year_end
                + 1
                - analysis_main.NGFS_PEG_YEAR
            ]
        )

    label_map = {
        "30Y": "30Y, D, E",
        "30Y_noE": "30Y, D, no E",
        "50Y": "50Y, D, E",
        "200Y": "Lifetime\nby D, E",
    }

    def reset():
        with_learning.ENABLE_WRIGHTS_LAW = 1
        with_learning.RENEWABLE_LIFESPAN = 30

    data_for_barchart = {
        (analysis_main.NGFS_PEG_YEAR + 1, 2050): {},
        (2051, 2070): {},
        (2071, 2100): {},
    }
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(axs[0])
    for key, label in label_map.items():
        reset()
        if key.endswith("Y"):
            with_learning.RENEWABLE_LIFESPAN = int(key[:-1])
        else:
            assert key == "30Y_noE"
            with_learning.ENABLE_WRIGHTS_LAW = 0
        linestyle = "-" if key == "30Y" else "dotted"
        yearly = calculate_yearly_world_cost(chosen_s2_scenario, discounted=False)
        plt.plot(
            whole_years,
            yearly,
            label=label,
            linestyle=linestyle,
            linewidth=2.5,
        )

        yearly_discounted = calculate_yearly_world_cost(chosen_s2_scenario)
        for year_start, year_end in [
            (analysis_main.NGFS_PEG_YEAR + 1, 2050),
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

    # Bar chart
    plt.sca(axs[1])
    xticks = None
    stacked_bar_data = []
    for year_pair, data in data_for_barchart.items():
        xticks = list(data.keys())
        stacked_bar_data.append((f"{year_pair[0]}-{year_pair[1]}", list(data.values())))
    util.plot_stacked_bar(
        xticks,
        stacked_bar_data,
    )
    plt.xticks(xticks, rotation=45, ha="right")
    plt.ylabel("PV global climate financing\n(trillion dollars)")
    plt.legend(
        loc="upper left",
    )
    plt.tight_layout()
    util.savefig("climate_financing_sensitivity", tight=True)


if __name__ == "__main__":
    make_carbon_arbitrage_opportunity_plot()
    exit()
    make_carbon_arbitrage_opportunity_plot(relative_to_world_gdp=True)
