import matplotlib.pyplot as plt
import numpy as np

import util

irena = util.read_json("data/irena.json")

hours_in_1year = 24 * 365.25
one_hour = 3600
years = range(2010, 2020 + 1)


def kW_to_GJ(x):
    # kW to W
    W = x * 1e3
    # W to J
    J = W * hours_in_1year * one_hour
    # J to GJ
    GJ = J / 1e9
    return GJ


def do_plot_installed_cost():
    # plt.figure()
    plt.plot(
        years,
        irena["installed_cost_solar_2010_2020_$/kW"],
        label="Solar",
    )
    plt.plot(
        years,
        irena["installed_cost_onshore_wind_2010_2020_$/kW"],
        label="Wind onshore",
    )
    plt.plot(
        years,
        irena["installed_cost_offshore_wind_2010_2020_$/kW"],
        label="Wind offshore",
    )
    plt.xlabel("Time")
    plt.ylabel("2020 USD/kW")
    # plt.legend()
    # plt.savefig("plots/experience_curve_kW.png")


fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plt.sca(axs[0])
do_plot_installed_cost()
util.set_integer_xaxis()
plt.sca(axs[1])

years_capacity = range(2011, 2020 + 1)
mul = 1 / 1e3
plt.plot(
    years_capacity,
    np.array(irena["solar_MW_world_cumulative_total_installed_capacity_2011_2020"])
    * mul,
    label="Solar",
)
plt.plot(
    years_capacity,
    np.array(
        irena["onshore_wind_MW_world_cumulative_total_installed_capacity_2011_2020"]
    )
    * mul,
    label="Wind onshore",
)
plt.plot(
    years_capacity,
    np.array(
        irena["offshore_wind_MW_world_cumulative_total_installed_capacity_2011_2020"]
    )
    * mul,
    label="Wind offshore",
)
plt.xlabel("Time")
plt.ylabel("Cumulative installed capacity (GW)")
# Deduplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# See https://stackoverflow.com/a/43439132
# To auto-scale the legend:
# 1. Do fig.legend(), with loc="center left", bbox_to_anchor=(0.9, 0.5)
# 2. In savefig, do bbox_inches="tight"
fig.legend(
    by_label.values(),
    by_label.keys(),
    loc="upper center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=len(by_label.values()),
)
plt.tight_layout()
plt.savefig("plots/irena_cumulative_installed_capacity.png", bbox_inches="tight")
