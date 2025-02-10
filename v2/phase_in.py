import glob
import os
import json
import pathlib
import sys

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pycountry

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import util  # noqa

# Function to read JSON files and convert kW to GW
scenario = "Net Zero 2050"


def read_json_files(file_pattern, keyword):
    data = {}
    for file in glob.glob(file_pattern):
        country_code = os.path.basename(file).split(
            f"battery_yearly_{keyword}_capacity_{scenario}_"
        )
        country_code = country_code[1].replace(".json", "")
        with open(file, "r") as f:
            json_data = json.load(f)
            # Convert kW to GW
            # IMPORTANT: battery short is the only once which stock in GJ, the rest is in kW
            # GJ to kWh, then divide by seconds in 1 day to kW, then to GW
            unit_conversion_battery_short = (util.GJ2MWh(1) * 1000) / (24 * 3600) / 1e6
            for key, value in json_data.items():
                if key == "short":
                    json_data[key] = {
                        int(year): val * unit_conversion_battery_short
                        for year, val in value.items()
                    }
                else:
                    json_data[key] = {
                        int(year): val / 1e6 for year, val in value.items()
                    }
        data[country_code] = json_data
    return data


# Read the JSON files
installed_capacity_all = read_json_files(
    f"plots/phase_in/battery_yearly_installed_capacity_{scenario}_*.json", "installed"
)
available_capacity_all = read_json_files(
    f"plots/phase_in/battery_yearly_available_capacity_{scenario}_*.json", "available"
)
print(list(installed_capacity_all.keys()))


def do_plot(ax, region_name, ylabel=None):
    ax.set_title(f"{region_name}", fontsize=12, fontweight="bold", pad=10)
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
    installed_capacity = installed_capacity_all[region_key]
    available_capacity = available_capacity_all[region_key]
    years = list(installed_capacity["solar"].keys())
    techs = list(installed_capacity.keys())
    values = [sum(installed_capacity[tech][y] for tech in techs) for y in years]
    # We start from 0 because the sum goes only up to t-1 for a given year t, see equation 15 in GCA.
    cumulative = np.cumsum([0] + values[:-1])
    # Already cumulative
    values_available = [
        sum(available_capacity[tech][y] for tech in techs) for y in years
    ]
    ax.plot(years, values, color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(years, cumulative, color="tab:green", linestyle="--")
    ax2.plot(years, values_available, color="tab:orange", linestyle="dashdot")
    if ylabel == "left":
        ax.set_ylabel("Annual (GW)")
    elif ylabel == "right":
        ax2.set_ylabel("Cumulative (GW)")


tech_colors = {
    "solar": "tab:red",
    "onshore_wind": "tab:orange",
    "offshore_wind": "tab:green",
    "hydropower": "tab:blue",
    "geothermal": "tab:brown",
    "short": "tab:pink",
    "long": "tab:olive",
}


def do_plot_split_energy(ax, region_name, ylabel=None):
    ax.set_title(f"{region_name}", fontsize=12, fontweight="bold", pad=10)
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
    installed_capacity = installed_capacity_all[region_key]
    years = list(installed_capacity["solar"].keys())

    ax2 = ax.twinx()
    for tech, ic in installed_capacity.items():
        val = list(ic.values())
        cumulative = np.cumsum(val)
        color = tech_colors[tech]
        ax.plot(years, val, color=color)
        ax2.plot(years, cumulative, color=color, linestyle="--")
    if ylabel == "left":
        ax.set_ylabel("Annual (GW)")
    elif ylabel == "right":
        ax2.set_ylabel("Cumulative (GW)")


fig, axes = plt.subplots(1, 3, figsize=(12, 5))
do_plot(axes[0], "Global", ylabel="left")
do_plot(axes[1], "Developed Countries")
do_plot(axes[2], "Developing Countries", ylabel="right")
plt.tight_layout()
plt.savefig("plots/phase_in/phase_in_combined.png")
plt.close()

# Second Graph: 8 Countries
fig = plt.figure(figsize=(12, 6))  # Adjusted figure size for grid layout
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])
do_plot(fig.add_subplot(gs[0, 0]), "India", ylabel="left")
do_plot(fig.add_subplot(gs[0, 1]), "Indonesia")
do_plot(fig.add_subplot(gs[0, 2]), "South Africa")
do_plot(fig.add_subplot(gs[0, 3]), "Mexico", ylabel="right")
do_plot(fig.add_subplot(gs[1, 0]), "Viet Nam", ylabel="left")
do_plot(fig.add_subplot(gs[1, 1]), "Iran")
do_plot(fig.add_subplot(gs[1, 2]), "Thailand")
do_plot(fig.add_subplot(gs[1, 3]), "Egypt", ylabel="right")
# Legend
solid_line = Line2D([0], [0], color="tab:blue", lw=2, linestyle="-")
dashed_line = Line2D([0], [0], color="tab:green", lw=2, linestyle="--")
dashed_line_available = Line2D([0], [0], color="tab:orange", lw=2, linestyle="dashdot")
fig.legend(
    [solid_line, dashed_line, dashed_line_available],
    [
        "Annual Capacity (Left axis)",
        "Cumulative Capacity (Right axis)",
        "Cumulative Operational Capacity (Right axis)",
    ],
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.1),
    fontsize=10,
    frameon=False,
)
plt.subplots_adjust(bottom=0.01, hspace=0.4, wspace=0.6)
plt.savefig("plots/phase_in/phase_in_8countries.png", bbox_inches="tight")
plt.close()


# Now, same plot as above, except it is split by energy type
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
do_plot_split_energy(axes[0], "Global", ylabel="left")
do_plot_split_energy(axes[1], "Developed Countries")
do_plot_split_energy(axes[2], "Developing Countries", ylabel="right")
plt.tight_layout()
plt.savefig("plots/phase_in/phase_in_combined_split.png")
plt.close()

# Second Graph: 8 Countries
fig = plt.figure(figsize=(12, 6))  # Adjusted figure size for grid layout
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])
do_plot_split_energy(fig.add_subplot(gs[0, 0]), "India", ylabel="left")
do_plot_split_energy(fig.add_subplot(gs[0, 1]), "Indonesia")
do_plot_split_energy(fig.add_subplot(gs[0, 2]), "South Africa")
do_plot_split_energy(fig.add_subplot(gs[0, 3]), "Mexico", ylabel="right")
do_plot_split_energy(fig.add_subplot(gs[1, 0]), "Viet Nam", ylabel="left")
do_plot_split_energy(fig.add_subplot(gs[1, 1]), "Iran")
do_plot_split_energy(fig.add_subplot(gs[1, 2]), "Thailand")
do_plot_split_energy(fig.add_subplot(gs[1, 3]), "Egypt", ylabel="right")
# 2 Legends
lines = []
labels = []
tech_names = {
    "solar": "Solar",
    "onshore_wind": "Wind onshore",
    "offshore_wind": "Wind offshore",
    "hydropower": "Hydropower",
    "geothermal": "Geothermal",
    "short": "Short-term storage (Li-ion batteries)",
    "long": "Long-term storage (Electrolyzers)",
}
for tech, color in tech_colors.items():
    lines.append(Line2D([0], [0], color=color, lw=2, linestyle="-"))
    labels.append(tech_names[tech])
fig.legend(
    lines,
    labels,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.1),
    fontsize=10,
    frameon=False,
)

solid_line = Line2D([0], [0], color="black", lw=2, linestyle="-")
dashed_line = Line2D([0], [0], color="black", lw=2, linestyle="--")
dashed_line_available = Line2D([0], [0], color="black", lw=2, linestyle="dashdot")
fig.legend(
    [solid_line, dashed_line, dashed_line_available],
    [
        "Annual Capacity (Left axis)",
        "Cumulative Capacity (Right axis)",
        "Cumulative Operational Capacity (Right axis)",
    ],
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.2),
    fontsize=10,
    frameon=False,
)

plt.subplots_adjust(bottom=0.01, hspace=0.4, wspace=0.6)
plt.savefig("plots/phase_in/phase_in_8countries_split.png", bbox_inches="tight")
plt.close()
