import pathlib
import sys
from collections import defaultdict

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
from adjustText import adjust_text
from matplotlib.lines import Line2D

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import analysis_main  # noqa
import util  # noqa
import with_learning  # noqa

# solar_and_onshore_wind: Herbaceous crops, Grassland, Shrub-covered areas, Sparsely natural vegetated areas, Terrestrial barren land
# Solar: Woody crops
# Wind Offshore: Inland water bodies
# Hydropower: Inland water bodies
# In terms of inland water bodies, hydropower is more appropriate than wind offshore.

# Data source: https://www.fao.org/faostat/en/#data/LC (picked year 2022)
df_land_use = pd.read_csv("./data/FAOSTAT_data_en_11-4-2024.csv.gz", compression="gzip")

# territorial_area = df_land_use.groupby("Area")["Value"].sum() * 1e3 * 1e4
# Taken from https://data.worldbank.org/indicator/AG.LND.TOTL.K2
# Last Updated Date 2024-12-16
# In km^2
territorial_area = (
    pd.read_csv(
        "./data/API_AG.LND.TOTL.K2_DS2_en_csv_v2_1036.csv",
        usecols=["Country Code", "2022"],
    )
    .set_index("Country Code")["2022"]
    .to_dict()
)
iso3166_df = util.read_iso3166()
alpha2_to_alpha3 = iso3166_df.set_index("alpha-2")["alpha-3"].to_dict()
alpha2_to_alpha3["XK"] = "XKK"

# m^2 / MWh
us_power_density_mwh = {
    "solar": 19.54713204,  # Zalk & Beheren
    "hydropower": 33,  # Our World in Data
    # "hydropower_large": 14,  # (Our World in Data)
    "onshore_wind": 56.51250057,  # Zalk & Beheren
    "offshore_wind": 43.40503846,  # Zalk & Beheren
}
hours_in_1year = 365.25 * 24


# Create a mapping of country names to alpha-2 codes
def get_alpha2(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_2
    except LookupError:
        match country_name:
            case "Kosovo":
                return "XK"
            case "Vatican City":
                return "VA"
        return None  # Handle cases where the name isn't matched


def get_capacity_factor(tech, alpha2):
    return with_learning.get_capacity_factor(tech, alpha2)


# Load the country EEZ data
if 0:
    # Source: https://marineregions.org/downloads.php
    eez_gdf = gpd.read_file("./data/World_24NM_v4_20231025/eez_24nm_v4.shp")
    eez_gdf["alpha2"] = eez_gdf["SOVEREIGN1"].apply(get_alpha2)
    # Merge all the polygons of each country into 1
    eez_dissolved = eez_gdf.dissolve(by="SOVEREIGN1").reset_index()
    # Project the EEZs to an equal-area projection
    eez_truncated_projected = eez_dissolved.to_crs(epsg=6933)
    eez_truncated_projected["area_km2"] = eez_truncated_projected.geometry.area / 1e6
    eez = eez_truncated_projected.set_index("alpha2")["area_km2"]
    eez.to_csv("./plots/eez_area.csv")
    exit()
eez = pd.read_csv("./plots/eez_area.csv").set_index("alpha2")["area_km2"].to_dict()


def make_4_panel_plot(
    land_use, percent_land_use, country_names, exclude_from_annotation
):
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 15), sharex=True)
    # Only use the last year value
    for a2, v in land_use.items():
        for energy_type, vv in v.items():
            land_use[a2][energy_type] = vv[-1]

    # Panel 1: "Total renewable space"
    land_use_all = [
        sum(v for k, v in d.items() if k != "wind_offshore") for d in land_use.values()
    ]
    percent_land_use_all = [
        sum(v for k, v in d.items() if k != "wind_offshore")
        for d in percent_land_use.values()
    ]
    axes[0].scatter(land_use_all, percent_land_use_all)
    axes[0].set_title("Total renewable space  (excl. offshore wind)")
    plt.sca(axes[0])
    # annotate with alpha2's
    texts = []
    for x, y, s in zip(land_use_all, percent_land_use_all, country_names):
        if s in exclude_from_annotation:
            # No need to annotate the developing countries
            continue
        texts.append(plt.text(x, y, s))
    adjust_text(
        texts,
        only_move={"points": "y", "texts": "y"},
        arrowprops=dict(arrowstyle="->", color="r", lw=0.5),
    )
    # for i, txt in enumerate(country_names):
    #     axes[0].annotate(txt, (land_use_all[i], percent_land_use_all[i]))

    # Note: x-axis label could be omitted here if you're sharing x-axis across subplots
    # but let's put the main x-axis label on the bottom-most subplot

    def _do_plot(energy_type, title, ax):
        land_use_energy_type = [d[energy_type] for d in land_use.values()]
        percent_land_use_energy_type = [
            d[energy_type] for d in percent_land_use.values()
        ]
        ax.scatter(land_use_energy_type, percent_land_use_energy_type)
        plt.sca(ax)
        # annotate with alpha2's
        texts = []
        for x, y, s in zip(
            land_use_energy_type, percent_land_use_energy_type, country_names
        ):
            if s in exclude_from_annotation:
                # No need to annotate the developing countries
                continue
            texts.append(plt.text(x, y, s))
        adjust_text(
            texts,
            only_move={"points": "y", "texts": "y"},
            arrowprops=dict(arrowstyle="->", color="r", lw=0.5),
        )
        # for i, txt in enumerate(country_names):
        #     ax.annotate(txt, (land_use_energy_type[i], percent_land_use_energy_type[i]))
        ax.set_title(title)

    _do_plot("solar", "Solar space", axes[1])
    _do_plot("onshore_wind", "Wind onshore space", axes[2])
    _do_plot("hydropower", "Hydro space", axes[3])
    _do_plot("offshore_wind", "Wind offshore space", axes[4])

    for i in range(5):
        axes[i].set_ylabel("% of country's land")
        # axes[i].set_xscale("log")
    axes[4].set_xlabel("Country's land use ($m^2$)")
    plt.savefig("plots/land_use_combined.png")


world_power_density_mw = {
    k: v * get_capacity_factor(k, "US") * hours_in_1year
    for k, v in us_power_density_mwh.items()
}

# landlocked_countries = pd.read_csv(
#     "./data/landlocked_countries_wikipedia.csv", header=None
# )
# landlocked_countries_alpha2 = landlocked_countries[0].apply(get_alpha2).tolist()

energy_types = list(world_power_density_mw.keys())
land_use_all_countries = {}
percent_land_use_all_countries = {}
country_names = []

exclude_from_annotation = []


def process_1_country(alpha2, ax, ylabel=None):
    if alpha2 in "CI ET GN CD HT MW NP SD BT BI KM GM GW LS LR ST PS".split():
        # Forward analytics doesn't have these countries
        return
    if alpha2 in ["EH"]:
        # Excluded because it causes error when saving the figure
        return
    if alpha2 == "XK":
        country_name = "Kosovo"
    else:
        country_name = pycountry.countries.get(alpha_2=alpha2).name
    if alpha2 == "IR":
        # This is the name from FAOSTAT data
        country_name = "Iran (Islamic Republic of)"
    df_land_use_country = df_land_use[df_land_use.Area == country_name]
    # In m^2
    territorial_area_country = territorial_area.get(alpha2_to_alpha3[alpha2], 0) * 1e6
    if alpha2 == "IR":
        # The name provided by FAOSTAT is too long.
        country_name = "Iran"

    if alpha2 in with_learning.DEVELOPING_UNFCCC:
        exclude_from_annotation.append(country_name)

    available_land = defaultdict(float)
    available_land_two_ticks = defaultdict(float)
    for i, row in df_land_use_country.iterrows():
        assert row["Unit"] == "1000 ha"
        value = row["Value"] * 1e3 * 1e4
        # Disable 1 tick for now
        # match row["Item"]:
        #     case "Woody crops":
        #         # In m^2
        #         # 1 ha is 1e4 m^2
        #         available_land["solar"] += value
        #     case (
        #         "Herbaceous crops"
        #         | "Grassland"
        #         | "Shrub-covered areas"
        #         | "Sparsely natural vegetated areas"
        #         | "Terrestrial barren land"
        #     ):
        #         available_land["solar_and_onshore_wind"] += value
        #     case "Inland water bodies":
        #         available_land["hydropower"] += value

        match row["Item"]:
            case "Herbaceous crops" | "Grassland" | "Sparsely natural vegetated areas":
                available_land_two_ticks["onshore_wind"] += value
            case "Terrestrial barren land":
                available_land_two_ticks["solar"] += value
            case "Inland water bodies":
                available_land_two_ticks["hydropower"] += value

    # Power density m^2/MWh
    # Wind off shore 43.40503846 (Zalk & Beheren)
    # m2/MW
    # 102731.9131278972
    eez_area = eez.get(alpha2, 0) * 1e6  # m^2

    # in kW
    yearly_installed_capacity = util.read_json(
        f"./plots/phase_in/battery_yearly_available_capacity_Net Zero 2050_{alpha2}.json"
    )
    country_power_density_mwh = {
        k: v / (get_capacity_factor(k, alpha2) * hours_in_1year)
        for k, v in world_power_density_mw.items()
    }
    land_use_yearly_all_energies = defaultdict(list)
    land_use_yearly_all_energies_adjusted = defaultdict(lambda: defaultdict(float))
    years = list(range(2024, 2050 + 1))
    for year in years:
        for energy_type in energy_types:
            installed_capacity_kw = yearly_installed_capacity[energy_type][str(year)]
            installed_capacity_mw = installed_capacity_kw / 1e3
            country_pd_mwh = country_power_density_mwh[energy_type]
            cf = get_capacity_factor(energy_type, alpha2)
            land_use = installed_capacity_mw * world_power_density_mw[energy_type]
            land_use_adjusted = land_use
            # Don't reallocate for now
            # if energy_type == "hydropower":
            #     if land_use > available_land_two_ticks[energy_type]:
            #         excess_capacity_mw = (
            #             land_use - available_land_two_ticks[energy_type]
            #         ) / world_power_density_mw[energy_type]
            #         excess_capacity_mwh = excess_capacity_mw * hours_in_1year * cf
            #         # Truncate actual land use to not exceed available land
            #         land_use_adjusted = available_land_two_ticks[energy_type]
            #         installed_capacity_all_other = sum(
            #             yearly_installed_capacity[energy_type][str(year)]
            #             for energy_type in energy_types
            #             if energy_type != "hydropower"
            #         )
            #         for et_other in energy_types:
            #             if et_other == "hydropower":
            #                 continue
            #             installed_capacity_other = yearly_installed_capacity[et_other][
            #                 str(year)
            #             ]
            #             cf_other = get_capacity_factor(et_other, alpha2)
            #             fraction_other = (
            #                 with_learning.get_renewable_weight(et_other, alpha2)
            #                 / (
            #                     1
            #                     - with_learning.get_renewable_weight(
            #                         "hydropower", alpha2
            #                     )
            #                 )
            #             ) * (excess_capacity_mwh / (hours_in_1year * cf_other))
            #             land_use_reallocation = (
            #                 fraction_other
            #                 * country_power_density_mwh[et_other]
            #                 * hours_in_1year
            #             )
            #             land_use_yearly_all_energies_adjusted[et_other][year] += (
            #                 land_use_reallocation
            #             )
            land_use_yearly_all_energies[energy_type].append(land_use)
            land_use_yearly_all_energies_adjusted[energy_type][year] += (
                land_use_adjusted
            )

    # Convert values to NumPy arrays
    for k, v in land_use_yearly_all_energies.items():
        land_use_yearly_all_energies[k] = np.array(v)
    for k, v in land_use_yearly_all_energies_adjusted.items():
        land_use_yearly_all_energies_adjusted[k] = np.array(list(v.values()))

    percent_land_use_all_countries[alpha2] = {}
    plt.sca(ax)

    # For dual y axis
    def percentage2area(y):
        return y / 100 * territorial_area_country

    def area2percentage(y):
        return y / territorial_area_country * 100 if territorial_area_country > 0 else 0

    for energy_type, color, label in [
        ("solar", "tab:blue", "Solar"),
        ("onshore_wind", "tab:orange", "Wind onshore"),
        ("hydropower", "tab:red", "Hydropower"),
        ("offshore_wind", "tab:green", "Wind offshore"),
    ]:
        plt.plot(
            years,
            land_use_yearly_all_energies[energy_type],
            label=label,
            color=color,
        )
        plt.plot(
            years,
            land_use_yearly_all_energies_adjusted[energy_type],
            color=color,
            linestyle="dashdot",
        )
        if energy_type == "offshore_wind":
            available = eez_area
        else:
            available = available_land_two_ticks[energy_type]
        plt.axhline(
            available,
            linestyle="dotted",
            color=color,
        )
        percent_land_use_all_countries[alpha2][energy_type] = area2percentage(available)

    # Add dual y axis
    ax2 = plt.gca().secondary_yaxis(
        "right", functions=(area2percentage, percentage2area)
    )
    if ylabel == "left":
        ax.set_ylabel("Cumulative land use ($m^2$)")
    elif ylabel == "right":
        ax2.set_ylabel("Cumulative land\nuse/Total land (%)")
    elif ylabel == "both":
        ax.set_ylabel("Cumulative land use ($m^2$)")
        ax2.set_ylabel("Cumulative land\nuse/Total land (%)")

    # plt.xlabel("Time (years)")
    if alpha2 == "BD":
        plt.legend()

    plt.title(country_name)

    ax.set_yscale("log")
    if ylabel == "both":
        plt.tight_layout()
    try:
        plt.savefig(f"plots/land_use/land_use_{alpha2}.png")
    except Exception as e:
        print("Not possible to save", alpha2, e)
    # plt.close()
    land_use_all_countries[alpha2] = land_use_yearly_all_energies_adjusted
    country_names.append(country_name)


if 0:
    fig = plt.figure(figsize=(12, 6))  # Adjusted figure size for grid layout
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

    process_1_country("IN", fig.add_subplot(gs[0, 0]), ylabel="left")
    process_1_country("ID", fig.add_subplot(gs[0, 1]))
    process_1_country("ZA", fig.add_subplot(gs[0, 2]))
    process_1_country("MX", fig.add_subplot(gs[0, 3]), ylabel="right")
    process_1_country("VN", fig.add_subplot(gs[1, 0]), ylabel="left")
    process_1_country("IR", fig.add_subplot(gs[1, 1]))
    process_1_country("TH", fig.add_subplot(gs[1, 2]))
    process_1_country("EG", fig.add_subplot(gs[1, 3]), ylabel="right")
    plt.tight_layout()
    fig.legend(
        [
            Line2D([0], [0], color=c, lw=2, linestyle="-")
            for c in ["tab:blue", "tab:orange", "tab:red", "tab:green"]
        ],
        ["Solar", "Wind onshore", "Hydropower", "Wind offshore"],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.2),
        fontsize=10,
        frameon=False,
    )
    plt.subplots_adjust(bottom=0.001, hspace=0.4, wspace=0.6)
    plt.savefig("plots/land_use_combined.png", bbox_inches="tight")

alpha2s = "DE ID IN KZ PL TR US VN".split()
alpha2s = "EG IN ID ZA MX VN IR TH TR BD".split()
alpha2s += with_learning.DEVELOPING_UNFCCC
# 15 African countries
alpha2s = "BW CI DJ GH GN KE NG RW SN SL SC TZ UG ZM ZW".split()
# All of FA countries
alpha2s = sorted(list(set(analysis_main.df_sector.asset_country.tolist())))

for alpha2 in alpha2s:
    plt.figure(figsize=(7, 4.8))
    ax = plt.gca()
    process_1_country(alpha2, ax, ylabel="both")
    plt.close()
#
# make_4_panel_plot(
#     land_use_all_countries,
#     percent_land_use_all_countries,
#     country_names,
#     exclude_from_annotation,
# )
