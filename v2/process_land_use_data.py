import math
import pathlib
import sys
from collections import defaultdict

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

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


def make_4_panel_plot(land_use):
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 15), sharex=True)

    # Panel 1: "Total renewable space"
    land_use_all = sum(land_use.values())
    axes[0].plot(land_use_all, y_data_total_renewable)
    axes[0].set_title("Total renewable space")
    axes[0].set_ylabel("% of country's land")
    axes[0].set_xscale("log")  # log scale for x-axis
    # Note: x-axis label could be omitted here if you're sharing x-axis across subplots
    # but let's put the main x-axis label on the bottom-most subplot

    # Panel 2: "Solar space"
    axes[1].plot(x_data, y_data_solar)
    axes[1].set_title("Solar space")
    axes[1].set_ylabel("% of country's land")
    axes[1].set_xscale("log")

    # Panel 3: "Wind onshore space"
    axes[2].plot(x_data, y_data_wind_onshore)
    axes[2].set_title("Wind onshore space")
    axes[2].set_ylabel("% of country's land")
    axes[2].set_xscale("log")

    # Panel 5: "Wind offshore space"
    # (Normally you'd list this as panel 4, but following the given numbering)
    axes[3].plot(x_data, y_data_wind_offshore)
    axes[3].set_title("Wind offshore space")
    axes[3].set_ylabel("% of country's land")
    axes[3].set_xscale("log")

    # Panel 4: "Hydro space"
    axes[4].plot(x_data, y_data_hydro)
    axes[4].set_title("Hydro space")
    axes[4].set_xlabel("Country's land (m^2) [log scale]")
    axes[4].set_ylabel("% of country's land")
    axes[4].set_xscale("log")


world_power_density_mw = {
    k: v * get_capacity_factor(k, "US") * hours_in_1year
    for k, v in us_power_density_mwh.items()
}

landlocked_countries = pd.read_csv(
    "./data/landlocked_countries_wikipedia.csv", header=None
)
landlocked_countries_alpha2 = landlocked_countries[0].apply(get_alpha2).tolist()
print(landlocked_countries_alpha2)

energy_types = list(world_power_density_mw.keys())
land_use_all_countries = {}

alpha2s = "DE ID IN KZ PL TR US VN".split()
alpha2s = "EG IN ID ZA MX VN IR TH TR BD NL".split()
for alpha2 in alpha2s:
    country_name = pycountry.countries.get(alpha_2=alpha2).name
    if alpha2 == "IR":
        # This is the name from FAOSTAT data
        country_name = "Iran (Islamic Republic of)"
    df_land_use_country = df_land_use[df_land_use.Area == country_name]
    # In m^2
    territorial_area_country = territorial_area[alpha2_to_alpha3[alpha2]] * 1e6
    if alpha2 == "IR":
        # The name provided by FAOSTAT is too long.
        country_name = "Iran"
    available_land = defaultdict(float)
    available_land_two_ticks = defaultdict(float)
    for i, row in df_land_use_country.iterrows():
        assert row["Unit"] == "1000 ha"
        value = row["Value"] * 1e3 * 1e4
        match row["Item"]:
            case "Woody crops":
                # In m^2
                # 1 ha is 1e4 m^2
                available_land["solar"] += value
            case (
                "Herbaceous crops"
                | "Grassland"
                | "Shrub-covered areas"
                | "Sparsely natural vegetated areas"
                | "Terrestrial barren land"
            ):
                available_land["solar_and_onshore_wind"] += value
            case "Inland water bodies":
                available_land["hydropower"] += value

        match row["Item"]:
            case "Herbaceous crops" | "Grassland" | "Sparsely natural vegetated areas":
                available_land_two_ticks["onshore_wind"] += value
            case "Terrestrial barren land":
                available_land_two_ticks["solar_and_onshore_wind"] += value
            case "Inland water bodies":
                available_land_two_ticks["hydropower"] += value

    # Power density m^2/MWh
    # Wind off shore 43.40503846 (Zalk & Beheren)
    # m2/MW
    # 102731.9131278972
    eez_area = eez.get(alpha2, 0) * 1e6  # m^2

    # in kW
    yearly_installed_capacity = util.read_json(
        f"./plots/battery_yearly_available_capacity_Net Zero 2050_{alpha2}.json"
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
            if energy_type == "hydropower":
                if land_use > available_land[energy_type]:
                    excess_capacity_mw = (
                        land_use - available_land[energy_type]
                    ) / world_power_density_mw
                    excess_capacity_mwh = excess_capacity_mw * hours_in_1year * cf
                    # Truncate actual land use to not exceed available land
                    land_use_adjusted = available_land[energy_type]
                    installed_capacity_all_other = sum(
                        yearly_installed_capacity[energy_type][str(year)]
                        for energy_type in energy_types
                        if energy_type != "hydropower"
                    )
                    for et_other in energy_types:
                        if et_other == "hydropower":
                            continue
                        installed_capacity_other = yearly_installed_capacity[et_other][
                            str(year)
                        ]
                        cf_other = get_capacity_factor(et_other, alpha2)
                        fraction_other = (
                            with_learning.get_renewable_weight(et_other, alpha2)
                            / (
                                1
                                - with_learning.get_renewable_weight(
                                    "hydropower", alpha2
                                )
                            )
                        ) * (excess_capacity_mwh / (hours_in_1year * cf_other))
                        land_use_reallocation = (
                            fraction_other
                            * country_power_density_mwh[et_other]
                            * hours_in_1year
                        )
                        land_use_yearly_all_energies_adjusted[et_other][year] += (
                            land_use_reallocation
                        )
            land_use_yearly_all_energies[energy_type].append(land_use)
            land_use_yearly_all_energies_adjusted[energy_type][year] += (
                land_use_adjusted
            )

    # Convert values to NumPy arrays
    for k, v in land_use_yearly_all_energies.items():
        land_use_yearly_all_energies[k] = np.array(v)
    for k, v in land_use_yearly_all_energies_adjusted.items():
        land_use_yearly_all_energies_adjusted[k] = np.array(list(v.values()))

    plt.figure()
    # solar and onshore wind
    plt.plot(
        years,
        land_use_yearly_all_energies["solar"]
        + land_use_yearly_all_energies["onshore_wind"],
        label="Solar & wind onshore",
        color="tab:blue",
    )
    plt.plot(
        years,
        land_use_yearly_all_energies_adjusted["solar"]
        + land_use_yearly_all_energies_adjusted["onshore_wind"],
        color="tab:blue",
        linestyle="dashdot",
    )
    plt.axhline(
        available_land_two_ticks["onshore_wind"]
        + available_land_two_ticks["solar_and_onshore_wind"],
        linestyle="dotted",
        color="tab:blue",
    )

    plt.plot(
        years,
        land_use_yearly_all_energies["hydropower"],
        label="Hydropower",
        color="tab:orange",
    )
    plt.plot(
        years,
        land_use_yearly_all_energies_adjusted["hydropower"],
        color="tab:orange",
        linestyle="dashdot",
    )
    plt.axhline(
        available_land["hydropower"],
        linestyle="dotted",
        color="tab:orange",
    )

    plt.plot(
        years,
        land_use_yearly_all_energies["offshore_wind"],
        label="Wind Offshore",
        color="tab:red",
    )
    plt.plot(
        years,
        land_use_yearly_all_energies_adjusted["offshore_wind"],
        color="tab:red",
        linestyle="dashdot",
    )
    plt.axhline(
        eez_area,
        linestyle="dotted",
        color="tab:red",
    )

    # Add dual y axis
    def percentage2area(y):
        return y / 100 * territorial_area_country

    def area2percentage(y):
        return y / territorial_area_country * 100

    ax2 = plt.gca().secondary_yaxis(
        "right", functions=(area2percentage, percentage2area)
    )
    ax2.set_ylabel("Cumulative land use/Total land (%)")

    # print(
    #     "???",
    #     alpha2,
    #     eez_area,
    #     available_land["hydropower"],
    #     land_use_yearly_all_energies["offshore_wind"][-1],
    # )
    # for energy_type, land_use in land_use_yearly_all_energies.items():
    #     plt.plot(years, np.cumsum(land_use), label=energy_type)
    # available_land_summed = sum(available_land.values())
    # plt.axhline(available_land_summed, color="black", label="Available land")
    plt.xlabel("Time (years)")
    plt.ylabel("Cumulative land use ($m^2$)")
    if alpha2 == "BD":
        plt.legend()
    plt.title(country_name)

    # plt.gca().set_yscale("log")
    plt.savefig(f"plots/land_use_{alpha2}.png")
    land_use_all_countries[alpha2] = land_use_yearly_all_energies
# make_4_panel_plot(land_use_all_countries)
