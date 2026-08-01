"""Self-contained script that generates plots/exp16_34_combined.png.

Extracted from analysis_data_section.py (exp 16 + exp 34 combined figure).
Depends only on util.py and the datasets it reads.

Every result is produced twice: once with the global NGFS trajectory applied to
everything (the original convention), and once with each country following the
trajectory of the NGFS region it belongs to (see compute_2dii_ngfs_over_time and
compute_2dii_ngfs_over_time_by_region).

Outputs:
- plots/exp16_{production,emissions}.png  (non-power, intermediate)
- plots/exp34_{production,emissions}.png  (power, intermediate)
- plots/for_comparison_yearly_exp16_34.json
- plots/for_comparison_yearly_exp16_34_by_development_level.json
- plots/for_comparison_yearly_exp16_34_by_region.json
- plots/exp16_34_combined.png             (the figure of interest)
- plots/for_comparison_yearly_exp16_34_regional_ngfs.json
- plots/for_comparison_yearly_exp16_34_by_development_level_regional_ngfs.json
- plots/for_comparison_yearly_exp16_34_by_region_regional_ngfs.json
- plots/exp16_34_combined_regional_ngfs.png
- plots/exp16_34_ngfs_region_vs_world.png (the two conventions, side by side)
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import util

# Ensure that plots directory exists
os.makedirs("plots", exist_ok=True)


def convert2Gtonnes(sector, x):
    if sector == "nonpower":
        # The initial unit is EJ
        # The resulting unit is in Giga tonnes of coal
        return util.GJ2coal(x)
    else:
        # The initial unit is GW
        # The resulting unit is in Giga tonnes of coal
        assert sector == "power"
        return util.GJ2coal(x * util.hours_in_1year * util.seconds_in_1hour / 1e9)


def iter_plotted_scenarios():
    """The subset of util.scenarios that ends up in the figures, in that order."""
    for scenario in util.scenarios:
        if scenario in ["Below 2Â°C", "Divergent Net Zero", "Delayed transition"]:
            # Skip this scenario
            continue
        yield scenario


def get_scenario_label(scenario):
    # Remove weird character
    label = scenario.replace("Capacity|", "").replace("Â", "")
    if label == "Nationally Determined Contributions (NDCs) ":
        label = "Nationally Determined\nContributions (NDCs)"
    return label


def get_ngfs_peg_year(scenario):
    """The year where the NGFS value is pegged to the masterdata value."""
    if scenario.replace("Capacity|", "") == "Current Policies ":
        ngfs_peg_year = 2026
    else:
        ngfs_peg_year = 2023
    # Assert the peg year to be at most the last year of masterdata.
    assert ngfs_peg_year <= 2026, ngfs_peg_year
    return ngfs_peg_year


def get_fraction_increase_over_peg_year(ngfs_row, sector, ngfs_peg_year):
    """One NGFS trajectory (a region, or the world), relative to its peg-year value.

    Returns (years, fractions) over the NGFS 5-yearly years after the peg year.
    The peg year itself is usually not one of those, so its value is linearly
    interpolated between the two NGFS years surrounding it.
    """
    ngfs_left_year, ngfs_right_year = util.get_in_between_year(ngfs_peg_year)
    ngfs_value_left = convert2Gtonnes(sector, ngfs_row[str(ngfs_left_year)])
    ngfs_value_right = convert2Gtonnes(sector, ngfs_row[str(ngfs_right_year)])
    # Do linear interpolation once
    ngfs_value_peg = (
        ngfs_value_left
        + (ngfs_peg_year - ngfs_left_year) * (ngfs_value_right - ngfs_value_left) / 5
    )

    # Get the fraction
    ngfs_years_after_peg = list(range(ngfs_right_year, 2105, 5))
    ngfs_values = [
        convert2Gtonnes(sector, ngfs_row[str(year)]) for year in ngfs_years_after_peg
    ]
    fraction_increase_over_peg_year = np.array(
        [(v / ngfs_value_peg) for v in ngfs_values]
    )
    return ngfs_years_after_peg, fraction_increase_over_peg_year


def compute_2dii_ngfs_over_time(_ngfs_global_coal, total_by_year, sector):
    """Patch the masterdata history with the NGFS scenario trajectories.

    Returns {scenario label: {"x": years, "y": values}}. The NGFS trajectories
    are the global ones; when total_by_year is a subset of the world (e.g. only
    the developed countries), the global fractional increase over the peg year
    is applied to that subset, which is the same convention as the rest of the
    codebase (see util.calculate_ngfs_fractional_increase).
    """
    out = {}
    for scenario in iter_plotted_scenarios():
        ngfs_global_coal_scenario = _ngfs_global_coal[
            _ngfs_global_coal.Scenario == scenario
        ].iloc[0]

        ngfs_peg_year = get_ngfs_peg_year(scenario)
        (
            ngfs_years_after_peg,
            fraction_increase_over_peg_year,
        ) = get_fraction_increase_over_peg_year(
            ngfs_global_coal_scenario, sector, ngfs_peg_year
        )
        rescaled_ngfs_value_after_2025 = list(
            total_by_year[(ngfs_peg_year - 2013)] * fraction_increase_over_peg_year
        )

        masterdata_years = list(range(2013, ngfs_peg_year + 1))
        patched_years = masterdata_years + ngfs_years_after_peg
        whole_range_production = (
            total_by_year[: len(masterdata_years)] + rescaled_ngfs_value_after_2025
        )
        out[get_scenario_label(scenario)] = {
            "x": patched_years,
            "y": np.array(whole_range_production),
        }
    return out


# The NGFS regions are prefixed by the model they belong to, e.g.
# "GCAM5.3_NGFS|India". The same file also has unprefixed aggregates (World,
# R5ASIA, ...) which would double count, hence the prefix filter below.
NGFS_REGION_PREFIX = f"{util.NGFS_MODEL}|"


def get_ngfs_region_map():
    """NGFS region -> alpha-2 country codes it covers."""
    region_map = util.read_json("data/NGFS_region.json")
    # Manually add the countries that the raw file leaves out but that the
    # masterdata has assets in.
    # Puerto Rico
    region_map[NGFS_REGION_PREFIX + "USA"].append("PR")
    # Hong Kong
    region_map[NGFS_REGION_PREFIX + "China"].append("HK")
    # Kosovo
    region_map[NGFS_REGION_PREFIX + "Europe_Non_EU"].append("XK")
    return region_map


def get_country_to_ngfs_region():
    return {
        country: region
        for region, countries in get_ngfs_region_map().items()
        for country in countries
    }


def get_total_by_year_by_country(coal_df, years, sector):
    """Masterdata production and emissions per year, broken down by asset country.

    Returns {mode: [Series indexed by asset_country, one per year]}, in the same
    units as the util.get_coal_*_global_*_across_years helpers, of which these
    are the per-country counterparts.
    """
    grouped = coal_df.groupby("asset_country")
    out = {"production": [], "emissions": []}
    for year in years:
        production = grouped[f"_{year}"].sum()
        emissions = (
            (coal_df[f"_{year}"] * coal_df.emissions_factor)
            .groupby(coal_df.asset_country)
            .sum()
        )
        if sector == "nonpower":
            # tonnes of coal -> Giga tonnes of coal, and the emissions_factor
            # (tonnes of CO2 per tonnes of coal) -> GtCO2.
            out["production"].append(production / 1e9)
            out["emissions"].append(emissions / 1e9)
        else:
            # MW -> Giga tonnes of coal, and the emissions_factor (tonnes of
            # CO2 per MWh) -> GtCO2.
            out["production"].append(util.MW2Gigatonnes_of_coal(production))
            out["emissions"].append(emissions * util.hours_in_1year / 1e9)
    return out


def compute_2dii_ngfs_over_time_by_region(_ngfs_coal, total_by_year_by_country, sector):
    """Same as compute_2dii_ngfs_over_time, but region by region.

    Instead of growing everything by the global NGFS fractional increase, each
    country's peg-year value is grown by the fractional increase of the NGFS
    region it belongs to, and the results are summed back up. The masterdata
    years are unaffected, so the two conventions only differ after the peg year.

    total_by_year_by_country is one Series indexed by asset_country per
    masterdata year, as returned by get_total_by_year_by_country.
    """
    country_to_region = get_country_to_ngfs_region()
    regional = _ngfs_coal[_ngfs_coal.Region.str.startswith(NGFS_REGION_PREFIX)]

    countries = set(total_by_year_by_country[0].index)
    unmapped = countries - set(country_to_region)
    assert not unmapped, f"Countries not in any NGFS region: {sorted(unmapped)}"

    out = {}
    for scenario in iter_plotted_scenarios():
        ngfs_peg_year = get_ngfs_peg_year(scenario)
        masterdata_years = list(range(2013, ngfs_peg_year + 1))
        history = [
            float(t.sum()) for t in total_by_year_by_country[: len(masterdata_years)]
        ]

        # Aggregate the peg year masterdata values by NGFS region.
        peg_by_region = defaultdict(float)
        peg_by_country = total_by_year_by_country[ngfs_peg_year - 2013]
        for country, value in peg_by_country.items():
            peg_by_region[country_to_region[country]] += value

        ngfs_years_after_peg = []
        contributions = []
        for _, ngfs_region_row in regional[regional.Scenario == scenario].iterrows():
            if ngfs_region_row.Region not in peg_by_region:
                # No masterdata asset in this NGFS region
                continue
            years, fraction = get_fraction_increase_over_peg_year(
                ngfs_region_row, sector, ngfs_peg_year
            )
            assert not ngfs_years_after_peg or years == ngfs_years_after_peg
            ngfs_years_after_peg = years
            contributions.append(peg_by_region[ngfs_region_row.Region] * fraction)
        assert contributions, f"No NGFS region matched, scenario {scenario}"
        rescaled_ngfs_value_after_peg = np.sum(contributions, axis=0)

        patched_years = masterdata_years + ngfs_years_after_peg
        out[get_scenario_label(scenario)] = {
            "x": patched_years,
            "y": np.array(history + list(rescaled_ngfs_value_after_peg)),
        }
    return out


def get_ylabel(mode):
    if mode == "production":
        return "Coal production (Giga tonnes / year)"
    return "Coal emissions (GtCO2 / year)"


def get_emissions_per_production(production, emissions):
    """Median tCO2 per tonne of coal across all scenarios and years.

    Production and emissions differ only by the production-weighted mean
    emissions factor, which is why the two are plottable on a shared vertical
    axis. That factor is not perfectly constant though (the asset mix shifts
    over the masterdata years), so the caller also gets the drift to report.
    """
    ratios = []
    for label, content in production.items():
        p = np.array(content["y"], dtype=float)
        e = np.array(emissions[label]["y"], dtype=float)
        mask = p > 0
        ratios.extend(list(e[mask] / p[mask]))
    ratios = np.array(ratios)
    factor = float(np.median(ratios))
    drift = float(ratios.max() / ratios.min() - 1)
    return factor, drift


def plot_production_and_emissions(ax, production, emissions, title):
    """Draw production (left axis, solid) and emissions (right axis, dashed).

    The right axis is pinned to the left one times the median emissions factor,
    so the two sets of lines coincide wherever that factor holds exactly, and
    visibly separate where it does not.
    """
    ax2 = ax.twinx()
    colors = {}
    for i, (label, content) in enumerate(production.items()):
        color = f"C{i}"
        colors[label] = color
        ax.plot(content["x"], content["y"], color=color, label=label)
    for label, content in emissions.items():
        ax2.plot(
            content["x"], content["y"], color=colors[label], linestyle="--", alpha=0.8
        )
    factor, drift = get_emissions_per_production(production, emissions)
    ax2.set_ylim(np.array(ax.get_ylim()) * factor)
    ax.set_title(title)
    ax.set_xlabel("Time")
    print(
        f"{title}: {factor:.3f} tCO2 per tonne of coal, "
        f"drift across scenarios/years {drift * 100:.2f}%"
    )
    return ax2


def plot_combined_2dii_ngfs_over_time(
    _ngfs_global_coal, figname, total_by_year, sector, mode
):
    assert mode in ["production", "emissions"]
    out = compute_2dii_ngfs_over_time(_ngfs_global_coal, total_by_year, sector)
    fig = plt.figure(figsize=(7, 5))
    for label, content in out.items():
        plt.plot(content["x"], content["y"], label=label)
    plt.xlabel("Time")
    plt.ylabel(get_ylabel(mode))
    fig.subplots_adjust(right=0.68)
    fig.legend(title="Scenario:", loc=7)
    plt.savefig(figname)
    plt.close()
    return out


DEVELOPMENT_LEVELS = [
    "Developed Countries",
    "Developing Countries",
    "Emerging Market Countries",
]

# Same regions as analysis_main.prepare_regions_for_climate_financing.
REGIONS = [
    "Asia",
    "Africa",
    "North America",
    "Latin America & the Carribean",
    "Europe",
    "Australia & New Zealand",
]
# Line-wrapped for the narrow region panels.
REGION_TITLES = {
    "Latin America & the Carribean": "Latin America &\nthe Carribean",
    "Australia & New Zealand": "Australia &\nNew Zealand",
}


def get_countries_by_development_level():
    (
        _,
        _,
        _,
        _,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()
    developING_country_shortnames = util.get_developing_countries()
    emerging_country_shortnames = util.get_emerging_countries()
    return {
        "Developed Countries": developed_country_shortnames,
        "Developing Countries": developING_country_shortnames,
        "Emerging Market Countries": emerging_country_shortnames,
    }


def get_countries_by_region():
    """Region -> alpha-2 country codes.

    Mirrors analysis_main.prepare_regions_for_climate_financing: the ISO-3166
    regions, except that the Americas are split into Northern America and Latin
    America & the Caribbean, and Oceania is narrowed to Australia & New Zealand.
    """
    iso3166_df = util.read_iso3166()
    by_region = {
        "Asia": iso3166_df[iso3166_df.region == "Asia"],
        "Africa": iso3166_df[iso3166_df.region == "Africa"],
        "North America": iso3166_df[iso3166_df["sub-region"] == "Northern America"],
        "Latin America & the Carribean": iso3166_df[
            iso3166_df["sub-region"] == "Latin America and the Caribbean"
        ],
        "Europe": iso3166_df[iso3166_df.region == "Europe"],
        "Australia & New Zealand": iso3166_df[
            iso3166_df["sub-region"] == "Australia and New Zealand"
        ],
    }
    # Iterate over REGIONS just to make sure that the order is deterministic.
    return {region: list(by_region[region]["alpha-2"]) for region in REGIONS}


def get_by_group(
    nonpower_coal, ngfs_nonpower, countries_by_group, group_kind, ngfs_by_region=False
):
    """Same as the global exp 16 result, but restricted to each group of countries.

    Returns {mode: {group: {scenario label: {"x": ..., "y": ...}}}}.
    Non-power only, to match out_combined in main(). With ngfs_by_region, each
    country in the group grows by its own NGFS region instead of by the global
    NGFS trajectory; ngfs_nonpower then has to include the regional rows.
    """
    uncategorized = set(nonpower_coal.asset_country.dropna()) - set(
        sum(countries_by_group.values(), [])
    )
    if uncategorized:
        print(f"Countries not in any {group_kind}:", sorted(uncategorized))

    years_masterdata = range(2013, 2027)
    out = {}
    for mode in ["production", "emissions"]:
        out[mode] = {}
        for level, shortnames in countries_by_group.items():
            subset = nonpower_coal[nonpower_coal.asset_country.isin(shortnames)]
            if ngfs_by_region:
                total_by_year = get_total_by_year_by_country(
                    subset, years_masterdata, "nonpower"
                )[mode]
                content = compute_2dii_ngfs_over_time_by_region(
                    ngfs_nonpower, total_by_year, "nonpower"
                )
            else:
                if mode == "production":
                    total_by_year = (
                        util.get_coal_nonpower_global_generation_across_years(
                            subset, years_masterdata
                        )
                    )
                else:
                    total_by_year = (
                        util.get_coal_nonpower_global_emissions_across_years(
                            subset, years_masterdata
                        )
                    )
                content = compute_2dii_ngfs_over_time(
                    ngfs_nonpower, total_by_year, "nonpower"
                )
            out[mode][level] = {
                label: {"x": v["x"], "y": list(v["y"])} for label, v in content.items()
            }
    return out


def main():
    _, nonpower_coal, power_coal = util.read_masterdata()

    # Non-power NGFS. Unit is EJ/yr.
    ngfs = util.read_ngfs_coal_and_power()["Coal"]
    # Constrain to a particular NGFS model
    ngfs = ngfs[ngfs.Model == util.NGFS_MODEL]
    ngfs_nonpower = ngfs[ngfs.Variable == "Primary Energy|Coal"]
    ngfs_nonpower_global = ngfs_nonpower[ngfs_nonpower.Region == "World"]

    # Power NGFS. Initial unit is GW.
    ngfs_power = util.read_ngfs_coal_and_power()["Power"]
    # Constrain to a particular NGFS model
    ngfs_power = ngfs_power[ngfs_power.Model == util.NGFS_MODEL]
    ngfs_power = ngfs_power[ngfs_power.Variable == "Capacity|Electricity|Coal"]
    ngfs_power_global = ngfs_power[ngfs_power.Region == "World"]

    out_combined = {}
    for mode in ["production", "emissions"]:
        print("# exp 16")
        years_masterdata = range(2013, 2027)
        if mode == "production":
            total_by_year = util.get_coal_nonpower_global_generation_across_years(
                nonpower_coal, years_masterdata
            )
        else:
            total_by_year = util.get_coal_nonpower_global_emissions_across_years(
                nonpower_coal, years_masterdata
            )
        out16_nonpower = plot_combined_2dii_ngfs_over_time(
            ngfs_nonpower_global,
            f"plots/exp16_{mode}.png",
            total_by_year,
            "nonpower",
            mode,
        )

        print("# exp 34")
        # Non-power is already done in exp 16
        years_masterdata = range(2013, 2027)
        if mode == "production":
            power_total_by_year = util.get_coal_power_global_generation_across_years(
                power_coal, years_masterdata
            )
        else:
            power_total_by_year = util.get_coal_power_global_emissions_across_years(
                power_coal, years_masterdata
            )
        out34_power = plot_combined_2dii_ngfs_over_time(
            ngfs_power_global,
            f"plots/exp34_{mode}.png",
            power_total_by_year,
            "power",
            mode,
        )

        # For combined plot
        out_combined[mode] = {}
        for label, v16 in out16_nonpower.items():
            v34 = out34_power[label]
            x = v16["x"]
            assert x == v34["x"]
            # out_combined[mode][label] = {"x": x, "y": list(v16["y"] + v34["y"])}
            # Only nonpower
            out_combined[mode][label] = {"x": x, "y": list(v16["y"])}

    print("# By level of development")
    out_by_development_level = get_by_group(
        nonpower_coal,
        ngfs_nonpower_global,
        get_countries_by_development_level(),
        "level of development",
    )

    print("# By region")
    out_by_region = get_by_group(
        nonpower_coal, ngfs_nonpower_global, get_countries_by_region(), "region"
    )

    print("# exp 16 + 34")
    plot_combined_figure(
        out_combined,
        out_by_development_level,
        out_by_region,
        "plots/exp16_34_combined.png",
    )

    with open("plots/for_comparison_yearly_exp16_34.json", "w") as f:
        json.dump(out_combined, f)
    with open(
        "plots/for_comparison_yearly_exp16_34_by_development_level.json", "w"
    ) as f:
        json.dump(out_by_development_level, f)
    with open("plots/for_comparison_yearly_exp16_34_by_region.json", "w") as f:
        json.dump(out_by_region, f)

    print("# NGFS by region")
    # Same 3 breakdowns again, but with each country following the NGFS
    # trajectory of its own region instead of the global one.
    years_masterdata = range(2013, 2027)
    total_by_year_by_country = get_total_by_year_by_country(
        nonpower_coal, years_masterdata, "nonpower"
    )
    out_combined_regional = {}
    for mode in ["production", "emissions"]:
        content = compute_2dii_ngfs_over_time_by_region(
            ngfs_nonpower, total_by_year_by_country[mode], "nonpower"
        )
        # Only nonpower, same as out_combined
        out_combined_regional[mode] = {
            label: {"x": v["x"], "y": list(v["y"])} for label, v in content.items()
        }

    print("# By level of development (NGFS by region)")
    out_by_development_level_regional = get_by_group(
        nonpower_coal,
        ngfs_nonpower,
        get_countries_by_development_level(),
        "level of development",
        ngfs_by_region=True,
    )

    print("# By region (NGFS by region)")
    out_by_region_regional = get_by_group(
        nonpower_coal,
        ngfs_nonpower,
        get_countries_by_region(),
        "region",
        ngfs_by_region=True,
    )

    plot_combined_figure(
        out_combined_regional,
        out_by_development_level_regional,
        out_by_region_regional,
        "plots/exp16_34_combined_regional_ngfs.png",
    )

    with open("plots/for_comparison_yearly_exp16_34_regional_ngfs.json", "w") as f:
        json.dump(out_combined_regional, f)
    with open(
        "plots/for_comparison_yearly_exp16_34_by_development_level_regional_ngfs.json",
        "w",
    ) as f:
        json.dump(out_by_development_level_regional, f)
    with open(
        "plots/for_comparison_yearly_exp16_34_by_region_regional_ngfs.json", "w"
    ) as f:
        json.dump(out_by_region_regional, f)

    print("# NGFS by region vs global NGFS")
    plot_ngfs_region_vs_world(
        out_combined, out_combined_regional, "plots/exp16_34_ngfs_region_vs_world.png"
    )


def plot_combined_figure(
    out_combined, out_by_development_level, out_by_region, figname
):
    """The figure of interest: world, then by level of development, then by region.

    Production and emissions share each panel: production on the left axis,
    emissions on the right one. Row 0 is the global result (1 panel, centered),
    row 1 breaks it down by level of development (3 panels), and row 2 by
    region (6 panels). The 6-column grid lets the 3 and the 6 panels line up;
    the region panels are consequently half as wide.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 6, hspace=0.4, wspace=0.85)

    ax_world = fig.add_subplot(gs[0, 2:4])
    ax2 = plot_production_and_emissions(
        ax_world, out_combined["production"], out_combined["emissions"], "World"
    )
    ax_world.set_ylabel(get_ylabel("production"))
    ax2.set_ylabel(get_ylabel("emissions"))

    for j, level in enumerate(DEVELOPMENT_LEVELS):
        ax = fig.add_subplot(gs[1, (2 * j) : (2 * j + 2)])
        ax2 = plot_production_and_emissions(
            ax,
            out_by_development_level["production"][level],
            out_by_development_level["emissions"][level],
            level,
        )
        if j == 0:
            ax.set_ylabel(get_ylabel("production"))
        if j == len(DEVELOPMENT_LEVELS) - 1:
            ax2.set_ylabel(get_ylabel("emissions"))

    for j, region in enumerate(REGIONS):
        ax = fig.add_subplot(gs[2, j])
        ax2 = plot_production_and_emissions(
            ax,
            out_by_region["production"][region],
            out_by_region["emissions"][region],
            REGION_TITLES.get(region, region),
        )
        if j == 0:
            ax.set_ylabel(get_ylabel("production"))
        if j == len(REGIONS) - 1:
            ax2.set_ylabel(get_ylabel("emissions"))

    # Deduplicate labels
    handles, labels = ax_world.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # The left/right axis of each panel is distinguished by line style
    by_label["Production (left axis)"] = Line2D([], [], color="gray", linestyle="-")
    by_label["Emissions (right axis)"] = Line2D([], [], color="gray", linestyle="--")
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
    )
    plt.savefig(figname, bbox_inches="tight")
    plt.close()


def plot_ngfs_region_vs_world(out_combined, out_combined_regional, figname):
    """How much the world total moves when the NGFS regions are used instead.

    Solid is the global NGFS trajectory applied to everything, dashed is the sum
    over the NGFS regions. They only differ after the peg year, and by how much
    depends on where the coal assets sit relative to the regions that the NGFS
    model phases out first.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, mode in enumerate(["production", "emissions"]):
        ax = axs[i]
        for j, (label, content) in enumerate(out_combined[mode].items()):
            ax.plot(content["x"], content["y"], color=f"C{j}", label=label)
            regional = out_combined_regional[mode][label]
            ax.plot(regional["x"], regional["y"], color=f"C{j}", linestyle="--")
            last = np.array(content["y"])[-1]
            print(
                f"{mode} {label.replace(chr(10), ' ')} in {content['x'][-1]}: "
                f"{last:.3f} global NGFS, {regional['y'][-1]:.3f} NGFS by region "
                f"({(regional['y'][-1] / last - 1) * 100:+.1f}%)"
            )
        ax.set_xlabel("Time")
        ax.set_ylabel(get_ylabel(mode))

    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label["Global NGFS"] = Line2D([], [], color="gray", linestyle="-")
    by_label["NGFS by region"] = Line2D([], [], color="gray", linestyle="--")
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
    )
    plt.tight_layout()
    plt.savefig(figname, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
