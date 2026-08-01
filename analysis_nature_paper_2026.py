"""Self-contained script that generates plots/exp16_34_combined.png.

Extracted from analysis_data_section.py (exp 16 + exp 34 combined figure).
Depends only on util.py and the datasets it reads.

Outputs:
- plots/exp16_{production,emissions}.png  (non-power, intermediate)
- plots/exp34_{production,emissions}.png  (power, intermediate)
- plots/for_comparison_yearly_exp16_34.json
- plots/for_comparison_yearly_exp16_34_by_development_level.json
- plots/exp16_34_combined.png             (the figure of interest)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

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


def compute_2dii_ngfs_over_time(_ngfs_global_coal, total_by_year, sector):
    """Patch the masterdata history with the NGFS scenario trajectories.

    Returns {scenario label: {"x": years, "y": values}}. The NGFS trajectories
    are the global ones; when total_by_year is a subset of the world (e.g. only
    the developed countries), the global fractional increase over the peg year
    is applied to that subset, which is the same convention as the rest of the
    codebase (see util.calculate_ngfs_fractional_increase).
    """
    out = {}
    for scenario in util.scenarios:
        if scenario in ["Below 2Â°C", "Divergent Net Zero", "Delayed transition"]:
            # Skip this scenario
            continue
        ngfs_global_coal_scenario = _ngfs_global_coal[
            _ngfs_global_coal.Scenario == scenario
        ].iloc[0]

        # clean up scenario
        scenario = scenario.replace("Capacity|", "")

        # ngfs_peg_year is the year where the NGFS value is pegged to be the
        # same as masterdata global production value.
        if scenario == "Current Policies ":
            ngfs_peg_year = 2026
        else:
            ngfs_peg_year = 2023
        # Assert the peg year to be at most the last year of masterdata.
        assert ngfs_peg_year <= 2026, ngfs_peg_year
        ngfs_left_year, ngfs_right_year = util.get_in_between_year(ngfs_peg_year)
        ngfs_value_left = convert2Gtonnes(
            sector, ngfs_global_coal_scenario[str(ngfs_left_year)]
        )
        ngfs_value_right = convert2Gtonnes(
            sector, ngfs_global_coal_scenario[str(ngfs_right_year)]
        )
        # Do linear interpolation once
        ngfs_value_peg = (
            ngfs_value_left
            + (ngfs_peg_year - ngfs_left_year)
            * (ngfs_value_right - ngfs_value_left)
            / 5
        )

        # Get the fraction
        ngfs_years_after_peg = list(range(ngfs_right_year, 2105, 5))
        ngfs_values = [
            convert2Gtonnes(sector, ngfs_global_coal_scenario[str(year)])
            for year in ngfs_years_after_peg
        ]
        fraction_increase_over_peg_year = np.array(
            [(v / ngfs_value_peg) for v in ngfs_values]
        )
        rescaled_ngfs_value_after_2025 = list(
            total_by_year[(ngfs_peg_year - 2013)] * fraction_increase_over_peg_year
        )

        masterdata_years = list(range(2013, ngfs_peg_year + 1))
        patched_years = masterdata_years + ngfs_years_after_peg
        whole_range_production = (
            total_by_year[: len(masterdata_years)] + rescaled_ngfs_value_after_2025
        )
        # Remove weird character
        label = scenario.replace("Â", "")
        if label == "Nationally Determined Contributions (NDCs) ":
            label = "Nationally Determined\nContributions (NDCs)"
        out[label] = {"x": patched_years, "y": np.array(whole_range_production)}
    return out


def get_ylabel(mode):
    if mode == "production":
        return "Coal production (Giga tonnes / year)"
    return "Coal emissions (GtCO2 / year)"


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


def get_by_development_level(nonpower_coal, ngfs_nonpower_global):
    """Same as the global exp 16 result, but restricted to each level of development.

    Returns {mode: {level of development: {scenario label: {"x": ..., "y": ...}}}}.
    Non-power only, to match out_combined in main().
    """
    countries_by_level = get_countries_by_development_level()
    uncategorized = set(nonpower_coal.asset_country.dropna()) - set(
        sum(countries_by_level.values(), [])
    )
    if uncategorized:
        print("Countries not in any level of development:", sorted(uncategorized))

    years_masterdata = range(2013, 2027)
    out = {}
    for mode in ["production", "emissions"]:
        out[mode] = {}
        for level, shortnames in countries_by_level.items():
            subset = nonpower_coal[nonpower_coal.asset_country.isin(shortnames)]
            if mode == "production":
                total_by_year = util.get_coal_nonpower_global_generation_across_years(
                    subset, years_masterdata
                )
            else:
                total_by_year = util.get_coal_nonpower_global_emissions_across_years(
                    subset, years_masterdata
                )
            content = compute_2dii_ngfs_over_time(
                ngfs_nonpower_global, total_by_year, "nonpower"
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
    out_by_development_level = get_by_development_level(
        nonpower_coal, ngfs_nonpower_global
    )

    print("# exp 16 + 34")
    # Row 0 is the global result (2 panels), rows 1 and 2 break it down by level
    # of development (3 panels each). The 6-column grid is the least common
    # multiple of the 2 and 3 panel rows.
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 6, hspace=0.35, wspace=0.9)

    for i, mode in enumerate(["production", "emissions"]):
        ax = fig.add_subplot(gs[0, (3 * i) : (3 * i + 3)])
        for label, content in out_combined[mode].items():
            ax.plot(content["x"], content["y"], label=label)
        ax.set_title("World")
        ax.set_xlabel("Time")
        ax.set_ylabel(get_ylabel(mode))

    for i, mode in enumerate(["production", "emissions"]):
        for j, level in enumerate(DEVELOPMENT_LEVELS):
            ax = fig.add_subplot(gs[1 + i, (2 * j) : (2 * j + 2)])
            for label, content in out_by_development_level[mode][level].items():
                ax.plot(content["x"], content["y"], label=label)
            ax.set_title(level)
            ax.set_xlabel("Time")
            if j == 0:
                ax.set_ylabel(get_ylabel(mode))

    with open("plots/for_comparison_yearly_exp16_34.json", "w") as f:
        json.dump(out_combined, f)
    with open(
        "plots/for_comparison_yearly_exp16_34_by_development_level.json", "w"
    ) as f:
        json.dump(out_by_development_level, f)

    # Deduplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
    )
    plt.savefig("plots/exp16_34_combined.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
