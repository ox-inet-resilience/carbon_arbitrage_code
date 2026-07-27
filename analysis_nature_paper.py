"""Self-contained script that generates plots/exp16_34_combined.png.

Extracted from analysis_data_section.py (exp 16 + exp 34 combined figure).
Depends only on util.py and the datasets it reads.

Outputs:
- plots/exp16_{production,emissions}.png  (non-power, intermediate)
- plots/exp34_{production,emissions}.png  (power, intermediate)
- plots/for_comparison_yearly_exp16_34.json
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


def plot_combined_2dii_ngfs_over_time(
    _ngfs_global_coal, figname, total_by_year, sector, mode
):
    assert mode in ["production", "emissions"]
    out = {}
    fig = plt.figure(figsize=(7, 5))
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
        label = scenario.replace("Â", "")
        if label == "Nationally Determined Contributions (NDCs) ":
            label = "Nationally Determined\nContributions (NDCs)"
        plt.plot(
            patched_years,
            whole_range_production,
            # Remove weird character
            label=label,
        )
        out[label] = {"x": patched_years, "y": np.array(whole_range_production)}
    plt.xlabel("Time")
    if mode == "production":
        ylabel = "Coal production (Giga tonnes / year)"
    else:
        ylabel = "Coal emissions (GtCO2 / year)"
    plt.ylabel(ylabel)
    fig.subplots_adjust(right=0.68)
    fig.legend(title="Scenario:", loc=7)
    plt.savefig(figname)
    plt.close()
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

    print("# exp 16 + 34")
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))

    for i, mode in enumerate(["production", "emissions"]):
        plt.sca(axs[i])
        for label, content in out_combined[mode].items():
            plt.plot(content["x"], content["y"], label=label)
        plt.xlabel("Time")
        if mode == "production":
            ylabel = "Coal production (Giga tonnes / year)"
        else:
            ylabel = "Coal emissions (GtCO2 / year)"
        plt.ylabel(ylabel)
    with open("plots/for_comparison_yearly_exp16_34.json", "w") as f:
        json.dump(out_combined, f)

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
