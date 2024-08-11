import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import util  # noqa


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


def plot_breakdown_fa_ngfs_over_time(
    ngfs_df, df_sector, value_fa, figname, sector, mode
):
    assert mode in ["production", "emissions"]
    assert mode == "emissions"
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    scenarios = list(set(ngfs_df[mode].Scenario))

    for i, subsector in enumerate(["Oil", "Gas", "Coal"]):
        plt.sca(axs[i])
        plt.title(subsector)

        for scenario in scenarios:
            if scenario in [
                "Below 2°C",
                "Divergent Net Zero",
                "Low demand",
                "Fragmented World",
            ]:
                # Skip this scenario
                continue
            with_ngfs_projection = util.calculate_ngfs_projection(
                mode,
                value_fa,
                ngfs_df,
                SECTOR_INCLUDED,
                scenario,
                NGFS_PEG_YEAR,
                last_year,
                alpha2_to_alpha3,
                filter_subsector=subsector,
            )

            plt.plot(
                years_interpolated,
                with_ngfs_projection,
                label=scenario,
            )
    plt.xlabel("Time")
    if mode == "production":
        ylabel = "Coal production (Giga tonnes / year)"
    else:
        ylabel = "Coal emissions (GtCO2 / year)"
    plt.ylabel(ylabel)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
    )

    plt.savefig(figname, bbox_inches="tight")
    plt.close()


def plot_combined_fa_ngfs_over_time(
    ngfs_df, df_sector, value_fa, figname, sector, mode
):
    assert mode == "emissions"
    fig = plt.figure()
    scenarios = list(set(ngfs_df[mode].Scenario))
    for scenario in scenarios:
        if scenario in [
            "Below 2°C",
            "Divergent Net Zero",
            "Low demand",
            "Fragmented World",
        ]:
            # Skip this scenario
            continue
        with_ngfs_projection = util.calculate_ngfs_projection(
            mode,
            value_fa,
            ngfs_df,
            SECTOR_INCLUDED,
            scenario,
            NGFS_PEG_YEAR,
            last_year,
            alpha2_to_alpha3,
        )

        plt.plot(
            years_interpolated,
            with_ngfs_projection,
            label=scenario,
        )
    plt.xlabel("Time")
    plt.ylabel("Emissions (GtCO2 / year)")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
    )

    plt.savefig(figname, bbox_inches="tight")
    plt.close()


# todo 2.2
SECTOR_INCLUDED = "Power"
NGFS_PEG_YEAR = 2024
ngfs_df = util.read_ngfs()
df, df_sector = util.read_forward_analytics_data(SECTOR_INCLUDED)
iso3166_df = util.read_iso3166()
alpha2_to_alpha3 = iso3166_df.set_index("alpha-2")["alpha-3"].to_dict()
# GtCO2
emissions_fa = util.get_emissions_by_country(df_sector)
print("Emissions 2024", emissions_fa.sum())
last_year = 2050
years_interpolated = list(range(NGFS_PEG_YEAR, last_year))

mode = "emissions"
plot_breakdown_fa_ngfs_over_time(
    ngfs_df,
    df_sector,
    emissions_fa,
    f"plots/v2.2_{mode}.png",
    "power",
    mode,
)
plot_combined_fa_ngfs_over_time(
    ngfs_df,
    df_sector,
    emissions_fa,
    f"plots/v2.2_{mode}_combined.png",
    "power",
    mode,
)
