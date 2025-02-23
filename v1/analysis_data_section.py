import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import util

# Ensure that plots directory exists
os.makedirs("plots", exist_ok=True)


def round4(x):
    return round(x, 4)


def round2(x):
    return round(x, 2)


df, nonpower_coal, power_coal = util.read_masterdata()

country_to_region, iso2_to_country_name = util.get_country_to_region()


def convert_country_to_region(c):
    if isinstance(c, float) and np.isnan(c):
        return "N/A"
    if pd.isna(c):
        return "N/A"
    return country_to_region[c]


def convert_country_id_to_country_name(c):
    return "N/A" if (isinstance(c, float) and np.isnan(c)) else iso2_to_country_name[c]


# https://stackoverflow.com/questions/47776516/where-does-this-pandas-warning-come-from
# https://stackoverflow.com/questions/42379818/correct-way-to-set-new-column-in-pandas-dataframe-to-avoid-settingwithcopywarnin
# nonpower_coal.reset_index(drop=True, inplace=True)
nonpower_coal.loc[:, "region"] = nonpower_coal.asset_country.apply(
    lambda c: country_to_region[c]
)

power_coal.loc[:, "region"] = power_coal.asset_country.apply(convert_country_to_region)
power_years = range(2013, 2031)


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


def plot_pure_ngfs_over_time(_ngfs, figname, sector):
    for scenario in util.scenarios:
        if scenario not in ["Net Zero 2050", "Current Policies"]:
            continue
        # Remove weird character
        cleaned_scenario = scenario.replace("Â", "")
        _ngfs_scenario = _ngfs[_ngfs.Scenario == scenario]
        plt.figure()
        for idx, row in _ngfs_scenario.iterrows():
            Gtonnes = [
                convert2Gtonnes(sector, row[str(year)]) for year in util.ngfs_years
            ]
            plt.plot(
                util.ngfs_years,
                Gtonnes,
                label=row.Region.replace(util.NGFS_MODEL + "|", ""),
            )
        plt.xlabel("Time (years)")
        plt.ylabel("Coal production (Giga tonnes of coal)")
        plt.legend()
        if sector == "nonpower":
            plt.title("Primary Energy|Coal")
        else:
            plt.title("Capacity|Electricity|Coal")
        plt.savefig(f"plots/{figname}_{cleaned_scenario}.png")
        plt.close()


def plot_pure_ngfs_over_time_GLOBAL(_ngfs, figname, sector):
    plt.figure()
    for scenario in util.scenarios:
        if scenario not in ["Net Zero 2050", "Current Policies"]:
            continue
        # Remove weird character
        cleaned_scenario = scenario.replace("Â", "")
        _ngfs_scenario = _ngfs[_ngfs.Scenario == scenario].iloc[0]
        Gtonnes = [
            convert2Gtonnes(sector, _ngfs_scenario[str(year)])
            for year in util.ngfs_years
        ]
        plt.plot(
            util.ngfs_years,
            Gtonnes,
            label=cleaned_scenario,
        )
    plt.xlabel("Time (years)")
    plt.ylabel("Coal production (Giga tonnes of coal)")
    plt.legend()
    if sector == "nonpower":
        plt.title("Primary Energy|Coal")
    else:
        plt.title("Capacity|Electricity|Coal")
    plt.savefig(f"plots/{figname}.png")
    plt.close()


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
        if scenario == "Current Policies":
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
    # if sector == "nonpower":
    #    plt.title("Primary Energy|Coal (2DII for 2013-2026)")
    # else:
    #    plt.title("Capacity|Electricity|Coal (2DII for 2013-2026)")
    plt.savefig(figname)
    plt.close()
    return out


if 0:

    def get_ngfs_regional(_df):
        _df_regional = _df[_df.Region != "World"]
        # Remove duplicate regions
        _df_regional = _df_regional[_df_regional.Region.str.startswith(util.NGFS_MODEL)]
        return _df_regional

    print("# exp 15")
    ngfs = pd.read_csv("data/ngfs_scenario_production_fossil.csv")
    # Constrain to a particular NGFS model
    ngfs = ngfs[ngfs.Model == util.NGFS_MODEL]
    # Unit is EJ/yr
    ngfs_nonpower = ngfs[ngfs.Variable == "Primary Energy|Coal"]
    # ngfs_nonpower_regional = get_ngfs_regional(ngfs_nonpower)
    ngfs_nonpower_global = ngfs_nonpower[ngfs_nonpower.Region == "World"]
    plot_pure_ngfs_over_time_GLOBAL(ngfs_nonpower_global, "exp15", "nonpower")

    print("# exp 33")
    # Power
    # Redundant, just use plot_ngfs_scenario("NGFS Scenario Data 2021 Power")
    # in analysis_comprehensive.py
    ngfs_power = pd.read_csv(
        "data/NGFS-Power-Sector-Scenarios.csv.gz", compression="gzip"
    )
    # Constrain to a particular NGFS model
    ngfs_power = ngfs_power[ngfs_power.Model == util.NGFS_MODEL]
    # Initial unit is GW
    ngfs_power = ngfs_power[ngfs_power.Variable == "Capacity|Electricity|Coal"]
    # ngfs_power_regional = get_ngfs_regional(ngfs_power)
    ngfs_power_global = ngfs_power[ngfs_power.Region == "World"]
    plot_pure_ngfs_over_time_GLOBAL(ngfs_power_global, "exp33", "power")

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
    ngfs_peg_year = 2023
    start_year = 2013

    def plot_halt_to_coal_production(x, y):
        halt_y = [0] * len(x)
        for i in range(ngfs_peg_year - start_year + 1):
            halt_y[i] = y[i]
        plt.plot(x, halt_y, label="Halt to coal production")

    for i, mode in enumerate(["production", "emissions"]):
        plt.sca(axs[i])
        for label, content in out_combined[mode].items():
            plt.plot(content["x"], content["y"], label=label)
        current_policies = out_combined[mode]["Current Policies"]
        plot_halt_to_coal_production(current_policies["x"], current_policies["y"])
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
    # plt.tight_layout()
    plt.savefig("plots/exp16_34_combined.png", bbox_inches="tight")
    exit()


def get_pure_coal_ids(df):
    nonpower_noncoal = set(df[~df.sector.isin(["Coal", "Power"])].company_id)
    power_companies = df[df.sector == "Power"]
    power_noncoal = set(
        power_companies[power_companies.technology != "CoalCap"].company_id
    )
    all_noncoal = nonpower_noncoal.union(power_noncoal)
    pure_coal_ids = all_coal - all_noncoal
    return all_noncoal, pure_coal_ids


if 0:
    print("Breakdown of coal energy types. Unit is tonnes of CO2 per tonnes of coal.")

    def print_tech(_df, techs):
        np_tech = _df[_df.technology.isin(techs)].copy()
        print(techs)
        print("min", np_tech.emissions_factor.min())
        print("max", np_tech.emissions_factor.max())
        print("median", np_tech.emissions_factor.median())
        print("mean unweighed", np_tech.emissions_factor.mean())
        print(
            "mean weighted",
            (np_tech.emissions_factor * np_tech._2020).sum() / np_tech._2020.sum(),
        )
        print(
            "quantiles 5%, 95%", np.nanquantile(np_tech.emissions_factor, [0.05, 0.95])
        )
        print()

    print_tech(nonpower_coal, ["Anthracite"])
    print_tech(nonpower_coal, ["Lignite"])
    print_tech(nonpower_coal, ["Bituminous"])
    print_tech(nonpower_coal, ["Sub-Bituminous"])
    print_tech(nonpower_coal, ["Unknown"])
    print("Power sector")
    print_tech(power_coal, ["CoalCap"])
    exit()


def print_stat(series):
    print("mean", series.mean())
    print("min", series.min())
    print("max", series.max())
    print("quantiles 5%, 95%", np.nanquantile(series, [0.05, 0.95]))


if 1:
    print("Pure coal companies")
    print("Coal company energy composition")
    all_companies = set(df.company_id)
    all_coal = set(nonpower_coal.company_id).union(set(power_coal.company_id))
    all_noncoal, pure_coal_ids = get_pure_coal_ids(df)

    coal_with_noncoal = all_coal.intersection(all_noncoal)
    print("All coal", len(all_coal))
    print("All noncoal", len(all_noncoal))
    print("Pure coal", len(pure_coal_ids))
    print("Coal with noncoal", len(coal_with_noncoal))

    # Sanity check
    power_pure_coal = power_coal[power_coal.company_id.isin(pure_coal_ids)]

    import processed_revenue

    # TODO remove this line once we get the new revenue data.
    df = df[df.company_id.isin(processed_revenue.revenue_data_companies)]

    df_pure_coal = df[df.company_id.isin(pure_coal_ids)].copy()

    # pure power
    # df_pure_coal = df_pure_coal[df_pure_coal.sector == "Power"]
    # df_pure_coal = df_pure_coal[~df_pure_coal.company_id.isin(set(nonpower_coal.company_id))]

    # pure nonpower
    # df_pure_coal = df_pure_coal[df_pure_coal.sector == "Coal"]
    # df_pure_coal = df_pure_coal[~df_pure_coal.company_id.isin(set(power_coal.company_id))]

    processed_revenue.prepare_average_unit_profit(df_pure_coal)

    df_pure_coal["prod"] = df_pure_coal.apply(
        lambda row: util.GJ2coal(processed_revenue.toGJ(row, 2020)), axis=1
    )

    df_pure_coal_by_company = df_pure_coal.groupby("company_id").agg(
        {
            "company_name": "first",
            "prod": "sum",
            "energy_type_specific_average_unit_profit": "first",
        },
        axis=1,
    )
    df_pure_coal_by_company = df_pure_coal_by_company.sort_values(
        by="prod", ascending=False
    )

    for topx in [10, 100, 200, 500, 1000, 1500]:
        df_pure_coal_topx = df_pure_coal_by_company.iloc[1 : topx + 1]
        median = df_pure_coal_topx.energy_type_specific_average_unit_profit.median()
        print(
            f"Median of top {topx} pure coal aup:",
            median,
            "$/GJ",
            median / util.GJ2coal(1),
            "$/tce",
        )
        # if topx == 10:
        #    print("Top 10 pure coal companies", list(df_pure_coal_topx.company_name))

    median = df_pure_coal_by_company.energy_type_specific_average_unit_profit.median()
    # Division by util.GJ2coal(1) converts the average unit profit to per tce.
    print(
        "Median of all pure coal aup:",
        median,
        "$/GJ",
        median / util.GJ2coal(1),
        "$/tce",
    )
    print_stat(df_pure_coal_by_company.energy_type_specific_average_unit_profit)
    exit()
