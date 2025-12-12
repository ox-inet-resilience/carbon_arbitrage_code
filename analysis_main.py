import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler

import util
from util import world_gdp_2023
import with_learning
import gca.table1 as table1
import gca.parameters as parameters

sns.set_theme(style="ticks")


# Ensure that plots directory exists
os.makedirs("plots", exist_ok=True)
os.makedirs("plots/table2", exist_ok=True)

# Params that can be modified
lcoe_mode = "solar+wind"
# lcoe_mode="solar+wind+gas"

print("Renewable degradation:", with_learning.ENABLE_RENEWABLE_GRADUAL_DEGRADATION)
print("30 year lifespan:", with_learning.ENABLE_RENEWABLE_30Y_LIFESPAN)
print("Wright's law", with_learning.ENABLE_WRIGHTS_LAW)
print("Residual benefit", with_learning.ENABLE_RESIDUAL_BENEFIT)
print("Sector included", parameters.SECTOR_INCLUDED)
print("BATTERY_SHORT", with_learning.ENABLE_BATTERY_SHORT)
print("BATTERY_LONG", with_learning.ENABLE_BATTERY_LONG)


def run_table2(name="", included_countries=None):
    result = {}
    sccs = [
        util.social_cost_of_carbon_imf,
        util.scc_biden_administration,
        util.scc_bilal,
    ]
    scc_default = util.social_cost_of_carbon_imf
    last_years = [2035, 2050]
    for last_year in last_years:
        util.social_cost_of_carbon = scc_default
        parameters.LAST_YEAR = last_year
        result[last_year] = table1.run_table1(included_countries=included_countries)
    gc_benefit_old_name = "Benefits of avoiding coal emissions including residual benefit (in trillion dollars)"
    subsectors = ["Coal", "Oil", "Gas"]
    mapper_worker = {}
    for subsector in subsectors:
        val = f"OC owner {subsector}"
        mapper_worker[val] = val
        val = f"OC workers lost wages {subsector}"
        mapper_worker[val] = val
        val = f"OC workers retraining cost {subsector}"
        mapper_worker[val] = val

    # Rename the key
    mapper = {
        "Time Period": "Time Period of Carbon Arbitrage",
        "Avoided fossil fuel electricity generation (PWh)": "Electricity generation avoided including residual (PWh)",
        "Avoided emissions (GtCO2e)": "Total emissions avoided including residual (GtCO2)",
        **{f"Avoided emissions {s}": f"AE+residual {s}" for s in util.SUBSECTORS},
        "Costs of power sector decarbonization (in trillion dollars)": "Costs of avoiding coal emissions (in trillion dollars)",
        "Opportunity costs (in trillion dollars)": "Opportunity costs (in trillion dollars)",
        "OC owner (in trillion dollars)": "OC owner (in trillion dollars)",
        **mapper_worker,
        "Investment costs (in trillion dollars)": "Investment costs (in trillion dollars)",
        "Investment costs in renewable energy": "Investment costs in renewable energy",
        "Investment costs short-term storage": "investment_cost_battery_short_trillion",
        "Investment costs long-term storage": "investment_cost_battery_long_trillion",
        "Investment costs renewables to power electrolyzers": "investment_cost_battery_pe_trillion",
        "Investment costs grid extension": "investment_cost_battery_grid_trillion",
    }

    def _s(y):
        return f"2024-{y} FA + Net Zero 2050 Scenario"

    table = defaultdict(list)
    for k, v in mapper.items():
        for y in last_years:
            try:
                table[k].append(result[y][v][_s(y)])
            except KeyError:
                print("Missing either of", k, y, v)
                continue
        if len(table[k]) == 0:
            # Some countries may have missing at least one of coal/oil/gas
            # e.g. Viet Nam has 0 oil
            del table[k]

    gdp_2023 = world_gdp_2023
    if included_countries is not None:
        gdp_marketcap_dict = util.read_json(util.gdp_marketcap_path)
        # actual = np.nansum(list(gdp_marketcap_dict.values()))
        # actual's value is 103.74, different from worldbank's own data of 105.44.
        # Both come from worldbank.
        # Convert to trillion dollars
        gdp_2023 = (
            np.nansum(
                [
                    gdp
                    for country, gdp in gdp_marketcap_dict.items()
                    if country in included_countries
                ]
            )
            / 1e12
        )
    for i, y in enumerate(last_years):
        ae = table["Avoided emissions (GtCO2e)"][i] * 1e9
        cost_per_ae = (
            (
                table["Costs of power sector decarbonization (in trillion dollars)"][i]
                * 1e12
                / ae
            )
            if ae > 0
            else 0
        )
        table["Costs per avoided tCO2e ($/tCO2e)"].append(cost_per_ae)
        arbitrage_period = y - parameters.NGFS_PEG_YEAR
        for scc in sccs:
            scc_scale = scc / scc_default
            table[f"scc {scc} GC benefit (in trillion dollars)"].append(
                result[y][gc_benefit_old_name][_s(y)] * scc_scale
            )
            table[f"scc {scc} CC benefit (in trillion dollars)"].append(
                result[y]["country_benefit_country_reduction"][_s(y)] * scc_scale
            )

            table[f"scc {scc} GC benefit per avoided tCO2e ($/tCO2e)"].append(
                result[y][gc_benefit_old_name][_s(y)] * scc_scale * 1e12 / ae
                if ae > 0
                else 0
            )
            table[f"scc {scc} CC benefit per avoided tCO2e ($/tCO2e)"].append(
                result[y]["country_benefit_country_reduction"][_s(y)]
                * scc_scale
                * 1e12
                / ae
                if ae > 0
                else 0
            )

            table[f"scc {scc} GC net benefit (in trillion dollars)"].append(
                table[f"scc {scc} GC benefit (in trillion dollars)"][i]
                - table["Costs of power sector decarbonization (in trillion dollars)"][
                    i
                ]
            )
            table[f"scc {scc} CC net benefit (in trillion dollars)"].append(
                table[f"scc {scc} CC benefit (in trillion dollars)"][i]
                - table["Costs of power sector decarbonization (in trillion dollars)"][
                    i
                ]
            )

            table[f"scc {scc} CC Net benefit relative to GDP (%)"].append(
                table[f"scc {scc} CC net benefit (in trillion dollars)"][i]
                * 100
                / (gdp_2023 * arbitrage_period)
            )

            table[f"scc {scc} GC Net benefit per avoided tCO2e ($/tCO2e)"].append(
                table[f"scc {scc} GC net benefit (in trillion dollars)"][i] * 1e12 / ae
                if ae > 0
                else 0
            )
            table[f"scc {scc} CC Net benefit per avoided tCO2e ($/tCO2e)"].append(
                table[f"scc {scc} CC net benefit (in trillion dollars)"][i] * 1e12 / ae
                if ae > 0
                else 0
            )
        table["GDP over time period (in trillion dollars)"].append(
            gdp_2023 * arbitrage_period
        )
    scc_share_percent = 100
    if included_countries is not None:
        scc_share_percent = (
            100
            * sum(table1.country_sccs.get(c, 0) for c in included_countries)
            / table1.country_sccs.sum()
        )

    table["scc_share (%)"] = scc_share_percent
    uid = util.get_unique_id(include_date=False)
    df = pd.DataFrame(table).round(6).T
    df.to_csv(f"plots/table2/table2_{name}_{uid}.csv")
    return df


def calculate_each_countries_with_cache(
    chosen_s2_scenario,
    cache_json_path,
    ignore_cache=False,
    info_name="cost",
    last_year=None,
    scc=None,
):
    # IMPORTANT: the chosen s2 scenario indicates whether the yearly cost for
    # avoiding is discounted or not.
    if last_year is not None:
        parameters.LAST_YEAR = last_year
    use_cache = not ignore_cache
    if use_cache and os.path.isfile(cache_json_path):
        info_dict = util.read_json(cache_json_path)
    else:
        print("Cached json not found. Calculating from scratch...")
        info_dict = {}
        if scc is not None:
            util.social_cost_of_carbon = scc
        out = table1.run_table1(to_csv=False, do_round=False, return_yearly=True)
        for key, yearly in out[chosen_s2_scenario].items():
            if key in [
                "residual_benefit",
                "avoided_emissions_including_residual_emissions",
                "country_benefit_country_reduction",
                "global_benefit_country_reduction",
            ]:
                # Is already in the format of dict[str, float]
                # of country, value.
                value = yearly
                info_dict[key] = value
                continue
            # Collect all country names
            country_names = set()
            for e in yearly:
                if isinstance(e, float):
                    continue
                elif isinstance(e, dict):
                    country_names = country_names.union(e.keys())
                elif isinstance(e, pd.Series):
                    country_names = country_names.union(e.index)
                else:
                    print(e)
                    raise Exception("Should not happen")

            each_key_dict = {}
            for country_name in country_names:
                country_level_cost = 0.0
                for e in yearly:
                    if isinstance(e, float):
                        assert math.isclose(e, 0.0)
                    elif isinstance(e, dict):
                        # We use .get instead of [], for battery's case
                        country_level_cost += e.get(country_name, 0.0)
                    else:
                        # pandas series
                        country_level_cost += e.loc[country_name]
                each_key_dict[country_name] = country_level_cost
            info_dict[key] = each_key_dict.copy()
        if use_cache:
            with open(cache_json_path, "w") as f:
                json.dump(info_dict, f)
    return info_dict[info_name]


def annotate(xs, ys, labels, filter_labels=None, no_zero_x=False, fontsize=None):
    for x, y, label in zip(xs, ys, labels):
        if (filter_labels is not None) and (label not in filter_labels):
            continue
        if no_zero_x and math.isclose(x, 0):
            continue
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",  # how to position the text
            xytext=(0, 5),  # distance from text to points (x,y)
            ha="center",  # horizontal alignment
            fontsize=fontsize,
        )


def make_climate_financing_SCATTER_plot():
    gdp_per_capita_dict = util.read_json(util.gdp_per_capita_path)
    # Taiwan in 2024
    # https://www.imf.org/external/datamapper/NGDPDPC@WEO/ADVEC/WEOWORLD/TWN/CHN
    # gdp_per_capita_dict["TW"] = 34430
    # Kosovo in 2023
    # https://data.worldbank.org/indicator/NY.GDP.PCAP.CD?locations=XK
    # gdp_per_capita_dict["XK"] = 5943.1

    gdp_marketcap_dict = util.read_json(util.gdp_marketcap_path)
    # Taiwan in 2023
    # https://www.statista.com/statistics/727589/gross-domestic-product-gdp-in-taiwan/
    # gdp_marketcap_dict["TW"] = 756.59 * 1e9  # billion USD
    # Kosovo in 2023
    # https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?locations=XK
    # gdp_marketcap_dict["XK"] = 10.44 * 1e9  # to billion uSD

    worldbank_set = set(gdp_per_capita_dict.keys())
    masterdata_coal_set = set(table1.df_sector.asset_country)
    divide_by_marketcap = True

    print("MODE divide by marketcap", divide_by_marketcap)
    # Data checking
    print("worldbank.org", len(worldbank_set))
    print("masterdata", len(masterdata_coal_set))
    print("intersection", len(worldbank_set.intersection(masterdata_coal_set)))
    print("masterdata - worldbank", masterdata_coal_set - worldbank_set)
    # Only in masterdata: {nan, 'TW', 'XK'}

    # country_shortnames = list(masterdata_coal_set - {np.nan, "TW", "XK"})
    chosen_s2_scenario = f"{parameters.NGFS_PEG_YEAR}-2100 FA + Net Zero 2050 Scenario"
    cache_json_path = "plots/climate_financing.json"

    costs_dict = calculate_each_countries_cost_with_cache(
        chosen_s2_scenario, cache_json_path
    )

    developing_shortnames = util.get_developing_countries()
    emerging_shortnames = util.get_emerging_countries()
    raise Exception("developed_gdp uses outdated GDP here")
    developed_gdp = pd.read_csv("data/GDP-Developed-World.csv", thousands=",")
    colname_for_gdp = "2023 GDP (million dollars)"
    developed_country_shortnames = list(
        developed_gdp.sort_values(by=colname_for_gdp, ascending=False).country_shortcode
    )
    # multiplier_mode = "trillion"
    # If the multiplier mode is billion, we restrict the ylim to up to 1
    # trillion.
    multiplier_mode = "billion"

    def plot_scatter(shortnames, label):
        x = []
        y = []
        if divide_by_marketcap:
            # one trillion
            # We convert the cost to just dollars.
            mul = 1e12
        else:
            mul = 1 if multiplier_mode == "trillion" else 1e3

        arbitrage_period = 1 + (2100 - (parameters.NGFS_PEG_YEAR + 1))
        print("Peg year", parameters.NGFS_PEG_YEAR, "arbitrage period", arbitrage_period)
        plot_labels = []
        filter_labels = []
        for country_shortname in shortnames:
            if country_shortname not in costs_dict:
                # print(f"{country_shortname} is missing")
                continue
            if country_shortname in ["TW", "XK"]:
                print("Intentionally skipping", country_shortname)
                continue
            x_val = gdp_per_capita_dict[country_shortname]
            x.append(x_val)
            plot_labels.append(country_shortname)
            if divide_by_marketcap:
                mul_marketcap = (
                    1 / (gdp_marketcap_dict[country_shortname] * arbitrage_period) * 100
                )
            else:
                mul_marketcap = 1
            cost = costs_dict[country_shortname]
            val = cost * mul * mul_marketcap
            # Sanity check
            if cost > 4.0:
                print(
                    "Beyond 4 trillion dollars:",
                    country_shortname,
                    f"cost {cost:.2f} trillion dollars",
                    "GDP",
                    gdp_marketcap_dict[country_shortname],
                )
            if divide_by_marketcap and val > 200:
                print(
                    f"Beyond 200% of GDP: {country_shortname} {val:.2f}%",
                )
            # End of sanity check
            y.append(val)
            if not ((x_val <= 20_000) and (val <= 5)):
                filter_labels.append(country_shortname)
        plt.plot(x, y, label=label, linewidth=0, marker="o", fillstyle="none")
        annotate(x, y, plot_labels, fontsize=10, filter_labels=filter_labels)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # By level of development
    # Sanity check
    by_development = set(
        developing_shortnames + emerging_shortnames + developed_country_shortnames
    )
    print("Not in level of development", masterdata_coal_set - by_development)
    # End of sanity check
    plt.sca(axs[0])
    plot_scatter(developing_shortnames, "Developing country")
    plot_scatter(emerging_shortnames, "Emerging country")
    plot_scatter(developed_country_shortnames, "Developed country")
    plt.xlabel("GDP per capita (dollars)")
    if divide_by_marketcap:
        ylabel = "PV Climate financing / country GDP (%)"
    else:
        ylabel = f"PV Climate financing ({multiplier_mode} dollars)"
    plt.ylabel(ylabel)
    if not divide_by_marketcap and multiplier_mode == "billion":
        plt.ylim(0, 1e3)
    plt.legend(loc="upper right")

    # By region
    iso3166_df = util.read_iso3166()
    region_countries_map, regions = prepare_regions_for_climate_financing(iso3166_df)
    # Sanity check
    by_region = []
    for country_names in region_countries_map.values():
        by_region += country_names
    by_region = set(by_region)
    print("Not in by region", masterdata_coal_set - by_region)
    # End of sanity check
    plt.sca(axs[1])
    for region_name, country_names in region_countries_map.items():
        if region_name == "Latin America & the Carribean":
            region_name = "Latin America &\nthe Carribean"
        elif region_name == "Australia & New Zealand":
            region_name = "Australia &\nNew Zealand"
        plot_scatter(country_names, region_name)
    plt.xlabel("GDP per capita (dollars)")
    plt.ylabel(ylabel)
    if not divide_by_marketcap and multiplier_mode == "billion":
        plt.ylim(0, 1e3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    util.savefig("climate_financing_scatter")


def calculate_yearly_info_dict(chosen_s2_scenario, info_name="cost"):
    yearly_costs_dict = {}
    out = table1.run_table1(to_csv=False, do_round=False, return_yearly=True)
    yearly_cost_for_avoiding = out[chosen_s2_scenario][info_name]
    country_names = list(yearly_cost_for_avoiding[-1].keys())
    for country_name in country_names:
        country_level_cost = []
        for e in yearly_cost_for_avoiding:
            if isinstance(e, float):
                country_level_cost.append(e)
            elif isinstance(e, dict):
                country_level_cost.append(e[country_name])
            else:
                # Pandas series
                country_level_cost.append(e.loc[country_name])
        yearly_costs_dict[country_name] = country_level_cost
    return yearly_costs_dict


def do_cf_battery_yearly():
    chosen_s2_scenario = f"{parameters.NGFS_PEG_YEAR}-2100 FA + Net Zero 2050 Scenario"

    whole_years = range(parameters.NGFS_PEG_YEAR, 2100 + 1)

    def calculate_yearly_world_cost(s2_scenario):
        yearly_costs_dict = calculate_yearly_info_dict(s2_scenario)
        # Calculating the cost for the whole world
        yearly_world_cost = np.zeros(len(whole_years))
        for v in yearly_costs_dict.values():
            yearly_world_cost += np.array(v)
        return yearly_world_cost

    def _get_year_range_cost(year_start, year_end, yearly_world_cost):
        return sum(
            yearly_world_cost[year_start - parameters.NGFS_PEG_YEAR : year_end + 1 - parameters.NGFS_PEG_YEAR]
        )

    labels = [
        "30Y, D, E",
        "30Y, D, E, S",
        "30Y, D, E, L",
        "30Y, D, E, O",
        "30Y, D, E, S+L",
        "30Y, D, E, S+L+O",
        "30Y, D, no E, S+L+O",
    ]
    import coal_worker

    cw_out = coal_worker.calculate("default", full_version=True)
    retraining_series = (
        cw_out["wage_lost_series"]
        * coal_worker.ic_usa
        / coal_worker.wage_usd_dict["US"]
    )
    # Set parameters.NGFS_PEG_YEAR value to 0
    retraining_series = np.insert(retraining_series, 0, 0)
    opportunity_cost_series = cw_out["opportunity_cost_series"]
    opportunity_cost_series = np.insert(opportunity_cost_series, 0, 0)

    # Just for bar chart and original
    data_for_barchart = {
        (parameters.NGFS_PEG_YEAR + 1, 2050): {},
        (2051, 2070): {},
        (2071, 2100): {},
    }
    label_map_original = {
        "30Y": "30Y, D, E",
        "30Y_noE": "30Y, D, no E",
        "LCOE": "LCOE proxy",
        "50Y": "50Y, D, E",
        "200Y": "Lifetime by D, E",
    }

    def reset():
        global ENABLE_NEW_METHOD
        ENABLE_NEW_METHOD = 1
        with_learning.ENABLE_WRIGHTS_LAW = 1
        with_learning.RENEWABLE_LIFESPAN = 30
        with_learning.ENABLE_BATTERY_SHORT = False
        with_learning.ENABLE_BATTERY_LONG = False
        with_learning.ENABLE_BATTERY_GRID = False

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(axs[0])
    global ENABLE_NEW_METHOD
    for key, label in label_map_original.items():
        reset()
        if key == "LCOE":
            ENABLE_NEW_METHOD = 0
        elif key.endswith("Y"):
            with_learning.RENEWABLE_LIFESPAN = int(key[:-1])
        else:
            assert key == "30Y_noE"
            with_learning.ENABLE_WRIGHTS_LAW = 0

        yearly = calculate_yearly_world_cost(chosen_s2_scenario, discounted=False)
        linestyle = "-" if label == "30Y, D, E" else "dotted"
        plt.plot(
            whole_years,
            yearly,
            label=label,
            linestyle=linestyle,
            linewidth=2.5,
        )

        yearly_discounted = calculate_yearly_world_cost(chosen_s2_scenario)
        for year_start, year_end in [
            (parameters.NGFS_PEG_YEAR + 1, 2050),
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
    reset()
    # End just for bar chart and original

    # Battery only
    plt.sca(axs[1])
    fname = "plots/cf_battery_cache.json"
    if os.path.isfile(fname):
        yearly_all, extra_data_for_barchart = util.read_json(fname)
    else:
        extra_data_for_barchart = {
            f"{parameters.NGFS_PEG_YEAR + 1}-2050": {},
            "2051-2070": {},
            "2071-2100": {},
        }
        yearly_all = {}
        for label in labels:
            with_learning.ENABLE_WRIGHTS_LAW = "no E" not in label
            with_learning.ENABLE_BATTERY_SHORT = "S" in label
            with_learning.ENABLE_BATTERY_LONG = "L" in label

            yearly = calculate_yearly_world_cost(chosen_s2_scenario, discounted=False)
            if "O" in label:
                # Division by 1e3 converts to trillion dollars
                yearly += opportunity_cost_series / 1e3
                yearly += retraining_series / 1e3
            yearly_all[label] = list(yearly)

            yearly_discounted = calculate_yearly_world_cost(chosen_s2_scenario)
            for year_start, year_end in [
                (parameters.NGFS_PEG_YEAR + 1, 2050),
                (2051, 2070),
                (2071, 2100),
            ]:
                aggregate = _get_year_range_cost(
                    year_start, year_end, yearly_discounted
                )
                extra_data_for_barchart[f"{year_start}-{year_end}"][label] = aggregate
        with open(fname, "w") as f:
            json.dump([yearly_all, extra_data_for_barchart], f)
    for label in labels:
        linestyle = "-" if label == "30Y, D, E" else "dotted"
        if label != "30Y, D, no E, S+L+O":
            plt.plot(
                whole_years,
                yearly_all[label],
                label=label,
                linestyle=linestyle,
                linewidth=2.5,
            )

    plt.xlabel("Time")
    plt.ylabel("Global annual climate financing\n(trillion dollars)")
    plt.legend(
        bbox_to_anchor=(0.5, -0.2),
        loc="upper center",
        ncol=2,
    )
    plt.tight_layout()

    util.savefig("cf_battery_yearly", tight=True)

    # Bar plot
    plt.figure()
    # Merge the barchart data
    for k in data_for_barchart:
        k_str = f"{k[0]}-{k[1]}"
        for _k, _v in extra_data_for_barchart[k_str].items():
            data_for_barchart[k][_k] = _v
    xticks = None
    stacked_bar_data = []
    for year_pair, data in data_for_barchart.items():
        xticks = list(data.keys())
        stacked_bar_data.append((f"{year_pair[0]}-{year_pair[1]}", list(data.values())))
    util.plot_stacked_bar(
        xticks,
        stacked_bar_data,
    )
    # For separating the baseline
    plt.axvline(0.5, color="gray", linestyle="dashed")
    # For separating the battery ones
    plt.axvline((4 + 5) / 2, color="gray", linestyle="dashed")
    plt.xticks(xticks, rotation=90, ha="center")
    plt.ylabel("PV global climate financing\n(trillion dollars)")
    plt.legend(loc="upper center")
    plt.tight_layout()
    util.savefig("cf_battery_pv")


def run_3_level_scc():
    # Run for 3 levels of social cost of carbon.
    mode = "cao"
    # mode = "cao_relative"
    # mode = "cost"
    # mode = "benefit"
    if mode == "cao":
        cao_name = "Carbon arbitrage opportunity (in trillion dollars)"
        cao_name_with_residual = (
            "Carbon arbitrage including residual benefit (in trillion dollars)"
        )
    elif mode == "cao_relative":
        cao_name = "Carbon arbitrage opportunity relative to world GDP (%)"
        cao_name_with_residual = (
            "Carbon arbitrage including residual benefit relative to world GDP (%)"
        )
    elif mode == "cost":
        cao_name = "Costs of avoiding coal emissions (in trillion dollars)"
        cao_name_with_residual = cao_name
    print(cao_name)
    for last_year in [2050, 2070, 2100]:
        parameters.LAST_YEAR = last_year
        caos = []
        caos_with_residual = []
        condition = f"{parameters.NGFS_PEG_YEAR}-{last_year} FA + Net Zero 2050 Scenario"
        # condition = f"{parameters.NGFS_PEG_YEAR}-{last_year} FA + Current Policies  Scenario"
        scs = [
            util.social_cost_of_carbon_lower_bound,
            util.social_cost_of_carbon_imf,
            util.social_cost_of_carbon_upper_bound,
        ]
        for sc in scs:
            util.social_cost_of_carbon = sc
            out = table1.run_table1(to_csv=False, do_round=True)
            cao = out[cao_name][condition]
            caos.append(f"{cao:.2f}")
            cao_with_residual = out[cao_name_with_residual][condition]
            caos_with_residual.append(f"{cao_with_residual:.2f}")
        if "Net Zero 2050" in condition:
            info = "NZ2050"
        elif "Current Policies" in condition:
            info = "CPS"
        else:
            raise Exception(f"condition not expected: {condition}")
        print(last_year, info, "with residual:")
        # print(" & ".join(caos))
        print(" & ".join(caos_with_residual))


def get_yearly_by_country():
    # Ensure plot output dir exists
    os.makedirs("plots/bruegel", exist_ok=True)

    for enable in [False, True]:
        parameters.ENABLE_COAL_EXPORT = enable
        out = table1.run_table1(to_csv=False, do_round=True, return_yearly=True)
        nz2050 = out[f"{parameters.NGFS_PEG_YEAR}-2100 FA + Net Zero 2050 Scenario"]
        series_ics = []
        series_ocs = []
        for i in range(2, 2100 - parameters.NGFS_PEG_YEAR + 1):
            # Trillions
            series_ocs.append(
                nz2050["opportunity_cost_non_discounted"][i].rename(parameters.NGFS_PEG_YEAR + i)
            )
            series_ics.append(
                pd.Series(
                    nz2050["investment_cost_non_discounted"][i],
                    name=(parameters.NGFS_PEG_YEAR + i),
                )
            )
        git_branch = util.get_git_branch()
        a2_to_full_name = util.prepare_alpha2_to_full_name_concise()

        df = pd.concat(series_ocs, axis=1)
        df.index = df.index.to_series().apply(lambda a2: a2_to_full_name[a2])
        suffix = f"{git_branch}_coalexport_{enable}"
        df.to_csv(
            f"plots/bruegel/yearly_by_country_opportunity_cost_NONDISCOUNTED_{suffix}.csv"
        )

        df = pd.concat(series_ics, axis=1)
        df.index = df.index.to_series().apply(lambda a2: a2_to_full_name[a2])
        df.to_csv(
            f"plots/bruegel/yearly_by_country_investment_cost_NONDISCOUNTED_{suffix}.csv"
        )

        yearly_ae = util.read_json(
            "./cache/unilateral_benefit_yearly_avoided_emissions_GtCO2_2100.json"
        )
        if parameters.ENABLE_COAL_EXPORT:
            from coal_export.common import modify_avoided_emissions_based_on_coal_export

            yearly_ae = modify_avoided_emissions_based_on_coal_export(yearly_ae)
        df = pd.DataFrame(
            yearly_ae, index=list(range(parameters.NGFS_PEG_YEAR, 2100 + 1))
        ).transpose()
        df.index = df.index.to_series().apply(lambda a2: a2_to_full_name[a2])
        df.to_csv(f"plots/bruegel/yearly_by_country_avoided_emissions_{suffix}.csv")


def get_yearly_by_country_power():
    out = table1.run_table1(to_csv=False, do_round=True, return_yearly=True)
    nz2050 = out[f"{parameters.NGFS_PEG_YEAR}-{parameters.LAST_YEAR} FA + Net Zero 2050 Scenario"]
    series = defaultdict(list)
    ignore = [
        "avoided_emissions_including_residual_emissions",
        "country_benefit_country_reduction",
        "global_benefit_country_reduction",
        "residual_benefit",
    ]
    for i in range(parameters.LAST_YEAR - parameters.NGFS_PEG_YEAR + 1):
        # Trillions
        for key, value in nz2050.items():
            if key in ignore:
                continue
            if isinstance(value[i], pd.Series):
                element = value[i].rename(parameters.NGFS_PEG_YEAR + i)
            else:
                element = pd.Series(
                    value[i],
                    name=(parameters.NGFS_PEG_YEAR + i),
                )
            series[key].append(element)
    git_branch = util.get_git_branch()
    for key, value in series.items():
        df = pd.concat(value, axis=1)
        df.to_csv(f"plots/bruegel/yearly_by_country_{key}_{git_branch}.csv")


def make_battery_unit_ic_plot(scenario, countries_included):
    # Ensure plot output dir exists
    os.makedirs("plots/phase_in", exist_ok=True)

    parameters.MEASURE_GLOBAL_VARS_SCENARIO = scenario
    parameters.MEASURE_GLOBAL_VARS = True
    with_learning.VERBOSE_ANALYSIS = True
    util.CARBON_BUDGET_CONSISTENT = "15-50"
    # util.CARBON_BUDGET_CONSISTENT = "strictly_declining"
    years = list(range(2024, parameters.LAST_YEAR + 1))
    years_plus_renewable_lifetime = years + list(
        range(2050 + 1, 2050 + 1 + with_learning.RENEWABLE_LIFESPAN)
    )

    def kW2GW(arr):
        return [i / 1e6 for i in arr]

    def GJ2TW(arr):
        return [util.GJ2MW(i) / 1e6 for i in arr]

    name_labels = {
        "solar": "Solar",
        "onshore_wind": "Wind onshore",
        "offshore_wind": "Wind Offshore",
        "geothermal": "Geothermal",
        "hydropower": "Hydropower",
    }

    a2_to_full_name = util.prepare_alpha2_to_full_name_concise()
    # for country in "WORLD EMDE IN ID DE US TR VN PL KZ".split():
    for country in countries_included:
        with_learning.VERBOSE_ANALYSIS_COUNTRY = country
        title = (
            a2_to_full_name[country]
            if country not in ["WORLD", "EMDE", "Developed_UNFCCC", "Developing_UNFCCC"]
            else country
        )
        print(title)
        try:
            table1.run_table1(to_csv=False, do_round=False)
        except Exception as e:
            print("FA doesn't have this country's data:", e)
            continue
        fig, axs = plt.subplots(1, 2, figsize=(8, 5))

        plt.sca(axs[0])
        for name, label in name_labels.items():
            plt.plot(
                years,
                parameters.global_cost_with_learning.cached_investment_costs[name].values(),
                label=label,
            )

        plt.plot(
            years,
            parameters.global_cost_with_learning.battery_unit_ic["short"].values(),
            label="Short",
        )
        plt.plot(
            years,
            parameters.global_cost_with_learning.battery_unit_ic["long"].values(),
            label="Battery long",
        )
        plt.xlabel("Time")
        plt.ylabel("Unit investment cost ($/kW)")

        plt.sca(axs[1])
        # Define a custom color and marker cycle
        colors = plt.cm.tab10.colors[:9]
        markers = ["^", "v", ">", "<", "o", "s", "D", "P", "*"]
        markersize = 2
        axs[1].set_prop_cycle(cycler(color=colors) + cycler(marker=markers))
        country_name = with_learning.VERBOSE_ANALYSIS_COUNTRY
        plt.suptitle(title)
        for tech, label in {
            **name_labels,
            "short": "Battery short",
            "long": "Battery long",
        }.items():
            y = np.cumsum(
                kW2GW(
                    list(
                        parameters.global_cost_with_learning.cached_stock_without_degradation[
                            tech
                        ].values()
                    )
                )
            )
            plt.plot(
                years_plus_renewable_lifetime,
                y,
                label=label,
                markersize=markersize,
                linewidth=0.8,
            )

        # Reset color cycler
        axs[1].set_prop_cycle(cycler(color=colors) + cycler(marker=markers))
        energy_produced_1country = [
            {
                tech: e[country_name][tech] if country_name in e else 0
                for tech in with_learning.TECHS
            }
            for e in parameters.global_cost_with_learning.green_energy_produced_by_country
        ]
        for tech, label in name_labels.items():
            plt.plot(
                years,
                util.GJ2MW(np.array([e[tech] for e in energy_produced_1country])) / 1e3,
                linestyle="dotted",
                label=tech,
                markersize=markersize,
            )

        plt.xlabel("Time")
        plt.ylabel("Capacity (GW)")

        # Deduplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncol=5,
        )
        plt.tight_layout()
        plt.savefig(
            f"plots/phase_in/battery_unit_ic_{parameters.MEASURE_GLOBAL_VARS_SCENARIO}_{country}.png",
            bbox_inches="tight",
        )
        plt.close()
        util.write_small_json(
            {
                **parameters.global_cost_with_learning.cached_investment_costs,
                **parameters.global_cost_with_learning.battery_unit_ic,
            },
            f"plots/phase_in/battery_unit_ic_{country}.json",
        )

        # 2nd file
        fig = plt.figure()
        plt.title(title)
        for tech, label in {
            **name_labels,
            "short": "Battery short",
            "long": "Battery long",
        }.items():
            y = kW2GW(
                list(
                    parameters.global_cost_with_learning.cached_stock_without_degradation[
                        tech
                    ].values()
                )
            )
            plt.plot(
                years_plus_renewable_lifetime,
                y,
                label=label,
                markersize=markersize,
                linewidth=0.8,
            )
        plt.xlabel("Time")
        plt.ylabel("Annual installed capacity (GW)")
        # Deduplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncol=5,
        )
        plt.tight_layout()
        plt.savefig(
            f"plots/phase_in/battery_yearly_installed_capacity_{parameters.MEASURE_GLOBAL_VARS_SCENARIO}_{country}.png",
            bbox_inches="tight",
        )
        plt.close()
        util.write_small_json(
            dict(parameters.global_cost_with_learning.cached_stock_without_degradation),
            f"plots/phase_in/battery_yearly_installed_capacity_{parameters.MEASURE_GLOBAL_VARS_SCENARIO}_{country}.json",
        )
        util.write_small_json(
            dict(parameters.global_cost_with_learning.cached_stock),
            f"plots/phase_in/battery_yearly_available_capacity_{parameters.MEASURE_GLOBAL_VARS_SCENARIO}_{country}.json",
        )

        # 3rd file
        fig = plt.figure()
        plt.title(title)
        y = sum(
            np.array(list(d.values()))
            for d in parameters.global_cost_with_learning.cached_stock_without_degradation.values()
        )
        y_cumsum = np.cumsum(y)

        plt.plot(years_plus_renewable_lifetime, kW2GW(y), label="Annual")
        plt.plot(years_plus_renewable_lifetime, kW2GW(y_cumsum), label="Cumulative")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Annual installed capacity (GW)")
        plt.savefig(
            f"plots/phase_in/battery_yearly_installed_capacity_{parameters.MEASURE_GLOBAL_VARS_SCENARIO}_{country}_summed.png"
        )
        plt.close()

    parameters.MEASURE_GLOBAL_VARS = False
    parameters.MEASURE_GLOBAL_VARS_SCENARIO = "Net Zero 2050"
    with_learning.VERBOSE_ANALYSIS = False
    util.CARBON_BUDGET_CONSISTENT = False


if __name__ == "__main__":
    if 0:
        get_yearly_by_country_power()
        # get_yearly_by_country()
        exit()

    if 1:
        run_table2()
        exit()

    if 0:
        sccs = [
            util.social_cost_of_carbon_imf,
            util.scc_biden_administration,
            util.scc_bilal,
        ]
        for scc in sccs:
            util.social_cost_of_carbon = scc
            out = table1.run_table1(to_csv=True, do_round=True)
        exit()
    if 1:
        # Battery yearly
        # do_cf_battery_yearly()
        # make_battery_plot()
        countries = with_learning.DEVELOPING_UNFCCC
        countries = (
            "WORLD Developed_UNFCCC Developing_UNFCCC EG IN ID ZA MX VN IR TH".split()
        )
        countries = ["CA"]
        # 15 Africa countries
        countries = "BW CI DJ GH GN KE NG RW SN SL SC TZ UG ZM ZW".split()
        countries = "ID IN VN ZA".split()
        countries = sorted(list(set(table1.df_sector.asset_country.tolist())))
        make_battery_unit_ic_plot("Net Zero 2050", countries)
        # Halt to coal production
        # make_battery_unit_ic_plot("Current Policies", countries)
        exit()
    if 0:
        run_3_level_scc()
        exit()
