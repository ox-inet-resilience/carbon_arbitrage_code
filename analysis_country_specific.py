import csv
from collections import defaultdict
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

import util
import analysis_main
from coal_export.common import modify_based_on_coal_export

matplotlib.use("agg")


def flatten_list_of_list(xss):
    return [x for xs in xss for x in xs]


# Countries that we have data for, for unilateral action.
# We intentionally exclude XK.
unilateral_countries = {
    "MM",
    "TZ",
    "GB",
    "HU",
    "RO",
    "TH",
    "CO",
    "US",
    "ZM",
    "MN",
    "PL",
    "KG",
    "ZW",
    "DE",
    "VN",
    "TJ",
    "RU",
    "CA",
    "TR",
    "LA",
    "BR",
    "NG",
    "IR",
    "MK",
    "RS",
    "UA",
    "MW",
    "ID",
    "PK",
    "KZ",
    "ME",
    "AU",
    "MX",
    "BG",
    "SK",
    "MZ",
    "JP",
    "CN",
    "ES",
    "MG",
    "ZA",
    "ET",
    "CL",
    "GR",
    "CD",
    "NE",
    "NO",
    "PE",
    "SI",
    "UZ",
    "IN",
    "CZ",
    "PH",
    "VE",
    "AR",
    "BA",
    "BD",
    "BW",
    "GE",
    "NZ",
}
G7 = "US JP DK GB DE IT NO".split()
EU = "AT BE BG CY CZ DK EE FI FR DE GR HU HR IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split()
eurozone = "AT BE HR CY EE FI FR DE GR IE IT LV LT LU MT NL PT SK SI ES".split()
# Largest by avoided emissions
# Taken from avoided_emissions_nonadjusted.csv (see Zulip).
by_avoided_emissions = "CN AU US IN RU ID ZA CA PL KZ CO DE MZ MN UA TR VN BW GR BR CZ BG RO TH RS GB UZ PH ZW NZ MX BD BA LA IR CL ES PK VE TZ HU ME SK ZM SI MG TJ MK GE AR MM JP KG MW NG NE PE NO ET CD".split()
countries_after_coal_export = "AE AF AG AM AO AR AT AU AW AZ BA BB BD BE BF BG BH BI BJ BM BN BO BR BS BW BY BZ CA CD CG CH CI CL CN CO CR CY CZ DE DK DO EC EE EG ES ET FI FJ FR GB GD GE GH GR GT GY HK HN HR HU ID IE IL IN IR IS IT JM JO JP KE KG KH KM KR KW KZ LA LB LC LK LS LT LU LV LY MA MD ME MG MK ML MM MN MS MT MU MV MW MX MY MZ NA NE NG NI NL NO NP NZ OM PA PE PF PH PK PL PS PT PY QA RO RS RU RW SA SC SE SG SI SK SN SV SZ TG TH TJ TN TR TT TZ UA UG US UY UZ VC VE VN XK YE ZA ZM ZW".split()


def apply_last_year(last_year):
    if last_year in [2030, 2035, 2070]:
        analysis_main.MID_YEAR = last_year
    else:
        # 2100 is included by default in the mid year of 2050.
        assert last_year in [2050, 2100], last_year
        analysis_main.MID_YEAR = 2050


def prepare_alpha2_to_full_name_concise():
    iso3166_df = util.read_iso3166()
    iso3166_df_alpha2 = iso3166_df.set_index("alpha-2")
    alpha2_to_full_name = iso3166_df_alpha2["name"].to_dict()
    alpha2_to_full_name["GB"] = "Great Britain"
    alpha2_to_full_name["US"] = "USA"
    alpha2_to_full_name["RU"] = "Russia"
    alpha2_to_full_name["TW"] = "Taiwan"
    alpha2_to_full_name["KR"] = "South Korea"
    alpha2_to_full_name["LA"] = "Laos"
    alpha2_to_full_name["VE"] = "Venezuela"
    alpha2_to_full_name["CD"] = "Congo-Kinshasa"
    alpha2_to_full_name["IR"] = "Iran"
    alpha2_to_full_name["TZ"] = "Tanzania"
    alpha2_to_full_name["BA"] = "B&H"
    return alpha2_to_full_name


def prepare_level_development():
    (
        iso3166_df,
        _,
        _,
        _,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()
    developING_country_shortnames = util.get_developing_countries()
    emerging_country_shortnames = util.get_emerging_countries()

    levels = [
        "Developed Countries",
        "Developing Countries",
        "Emerging Market Countries",
    ]
    levels_map = {
        "Developed Countries": developed_country_shortnames,
        "Developing Countries": developING_country_shortnames,
        "Emerging Market Countries": emerging_country_shortnames,
    }
    return levels, levels_map, iso3166_df


def read_country_specific_scc_filtered():
    country_specific_scc = util.read_json("plots/country_specific_scc.json")
    # Remove these countries
    # Because they are in Oceania:
    # New Caledonia
    # Fiji
    # Solomon Islands
    # Vanuatu
    # They are not part of the 6 regions (Asia, Africa, NA,
    # LAC, Europe, AUS&NZ), nor are they part of the
    # developing, emerging, developed world.
    for country in ["NC", "FJ", "SB", "VU"]:
        del country_specific_scc[country]
    # NaN is also removed
    del country_specific_scc["NaN"]
    return country_specific_scc


def _do_sanity_check_for_calculate_cs_scc_data(
    unilateral_actor,
    isa_climate_club,
    bs,
    cs,
    bs_region,
    cs_region,
    unilateral_emissions,
    cumulative_benefit,
    global_benefit,
    global_emissions,
    unilateral_emissions_GtCO2,
    levels,
    cost_climate_club,
    benefit_climate_club,
):
    # Sanity check
    if unilateral_actor is None:
        return
    if unilateral_actor == "EMDE":
        return
    # SC1
    # We don't test based on region, because PG and WS are not part of
    # the 6 regions.
    # world_benefit_but_unilateral_action = sum(sum(v) for v in bs_region.values())
    world_benefit_but_unilateral_action = sum(sum(v) for v in bs.values())
    # Check that the computed world scc corresponds to the original world scc.
    actual_world_scc = world_benefit_but_unilateral_action / unilateral_emissions
    assert math.isclose(actual_world_scc, util.social_cost_of_carbon), actual_world_scc
    # SC2
    if not isa_climate_club:
        # The unilateral actor is a country
        # Cumulative benefit is the unilateral benefit of the country.
        ratio1 = cumulative_benefit / global_benefit
        ratio2 = unilateral_emissions_GtCO2 / global_emissions
        assert math.isclose(ratio1, ratio2), (ratio1, ratio2)
    else:
        # Unilateral action is either a region or level of development.
        # Double checking that they add up to the same amount
        if unilateral_actor in levels:
            assert math.isclose(cost_climate_club, sum(cs[unilateral_actor]))
            assert math.isclose(benefit_climate_club, sum(bs[unilateral_actor]))
        else:
            # Region
            actual = sum(cs_region[unilateral_actor])
            assert math.isclose(cost_climate_club, actual), (
                cost_climate_club,
                actual,
            )
            actual = sum(bs_region[unilateral_actor])
            assert math.isclose(benefit_climate_club, actual), (
                benefit_climate_club,
                actual,
            )


def calculate_country_specific_scc_data(
    unilateral_actor=None,
    ext="",
    to_csv=True,
    last_year=None,
    cost_name="cost",
):
    if last_year is None:
        last_year = 2100
    chosen_s2_scenario = f"2022-{last_year} 2DII + Net Zero 2050 Scenario"
    apply_last_year(last_year)
    costs_dict = analysis_main.calculate_each_countries_cost_with_cache(
        chosen_s2_scenario,
        "plots/country_specific_cost.json",
        ignore_cache=True,
        cost_name=cost_name,
    )
    out = analysis_main.run_cost1(x=1, to_csv=False, do_round=False)
    global_benefit = out[
        "Benefits of avoiding coal emissions including residual benefit (in trillion dollars)"
    ][chosen_s2_scenario]
    global_emissions = out["Total emissions avoided including residual (GtCO2)"][
        chosen_s2_scenario
    ]

    # In dollars/tCO2
    country_specific_scc = read_country_specific_scc_filtered()
    total_scc = sum(country_specific_scc.values())

    levels, levels_map, iso3166_df = prepare_level_development()
    region_countries_map, regions = analysis_main.prepare_regions_for_climate_financing(
        iso3166_df
    )
    _, iso2_to_country_name = util.get_country_to_region()

    unilateral_benefit = None
    unilateral_emissions = None

    isa_climate_club = unilateral_actor in (levels + regions + ["EMDE"])
    cost_climate_club = None
    benefit_climate_club = None
    benefit_of_country_doing_the_action = None
    if unilateral_actor is not None:
        # Generated from the Git branch unilateral_action_benefit
        if util.USE_NATURE_PAPER_SCC:
            cache_name = f"cache/unilateral_benefit_scc_nature_paper/unilateral_benefit_trillion_{last_year}.json"
        else:
            suffix = "_with_coal_export" if analysis_main.ENABLE_COAL_EXPORT else ""
            cache_name = (
                f"cache/unilateral_benefit_total_trillion_{last_year}{suffix}.json"
            )
        unilateral_benefit = util.read_json(cache_name)
        if isa_climate_club:
            unilateral_emissions = 0.0
            cost_climate_club = 0.0
            benefit_climate_club = 0.0
            if unilateral_actor in levels:
                group = levels_map[unilateral_actor]
            elif unilateral_actor == "EMDE":
                group = (
                    levels_map["Emerging Market Countries"]
                    + levels_map["Developing Countries"]
                )
            else:
                group = region_countries_map[unilateral_actor]
            for country in group:
                if country not in unilateral_benefit:
                    # Skip if we don't have the data for it.
                    continue
                # TODO we should have just used a JSON data of unilateral
                # emissions.
                ub = unilateral_benefit[country]
                scc = (
                    country_specific_scc[country]
                    / total_scc
                    * util.social_cost_of_carbon
                )
                unilateral_emissions += ub / scc
                cost_climate_club += costs_dict[country]
            # We have to first sum the emissions across the countries, in order
            # to get the benefit of the climate club.
            for country in group:
                if country not in country_specific_scc:
                    # Skip if we don't have scc data of the country.
                    continue
                scc = (
                    country_specific_scc[country]
                    / total_scc
                    * util.social_cost_of_carbon
                )
                benefit_climate_club += unilateral_emissions * scc
        else:
            benefit_of_country_doing_the_action = unilateral_benefit[unilateral_actor]
            scc = (
                country_specific_scc[unilateral_actor]
                / total_scc
                * util.social_cost_of_carbon
            )
            unilateral_emissions = benefit_of_country_doing_the_action / scc
    # From "trillion" tCO2 to Giga tCO2
    if unilateral_emissions is not None:
        unilateral_emissions_GtCO2 = unilateral_emissions * 1e3
    else:
        unilateral_emissions_GtCO2 = None
    # print("emissions Giga tCO2", unilateral_actor, unilateral_emissions_GtCO2)

    names = defaultdict(list)
    cs_level = defaultdict(list)
    bs_level = defaultdict(list)
    names_region = defaultdict(list)
    cs_region = defaultdict(list)
    bs_region = defaultdict(list)
    no_cost = []
    benefit_greater_than_cost = []
    costly = []
    table = pd.DataFrame(
        columns=[
            "iso2",
            # Commented out so for the website purpose, so that we use
            # consistent country naming according to ISO 3166.
            # "country_name",
            "net_benefit",
            "benefit",
            "cost",
            "country_specific_scc",
        ]
    )
    climate_club_countries = None
    if isa_climate_club:
        if unilateral_actor in levels:
            climate_club_countries = levels_map[unilateral_actor]
        elif unilateral_actor == "EMDE":
            climate_club_countries = (
                levels_map["Emerging Market Countries"]
                + levels_map["Developing Countries"]
            )
        else:
            climate_club_countries = region_countries_map[unilateral_actor]

    cumulative_benefit = 0.0  # For sanity check
    actual_size = 0  # For sanity check
    for country, unscaled_scc in country_specific_scc.items():
        if country not in costs_dict:
            c = 0.0
        else:
            c = costs_dict[country]
            if unilateral_actor is not None:
                if isa_climate_club:
                    if country not in climate_club_countries:
                        c = 0.0
                elif country != unilateral_actor:
                    # unilateral_actor is 1 country
                    c = 0.0
        cs_scc_scale = unscaled_scc / total_scc
        if unilateral_actor is not None:
            scc = util.social_cost_of_carbon * cs_scc_scale
            b = unilateral_emissions * scc
            if not isa_climate_club and unilateral_actor == country:
                # SC3
                # Sanity check for when the unilateral actor is 1 country only.
                assert math.isclose(b, benefit_of_country_doing_the_action)
        else:
            # Global action
            b = cs_scc_scale * global_benefit
        cumulative_benefit += b

        net_benefit = b - c
        table = pd.concat(
            [
                table,
                pd.DataFrame(
                    [
                        {
                            "iso2": country,
                            # Commented out so for the website purpose, so that we use
                            # consistent country naming according to ISO 3166.
                            # "country_name": iso2_to_country_name[country],
                            "net_benefit": net_benefit,
                            "benefit": b,
                            "cost": c,
                            "country_specific_scc": unscaled_scc,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        if math.isclose(c, 0.0):
            no_cost.append(country)
        elif b >= c:
            benefit_greater_than_cost.append(country)
        else:
            costly.append(country)
        if unilateral_actor == "EMDE":
            if (country in levels_map["Developing Countries"]) or (
                country in levels_map["Emerging Market Countries"]
            ):
                cs_level["EMDE"].append(c)
                bs_level["EMDE"].append(b)
                names["EMDE"].append(country)
            elif country in levels_map["Developed Countries"]:
                cs_level["Developed Countries"].append(c)
                bs_level["Developed Countries"].append(b)
                names["Developed Countries"].append(country)
        else:
            if country in levels_map["Developed Countries"]:
                cs_level["Developed Countries"].append(c)
                bs_level["Developed Countries"].append(b)
                names["Developed Countries"].append(country)
            elif country in levels_map["Developing Countries"]:
                cs_level["Developing Countries"].append(c)
                bs_level["Developing Countries"].append(b)
                names["Developing Countries"].append(country)
            elif country in levels_map["Emerging Market Countries"]:
                cs_level["Emerging Market Countries"].append(c)
                bs_level["Emerging Market Countries"].append(b)
                names["Emerging Market Countries"].append(country)
            else:
                print("Skipping", country)

        part_of_region = False
        for region in regions:
            if country in region_countries_map[region]:
                cs_region[region].append(c)
                bs_region[region].append(b)
                names_region[region].append(country)
                part_of_region = True
                break
        if not part_of_region:
            print("Not part of any region", country)
    print("country-specific country count", len(country_specific_scc))
    print("No cost", len(no_cost))
    print("benefit >= cost", len(benefit_greater_than_cost), benefit_greater_than_cost)
    print("cost > benefit", len(costly), costly)
    actual_size += len(no_cost) + len(benefit_greater_than_cost) + len(costly)
    assert actual_size == len(country_specific_scc), (
        actual_size,
        len(country_specific_scc),
    )

    # Sanity check
    _do_sanity_check_for_calculate_cs_scc_data(
        unilateral_actor,
        isa_climate_club,
        bs_level,
        cs_level,
        bs_region,
        cs_region,
        unilateral_emissions,
        cumulative_benefit,
        global_benefit,
        global_emissions,
        unilateral_emissions_GtCO2,
        levels,
        cost_climate_club,
        benefit_climate_club,
    )

    if to_csv:
        table = table.sort_values(by="net_benefit", ascending=False)
        table.to_csv(
            f"plots/country_specific_table{ext}.csv", index=False, float_format="%.5f"
        )
    return (
        cs_level,
        bs_level,
        names,
        cs_region,
        bs_region,
        names_region,
        unilateral_emissions_GtCO2,
    )


def do_country_specific_scc_part3():
    raise Exception("This is not yet sanity checked")
    emerging = "Emerging Market Countries"
    (
        cs_emerging,
        bs_emerging,
        names_emerging,
        _,
        _,
        _,
        _,
    ) = calculate_country_specific_scc_data(
        unilateral_actor=emerging,
        to_csv=False,
    )
    cs_emerging = dict(sorted(cs_emerging.items()))
    bs_emerging = dict(sorted(bs_emerging.items()))
    names_emerging = dict(sorted(names_emerging.items()))
    developing = "Developing Countries"
    (
        cs_developing,
        bs_developing,
        names_developing,
        _,
        _,
        _,
        _,
    ) = calculate_country_specific_scc_data(
        unilateral_actor=developing,
        to_csv=False,
    )
    cs_developing = dict(sorted(cs_developing.items()))
    bs_developing = dict(sorted(bs_developing.items()))
    names_developing = dict(sorted(names_developing.items()))
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(axs[0])
    for level, c in cs_emerging.items():
        total_c = sum(c)
        total_b = sum(bs_emerging[level])
        plt.plot(
            total_c, total_b, linewidth=0, marker="o", label=level, fillstyle="none"
        )
    # 45 degree line
    axs[0].axline([0, 0], [0.05, 0.05])
    plt.xlabel("PV country-specific costs (bln dollars)")
    plt.ylabel("PV country-specific benefits (bln dollars)")
    plt.title(emerging)

    plt.sca(axs[1])
    for level, c in cs_developing.items():
        total_c = sum(c)
        total_b = sum(bs_developing[level])
        plt.plot(
            total_c, total_b, linewidth=0, marker="o", label=level, fillstyle="none"
        )
    # 45 degree line
    axs[1].axline([0, 0], [0.05, 0.05])
    plt.xlabel("PV country-specific costs (bln dollars)")
    plt.ylabel("PV country-specific benefits (bln dollars)")
    plt.title(developing)

    # Deduplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(by_label.values()),
    )
    plt.tight_layout()

    util.savefig("country_specific_scatter_part3", tight=True)


def do_country_specific_scc_part4():
    # We save to CSV so that the data is shown in the website.
    cs, bs, _, _, bs_region, _, _ = calculate_country_specific_scc_data(
        unilateral_actor=None,
        ext="_part4",
        to_csv=False,
    )

    # Sanity check
    # SC4
    # The circles in the plot add up to baseline global benefit.
    # 114.04 is the global benefit under baseline settings.
    actual_global_benefit = sum(sum(b) for b in bs.values())
    assert math.isclose(
        actual_global_benefit, 114.04380627502013
    ), actual_global_benefit
    actual_global_benefit = sum(sum(b) for b in bs_region.values())
    # This is not 114.04, because PG and WS are not part of the 6 regions (they
    # are in Oceania).
    assert math.isclose(actual_global_benefit, 114.025120520287), actual_global_benefit
    # SC5
    # Cost
    global_cost = 29.03104345418096  # Hardcoded for fast computation.
    # Cost for Kosovo. To be excluded.
    cost_XK = 0.03241056086650181
    actual = sum(sum(c) for c in cs.values())
    assert math.isclose(actual, global_cost - cost_XK)

    plt.figure()
    ax = plt.gca()

    def mul_1000(x):
        return [i * 1e3 for i in x]

    # Multiplication by 1000 converts to billion dollars
    for level, c in cs.items():
        plt.plot(
            mul_1000(c),
            mul_1000(bs[level]),
            linewidth=0,
            marker="o",
            label=level,
            fillstyle="none",
        )
    # 45 degree line
    ax.axline([0, 0], [1, 1])
    plt.xlabel("PV country costs (bln dollars)")
    plt.ylabel("PV country benefits (bln dollars)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    axis_limit = 45_000
    plt.xlim(5e-2, axis_limit)
    plt.ylim(5e-2, axis_limit)
    plt.legend()
    util.savefig("country_specific_scatter_part4")


def calculate_global_benefit(last_year=None):
    if last_year is None:
        last_year = 2100
    out = analysis_main.run_cost1(x=1, to_csv=False, do_round=False, plot_yearly=False)
    chosen_s2_scenario = f"2022-{last_year} 2DII + Net Zero 2050 Scenario"
    property = "Benefits of avoiding coal emissions including residual benefit (in trillion dollars)"
    global_benefit = out[property][chosen_s2_scenario]
    return global_benefit


def do_country_specific_scc_part5():
    regions = [
        "Asia",
        "Africa",
        "North America",
        "Latin America & the Carribean",
        "Europe",
        "Australia & New Zealand",
    ]

    git_branch = util.get_git_branch()
    fname = f"cache/country_specific_data_part5_git_{git_branch}.json"
    if os.path.isfile(fname):
        content = util.read_json(fname)
        cs_region_combined = content["unilateral_cost"]
        bs_region_combined = content["unilateral_benefit"]
        zerocost = content["freeloader_benefit"]
        global_benefit_by_region = content["global_benefit"]
    else:
        cs_region_combined = {}
        bs_region_combined = {}
        zerocost = {}

        # We round everything to 6 digits, and so the precision is thousand dollars.
        def do_round(x):
            return round(x, 6)

        # This variable is for sanity check
        unilateral_emissions_cumulative = 0.0

        for group in regions:
            unilateral_actor = group
            (
                _,
                _,
                _,
                cs_region,
                bs_region,
                _,
                unilateral_emissions_GtCO2,
            ) = calculate_country_specific_scc_data(
                unilateral_actor=unilateral_actor,
                ext="",
                to_csv=False,
            )
            cs_region_combined[group] = do_round(sum(cs_region[group]))
            bs_region_combined[group] = do_round(sum(bs_region[group]))
            # group is the entity doing the unilateral action, while the
            # regions inside the dict is the one who gets benefit with zero
            # cost.
            zerocost[group] = {
                k: do_round(sum(v)) for k, v in bs_region.items() if k != group
            }
            unilateral_emissions_cumulative += unilateral_emissions_GtCO2

        # Sanity check
        # SC6
        # Make sure the unilateral emissions, if summed, is the same as baseline result.
        # It's not 1425.55 because XK is excluded from the 6 regions.
        assert math.isclose(
            unilateral_emissions_cumulative, 1424.1494924127742
        ), unilateral_emissions_cumulative

        # This code chunk is used to calculate global_benefit_by_region
        global_benefit = calculate_global_benefit()
        scc_dict = read_country_specific_scc_filtered()
        unscaled_global_scc = sum(scc_dict.values())
        iso3166_df = util.read_iso3166()
        region_countries_map, _ = analysis_main.prepare_regions_for_climate_financing(
            iso3166_df
        )
        global_benefit_by_region = {}
        # End of global_benefit_by_region preparation

        for region in cs_region_combined.keys():
            countries = region_countries_map[region]
            scc_scale = (
                sum(scc_dict.get(c, 0.0) for c in countries) / unscaled_global_scc
            )
            # Benefit to 1 region if everyone in the world takes action
            global_benefit_by_region[region] = do_round(global_benefit * scc_scale)

        with open(fname, "w") as f:
            json.dump(
                {
                    "unilateral_cost": cs_region_combined,
                    "unilateral_benefit": bs_region_combined,
                    "freeloader_benefit": zerocost,
                    "global_benefit": global_benefit_by_region,
                },
                f,
            )

    # Sanity check
    # SC7
    # The closed circles in the plot add up to baseline global benefit.
    # This is not 114.04 (baseline number), because PG and WS are not part of
    # the 6 regions (they are in Oceania).
    sum_global_benefit_by_region = sum(global_benefit_by_region.values())
    # rel_tol is 1e-6 because previously we round everything to 6 decimals.
    assert math.isclose(
        sum_global_benefit_by_region, 114.025120, rel_tol=1e-6
    ), sum_global_benefit_by_region
    # SC8
    # The open circles and the lots of circles in the left plot add up to
    # baseline global benefit.
    all_freeloader_benefit = sum(sum(g.values()) for g in zerocost.values())
    all_unilateral_benefit = sum(bs_region_combined.values())
    actual_benefit = all_freeloader_benefit + all_unilateral_benefit
    # This is not 114.04, because PG, WS, and XK are excluded. If they are
    # temporarily included, this should be 114.04.
    # rel_tol is 1e-6 because previously we round everything to 6 decimals.
    assert math.isclose(actual_benefit, 113.913292, rel_tol=1e-6), actual_benefit
    # SC9
    # Cost
    global_cost = 29.03104345418096  # Hardcoded for fast computation.
    # Cost for Kosovo. To be excluded.
    cost_XK = 0.03241056086650181
    actual = sum(cs_region_combined.values())
    # rel_tol is 1e-6 because previously we round everything to 6 decimals.
    assert math.isclose(actual, global_cost - cost_XK, rel_tol=1e-6)

    # For conversion from trillion to billion dollars
    def mul_1000(x):
        return [i * 1e3 for i in x]

    def mul_1000_scalar(x):
        return x * 1e3

    right = 2.5
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(4 * (1 + right) / right, 4),
        gridspec_kw={"width_ratios": [1, right]},
    )
    ax = axs[1]
    plt.sca(ax)
    # Reset color cycler
    ax.set_prop_cycle(None)
    # Unilateral benefit
    for region, c in cs_region_combined.items():
        label = region
        plt.plot(
            mul_1000_scalar(c),
            mul_1000_scalar(bs_region_combined[region]),
            linewidth=0,
            marker="o",
            label=label,
            fillstyle="none",
        )
        benefit = bs_region_combined[region]
        break_even_scc = c * util.social_cost_of_carbon / benefit

        print(
            "self",
            region,
            "cost",
            c,
            "benefit",
            benefit,
            "break-even scc",
            break_even_scc,
        )

    # We plot the global benefit
    ax.set_prop_cycle(None)
    for region, c in cs_region_combined.items():
        plt.plot(
            mul_1000_scalar(c),
            mul_1000_scalar(global_benefit_by_region[region]),
            linewidth=0,
            marker="o",
        )

    print("Freeloader benefit", zerocost)
    print("Unilateral benefit", bs_region_combined)
    print("Global benefit", global_benefit_by_region)

    # 45 degree line
    ax.axline([0.1, 0.1], [1, 1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    y_min, y_max = ax.get_ylim()
    axis_limit = y_max + 4
    plt.xlim(20, axis_limit)
    plt.ylim(20, axis_limit)
    plt.xlabel("PV country costs (bln dollars)")
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=2,
    )
    plt.tight_layout()

    # zerocost
    ax = axs[0]
    plt.sca(ax)
    ax.set_yscale("log")
    plt.ylim(20, axis_limit)
    transposed = defaultdict(list)
    transposed_xs = defaultdict(list)
    markersize = 8
    for i, (group, content) in enumerate(zerocost.items()):
        xs = [i / 5 for _ in range(len(content))]
        plt.plot(
            xs,
            mul_1000(content.values()),
            linewidth=0,
            marker="o",
            label=group,
            fillstyle="none",
            markersize=markersize,
        )
        for k, v in content.items():
            transposed[k].append(v)
            transposed_xs[k].append(i / 5)
    # Reset color cycler
    ax.set_prop_cycle(None)
    for k in zerocost:
        plt.plot(
            transposed_xs[k],
            mul_1000(transposed[k]),
            linewidth=0,
            marker="o",
            label=f"{k} t",
            # fillstyle="none",
            markersize=markersize * 0.3,
        )

    # Finishing touch
    x_min, x_max = ax.get_xlim()
    x_min -= 0.1
    x_max *= 1.45
    x_mid = (x_min + x_max) / 2
    plt.xlim(x_min, x_max)
    ax.set_yticklabels([])
    plt.xticks([x_mid], [0])
    plt.subplots_adjust(wspace=0)
    plt.ylabel("PV country benefits (bln dollars)")

    util.savefig("country_specific_scatter_part5", tight=True)


def do_country_specific_scc_part6():
    top9 = "CN US IN AU RU ID ZA DE KZ".split()

    (
        _,
        iso3166_df_alpha2,
        _,
        _,
        _,
    ) = util.prepare_from_climate_financing_data()
    alpha2_to_full_name = iso3166_df_alpha2["name"].to_dict()

    git_branch = util.get_git_branch()
    fname = f"cache/country_specific_data_part6_git_{git_branch}.json"
    (
        cs_combined,
        bs_combined,
        zerocost,
        global_benefit_by_country,
    ) = common_prepare_cost_benefit_by_country(fname, top9)

    # Sanity check
    # SC11
    # Hardcoded for fast sanity check computation.
    # Global deal but benefit only to top9
    global_benefit = 114.04380627502013
    sum_global_benefit_by_country = sum(global_benefit_by_country.values())
    scc_dict = read_country_specific_scc_filtered()
    unscaled_global_scc = sum(scc_dict.values())
    expected = global_benefit * sum(scc_dict[c] for c in top9) / unscaled_global_scc
    # This is global action, but benefit only to top9
    assert math.isclose(sum_global_benefit_by_country, expected)

    # For conversion from trillion to billion dollars
    def mul_1000(x):
        return [i * 1e3 for i in x]

    def mul_1000_scalar(x):
        return x * 1e3

    right = 2.5
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(4 * (1 + right) / right, 4),
        gridspec_kw={"width_ratios": [1, right]},
    )
    ax = axs[1]
    plt.sca(ax)
    # Reset color cycler
    ax.set_prop_cycle(None)
    fillstyle = "none"
    for group, c in cs_combined.items():
        label = alpha2_to_full_name[group]
        if label is not None:
            label = label.replace("United States of America", "USA")
        plt.plot(
            mul_1000_scalar(c),
            mul_1000_scalar(bs_combined[group]),
            linewidth=0,
            marker="o",
            label=label,
            fillstyle=fillstyle,
        )
        benefit = bs_combined[group]
        break_even_scc = c * util.social_cost_of_carbon / benefit
        print(
            "self",
            group,
            "cost",
            c,
            "benefit",
            benefit,
            "break-even scc",
            break_even_scc,
        )

    print("Unilateral benefit", bs_combined)
    print("Global benefit", global_benefit_by_country)

    # We plot the global benefit
    ax.set_prop_cycle(None)
    for group, c in cs_combined.items():
        plt.plot(
            mul_1000_scalar(c),
            mul_1000_scalar(global_benefit_by_country[group]),
            linewidth=0,
            marker="o",
        )

    # 45 degree line
    ax.axline([0.1, 0.1], [1, 1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.xlabel("PV country costs (bln dollars)")
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
    )
    plt.tight_layout()
    y_min_ori, y_max_ori = ax.get_ylim()

    # zerocost
    ax = axs[0]
    plt.sca(ax)
    ax.set_yscale("log")
    for i, (k, v) in enumerate(zerocost.items()):
        plt.plot(
            [i / 5 for _ in range(len(v))],
            mul_1000(v.values()),
            linewidth=0,
            marker="o",
            label=k,
            fillstyle="none",
            markersize=4.8,
        )
    y_min_zero, y_max_zero = ax.get_ylim()
    y_min, y_max = min(y_min_ori, y_min_zero), max(y_max_ori, y_max_zero)

    plt.ylim(y_min, y_max)

    # Finishing touch
    x_min, x_max = ax.get_xlim()
    x_min -= 0.1
    x_max *= 1.45
    x_mid = (x_min + x_max) / 2
    plt.xlim(x_min, x_max)
    ax.set_yticklabels([])
    plt.xticks([x_mid], [0])
    plt.subplots_adjust(wspace=0)
    plt.ylabel("PV country benefits (bln dollars)")

    # Rescale original plot to match zerocost range
    plt.sca(axs[1])
    plt.xlim(y_min, y_max)
    plt.ylim(y_min, y_max)

    util.savefig("country_specific_scatter_part6", tight=True)


def do_country_specific_scc_part7(
    country_doing_action, last_year=None, use_developed_for_zerocost=False
):
    """
    use_developed_for_zerocost makes it so that only the developed countries
    are shown in the zerocost plot.
    """
    (
        _,
        iso3166_df_alpha2,
        _,
        _,
        _,
    ) = util.prepare_from_climate_financing_data()
    alpha2_to_full_name = iso3166_df_alpha2["name"].to_dict()

    # We round to 3 decimal digits instead of 6 because the unit is billion
    # dollars.
    def round3(x):
        return round(x, 3)

    cs, bs, names, _, _, _, unilateral_emissions = calculate_country_specific_scc_data(
        unilateral_actor=country_doing_action,
        ext="",
        to_csv=False,
        last_year=last_year,
    )
    cost_country = None
    benefit_country = None
    for level, level_names in names.items():
        if country_doing_action not in level_names:
            continue
        location = level_names.index(country_doing_action)
        # Billion dollars
        cost_country = cs[level][location] * 1e3
        benefit_country = bs[level][location] * 1e3
        break

    # Calculating zerocost
    if use_developed_for_zerocost:
        level = "Developed Countries"
        # Billion dollars
        zerocost = [i * 1e3 for i in bs[level]]
    else:
        zerocost = defaultdict(float)
        zerocost_benefit_eu = 0.0
        zerocost_benefit_world = 0.0
        for level, level_names in names.items():
            location = None
            if country_doing_action in level_names:
                location = level_names.index(country_doing_action)
            for i, c in enumerate(level_names):
                # Billion dollars
                benefit_zc = bs[level][i] * 1e3
                assert c not in zerocost
                if i == location:
                    # This is the country doing the action, and so, not part of
                    # zerocost.
                    continue
                zerocost_benefit_world += benefit_zc
                if c in EU:
                    # WARNING This assumes the country doing the action is not part
                    # of EU.
                    zerocost_benefit_eu += benefit_zc
                    EU.remove(c)
                if c in G7:
                    zerocost[c] = benefit_zc
        zerocost["G7"] = sum(zerocost[c] for c in G7)
        zerocost["EU"] = zerocost_benefit_eu
        zerocost["ROW"] = zerocost_benefit_world - zerocost["G7"] - benefit_country
        # Round to 3 decimal places
        zerocost = {k: round3(v) for k, v in zerocost.items()}
    print(zerocost)

    # This code chunk is used to calculate global_benefit_by_country
    global_benefit = calculate_global_benefit(last_year=last_year)
    scc_dict = read_country_specific_scc_filtered()
    unscaled_global_scc = sum(scc_dict.values())
    # End of global_benefit_by_country preparation

    # Benefit to 1 country if everyone in the world takes action
    # Billion dollars
    global_benefit_country = (
        global_benefit * scc_dict[country_doing_action] / unscaled_global_scc * 1e3
    )

    if not use_developed_for_zerocost:
        # Sanity check
        # SC12
        world_benefit_from_unilateral_action = zerocost_benefit_world + benefit_country
        expected = unilateral_emissions * util.social_cost_of_carbon
        assert math.isclose(world_benefit_from_unilateral_action, expected), (
            world_benefit_from_unilateral_action,
            expected,
        )

    make_common_freeloader_plot(
        alpha2_to_full_name,
        country_doing_action,
        cost_country,
        benefit_country,
        global_benefit_country,
        zerocost,
        use_developed_for_zerocost=use_developed_for_zerocost,
    )

    util.savefig(f"country_specific_scatter_part7_{country_doing_action}", tight=True)
    return {
        "unilateral_cost": cost_country,
        "unilateral_benefit": benefit_country,
        "global_benefit": global_benefit_country,
        "freeloader_benefit": zerocost,
    }


def common_prepare_cost_benefit_by_country(
    fname, countries, last_year=None, cost_name="cost"
):
    if os.path.isfile(fname):
        content = util.read_json(fname)
        cs_combined = content["unilateral_cost"]
        bs_combined = content["unilateral_benefit"]
        zerocost = content["freeloader_benefit"]
        global_benefit_by_country = content["global_benefit"]
    else:
        cs_combined = {}
        bs_combined = {}
        zerocost = {}
        # We round everything to 6 digits, and so the precision is thousand dollars.
        cumulative_emissions = 0.0
        for country_doing_action in countries:
            unilateral_actor = country_doing_action
            cs, bs, names, _, _, _, emissions = calculate_country_specific_scc_data(
                unilateral_actor=unilateral_actor,
                ext="",
                to_csv=False,
                last_year=last_year,
                cost_name=cost_name,
            )
            cumulative_emissions += emissions
            for level, level_names in names.items():
                if country_doing_action not in level_names:
                    continue
                location = level_names.index(country_doing_action)
                cs_combined[country_doing_action] = round(cs[level][location], 6)
                bs_combined[country_doing_action] = round(bs[level][location], 6)
            # Calculating zerocost
            _dict = {}
            for level, level_names in names.items():
                location = None
                if country_doing_action in level_names:
                    location = level_names.index(country_doing_action)
                for i, c in enumerate(level_names):
                    assert c not in _dict
                    if i == location:
                        continue
                    _dict[c] = round(bs[level][i], 6)
            zerocost[country_doing_action] = _dict

        # Sanity check
        # SC10
        # This is unilateral action by `countries`, but benefit to the world.
        all_freeloader_benefit = sum(sum(g.values()) for g in zerocost.values())
        all_unilateral_benefit = sum(bs_combined.values())
        actual_benefit = all_freeloader_benefit + all_unilateral_benefit
        expected = cumulative_emissions / 1e3 * util.social_cost_of_carbon
        # It is only approximately the same with 1e-5 tolerance, because the
        # benefit is rounded to 6 decimal points. See the previous lines.
        assert math.isclose(actual_benefit, expected, rel_tol=2e-5), (
            actual_benefit,
            expected,
        )

        # This code chunk is used to calculate global_benefit_by_country
        global_benefit = calculate_global_benefit()
        scc_dict = read_country_specific_scc_filtered()
        unscaled_global_scc = sum(scc_dict.values())
        global_benefit_by_country = {}
        # End of global_benefit_by_country preparation

        for c in cs_combined.keys():
            # Benefit to 1 country if everyone in the world takes action
            global_benefit_by_country[c] = (
                global_benefit * scc_dict[c] / unscaled_global_scc
            )

        with open(fname, "w") as f:
            json.dump(
                {
                    "unilateral_cost": cs_combined,
                    "unilateral_benefit": bs_combined,
                    "freeloader_benefit": zerocost,
                    "global_benefit": global_benefit_by_country,
                },
                f,
            )
    return cs_combined, bs_combined, zerocost, global_benefit_by_country


def do_country_specific_scc_part8():
    levels, levels_map, iso3166_df = prepare_level_development()
    cs, bs, _, _, bs_region, _, _ = calculate_country_specific_scc_data(
        unilateral_actor=None,
        to_csv=False,
    )

    fig = plt.figure()
    ax = plt.gca()

    def mul_1000(x):
        return [i * 1e3 for i in x]

    # Global action
    # Multiplication by 1000 converts to billion dollars
    for level in levels:
        plt.plot(
            mul_1000(cs[level]),
            mul_1000(bs[level]),
            linewidth=0,
            marker="o",
            label=level,
            # fillstyle="none",
        )

    # Local action
    # Reset color cycler
    ax.set_prop_cycle(None)
    git_branch = util.get_git_branch()
    for level_name, countries in levels_map.items():
        # Filter countries to only for the ones we have data for
        countries = [c for c in countries if c in unilateral_countries]
        fname = f"cache/country_specific_data_part8_git_{git_branch}_{level_name}.json"
        (
            cs_combined,
            bs_combined,
            zerocost,
            global_benefit_by_country,
        ) = common_prepare_cost_benefit_by_country(fname, countries)
        plt.plot(
            mul_1000(cs_combined.values()),
            mul_1000(bs_combined.values()),
            linewidth=0,
            marker="o",
            fillstyle="none",
        )

    # 45 degree line
    ax.axline([0, 0], [1, 1])
    plt.xlabel("PV country costs (bln dollars)")
    plt.ylabel("PV country benefits (bln dollars)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    axis_limit = 45_000
    plt.xlim(5e-4, axis_limit)
    plt.ylim(5e-4, axis_limit)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
    )
    plt.tight_layout()
    util.savefig(f"country_specific_scatter_part8_git_{git_branch}", tight=True)


def make_common_freeloader_plot(
    alpha2_to_full_name,
    country_doing_action,
    cost,
    benefit,
    global_benefit,
    zerocost,
    use_developed_for_zerocost,
):
    right = 2.5
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(4 * (1 + right) / right, 4),
        gridspec_kw={"width_ratios": [1, right]},
    )
    ax = axs[1]
    plt.sca(ax)
    # Reset color cycler
    ax.set_prop_cycle(None)
    label = alpha2_to_full_name[country_doing_action]
    plt.plot(
        cost,
        benefit,
        linewidth=0,
        marker="o",
        label=label,
        fillstyle="none",
    )

    # We plot the global benefit
    ax.set_prop_cycle(None)
    plt.plot(
        cost,
        global_benefit,
        linewidth=0,
        marker="o",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.xlabel("PV country costs (bln dollars)")
    y_min_ori, y_max_ori = ax.get_ylim()
    # 45 degree line
    ax.axline([y_min_ori, y_min_ori], [8, 8])

    # zerocost
    ax = axs[0]
    plt.sca(ax)
    ax.set_yscale("log")
    if use_developed_for_zerocost:
        # This is for part 7 when the zerocost is the entire developed countries
        plt.plot(
            [0] * len(zerocost),
            zerocost,
            color="tab:orange",
            linewidth=0,
            marker="o",
            fillstyle="none",
            markersize=4.8,
        )
    else:
        for k, v in zerocost.items():
            k = k.replace("US", "USA")
            if k == "GB":
                label = "GB"
            else:
                label = alpha2_to_full_name.get(k, k)
            plt.plot(
                0,
                v,
                linewidth=0,
                marker="o",
                label=label,
                fillstyle="none",
                markersize=4.8,
            )
    y_min_zero, y_max_zero = ax.get_ylim()
    y_min, y_max = min(y_min_ori, y_min_zero), max(y_max_ori, y_max_zero)
    # Increase y_max because the right plot has the circles at the edges.
    y_max *= 1.1
    # Decrease y_min slightly
    y_min *= 0.9

    plt.ylim(y_min, y_max)

    # Finishing touch
    x_min, x_max = ax.get_xlim()
    x_mid = (x_min + x_max) / 2
    ax.set_yticklabels([])
    plt.xticks([x_mid], [0])
    plt.subplots_adjust(wspace=0)
    plt.ylabel("PV country benefits (bln dollars)")

    # Rescale original plot to match zerocost range
    plt.sca(axs[1])
    plt.xlim(y_min, y_max)
    plt.ylim(y_min, y_max)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
    )


def do_country_specific_scc_part9():
    (
        cs,
        bs,
        names,
        _,
        _,
        _,
        unilateral_emissions_GtCO2,
    ) = calculate_country_specific_scc_data(
        unilateral_actor="EMDE",
        ext="",
        to_csv=False,
    )
    git_branch = util.get_git_branch()
    cost_EMDE = (sum(cs["EMDE"]),)
    benefit_EMDE = sum(bs["EMDE"])
    # Global benefit
    global_benefit = calculate_global_benefit()
    scc_dict = read_country_specific_scc_filtered()
    unscaled_global_scc = sum(scc_dict.values())
    countries = names["EMDE"]
    scc_scale = sum(scc_dict.get(c, 0.0) for c in countries) / unscaled_global_scc
    global_benefit_EMDE = global_benefit * scc_scale
    # End global benefit

    right = 2.5
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(4 * (1 + right) / right, 4),
        gridspec_kw={"width_ratios": [1, right]},
    )
    ax = axs[1]
    plt.sca(ax)
    # Reset color cycler
    ax.set_prop_cycle(None)
    plt.plot(
        cost_EMDE,
        benefit_EMDE,
        linewidth=0,
        marker="o",
        label="EMDE",
        fillstyle="none",
    )

    # We plot the global benefit
    ax.set_prop_cycle(None)
    plt.plot(
        cost_EMDE,
        global_benefit_EMDE,
        linewidth=0,
        marker="o",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.xlabel("PV costs (bln dollars)")
    y_min_ori, y_max_ori = ax.get_ylim()
    # 45 degree line
    ax.axline([y_min_ori, y_min_ori], [8, 8])

    # zerocost
    freeloader_benefit_DE = sum(bs["Developed Countries"])
    # Sanity check
    assert math.isclose(freeloader_benefit_DE + global_benefit_EMDE, global_benefit)
    ax = axs[0]
    plt.sca(ax)
    ax.set_yscale("log")
    plt.plot(
        0,
        freeloader_benefit_DE,
        linewidth=0,
        marker="o",
        label="DE",
        color="tab:orange",
        fillstyle="none",
        markersize=4.8,
    )

    # Freeloader benefit for individual developed countries
    plt.plot(
        [0] * len(bs["Developed Countries"]),
        bs["Developed Countries"],
        linewidth=0,
        marker="o",
        label="Individual DE countries",
        color="tab:green",
        fillstyle="none",
        markersize=4.8,
    )

    y_min_zero, y_max_zero = ax.get_ylim()
    y_min, y_max = min(y_min_ori, y_min_zero), max(y_max_ori, y_max_zero)
    # Increase y_max because the right plot has the circles at the edges.
    y_max *= 1.1
    # Decrease y_min slightly
    y_min *= 0.9

    plt.ylim(y_min, y_max)

    # Finishing touch
    x_min, x_max = ax.get_xlim()
    x_mid = (x_min + x_max) / 2
    ax.set_yticklabels([])
    plt.xticks([x_mid], [0])
    plt.subplots_adjust(wspace=0)
    plt.ylabel("PV benefits (bln dollars)")

    # Rescale original plot to match zerocost range
    plt.sca(axs[1])
    plt.xlim(y_min, y_max)
    plt.ylim(y_min, y_max)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
    )

    util.savefig(f"country_specific_scatter_part9_git_{git_branch}", tight=True)


def do_bruegel_heatmap():
    (
        _,
        _,
        _,
        _,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()
    alpha2_to_full_name = prepare_alpha2_to_full_name_concise()

    # Exclude developed countries
    # Reduce from 60 to 47
    by_avoided_emissions_emde = [
        c for c in by_avoided_emissions if c not in developed_country_shortnames
    ]

    emerging_country_shortnames = util.get_emerging_countries()
    developING_country_shortnames = util.get_developing_countries()
    emde = emerging_country_shortnames + developING_country_shortnames
    emde_minus_cn = [c for c in emde if c != "CN"]
    emde_minus_cn_in = [c for c in emde if c not in ["CN", "IN"]]

    git_branch = util.get_git_branch()

    fname = f"cache/country_specific_data_bruegel_git_{git_branch}.json"
    (
        cs_combined,
        _,
        zerocost,
        _,
    ) = common_prepare_cost_benefit_by_country(fname, by_avoided_emissions_emde)

    EU_and_US = EU + ["US"]

    # fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    plt.figure(figsize=(15, 15))

    for i in range(1):
        # if i == 0:
        #    topn = by_avoided_emissions_emde[:23]
        # else:
        #    topn = by_avoided_emissions_emde[23:]
        topn = by_avoided_emissions_emde

        # We split the heatmap into 2: first 20 developed, and last 20.
        for j in range(1):
            # plt.sca(axs[i, j])

            # if j == 0:
            #    developed_subset_y = developed_country_shortnames[:20]
            # else:
            #    developed_subset_y = developed_country_shortnames[20:]
            developed_subset_y = developed_country_shortnames

            # y axis
            donor_groups = [
                developed_country_shortnames,
                G7,
                EU_and_US,
                EU,
                eurozone,
                EU[:15],
                EU[:10],
                EU[:5],
                *[[c] for c in developed_subset_y],
            ]

            def get_net_benefit_composite(action_group):
                net_benefit = 0
                for country_doing_action, v in zerocost.items():
                    if country_doing_action not in action_group:
                        continue
                    net_benefit += (
                        sum(vv for kk, vv in v.items() if kk in group)
                        - cs_combined[country_doing_action]
                    )
                return net_benefit

            matrix = []
            for group in donor_groups:
                net_benefits = []

                # EMDE
                net_benefits.append(get_net_benefit_composite(emde))

                # EMDE - CN
                net_benefits.append(get_net_benefit_composite(emde_minus_cn))

                # EMDE - CN - IN
                net_benefits.append(get_net_benefit_composite(emde_minus_cn_in))

                # Individual countries
                for country_doing_action in topn:
                    if country_doing_action in developed_country_shortnames:
                        # Skip developed countries
                        continue
                    freeloader_benefit = sum(
                        vv
                        for kk, vv in zerocost[country_doing_action].items()
                        if kk in group
                    )
                    # In trillion dollars
                    net_benefit = freeloader_benefit - cs_combined[country_doing_action]
                    # For conversion from trillion to billion dollars
                    # net_benefit *= 1e3
                    net_benefits.append(net_benefit)

                matrix.append(net_benefits)

            yticklabels = [
                "D",
                "G7",
                "EU+US",
                "EU",
                "Eurozone",
                "EU top 15",
                "EU top 10",
                "EU top 5",
                *[alpha2_to_full_name[c] for c in developed_subset_y],
            ]

            sns.heatmap(
                matrix,
                xticklabels=["EMDE", "EMDE-CN", "EMDE-CN-IN"]
                + [
                    alpha2_to_full_name[c]
                    for c in topn
                    if c not in developed_country_shortnames
                ],
                yticklabels=yticklabels,
                annot=True,
                cmap="tab20c",
                fmt=".1f",
                annot_kws={"fontsize": 5},
                vmin=-20.2,  # -20.177
                vmax=37.5,  # 37.3  this is for global scc of 80
                # vmax=489.1,  # 489.02  this is for global scc from the Nature paper
            )
    util.savefig("bruegel", tight=True)
    plt.close()


def do_bruegel_2():
    git_branch = util.get_git_branch()
    if analysis_main.ENABLE_COAL_EXPORT:
        ae_csv_path = "./plots/avoided_emissions_modified_by_coal_export.csv"
        suffix = "_modified_with_coal_export"
        all_obj = by_avoided_emissions
    else:
        ae_csv_path = "./plots/avoided_emissions_nonadjusted.csv"
        suffix = ""
        all_obj = by_avoided_emissions
    avoided_emissions = pd.read_csv(ae_csv_path)

    (
        _,
        iso3166_df_alpha2,
        _,
        _,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()
    alpha2_to_full_name = iso3166_df_alpha2["name"].to_dict()
    emerging_country_shortnames = util.get_emerging_countries()
    developING_country_shortnames = util.get_developing_countries()
    emde = emerging_country_shortnames + developING_country_shortnames
    emde_fullname = [alpha2_to_full_name.get(c, c) for c in emde]

    ae_emde = avoided_emissions[avoided_emissions.Country.isin(emde_fullname)]
    ae_emde.to_csv(
        f"plots/bruegel/bruegel_2_{git_branch}_avoided_emissions_emde{suffix}.csv"
    )

    # SCC of EMDE
    scc_dict = read_country_specific_scc_filtered()
    unscaled_global_scc = sum(scc_dict.values())
    scc_80_dict = None
    for name, filter_countries in [
        ("all", list(scc_dict.keys())),
        ("emde", emde),
        ("developed", developed_country_shortnames),
    ]:
        filtered_scc = {k: v for k, v in scc_dict.items() if k in filter_countries}
        data = {
            "name": [alpha2_to_full_name[c] for c in filtered_scc.keys()],
            "share (%)": [
                round(v / unscaled_global_scc * 100, 2) for v in filtered_scc.values()
            ],
            "absolute (total 80)": [
                round(v / unscaled_global_scc * 80, 2) for v in filtered_scc.values()
            ],
            "absolute (total nature paper)": [
                round(v, 2) for v in filtered_scc.values()
            ],
        }
        _df = pd.DataFrame.from_dict(data)
        _df.to_csv(f"plots/bruegel/bruegel_2_{git_branch}_scc_{name}{suffix}.csv")
        if name == "all":
            scc_80_dict = _df.set_index("name")["absolute (total 80)"].to_dict()

    def func(row):
        full_name = row.Country
        scc = scc_80_dict.get(full_name, 0)
        row["2100"] *= scc
        row["2050"] *= scc
        row["2030"] *= scc
        return row

    benefit_emde = ae_emde.apply(func, axis=1)
    benefit_emde.to_csv(
        f"plots/bruegel/bruegel_2_{git_branch}_benefit_emde{suffix}.csv"
    )
    benefit_all = avoided_emissions.apply(func, axis=1)
    benefit_all.to_csv(f"plots/bruegel/bruegel_2_{git_branch}_benefit_all{suffix}.csv")

    # Cost
    def get_cost(cost_name, obj, obj_name):
        fname = f"cache/country_specific_data_bruegel_git_{git_branch}_{last_year}_{cost_name}_{obj_name}{suffix}.json"
        (
            cs_combined,
            _,
            zerocost,
            _,
        ) = common_prepare_cost_benefit_by_country(
            fname,
            obj,
            last_year=last_year,
            cost_name=cost_name,
        )
        return cs_combined

    emde_sorted_by_avoided_emissions = [c for c in by_avoided_emissions if c in emde]
    for name, obj in [
        ("emde", emde_sorted_by_avoided_emissions),
        ("all", all_obj),
    ]:
        cs_by_last_year_total = {}
        cs_by_last_year_investment_cost = {}
        for last_year in [2030, 2050, 2100]:
            cs_combined_total = get_cost("cost", obj, name)
            cs_combined_investment_cost = get_cost("investment_cost", obj, name)
            cs_by_last_year_total[last_year] = cs_combined_total
            cs_by_last_year_investment_cost[last_year] = cs_combined_investment_cost

        with open(
            f"plots/bruegel/bruegel_2_{git_branch}_cost_{name}{suffix}.csv", "w"
        ) as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [
                    "index",
                    "country",
                    "opportunity_cost_2030",
                    "opportunity_cost_2050",
                    "opportunity_cost_2100",
                    "investment_cost_2030",
                    "investment_cost_2050",
                    "investment_cost_2100",
                    "total_cost_2030",
                    "total_cost_2050",
                    "total_cost_2100",
                ]
            )
            last_years = [2030, 2050, 2100]
            for i, a2 in enumerate(obj):
                full_name = alpha2_to_full_name[a2]
                opportunity_costs = [
                    round(
                        cs_by_last_year_total[last_year][a2]
                        - cs_by_last_year_investment_cost[last_year][a2],
                        6,
                    )
                    for last_year in last_years
                ]
                investment_costs = [
                    cs_by_last_year_investment_cost[last_year][a2]
                    for last_year in last_years
                ]
                total_costs = [
                    cs_by_last_year_total[last_year][a2] for last_year in last_years
                ]
                csvwriter.writerow(
                    [i, full_name, *opportunity_costs, *investment_costs, *total_costs]
                )


def do_bruegel_4(action_groups):
    git_branch = util.get_git_branch()

    # For gov cost breakdown
    gdp2022 = util.read_json("./data/all_countries_gdp_marketcap_2022.json")
    gdp2022["UK"] = gdp2022["GB"]
    # Data source https://www.imf.org/external/datamapper/GG_DEBT_GDP@GDD/CHN/FRA/DEU/ITA/JPN/GBR/USA/FADGDWORLD
    imf_gov_debt_2022 = (
        pd.read_csv("./bruegel/imf-dm-export-20240205.csv")[
            ["General Government Debt (Percent of GDP)", "2022"]
        ]
        .set_index("General Government Debt (Percent of GDP)")["2022"]
        .to_dict()
    )
    alpha2_to_full_name = prepare_alpha2_to_full_name_concise()
    full_name_to_alpha2 = {v: k for k, v in alpha2_to_full_name.items()}
    amendment = {
        "Türkiye, Republic of": "TR",
        "Bosnia and Herzegovina": "BA",
        "China, People's Republic of": "CN",
        "Congo, Dem. Rep. of the": "CD",
        "Czech Republic": "CZ",
        "Korea, Republic of": "KR",
        "Kyrgyz Republic": "KG",
        "Micronesia, Fed. States of": "FM",
        "Moldova": "MD",
        "North Macedonia ": "MK",
        "Russian Federation": "RU",
        "Slovak Republic": "SK",
        "Taiwan Province of China": "TW",
        "United Kingdom": "GB",
        "United States": "US",
        "Vietnam": "VN",
    }
    full_name_to_alpha2 = {**full_name_to_alpha2, **amendment}

    imf_gov_debt_2022_percent = {
        full_name_to_alpha2[k]: v for k, v in imf_gov_debt_2022.items()
    }
    imf_gov_debt_2022_percent["UK"] = imf_gov_debt_2022_percent["GB"]
    # Missing data
    # Taken from https://www.ceicdata.com/en/indicator/south-africa/government-debt--of-nominal-gdp
    imf_gov_debt_2022_percent["ZA"] = 71.17
    # Taken from https://www.ceicdata.com/en/mozambique/government-finance-statistics
    imf_gov_debt_2022_percent["MZ"] = 130.30
    # Taken from https://www.ceicdata.com/en/indicator/botswana/national-government-debt
    imf_gov_debt_2022_percent["BW"] = 3581.92
    # End of for gov cost breakdown

    # For info, can be ignored
    if False:
        print("Country", "Gov_debt_percent", "GDP_2022")
        for country in ["CN", "JP", "GB", "US"]:
            print(
                alpha2_to_full_name[country],
                round(imf_gov_debt_2022_percent[country], 2),
                round(gdp2022[country] / 1e12, 2),
            )
        eu_gdp = sum(gdp2022[country] for country in EU)
        print(
            "EU",
            round(
                sum(
                    gdp2022[country] * imf_gov_debt_2022_percent[country]
                    for country in EU
                )
                / eu_gdp,
                2,
            ),
            round(eu_gdp / 1e12, 2),
        )
        exit()

    for public_funding_fraction in [1, 0.5, 0.2, 0.1]:
        # for last_year in [2030, 2050, 2100]:
        for last_year in [2030, 2050]:
            fname = f"cache/country_specific_data_bruegel_git_{last_year}_{git_branch}_cost.json"
            (
                cs_combined,
                _,
                zerocost,
                _,
            ) = common_prepare_cost_benefit_by_country(
                fname, by_avoided_emissions, last_year=last_year
            )
            fname = f"cache/country_specific_data_bruegel_git_{last_year}_{git_branch}_investment_cost.json"
            investment_cost = common_prepare_cost_benefit_by_country(
                fname,
                by_avoided_emissions,
                last_year=last_year,
                cost_name="investment_cost",
            )[0]

            def get_net_benefit_composite(action_group, group):
                net_benefit = 0
                for country_doing_action, v in zerocost.items():
                    if country_doing_action not in action_group:
                        continue
                    net_benefit += (
                        sum(vv for kk, vv in v.items() if kk in group)
                        - cs_combined[country_doing_action]
                        + (1 - public_funding_fraction)
                        * investment_cost[country_doing_action]
                    )
                return net_benefit

            donor_groups = {
                "CN": ["CN"],
                "JP": ["JP"],
                "EU": EU,
                "US": ["US"],
                "EU,US,JP,CA,UK": EU + ["US", "JP", "CA", "UK"],
            }

            identifier = git_branch
            if git_branch == "main" and analysis_main.ENABLE_COAL_EXPORT:
                identifier = "coal_export"
            elif git_branch == "battery" and analysis_main.ENABLE_COAL_EXPORT:
                identifier = "coal_export_over_battery"

            with open(
                f"plots/bruegel/bruegel_4_{last_year}_{identifier}_public_funding_{public_funding_fraction}.csv",
                "w",
            ) as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["country"] + list(donor_groups))
                for ag_name, ag in action_groups.items():
                    benefits = []
                    for group_name, group in donor_groups.items():
                        # benefits.append(round(get_net_benefit_composite(ag, group), 2))
                        benefits.append(int(get_net_benefit_composite(ag, group) * 1e3))
                    csvwriter.writerow([ag_name, *benefits])

            # Gov cost breakdown

            with open(
                f"plots/bruegel/bruegel_4_gov_cost_{last_year}_{identifier}_{public_funding_fraction}.csv",
                "w",
            ) as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(
                    [
                        "Country name",
                        "PV gov cost (billion $)",
                        "PV opp costs (billion $)",
                        "PV inv. costs (billion $)",
                        *flatten_list_of_list(
                            [
                                [
                                    f"% of {donor_name} GDP",
                                    "% of 2022 gov debt",
                                    "Change in (2022 gov debt/2022 GDP)%",
                                ]
                                for donor_name in donor_groups
                            ]
                        ),
                    ]
                )
                duration = last_year - (analysis_main.NGFS_PEG_YEAR + 1)
                for ag_name, ag in action_groups.items():
                    # Ignore developed countries
                    if ag_name in ["AU", "US", "CA", "DE", "GR"]:
                        continue
                    # Billions USD
                    opportunity_cost = round(
                        sum(
                            cs_combined[country] - investment_cost[country]
                            for country in ag
                            if country in cs_combined
                        )
                        * 1e3,
                        2,
                    )
                    ic = int(
                        public_funding_fraction
                        * sum(
                            investment_cost[country]
                            for country in ag
                            if country in investment_cost
                        )
                        * 1e3
                    )
                    gov_cost = int(opportunity_cost + ic)

                    row = [ag_name, gov_cost, opportunity_cost, ic]

                    for donor_name, dg in donor_groups.items():
                        # In dollars
                        gdp2022_donor = sum(gdp2022[country] for country in dg)
                        # In dollars
                        gov_debt = sum(
                            gdp2022[country] * imf_gov_debt_2022_percent[country] / 100
                            for country in dg
                        )

                        gov_cost_dollars = gov_cost * 1e9
                        cumulative_gdp = gdp2022_donor * duration
                        percent_of_gdp = round(
                            gov_cost_dollars / cumulative_gdp * 100, 2
                        )
                        percent_of_gov_debt = round(
                            gov_cost_dollars / gov_debt * 100, 2
                        )
                        change_in_gov_debt_over_gdp = round(
                            gov_cost_dollars / gdp2022_donor * 100, 2
                        )

                        row += [
                            percent_of_gdp,
                            percent_of_gov_debt,
                            change_in_gov_debt_over_gdp,
                        ]
                    csvwriter.writerow(row)


def do_bruegel_5(action_groups, enable_coal_export):
    yearly_avoided_emissions_by_country = util.read_json(
        "cache/unilateral_benefit_yearly_avoided_emissions_GtCO2_2100.json"
    )
    if enable_coal_export:
        # This is a roundabout way to account for coal export, but there is just no other simpler way.
        length = len(yearly_avoided_emissions_by_country["US"])
        new = {}
        for i in range(length):
            new_element = modify_based_on_coal_export(
                {c: v[i] for c, v in yearly_avoided_emissions_by_country.items()}
            )
            for c, v in new_element.items():
                if c in new:
                    new[c].append(v)
                else:
                    new[c] = [v]
        yearly_avoided_emissions_by_country = new

    with open(
        f"plots/bruegel/bruegel_5_yearly_avoided_emissions_coal_export_{enable_coal_export}.csv",
        "w",
    ) as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["country"] + list(range(2022, 2100 + 1)))

        total = 0
        for group, countries in action_groups.items():
            if group == "World":
                for c in countries:
                    if c not in yearly_avoided_emissions_by_country:
                        continue
                    total += sum(yearly_avoided_emissions_by_country[c])
                continue
            timeseries = None
            idx = 0
            while True:
                try:
                    timeseries = np.array(
                        yearly_avoided_emissions_by_country[countries[idx]]
                    )
                    break
                except KeyError:
                    idx += 1

            for c in countries[idx + 1 :]:
                if c not in yearly_avoided_emissions_by_country:
                    continue
                timeseries += yearly_avoided_emissions_by_country[c]
            csvwriter.writerow([group] + list(timeseries))
        print("Sanity check", total)


if __name__ == "__main__":
    if 1:
        # country specific scc
        # do_country_specific_scc_part3()

        # do_country_specific_scc_part4()
        # do_country_specific_scc_part5()
        # do_country_specific_scc_part6()
        # do_country_specific_scc_part8()
        # do_country_specific_scc_part9()
        # exit()
        # do_country_specific_scc_part7("ID")
        # do_country_specific_scc_part7("ZA")
        # do_country_specific_scc_part7("VN")
        # do_country_specific_scc_part7("IN", use_developed_for_zerocost=True)
        # do_country_specific_scc_part7("CN", use_developed_for_zerocost=True)
        # do_bruegel_heatmap()
        # exit()
        analysis_main.ENABLE_COAL_EXPORT = True
        do_bruegel_2()
        analysis_main.ENABLE_COAL_EXPORT = False
        do_bruegel_2()
        exit()

        if 1:
            # EMDE-CN
            top20 = by_avoided_emissions[:20]
            emerging_country_shortnames = util.get_emerging_countries()
            developING_country_shortnames = util.get_developing_countries()
            emde = emerging_country_shortnames + developING_country_shortnames
            emde_minus_cn = [c for c in emde if c != "CN"]
            developed_country_shortnames = util.prepare_from_climate_financing_data()[4]
            top20_without_developed = [
                c for c in top20 if c not in developed_country_shortnames
            ]
            action_groups = {
                "EMDE-CN": emde_minus_cn,
                **{k: [k] for k in top20_without_developed},
            }
        else:
            # New grouping
            iso3166_df = util.read_iso3166()
            (
                region_countries_map,
                regions,
            ) = analysis_main.prepare_regions_for_climate_financing(iso3166_df)
            eurasia = "AM AZ GE KZ KG RU TJ TM UZ".split()
            southeast_asia = "BN KH ID LA MY MM PH SG TH VN".split()
            other_asia_min_cn_in = "AU BD JP KR KP MN NP NZ PK LK TW".split()
            action_groups = {
                "CN": ["CN"],
                "IN": ["IN"],
                # From https://www.iea.org/reports/scaling-up-private-finance-for-clean-energy-in-emerging-and-developing-economies
                "Southeast Asia": southeast_asia,
                # From https://www.iea.org/reports/scaling-up-private-finance-for-clean-energy-in-emerging-and-developing-economies
                "Other Asia-CN-IN": other_asia_min_cn_in,
                # "Total Asia-CN-IN": southeast_asia + other_asia_min_cn_in,
                "Africa": region_countries_map["Africa"],
                "Latin America & the Carribean": region_countries_map[
                    "Latin America & the Carribean"
                ],
                "Europe & Eurasia": list(set(region_countries_map["Europe"] + eurasia)),
                "Middle East": "BH IR IQ JO KW LB OM QA SA SY AE YE".split(),
                "World": flatten_list_of_list(list(region_countries_map.values())),
            }
        analysis_main.ENABLE_COAL_EXPORT = 1
        do_bruegel_4(action_groups)
        # for enable_coal_export in [True, False]:
        #     do_bruegel_5(action_groups, enable_coal_export)
