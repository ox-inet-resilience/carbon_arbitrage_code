from collections import defaultdict
import json
import math
import os

import matplotlib.pyplot as plt
import pandas as pd

import util
import analysis_main

def calculate_country_specific_scc_data(
    unilateral_actor=None,
    ext="",
    to_csv=True,
    do_beyond_61_countries_from_masterdata=False,
):
    chosen_s2_scenario = "2022-2100 2DII + Net Zero 2050 Scenario"
    costs_dict = analysis_main.calculate_each_countries_cost_with_cache(
        chosen_s2_scenario, "plots/country_specific_cost.json", ignore_cache=True
    )
    out = analysis_main.run_cost1(x=1, to_csv=False, do_round=False)
    global_benefit = out[
        "Benefits of avoiding coal emissions including residual benefit (in trillion dollars)"
    ][chosen_s2_scenario]

    # In dollars/tCO2
    country_specific_scc = util.read_json("plots/country_specific_scc.json")
    total_scc = sum(country_specific_scc.values())

    (
        iso3166_df,
        _,
        _,
        _,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()
    developING_country_shortnames = util.get_developing_countries()
    emerging_country_shortnames = util.get_emerging_countries()
    region_countries_map, regions = analysis_main.prepare_regions_for_climate_financing(iso3166_df)
    _, iso2_to_country_name = util.get_country_to_region()

    unilateral_benefit = None
    unilateral_emissions = None
    levels = [
        "Developed Countries",
        "Developing Countries",
        "Emerging Market Countries",
    ]
    isa_climate_club = unilateral_actor in (levels + regions)
    cost_climate_club = None
    benefit_climate_club = None
    if unilateral_actor is not None:
        # Generated from the Git branch unilateral_action_benefit
        unilateral_benefit = util.read_json("cache/unilateral_benefit_trillion.json")
        if isa_climate_club:
            unilateral_emissions = 0.0
            cost_climate_club = 0.0
            benefit_climate_club = 0.0
            if unilateral_actor in levels:
                group = {
                    "Developed Countries": developed_country_shortnames,
                    "Developing Countries": developING_country_shortnames,
                    "Emerging Market Countries": emerging_country_shortnames,
                }[unilateral_actor]
            else:
                group = region_countries_map[unilateral_actor]
            for country in group:
                if country not in unilateral_benefit:
                    # Skip if we don't have the data for it.
                    continue
                benefit_of_country_doing_the_action = unilateral_benefit[country]
                unilateral_emissions += (
                    benefit_of_country_doing_the_action / country_specific_scc[country]
                )
                benefit_climate_club += benefit_of_country_doing_the_action
                cost_climate_club += costs_dict[country]
        else:
            benefit_of_country_doing_the_action = unilateral_benefit[
                unilateral_actor
            ]
            unilateral_emissions = (
                benefit_of_country_doing_the_action
                / country_specific_scc[unilateral_actor]
            )
    # From "trillion" tCO2 to Giga tCO2
    # print("emissions Giga tCO2", unilateral_actor, unilateral_emissions * 1e3)

    names = defaultdict(list)
    cs = defaultdict(list)
    bs = defaultdict(list)
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
    if isa_climate_club:
        if unilateral_actor in levels:
            cs[unilateral_actor].append(cost_climate_club)
            bs[unilateral_actor].append(benefit_climate_club)
            names[unilateral_actor].append(
                unilateral_actor
            )
        else:
            cs_region[unilateral_actor].append(cost_climate_club)
            bs_region[unilateral_actor].append(benefit_climate_club)
            names_region[unilateral_actor].append(
                unilateral_actor
            )

    for country, cs_scc in country_specific_scc.items():
        if country in ["NC", "FJ", "SB", "VU"]:
            # Skipping, because they are in Oceania:
            # New Caledonia
            # Fiji
            # Solomon Islands
            # Vanuatu
            continue
        if country == "NaN":
            continue
        if country not in costs_dict:
            c = 0.0
        else:
            c = costs_dict[country]
            if unilateral_actor is not None:
                if country != unilateral_actor:
                    c = 0.0
        cs_scc_scale = cs_scc / total_scc
        if unilateral_actor is not None:
            # Unilateral action
            if (not do_beyond_61_countries_from_masterdata) and (country not in unilateral_benefit):
                continue
            # Freeloader benefit
            b = unilateral_emissions * cs_scc
            if unilateral_actor == country:
                # Sanity check for the country doing the action
                assert math.isclose(b, benefit_of_country_doing_the_action)
        else:
            # Global action
            b = cs_scc_scale * global_benefit

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
                            "country_specific_scc": cs_scc,
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
        if country in developed_country_shortnames:
            cs["Developed Countries"].append(c)
            bs["Developed Countries"].append(b)
            names["Developed Countries"].append(country)
        elif country in developING_country_shortnames:
            cs["Developing Countries"].append(c)
            bs["Developing Countries"].append(b)
            names["Developing Countries"].append(country)
        elif country in emerging_country_shortnames:
            cs["Emerging Market Countries"].append(c)
            bs["Emerging Market Countries"].append(b)
            names["Emerging Market Countries"].append(country)
        else:
            print("Skipping", country)

        for region in regions:
            if country in region_countries_map[region]:
                cs_region[region].append(c)
                bs_region[region].append(b)
                names_region[region].append(country)
                break
    print("country-specific country count", len(country_specific_scc))
    print("No cost", len(no_cost), no_cost)
    print("benefit >= cost", len(benefit_greater_than_cost), benefit_greater_than_cost)
    print("cost < benefit", len(costly), costly)

    if to_csv:
        table = table.sort_values(by="net_benefit", ascending=False)
        table.to_csv(f"plots/country_specific_table{ext}.csv", index=False, float_format="%.5f")
    return cs, bs, names, cs_region, bs_region, names_region


def do_country_specific_scc_part3():
    emerging = "Emerging Market Countries"
    (
        cs_emerging,
        bs_emerging,
        names_emerging,
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
        if level == emerging:
            idx = names_emerging[level].index(emerging)
            total_c = c[idx]
            total_b = bs_emerging[level][idx]
        else:
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
        if level == developing:
            idx = names_developing[level].index(developing)
            total_c = c[idx]
            total_b = bs_developing[level][idx]
        else:
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

    plt.savefig("plots/scc_part3.png", bbox_inches="tight")


def do_country_specific_scc_part4():
    # We save to CSV so that the data is shown in the website.
    cs, bs, _, _, bs_region, _ = calculate_country_specific_scc_data(
        unilateral_actor=None,
        ext="_part4",
        to_csv=False,
    )

    # Sanity check
    # The actual full global benefit is 114.04380627502013, but it is less
    # because we skip NC, FJ, SB, VU, and NaN country in the
    # country_specific_scc.json
    actual_global_benefit = sum(sum(b) for b in bs.values())
    assert math.isclose(actual_global_benefit, 113.9984096511258), actual_global_benefit
    actual_global_benefit = sum(sum(b) for b in bs_region.values())
    assert math.isclose(actual_global_benefit, 113.97973133450103), actual_global_benefit

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
    plt.savefig("plots/country_specific_scatter_part4.png")


def calculate_global_benefit():
    out = analysis_main.run_cost1(x=1, to_csv=False, do_round=False, plot_yearly=False)
    chosen_s2_scenario = "2022-2100 2DII + Net Zero 2050 Scenario"
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

    fname = "cache/country_specific_data_part5.json"
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

        for group in regions:
            unilateral_actor = group
            _, _, _, cs_region, bs_region, _ = calculate_country_specific_scc_data(
                unilateral_actor=unilateral_actor,
                ext="",
                to_csv=False,
            )
            cs_region_combined[group] = do_round(sum(cs_region[group]))
            bs_region_combined[group] = do_round(sum(bs_region[group]))
            # group is the country doing the unilateral action, while the
            # regions inside the dict is the one who gets benefit with zero
            # cost.
            zerocost[group] = {k: do_round(sum(v)) for k, v in bs_region.items() if k != group}

        # This code chunk is used to calculate global_benefit_by_region
        global_benefit = calculate_global_benefit()
        scc_dict = util.read_json("plots/country_specific_scc.json")
        unscaled_global_scc = sum(scc_dict.values())
        iso3166_df = util.read_iso3166()
        region_countries_map, _ = analysis_main.prepare_regions_for_climate_financing(iso3166_df)
        global_benefit_by_region = {}
        # End of global_benefit_by_region preparation

        for region, c in cs_region_combined.items():
            countries = region_countries_map[region]
            scc_scale = sum(scc_dict.get(c, 0.0) for c in countries) / unscaled_global_scc
            # Benefit to 1 region if everyone in the world takes action
            global_benefit_by_region[region] = do_round(global_benefit * scc_scale)

        with open(fname, "w") as f:
            json.dump({
                "unilateral_cost": cs_region_combined,
                "unilateral_benefit": bs_region_combined,
                "freeloader_benefit": zerocost,
                "global_benefit": global_benefit_by_region,
            }, f)

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

    plt.savefig("plots/country_specific_scatter_part5.png", bbox_inches="tight")


def do_country_specific_scc_part6():
    top9 = "CN US IN AU RU ID ZA DE KZ".split()

    (
        _,
        iso3166_df_alpha2,
        _,
        _,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()
    alpha2_to_full_name = iso3166_df_alpha2["name"].to_dict()

    fname = "cache/country_specific_data_part6.json"
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
        for country_doing_action in top9:
            unilateral_actor = country_doing_action
            cs, bs, names, _, _, _ = calculate_country_specific_scc_data(
                unilateral_actor=unilateral_actor,
                ext="",
                to_csv=False,
            )
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

        # This code chunk is used to calculate global_benefit_by_country
        global_benefit = calculate_global_benefit()
        scc_dict = util.read_json("plots/country_specific_scc.json")
        unscaled_global_scc = sum(scc_dict.values())
        global_benefit_by_country = {}
        # End of global_benefit_by_country preparation

        for c in cs_combined.keys():
            # Benefit to 1 country if everyone in the world takes action
            global_benefit_by_country[c] = (
                global_benefit * scc_dict[c] / unscaled_global_scc
            )

        with open(fname, "w") as f:
            json.dump({
                "unilateral_cost": cs_combined,
                "unilateral_benefit": bs_combined,
                "freeloader_benefit": zerocost,
                "global_benefit": global_benefit_by_country,
            }, f)

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

    plt.savefig("plots/country_specific_scatter_part6.png", bbox_inches="tight")


def do_country_specific_scc_part7():
    (
        _,
        iso3166_df_alpha2,
        _,
        _,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()
    alpha2_to_full_name = iso3166_df_alpha2["name"].to_dict()

    G7 = "US JP DK GB DE IT NO".split()

    country_doing_action = "ID"
    cs, bs, names, _, _, _ = calculate_country_specific_scc_data(
        unilateral_actor=country_doing_action,
        ext="",
        to_csv=False,
        do_beyond_61_countries_from_masterdata=True,
    )
    cost_country = None
    benefit_country = None
    for level, level_names in names.items():
        if country_doing_action not in level_names:
            continue
        location = level_names.index(country_doing_action)
        # Billion dollars
        cost_country = round(cs[level][location] * 1e3, 3)
        benefit_country = round(bs[level][location] * 1e3, 3)
        break

    # Calculating zerocost
    zerocost = defaultdict(float)
    EU = "AT BE BG CY CZ DK EE FI FR DE GR HU HR IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split()
    zerocost_benefit_eu = 0.0
    zerocost_benefit_world = 0.0
    for level, level_names in names.items():
        location = None
        if country_doing_action in level_names:
            location = level_names.index(country_doing_action)
        for i, c in enumerate(level_names):
            # Billion dollars
            benefit_zc = round(bs[level][i] * 1e3, 3)
            zerocost_benefit_world += benefit_zc
            assert c not in zerocost
            if i == location:
                # This is the country doing the action.
                continue
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
    print(zerocost)

    # This code chunk is used to calculate global_benefit_by_country
    global_benefit = calculate_global_benefit()
    scc_dict = util.read_json("plots/country_specific_scc.json")
    unscaled_global_scc = sum(scc_dict.values())
    # End of global_benefit_by_country preparation

    # Benefit to 1 country if everyone in the world takes action
    # Billion dollars
    global_benefit_country = global_benefit * scc_dict[country_doing_action] / unscaled_global_scc * 1e3

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
        cost_country,
        benefit_country,
        linewidth=0,
        marker="o",
        label=label,
        fillstyle="none",
    )

    # We plot the global benefit
    ax.set_prop_cycle(None)
    plt.plot(
        cost_country,
        global_benefit_country,
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
    # plt.tight_layout()

    plt.savefig("plots/country_specific_scatter_part7.png", bbox_inches="tight")


if __name__ == "__main__":
    if 1:
        # country specific scc
        # do_country_specific_scc_part3()
        # do_country_specific_scc_part4()
        # do_country_specific_scc_part5()
        do_country_specific_scc_part6()
        # do_country_specific_scc_part7()
        exit()
