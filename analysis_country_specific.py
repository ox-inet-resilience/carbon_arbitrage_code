from collections import defaultdict
import json
import math
import os

import matplotlib.pyplot as plt
import pandas as pd

import util
import analysis_main


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


def calculate_country_specific_scc_data(
    unilateral_actor=None,
    ext="",
    to_csv=True,
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
    country_specific_scc = read_country_specific_scc_filtered()
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
    region_countries_map, regions = analysis_main.prepare_regions_for_climate_financing(
        iso3166_df
    )
    _, iso2_to_country_name = util.get_country_to_region()

    unilateral_benefit = None
    unilateral_emissions = None
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

    isa_climate_club = unilateral_actor in (levels + regions)
    cost_climate_club = None
    benefit_climate_club = None
    benefit_of_country_doing_the_action = None
    if unilateral_actor is not None:
        # Generated from the Git branch unilateral_action_benefit
        unilateral_benefit = util.read_json("cache/unilateral_benefit_trillion.json")
        if isa_climate_club:
            unilateral_emissions = 0.0
            cost_climate_club = 0.0
            benefit_climate_club = 0.0
            if unilateral_actor in levels:
                group = levels_map[unilateral_actor]
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
    climate_club_countries = None
    if isa_climate_club:
        if unilateral_actor in levels:
            climate_club_countries = levels_map[unilateral_actor]
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
    print("cost < benefit", len(costly), costly)
    actual_size += len(no_cost) + len(benefit_greater_than_cost) + len(costly)
    assert actual_size == len(country_specific_scc), (actual_size, len(country_specific_scc))

    # Sanity check
    if unilateral_actor is not None:
        if not isa_climate_club:
            # The unilateral actor is a country
            ratio1 = cumulative_benefit / global_benefit
            global_emissions = 1425.5475784377522
            ratio2 = unilateral_emissions_GtCO2 / global_emissions
            assert math.isclose(ratio1, ratio2), (ratio1, ratio2)
        else:
            # Unilateral action is either a region or level of development.
            # We don't test based on region, because PG and WS are not part of
            # the 6 regions.
            # world_benefit_but_unilateral_action = sum(sum(v) for v in bs_region.values())
            world_benefit_but_unilateral_action = sum(sum(v) for v in bs.values())
            # Check that the computed world scc corresponds to the original world scc.
            actual_world_scc = world_benefit_but_unilateral_action / unilateral_emissions
            assert math.isclose(actual_world_scc, util.social_cost_of_carbon), actual_world_scc

            if unilateral_actor in levels:
                assert math.isclose(cost_climate_club, sum(cs[unilateral_actor]))
                assert math.isclose(benefit_climate_club, sum(bs[unilateral_actor]))
            else:
                # Region
                actual = sum(cs_region[unilateral_actor])
                assert math.isclose(cost_climate_club, actual), (cost_climate_club, actual)
                actual = sum(bs_region[unilateral_actor])
                assert math.isclose(benefit_climate_club, actual), (benefit_climate_club, actual)

    if to_csv:
        table = table.sort_values(by="net_benefit", ascending=False)
        table.to_csv(
            f"plots/country_specific_table{ext}.csv", index=False, float_format="%.5f"
        )
    return cs, bs, names, cs_region, bs_region, names_region, unilateral_emissions_GtCO2


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
    actual_global_benefit = sum(sum(b) for b in bs.values())
    assert math.isclose(actual_global_benefit, 114.04380627502013), actual_global_benefit
    actual_global_benefit = sum(sum(b) for b in bs_region.values())
    # This is not 114.04, because PG and WS are not part of the 6 regions (they
    # are in Oceania).
    assert math.isclose(
        actual_global_benefit, 114.025120520287
    ), actual_global_benefit

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
        # It's not 1425.55 because XK is excluded from the 6 regions.
        assert math.isclose(unilateral_emissions_cumulative, 1424.1494924127742), unilateral_emissions_cumulative

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

        for region, c in cs_region_combined.items():
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
    # It's not 114.04 likely because XK is not part of the 6 regions.
    sum_global_benefit_by_region = sum(global_benefit_by_region.values())
    assert math.isclose(sum_global_benefit_by_region, 114.02512000000002), sum_global_benefit_by_region

    all_freeloader_benefit = sum(sum(g.values()) for g in zerocost.values())
    all_unilateral_benefit = sum(bs_region_combined.values())
    actual_benefit = all_freeloader_benefit + all_unilateral_benefit
    # TODO this should have been 114
    assert math.isclose(actual_benefit, 113.91329200000001), actual_benefit

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
            cs, bs, names, _, _, _, _ = calculate_country_specific_scc_data(
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
    cs, bs, names, _, _, _, _ = calculate_country_specific_scc_data(
        unilateral_actor=country_doing_action,
        ext="",
        to_csv=False,
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
    scc_dict = read_country_specific_scc_filtered()
    unscaled_global_scc = sum(scc_dict.values())
    # End of global_benefit_by_country preparation

    # Benefit to 1 country if everyone in the world takes action
    # Billion dollars
    global_benefit_country = (
        global_benefit * scc_dict[country_doing_action] / unscaled_global_scc * 1e3
    )

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

    util.savefig("country_specific_scatter_part7", tight=True)


if __name__ == "__main__":
    if 1:
        # country specific scc
        # do_country_specific_scc_part3()
        do_country_specific_scc_part4()
        do_country_specific_scc_part5()
        do_country_specific_scc_part6()
        do_country_specific_scc_part7()
        exit()
