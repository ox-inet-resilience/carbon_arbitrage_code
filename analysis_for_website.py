from collections import defaultdict
import json
import multiprocessing as mp
import time

import util
import analysis_main


BATTERY_MODES = ["Not included", "Short-term storage", "Short-term + long-term storage"]


# For website sensitivity analysis
def nested_dict(n, _type):
    if n == 1:
        return defaultdict(_type)
    else:
        return defaultdict(lambda: nested_dict(n - 1, _type))


def setup_battery(battery_mode):
    if battery_mode == "Not included":
        analysis_main.ENABLE_BATTERY_SHORT = False
        analysis_main.ENABLE_BATTERY_LONG = False
    elif battery_mode == "Short-term storage":
        analysis_main.ENABLE_BATTERY_SHORT = True
        analysis_main.ENABLE_BATTERY_LONG = False
    else:
        assert battery_mode == "Short-term + long-term storage"
        analysis_main.ENABLE_BATTERY_SHORT = True
        analysis_main.ENABLE_BATTERY_LONG = True


def initialize_website_sensitivity_analysis_params():
    measure_map = {
        # We reduce the JSON output size by not including cao and cao_relative
        # "cao": "Carbon arbitrage including residual benefit (in trillion dollars)",
        # "cao_relative": "Carbon arbitrage including residual benefit relative to world GDP (%)",
        # We abbreviate the keys so that they take less space.
        # cost
        "c": "Costs of avoiding coal emissions (in trillion dollars)",
        # benefit
        "b": "Benefits of avoiding coal emissions including residual benefit (in trillion dollars)",
        # production_avoided
        "production": "Total coal production avoided including residual (Giga tonnes)",
        # emissions_avoided
        "emissions": "Total emissions avoided including residual (GtCO2)",
        # opportunity_cost
        "oc": "Opportunity costs (in trillion dollars)",
        # investment_cost
        "ic": "Investment costs (in trillion dollars)",
    }

    social_costs = list(range(20, 300 + 1, 20))
    time_horizons = [2030, 2050, 2070, 2100]
    # For these dicts, the keys are usually the string value as shown in the
    # demonstration website
    # (https://ox-inet-resilience.github.io/carbon_arbitrage/).
    coal_replacements = {
        "50% solar, 25% wind onshore, 25% wind offshore": (0.5, 0.25, 0.25),
        "100% solar, 0% wind": (1.0, 0.0, 0.0),
        "56% solar, 42% wind onshore, 2% wind offshore": (0.56, 0.42, 0.02),
        "0% solar, 100% wind onshore, 0% wind offshore": (0.0, 1.0, 0.0),
        "0% solar, 0% wind onshore, 100% wind offshore": (0.0, 0.0, 1.0),
    }
    lifetimes = [30, 50]
    learning_curve_map = {
        "Learning (investment cost drop because of learning)": True,
        "No learning (no investment cost drop)": False,
    }
    rho_mode_map = {
        "0%": "0%",
        "2.8% (WACC)": "default",
        "3.6% (WACC, average risk-premium 100 years)": "100year",
        "5%": "5%",
        "8%": "8%",
    }

    return (
        measure_map,
        social_costs,
        time_horizons,
        coal_replacements,
        lifetimes,
        learning_curve_map,
        rho_mode_map,
    )


# End of for website sensitivity analysis


def do_website_sensitivity_analysis():
    out_dict = nested_dict(6, dict)
    (
        measure_map,
        social_costs,
        time_horizons,
        coal_replacements,
        lifetimes,
        learning_curve_map,
        rho_mode_map,
    ) = initialize_website_sensitivity_analysis_params()
    # For time tracking
    start = time.time()
    progress_total = len(learning_curve_map) * len(lifetimes) * len(coal_replacements)
    count = 0

    for learning_curve in learning_curve_map:
        for lifetime in lifetimes:
            for coal_replacement in coal_replacements:
                elapsed = int((time.time() - start) / 60)
                print(f"Progress {count}/{progress_total}, {elapsed} mins")
                count += 1
                for last_year in time_horizons:
                    for rho_mode in rho_mode_map:
                        for battery_mode in BATTERY_MODES:
                            all_scs_output = mp.Manager().dict()

                            def fn(sc):
                                apply_last_year(last_year)

                                analysis_main.ENABLE_WRIGHTS_LAW = learning_curve_map[
                                    learning_curve
                                ]
                                analysis_main.RENEWABLE_LIFESPAN = lifetime
                                weights = coal_replacements[coal_replacement]
                                analysis_main.RENEWABLE_WEIGHTS = {
                                    "solar": weights[0],
                                    "onshore_wind": weights[1],
                                    "offshore_wind": weights[2],
                                }
                                analysis_main.RHO_MODE = rho_mode_map[rho_mode]
                                util.social_cost_of_carbon = sc
                                analysis_main.social_cost_of_carbon = sc  # noqa: F811
                                setup_battery(battery_mode)
                                out = analysis_main.run_table1(
                                    to_csv=False, do_round=True, plot_yearly=False
                                )

                                _scenario = (
                                    f"2022-{last_year} 2DII + Net Zero 2050 Scenario"
                                )

                                fn_output = defaultdict(dict)
                                for k, v in measure_map.items():
                                    fn_output[k] = out[v][_scenario]
                                all_scs_output[str(sc)] = fn_output

                            util.run_parallel(fn, social_costs, ())
                            out_dict[learning_curve][str(lifetime)][coal_replacement][
                                str(last_year)
                            ][rho_mode][battery_mode] = dict(all_scs_output)
    git_branch = util.get_git_branch()
    with open(f"cache/website_sensitivity_analysis_result_{git_branch}.json", "w") as f:
        json.dump(out_dict, f, separators=(",", ":"))


def common_set_website_sensitivity_analysis_params(
    param, learning_curve_map, coal_replacements
):
    learning_curve = param["learning_curve"]
    lifetime = param["lifetime"]
    coal_replacement = param["coal_replacement"]
    last_year = param.get("last_year", 2100)

    apply_last_year(last_year)

    analysis_main.ENABLE_WRIGHTS_LAW = learning_curve_map[learning_curve]
    analysis_main.RENEWABLE_LIFESPAN = lifetime
    weights = coal_replacements[coal_replacement]
    analysis_main.RENEWABLE_WEIGHTS = {
        "solar": weights[0],
        "onshore_wind": weights[1],
        "offshore_wind": weights[2],
    }


def do_website_sensitivity_analysis_climate_financing():
    analysis_main.ENABLE_RESIDUAL_BENEFIT = 0

    (
        measure_map,
        social_costs,
        time_horizons,
        coal_replacements,
        lifetimes,
        learning_curve_map,
        rho_mode_map,
    ) = initialize_website_sensitivity_analysis_params()

    params_flat = []
    for learning_curve in learning_curve_map:
        for lifetime in lifetimes:
            for coal_replacement in coal_replacements:
                for battery_mode in BATTERY_MODES:
                    params_flat.append(
                        {
                            "learning_curve": learning_curve,
                            "lifetime": lifetime,
                            "coal_replacement": coal_replacement,
                            "battery_mode": battery_mode,
                        }
                    )

    print("Total number of params", len(params_flat))

    output = mp.Manager().dict()

    def fn(param):
        common_set_website_sensitivity_analysis_params(
            param, learning_curve_map, coal_replacements
        )

        battery_mode = param["battery_mode"]
        setup_battery(battery_mode)

        param_key = "_".join(str(v) for v in param.values())

        s2_scenario = "2022-2100 2DII + Net Zero 2050 Scenario"

        # Important: must be non-discounted
        yearly_costs_dict = analysis_main.calculate_yearly_info_dict(
            s2_scenario, discounted=False
        )
        # Reduce the floating precision to save space
        yearly_costs_dict = {
            k: [float(f"{i:.8f}") for i in v] for k, v in yearly_costs_dict.items()
        }
        output[param_key] = yearly_costs_dict

    util.run_parallel_ncpus(8, fn, params_flat, ())

    git_branch = util.get_git_branch()
    with open(
        f"cache/website_sensitivity_climate_financing_{git_branch}.json", "w"
    ) as f:
        json.dump(dict(output), f, separators=(",", ":"))

    # Reenable residual benefit again
    analysis_main.ENABLE_RESIDUAL_BENEFIT = 1


def apply_last_year(last_year):
    if last_year == 2070:
        analysis_main.MID_YEAR = 2070
    elif last_year == 2030:
        analysis_main.MID_YEAR = 2030
    else:
        # 2100 is included by default in the mid year of 2050.
        assert last_year in [2050, 2100], last_year
        analysis_main.MID_YEAR = 2050


def do_website_sensitivity_analysis_opportunity_costs():
    (
        _,
        _,
        _,
        _,
        _,
        _,
        rho_mode_map,
    ) = initialize_website_sensitivity_analysis_params()
    time_horizons = [2030, 2050, 2070, 2100]

    params = []
    for rho_mode in rho_mode_map:
        for last_year in time_horizons:
            params.append(
                {
                    "rho_mode": rho_mode,
                    "last_year": last_year,
                }
            )
    print("Total number of params", len(params))

    output = mp.Manager().dict()

    def fn(param):
        rho_mode = param["rho_mode"]
        last_year = param["last_year"]
        analysis_main.RHO_MODE = rho_mode_map[rho_mode]
        apply_last_year(last_year)

        # round to 6 decimals to save space
        def do_round(x):
            return round(x, 6)

        s2_scenario = f"2022-{last_year} 2DII + Net Zero 2050 Scenario"

        out = analysis_main.run_table1(
            to_csv=False, do_round=False, return_yearly=True
        )
        yearly_opportunity_costs = out[s2_scenario]["opportunity_cost_non_discounted"]
        country_names = list(yearly_opportunity_costs[-1].keys())
        yearly_oc_dict = {}
        for country_name in country_names:
            country_level_oc = []
            for e in yearly_opportunity_costs:
                # Multiplication by 1e3 converts trillion to billion
                if isinstance(e, float):
                    country_level_oc.append(do_round(e * 1e3))
                elif isinstance(e, dict):
                    country_level_oc.append(do_round(e[country_name] * 1e3))
                else:
                    # Pandas series
                    country_level_oc.append(do_round(e.loc[country_name] * 1e3))
            yearly_oc_dict[country_name] = country_level_oc

        output[f"{rho_mode}_{last_year}"] = yearly_oc_dict

    util.run_parallel(fn, params, ())

    util.write_small_json(
        dict(output), "cache/website_sensitivity_opportunity_costs_phase_out.json"
    )


def do_website_sensitivity_cost_benefit_scatter_1country():
    # Adapted from do_country_specific_scc_part7
    import analysis_country_specific as acs

    time_horizons = [2030, 2050, 2070, 2100]
    out = {}
    for last_year in time_horizons:
        print(last_year)
        apply_last_year(last_year)

        result = acs.do_country_specific_scc_part7(last_year=last_year)
        out[last_year] = result
    util.write_small_json(out, "cache/website_sensitivity_coasian_1country.json")


if __name__ == "__main__":
    do_website_sensitivity_analysis()
    # do_website_sensitivity_analysis_climate_financing()
    exit()
    # do_website_sensitivity_analysis_opportunity_costs()
    do_website_sensitivity_cost_benefit_scatter_1country()
