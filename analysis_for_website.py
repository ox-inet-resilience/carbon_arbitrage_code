from collections import defaultdict
import json
import multiprocessing as mp

import util
import analysis_main

# For website sensitivity analysis
def nested_dict(n, _type):
    if n == 1:
        return defaultdict(_type)
    else:
        return defaultdict(lambda: nested_dict(n - 1, _type))


def initialize_website_sensitivity_analysis_params():
    measure_map = {
        "cao": "Carbon arbitrage including residual benefit (in trillion dollars)",
        "cao_relative": "Carbon arbitrage including residual benefit relative to world GDP (%)",
        "cost": "Costs of avoiding coal emissions (in trillion dollars)",
        "benefit": "Benefits of avoiding coal emissions including residual benefit (in trillion dollars)",
        "production_avoided": "Total coal production avoided including residual (Giga tonnes)",
        "emissions_avoided": "Total emissions avoided including residual (GtCO2)",
        "opportunity_cost": "Opportunity costs represented by missed coal revenues (in trillion dollars)",
        "investment_cost": "Investment costs in renewable energy (in trillion dollars)",
    }

    social_costs = list(range(20, 300 + 1, 20))
    time_horizons = [2050, 2070, 2100]
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
    out_dict = nested_dict(5, dict)
    (
        measure_map,
        social_costs,
        time_horizons,
        coal_replacements,
        lifetimes,
        learning_curve_map,
        rho_mode_map,
    ) = initialize_website_sensitivity_analysis_params()
    for learning_curve in learning_curve_map:
        for lifetime in lifetimes:
            for coal_replacement in coal_replacements:
                for last_year in time_horizons:
                    for rho_mode in rho_mode_map:
                        all_scs_output = mp.Manager().dict()

                        def fn(sc):
                            if last_year == 2070:
                                analysis_main.MID_YEAR = 2070
                            else:
                                analysis_main.MID_YEAR = 2050

                            analysis_main.ENABLE_WRIGHTS_LAW = learning_curve_map[learning_curve]
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
                            out = analysis_main.run_cost1(
                                x=1, to_csv=False, do_round=True, plot_yearly=False
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
                        ][rho_mode] = dict(all_scs_output)
    with open("cache/sensitivity_analysis_result.json", "w") as f:
        json.dump(out_dict, f, separators=(",", ":"))


def common_set_website_sensitiviy_analysis_params(
    param, learning_curve_map, coal_replacements
):
    learning_curve = param["learning_curve"]
    lifetime = param["lifetime"]
    coal_replacement = param["coal_replacement"]
    last_year = param.get("last_year", 2100)

    if last_year == 2070:
        analysis_main.MID_YEAR = 2070
    else:
        analysis_main.MID_YEAR = 2050

    analysis_main.ENABLE_WRIGHTS_LAW = learning_curve_map[learning_curve]
    analysis_main.RENEWABLE_LIFESPAN = lifetime
    weights = coal_replacements[coal_replacement]
    analysis_main.RENEWABLE_WEIGHTS = {
        "solar": weights[0],
        "onshore_wind": weights[1],
        "offshore_wind": weights[2],
    }


def do_website_sensitivity_analysis_climate_financing():
    raise Exception("You must run this in the 'battery' Git branch")
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
                params_flat.append(
                    {
                        "learning_curve": learning_curve,
                        "lifetime": lifetime,
                        "coal_replacement": coal_replacement,
                    }
                )

    print("Total number of params", len(params_flat))

    output = mp.Manager().dict()

    def fn(param):
        common_set_website_sensitiviy_analysis_params(
            param, learning_curve_map, coal_replacements
        )
        param_key = "_".join(str(v) for v in param.values())

        # Important: must be non-discounted
        s2_scenario = "2022-2100 2DII + Net Zero 2050 Scenario NON-DISCOUNTED"

        yearly_costs_dict = analysis_main.calculate_yearly_costs_dict(s2_scenario)
        # Reduce the floating precision to save space
        yearly_costs_dict = {
            k: [float(f"{i:.8f}") for i in v] for k, v in yearly_costs_dict.items()
        }
        output[param_key] = yearly_costs_dict

    util.run_parallel_ncpus(8, fn, params_flat, ())

    with open("cache/website_sensitivity_climate_financing.json", "w") as f:
        json.dump(dict(output), f, separators=(",", ":"))

    # Reenable residual benefit again
    analysis_main.ENABLE_RESIDUAL_BENEFIT = 1


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

    params = list(rho_mode_map.keys())
    print("Total number of params", len(params))

    output = {}

    def fn(param):
        analysis_main.RHO_MODE = rho_mode_map[param]

        s2_scenario = "2022-2100 2DII + Net Zero 2050 Scenario NON-DISCOUNTED"

        out = analysis_main.run_cost1(x=1, to_csv=False, do_round=False, return_yearly=True)
        yearly_opportunity_costs = out[s2_scenario]["opportunity_cost"]
        country_names = list(yearly_opportunity_costs[-1].keys())
        yearly_oc_dict = {}
        for country_name in country_names:
            country_level_oc = []
            for e in yearly_opportunity_costs:
                # Multiplication by 1e3 converts trillion to billion
                if isinstance(e, float):
                    country_level_oc.append(e * 1e3)
                elif isinstance(e, dict):
                    country_level_oc.append(e[country_name] * 1e3)
                else:
                    # Pandas series
                    country_level_oc.append(e.loc[country_name] * 1e3)
            yearly_oc_dict[country_name] = country_level_oc

        output[param] = yearly_oc_dict

    for param in params:
        fn(param)

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
        if last_year == 2070:
            analysis_main.MID_YEAR = 2070
        elif last_year == 2030:
            analysis_main.MID_YEAR = 2030
        else:
            analysis_main.MID_YEAR = 2050

        result = acs.do_country_specific_scc_part7(last_year=last_year)
        out[last_year] = result
    util.write_small_json(out, "cache/website_sensitivity_coasian_1country.json")


if __name__ == "__main__":
    # do_website_sensitivity_analysis()
    # do_website_sensitivity_analysis_climate_financing()
    # do_website_sensitivity_analysis_opportunity_costs()
    do_website_sensitivity_cost_benefit_scatter_1country()
    exit()
