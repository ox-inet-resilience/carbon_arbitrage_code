import math
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import analysis_main  # noqa
import util  # noqa

os.makedirs("plots/cop30", exist_ok=True)

developed, developing = util.get_countries_unfccc()
# To prepare the tables needed, uncomment these
# for country in developing:
#     analysis_main.run_table2(f"country_{country}", [country])
# for country in developed:
#     analysis_main.run_table2(f"country_{country}", [country])

a2_to_full_name = util.prepare_alpha2_to_full_name_concise()
full_name_to_a2 = {v: k for k, v in a2_to_full_name.items()}
_, df_sector = util.read_forward_analytics_data(analysis_main.SECTOR_INCLUDED)
total_production_fa = util.get_production_by_country(
    df_sector, analysis_main.SECTOR_INCLUDED
)
total_production_fa_summed = total_production_fa.groupby(level=0).sum()

brown_energies = ["Coal", "Oil", "Gas"]

measures_keys = {
    "costs": "Costs per avoided tCO2e ($/tCO2e)",
    "public_costs": "Public costs per avoided tCO2e ($/tCO2e)",
    "investment_costs": "Investment costs (in trillion dollars)",
    "opportunity_costs": "Opportunity costs (in trillion dollars)",
}
# file hashes to try, because the output of table2 filenames contain hashes
hashes = [
    "10f5db762fcfcd9cfcd6943ab22d029c639373ca",
    "37c38a58a68b92dc9f6a6da558a9a11ce50f6c5f",
]

top30 = {}
for last_year in [2035, 2050]:
    analysis_main.LAST_YEAR = last_year
    out_yearly = analysis_main.run_table1(return_yearly=True)
    ae_dict = out_yearly[f"2024-{last_year} FA + Net Zero 2050 Scenario"][
        "avoided_emissions_including_residual_emissions"
    ]
    # Sort by descending avoided emissions
    ae_dict = dict(sorted(ae_dict.items(), key=lambda item: item[1], reverse=True))
    top30[last_year] = list(ae_dict.keys())[:30]

country_map = {
    "developing": developing,
    "developed": developed,
    "combined": developing + developed,
    "top30ae_2035": top30[2035],
    "top30ae_2050": top30[2050],
}

for country_group in ["developing", "developed", "combined", "top30ae_2035", "top30ae_2050"]:
    countries = country_map[country_group]
    for measure, key in measures_keys.items():
        for time_period in ["2024-2035", "2024-2050"]:
            out = {}
            for country in countries:
                for hash in hashes:
                    try:
                        df = pd.read_csv(
                            f"./plots/table2/table2_country_{country}_{hash}.csv",
                            skiprows=1,
                            index_col=0,
                        ).T
                    except FileNotFoundError:
                        continue
                    break
                if measure == "public_costs":
                    value = (
                        0.25
                        * df[df.index == time_period][
                            "Investment costs (in trillion dollars)"
                        ].iloc[0]
                        + df[df.index == time_period][
                            "Opportunity costs (in trillion dollars)"
                        ].iloc[0]
                    )
                else:
                    value = df[df.index == time_period][key].iloc[0]
                if measure in ["investment_costs", "opportunity_costs", "public_costs"]:
                    ae = df[df.index == time_period]["Avoided emissions (GtCO2e)"].iloc[
                        0
                    ]
                    if math.isclose(ae, 0):
                        value = 0
                    else:
                        value *= 1e12 / ae / 1e9
                out[country] = float(value)

            sorted_desc = dict(
                sorted(out.items(), key=lambda item: item[1], reverse=True)
            )

            out_name = f"plots/cop30/{measure}_per_ae_dollar_per_tCO2e_{time_period}"
            plot_width = 35 if country_group == "combined" else 30
            plt.figure(figsize=(plot_width, 9))
            width = 1.0
            xs = np.arange(len(sorted_desc))
            if measure in ["costs", "public_costs"]:
                ys = {}
                for brown_energy in brown_energies:
                    y = []
                    for c, v in sorted_desc.items():
                        try:
                            denominator = total_production_fa_summed.loc[c]
                            if denominator > 0:
                                e = (
                                    v
                                    * total_production_fa.loc[(c, brown_energy)]
                                    / denominator
                                )
                            else:
                                e = 0
                        except KeyError:
                            e = 0
                        y.append(e)
                    ys[brown_energy] = y
                plt.bar(xs, ys["Coal"], width, label="Coal", color="peru")
                plt.bar(
                    xs,
                    ys["Oil"],
                    width,
                    bottom=ys["Coal"],
                    label="Oil",
                    color="dimgrey",
                )
                plt.bar(
                    xs,
                    ys["Gas"],
                    width,
                    bottom=np.array(ys["Coal"]) + ys["Oil"],
                    label="Gas",
                    color="orange",
                )
                for level in [25, 50, 75, 100, 125, 150, 175, 200]:
                    plt.axhline(level, color="gray", linestyle="dashed")
                plt.legend(fontsize=17, loc="upper right")
                dictionary = {
                    f"cost_{brown_energy}": ys[brown_energy]
                    for brown_energy in brown_energies
                }
                dictionary["country"] = list(sorted_desc.keys())
                df_out = pd.DataFrame(dictionary)
                df_out.to_csv(
                    f"{out_name}_breakdown_by_energy_type_{country_group}.csv"
                )
            else:
                plt.bar(xs, list(sorted_desc.values()), width)

            plt.xticks(
                xs,
                labels=[a2_to_full_name[e] for e in sorted_desc.keys()],
                rotation=45,
                ha="right",
            )
            if country_group in ["combined", "top30ae_2035", "top30ae_2050"]:
                for tick in plt.gca().get_xticklabels():
                    tick.set_color(
                        "green"
                        if full_name_to_a2[tick.get_text()] in developed
                        else "black"
                    )

            plt.ylabel("$/tCO2e", fontsize=17)
            plt.yticks(fontsize=17)
            plt.tight_layout()
            plt.savefig(f"{out_name}_{country_group}.png")
            plt.close()

            df_out = pd.DataFrame(sorted_desc, index=["cost_per_ae"]).T
            df_out["full_name"] = df_out.index.map(lambda x: a2_to_full_name[x])
            df_out.to_csv(f"{out_name}_{country_group}.csv")
