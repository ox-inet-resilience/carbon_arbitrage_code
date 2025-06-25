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

a2_to_full_name = util.prepare_alpha2_to_full_name_concise()

measures_keys = {
    "costs": "Costs per avoided tCO2e ($/tCO2e)",
    "investment_costs": "Investment costs (in trillion dollars)",
    "opportunity_costs": "Opportunity costs (in trillion dollars)",
}
for measure, key in measures_keys.items():
    for time_period in ["2024-2035", "2024-2050"]:
        out = {}
        for country in developing:
            df = pd.read_csv(
                f"./plots/table2/table2_country_{country}_10f5db762fcfcd9cfcd6943ab22d029c639373ca.csv",
                skiprows=1,
                index_col=0,
            ).T
            value = df[df.index == time_period][key].iloc[0]
            if measure in ["investment_costs", "opportunity_costs"]:
                ae = df[df.index == time_period]["Avoided emissions (GtCO2e)"].iloc[0]
                if math.isclose(ae, 0):
                    value = 0
                else:
                    value *= 1e12 / ae / 1e9
            out[country] = float(value)
        sorted_desc = dict(sorted(out.items(), key=lambda item: item[1], reverse=True))

        out_name = f"plots/cop30/{measure}_per_ae_dollar_per_tCO2e_{time_period}"
        plt.figure(figsize=(30, 9))
        width = 1.0
        xs = np.arange(len(sorted_desc))
        plt.bar(xs, list(sorted_desc.values()), width)
        plt.xticks(xs, labels=[a2_to_full_name[e] for e in sorted_desc.keys()], rotation=45, ha="right")
        plt.ylabel("$/tCO2e")
        plt.tight_layout()
        plt.savefig(f"{out_name}.png")
        plt.close()

        df_out = pd.DataFrame(sorted_desc, index=["cost_per_ae"]).T
        df_out.to_csv(f"{out_name}.csv")
