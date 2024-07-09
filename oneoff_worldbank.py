import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# From bruegel_2_battery_avoided_emissions_all_modified_with_coal_export.csv
# 2100, 2050, 2030
ae = {
    "PL": np.array([26.833528082259352, 11.713975363971072, 7.168344793678847]),
    "CL": np.array([3.791464813463716, 2.814316879795732, 2.668661980158804]),
}
scc_country_percent = {"PL": 0.53, "CL": 0.16}
scc_eu_percent = 11.169
scc_coalition_percent = 47.246496

# TODO 2
prefix = Path.home() / "Downloads/plots"
with open(prefix / "worldbank_todo2_unilateral_benefit_billions.csv", "w") as csvfile:
    # Create a CSV writer object
    csvwriter = csv.writer(csvfile)

    # Write the header row
    csvwriter.writerow(["Country", "SCC", "Percent Name", "2100", "2050", "2030"])

    # Your existing nested loops
    for country in ["PL", "CL"]:
        percent_names = [country, "EU", "coalition", "global"]
        # in billions
        for scc in [80, 190, 1056]:
            for i, percent in enumerate(
                [
                    scc_country_percent[country],
                    scc_eu_percent,
                    scc_coalition_percent,
                    100,
                ]
            ):
                # Calculate the value
                value = ae[country] * percent / 100 * scc

                # Write a row to the CSV file
                csvwriter.writerow([country, scc, percent_names[i], *value])

# From yearly_by_country_opportunity_cost_NONDISCOUNTED_battery.csv
years = range(2024, 2100 + 1)
for enable_coal_export in [False, True]:
    suffix = f"coalexport_{enable_coal_export}"
    with open(
        f"./plots/bruegel/yearly_by_country_investment_cost_NONDISCOUNTED_battery_{suffix}.csv"
    ) as f:
        yearly_ic_all = f.read().split("\n")
    with open(
        f"./plots/bruegel/yearly_by_country_opportunity_cost_NONDISCOUNTED_battery_{suffix}.csv"
    ) as f:
        yearly_oc_all = f.read().split("\n")
    for country in ["Poland", "Chile"]:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(country)
        plt.sca(axs[0])
        # trillion
        yearly_oc = [i for i in yearly_oc_all if i.startswith(country)][0]
        # million
        yearly_oc = [float(i) * 1e6 for i in yearly_oc.split(",")[1:]]
        plt.plot(years, yearly_oc)
        plt.ylabel("Opportunity cost (million dollars)")
        plt.xlabel("Time (years)")
        plt.sca(axs[1])
        # trillion
        yearly_ic = [i for i in yearly_ic_all if i.startswith(country)][0]
        # billion
        yearly_ic = [float(i) * 1e3 for i in yearly_ic.split(",")[1:]]
        plt.plot(years, yearly_ic)
        plt.ylabel("Investment cost (billion dollars)")
        plt.xlabel("Time (years)")
        plt.tight_layout()
        plt.savefig(prefix / f"worldbank_yearly_cost_{country}_{suffix}.png")
