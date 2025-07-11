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
import with_learning  # noqa

os.makedirs("plots/cop30", exist_ok=True)

a2_to_full_name = util.prepare_alpha2_to_full_name_concise()
developed, developing = util.get_countries_unfccc()

analysis_main.MEASURE_GLOBAL_VARS = True
analysis_main.MEASURE_GLOBAL_VARS_SCENARIO = "Net Zero 2050"
with_learning.VERBOSE_ANALYSIS = True
util.CARBON_BUDGET_CONSISTENT = "15-50"

renewables = [
    "solar",
    "onshore_wind",
    "offshore_wind",
    "geothermal",
    "hydropower",
]

year = 2024
if 0:
    output = []
    for country in developing[:2]:
        print(country)
        with_learning.VERBOSE_ANALYSIS_COUNTRY = country
        try:
            analysis_main.run_table1(to_csv=False, do_round=False, plot_yearly=False)
        except Exception:
            print("Skipping", country)
            continue

        output_1_country = {
            r: analysis_main.global_cost_with_learning.cached_investment_costs[r][year]
            for r in renewables
        }
        for battery_type in ["short", "long"]:
            output_1_country[battery_type] = (
                analysis_main.global_cost_with_learning.battery_unit_ic[battery_type][
                    year
                ]
            )
        output_1_country["country"] = country
        output.append(output_1_country)
    pd.DataFrame(output).to_csv(
        f"./plots/cop30/initial_unit_ic_{year}.csv", index=False
    )
    exit()
else:
    df_unit_ic = pd.read_csv(f"./plots/cop30/initial_unit_ic_{year}.csv")

df_costs_per_ae = pd.read_csv(
    "./plots/cop30/costs_per_ae_dollar_per_tCO2e_2024-2035.csv"
)
country_order = df_costs_per_ae["Unnamed: 0"].replace(pd.NA, "NA").tolist()
assert len(country_order) == len(developing)
developing = country_order

tech_names = {
    "solar": "Solar",
    "onshore_wind": "Wind onshore",
    "offshore_wind": "Wind offshore",
    "hydropower": "Hydropower",
    "geothermal": "Geothermal",
}

xs = range(len(developing))

# 1
plt.figure(figsize=(30, 9))
fa_energy_mix, fa_energy_mix_world = with_learning.prepare_fa_renewable_weights_data()
for k, label in tech_names.items():
    values = []
    for country in developing:
        value = fa_energy_mix[fa_energy_mix.index == country][k]
        if isinstance(value, pd.Series) and len(value) == 0:
            value = 0
        else:
            value = value.iloc[0]
        value = float(value) * 100
        values.append(value)
    plt.scatter(xs, values, label=label)
ax = plt.gca()
plt.xticks(xs, rotation=45, ha="right")
ax.set_xticklabels([a2_to_full_name[d] for d in developing])
for label in ax.get_xticklabels():
    label.set_visible(True)
plt.legend()
plt.ylabel("Percentage (%)")
plt.tight_layout()
plt.savefig("plots/cop30/energy_mix.png")
plt.close()

# 2
plt.figure(figsize=(30, 9))
fa_capacity_factor, fa_capacity_factor_world = (
    with_learning.prepare_fa_capacity_factor_data()
)
for k, label in tech_names.items():
    values = []
    for country in developing:
        value = fa_capacity_factor[fa_capacity_factor.index == country][k]
        if isinstance(value, pd.Series) and len(value) == 0:
            value = 0
        else:
            value = value.iloc[0]
        value = float(value) * 100
        values.append(value)
    plt.scatter(xs, values, label=label)
ax = plt.gca()
plt.xticks(xs, rotation=45, ha="right")
ax.set_xticklabels([a2_to_full_name[d] for d in developing])
for label in ax.get_xticklabels():
    label.set_visible(True)
plt.legend()
plt.ylabel("Percentage (%)")
plt.tight_layout()
plt.savefig("plots/cop30/capacity_factor.png")
plt.close()

# 3
plt.figure(figsize=(30, 9))
values_summed = np.zeros(len(developing))
for k, label in tech_names.items():
    values = []
    for country in developing:
        cf = fa_capacity_factor[fa_capacity_factor.index == country][k]
        if isinstance(cf, pd.Series) and len(cf) == 0:
            cf = 0
        else:
            cf = cf.iloc[0]
        cf = float(cf)
        em = fa_energy_mix[fa_energy_mix.index == country][k]
        if isinstance(em, pd.Series) and len(em) == 0:
            em = 0
        else:
            em = em.iloc[0]
        em = float(em)

        value = cf * em
        if np.isnan(value):
            value = 0
        values.append(value)
    values_summed += values
plt.scatter(xs, values_summed)
ax = plt.gca()
plt.xticks(xs, rotation=45, ha="right")
ax.set_xticklabels([a2_to_full_name[d] for d in developing])
for label in ax.get_xticklabels():
    label.set_visible(True)
plt.tight_layout()
plt.savefig("plots/cop30/capacity_factorxenergy_mix.png")
plt.close()

# 4
plt.figure(figsize=(30, 9))
values_summed = np.zeros(len(developing))
for k, label in tech_names.items():
    values = []
    for country in developing:
        cf = fa_capacity_factor[fa_capacity_factor.index == country][k]
        if isinstance(cf, pd.Series) and len(cf) == 0:
            cf = 0
        else:
            cf = cf.iloc[0]
        cf = float(cf)
        em = fa_energy_mix[fa_energy_mix.index == country][k]
        if isinstance(em, pd.Series) and len(em) == 0:
            em = 0
        else:
            em = em.iloc[0]
        em = float(em)

        value = cf * em * df_unit_ic.iloc[0][k]
        if np.isnan(value):
            value = 0
        values.append(value)
    values_summed += values
plt.scatter(xs, values_summed)
ax = plt.gca()
plt.xticks(xs, rotation=45, ha="right")
ax.set_xticklabels([a2_to_full_name[d] for d in developing])
for label in ax.get_xticklabels():
    label.set_visible(True)
plt.ylabel("Weighted investment costs ($/kW)")
plt.tight_layout()
plt.savefig("plots/cop30/capacity_factorxenergy_mixxinvestment_cost.png")
plt.close()
