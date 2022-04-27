import os
import matplotlib.pyplot as plt
import pandas as pd

# Ensure that plots directory exists
os.makedirs("plots", exist_ok=True)

ngfs_years = list(range(2005, 2105, 5))
ngfs = pd.read_csv("RenewableNGFSScenarios.csv.gz", compression="gzip")
ngfs_global = ngfs[ngfs.Region == "World"]
model = "GCAM5.3_NGFS"
by_model = ngfs_global[ngfs_global.Model == model]
scenarios = set(by_model.Scenario)
for scenario in scenarios:
    by_scenario = by_model[by_model.Scenario == scenario]
    # Remove weird character
    scenario = scenario.replace("Â", "")
    variables = set(ngfs_global.Variable)
    plt.figure()
    for variable in variables:
        if "CCS" in variable:
            continue
        if "CSP" in variable:
            continue
        cleaned_variable = variable.replace("Capacity|Electricity|", "")
        if cleaned_variable in ["Capacity|Electricity", "Solar|PV", "Gas", "Coal", "Oil"]:
            continue
        if "shore" in cleaned_variable:
            # We want wind only
            continue
        by_variable = by_scenario[by_scenario.Variable == variable]
        if len(by_variable) == 0:
            continue
        assert len(by_variable) == 1
        by_variable = by_variable.iloc[0]
        plt.plot(ngfs_years, [by_variable[str(y)] for y in ngfs_years], label=cleaned_variable)
    plt.title(f"{model}, {scenario}")
    plt.xlabel("Time (years)")
    plt.ylabel("Production (GW)")
    plt.legend()
    plt.savefig(f"plots/{scenario}.png")
    plt.close()
