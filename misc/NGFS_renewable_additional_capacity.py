import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import util


def read_json(filename):
    with open(filename) as f:
        obj = json.load(f)
    return obj


ngfs_renewable_additional_capacity = read_json(
    f"data/NGFS_renewable_additional_capacity_{util.NGFS_MODEL}.json"
)

data = {
    "solar": [],
    "offshore_wind": [],
    "onshore_wind": [],
}
years = range(2023, 2100 + 1)
years_included = []
for y in years:
    if y == 2100:
        # Exclude year 2100
        continue
    total = sum(
        ngfs_renewable_additional_capacity[tech][str(y)] for tech in data.keys()
    )
    include = False
    for tech in data.keys():
        if total > 0:
            fraction = ngfs_renewable_additional_capacity[tech][str(y)] / total
            data[tech].append(fraction * 100)
            include = True
    if include:
        years_included.append(y)

data_interpolated = {}
for tech in data.keys():
    # Assume that the value in 2020 is the same as in 2025
    _years = [2020] + years_included
    arr = [data[tech][0]] + data[tech]

    # Assume that year 2100 is the same as in 2095
    _years = _years + [2100]
    arr = arr + [data[tech][-1]]

    # FInally do the interpolation
    f = interp1d(_years, arr, kind="zero")
    interpolated = f(years)
    data_interpolated[tech] = {years[i]: interpolated[i] for i in range(len(years))}

label_map = {
    "solar": "Solar",
    "offshore_wind": "Offshore wind",
    "onshore_wind": "Onshore wind",
}
fig = plt.figure()
for tech, arr in data_interpolated.items():
    y = list(arr.values())
    print("Mean of", tech, np.mean(y))
    plt.plot(list(arr.keys()), y, label=label_map[tech])
all_wind = np.array(list(data_interpolated["offshore_wind"].values())) + np.array(
    list(data_interpolated["onshore_wind"].values())
)
plt.plot(
    list(data_interpolated["offshore_wind"].keys()),
    all_wind,
    label="Wind\n(onshore & offshore)",
)
plt.xlabel("Time")
plt.ylabel("Renewable weight (%)")
fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4)
plt.savefig("plots/additional_capacity_composition.png", bbox_inches="tight")
with open(f"data/NGFS_renewable_dynamic_weight_{util.NGFS_MODEL}.json", "w") as f:
    json.dump(data_interpolated, f)
