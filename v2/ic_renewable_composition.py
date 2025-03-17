import pathlib
import sys
from collections import defaultdict

import pandas as pd

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import analysis_main  # noqa
import util  # noqa


# All of FA countries
# alpha2s = sorted(list(set(analysis_main.df_sector.asset_country.tolist())))
df_ic_non_discounted = pd.read_csv(
    "./plots/bruegel/yearly_by_country_investment_cost_non_discounted_main.csv"
)
years = range(analysis_main.NGFS_PEG_YEAR, analysis_main.LAST_YEAR + 1)
energy_types = ["solar", "onshore_wind", "offshore_wind"]
for a2 in ["BW"]:
    data = util.read_json(f"./plots/phase_in/battery_unit_ic_{a2}.json")
    ic_country = df_ic_non_discounted[df_ic_non_discounted["Unnamed: 0"] == a2].iloc[0]
    composition_country = defaultdict(list)
    for year in years:
        total_unit_ic = sum(data[e][str(year)] for e in energy_types)
        ic = ic_country[str(year)]
        for e in energy_types:
            composition_country[e].append(
                float(data[e][str(year)] / total_unit_ic * ic)
            )
    util.write_small_json(
        composition_country,
        f"./plots/phase_in/yearly_investment_cost_renewable_composition_{a2}.json",
    )
