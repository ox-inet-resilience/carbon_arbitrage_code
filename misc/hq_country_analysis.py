import matplotlib.pyplot as plt
import util

before = util.read_json("plots_before/for_comparison_pv_climate_financing_main.json")
after = util.read_json("plots/for_comparison_pv_climate_financing_main.json")
diff = {}
for k in after.keys():
    diff[k] = after[k] - before[k]

xticks = list(diff.keys())
plt.bar(xticks, list(diff.values()))
plt.xticks(xticks, rotation=45, ha="right")

plt.ylabel("Difference of PV climate financing\n(trillion dollars)")
plt.tight_layout()
plt.savefig("plots/climate_financing_asset_country_vs_HQ_country.png")
