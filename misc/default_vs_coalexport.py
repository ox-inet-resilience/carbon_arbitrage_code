import math
import time

tic = time.time()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print("Elapsed to import matplotlib", time.time() - tic)
fig = plt.figure(constrained_layout=True, figsize=(8, 8))
gs = gridspec.GridSpec(2, 4, figure=fig)
gs.update(wspace=0.5)
axes = [
    plt.subplot(
        gs[0, :2],
    ),
    plt.subplot(gs[0, 2:]),
    plt.subplot(gs[1, 1:3]),
]

# We import this to ensure the plot style is properly configured.
import util

# Part 1: PV climate financing
pv_climate_financing_by_world = util.read_json(
    "plots/for_comparison_pv_climate_financing_main.json"
)

pv_climate_financing_by_coal_export = util.read_json(
    "plots/for_comparison_pv_climate_financing_coal_export.json"
)

pv_climate_financing_by_region = util.read_json(
    "plots/for_comparison_pv_climate_financing_ngfs_by_region.json"
)

# For sanity check
aggregate_main = 0.0
aggregate_coal_export = 0.0

for i, yaxis in enumerate(
    [pv_climate_financing_by_region, pv_climate_financing_by_coal_export]
):
    ax = axes[i]
    plt.sca(ax)
    for k in pv_climate_financing_by_world.keys():
        x = pv_climate_financing_by_world[k]
        y = yaxis[k]
        if i == 1:
            # Only aggregate the coal export case
            aggregate_main += x
            aggregate_coal_export += y
        if k == "World":
            marker = "o"
        elif "Countries" in k:
            marker = "s"
        else:
            marker = "^"
        # s is marker size
        plt.scatter(x, y, s=80, label=k, marker=marker)
    # 45 degree line
    plt.plot(ax.get_xlim(), ax.get_xlim(), color="tab:gray")

    plt.xlabel("PV climate financing (default)")
    if i == 0:
        plt.ylabel("PV climate financing (regional scenario)")
    else:
        plt.ylabel("PV climate financing (coal export)")


print("Aggregate main", aggregate_main, "Aggregate coal export", aggregate_coal_export)
assert math.isclose(aggregate_main, aggregate_coal_export), (
    aggregate_main,
    aggregate_coal_export,
)
# Deduplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(
    by_label.values(),
    by_label.keys(),
    bbox_to_anchor=(0.5, 0.05),
    loc="upper center",
    ncol=2,
)

# Part 2: Pv climate financing by country
pv_cf_yearly_countries_main = util.read_json(
    "plots/climate_financing_yearly_discounted_main.json"
)
pv_cf_yearly_countries_coal_export = util.read_json(
    "plots/climate_financing_yearly_discounted_coal_export.json"
)
pv_cf_countries_main = {k: sum(v) for k, v in pv_cf_yearly_countries_main.items()}
pv_cf_countries_coal_export = {
    k: sum(v) for k, v in pv_cf_yearly_countries_coal_export.items()
}

ax = axes[2]
plt.sca(ax)
# For sanity check
aggregate_main = 0.0
aggregate_coal_export = 0.0

for country in pv_cf_countries_main.keys():
    x = pv_cf_countries_main[country]
    y = pv_cf_countries_coal_export[country]
    aggregate_main += x
    aggregate_coal_export += y
    plt.scatter(x, y, s=80, marker="o")
    if country not in ["JP", "DE", "ZA", "ID", "RU", "AU", "US", "IN", "CN"]:
        continue
    plt.annotate(country, (x, y))
print("Aggregate main", aggregate_main, "Aggregate coal export", aggregate_coal_export)
assert math.isclose(aggregate_main, aggregate_coal_export), (
    aggregate_main,
    aggregate_coal_export,
)
# 45 degree line
plt.plot(ax.get_xlim(), ax.get_xlim(), color="tab:gray")
plt.xlabel("PV climate financing (default)")
plt.ylabel("PV climate financing (coal export)")

# Finishing touch
plt.tight_layout()

util.savefig("default_vs_coalexport", tight=True)
