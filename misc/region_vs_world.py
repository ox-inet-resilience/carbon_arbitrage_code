import time

tic = time.time()
import matplotlib.pyplot as plt

print("Elapsed to import matplotlib", time.time() - tic)

# We import this to ensure the plot style is properly configured.
import util

# Part 1: PV climate financing
pv_climate_financing_by_world = util.read_json(
    "plots/for_comparison_pv_climate_financing_main.json"
)

pv_climate_financing_by_region = util.read_json(
    "plots/for_comparison_pv_climate_financing_ngfs_by_region.json"
)

fig = plt.figure()
ax = plt.gca()
for k in pv_climate_financing_by_world.keys():
    x = pv_climate_financing_by_world[k]
    y = pv_climate_financing_by_region[k]
    if k == "World":
        marker = "o"
    elif "Countries" in k:
        marker = "s"
    else:
        marker = "^"
    # s is marker size
    plt.scatter(x, y, s=80, label=k, marker=marker)

plt.plot(ax.get_xlim(), ax.get_xlim(), color="tab:gray")

fig.legend(
    bbox_to_anchor=(0.5, 0),
    loc="upper center",
    ncol=2,
)

plt.xlabel("PV climate financing (global scenario)")
plt.ylabel("PV climate financing (regional scenario)")
util.savefig("region_vs_world", tight=True)

# Part 2: Yearly plot
yearly_by_region = util.read_json("plots/for_comparison_yearly_ngfs_by_region.json")
yearly_by_world = util.read_json("plots/for_comparison_yearly_main.json")

whole_years = range(2022, 2100 + 1)
for mode in ["by_development", "by_region"]:
    fig = plt.figure()
    for scope, yearly in yearly_by_world[mode].items():
        # if scope == "World":
        #     continue
        plt.plot(whole_years, yearly, label=scope)
    if mode == "by_region":
        fig.legend(
            bbox_to_anchor=(0.5, 0),
            loc="upper center",
            ncol=2,
        )
    else:
        plt.legend()
    # Reset color cycle
    plt.gca().set_prop_cycle(None)
    for scope, yearly in yearly_by_region[mode].items():
        # if scope == "World":
        #     continue
        plt.plot(whole_years, yearly, linestyle="dashed")

    plt.xlabel("Time")
    plt.ylabel("Annual climate financing\n(trillion dollars)")
    # plt.tight_layout()
    util.savefig(f"region_vs_world_yearly_{mode}", tight=True)
