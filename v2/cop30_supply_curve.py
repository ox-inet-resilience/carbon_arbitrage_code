import os
import pathlib
import sys

# https://github.com/Phlya/adjustText
from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import analysis_main  # noqa
import util  # noqa
import with_learning  # noqa

os.makedirs("plots/cop30", exist_ok=True)

developed, developing = util.get_countries_unfccc()

a2_to_full_name = util.prepare_alpha2_to_full_name_concise()
full_name_to_a2 = {v: k for k, v in a2_to_full_name.items()}

last_year = 2035
analysis_main.LAST_YEAR = last_year
out_yearly = analysis_main.run_table1(return_yearly=True)
ae_dict = out_yearly[f"2024-{last_year} FA + Net Zero 2050 Scenario"][
    "avoided_emissions_including_residual_emissions"
]
# Sort by descending avoided emissions
sorted_ae_dict_top10 = dict(
    sorted(ae_dict.items(), key=lambda item: item[1], reverse=True)[:20]
)
sorted_ae_dict_bot50 = dict(
    sorted(ae_dict.items(), key=lambda item: item[1], reverse=True)[-50:]
)


def filter_n_elements(dictionary, n, last=False):
    """
    Filters the first/last 'n' key-value pairs from a dictionary.

    Args:
        dictionary (dict): The input dictionary.
        n (int): The number of elements to retrieve from the beginning.

    Returns:
        dict: A new dictionary containing the first 'n' elements.
    """
    if last:
        return dict(list(dictionary.items())[-n:])
    return dict(list(dictionary.items())[:n])


def common_xticks(xs, labels, remove_marginal_number=True):
    plt.xticks(
        xs,
        labels=labels,
        rotation=45,
        ha="right",
    )
    for tick in plt.gca().get_xticklabels():
        country_name = tick.get_text()
        if remove_marginal_number:
            country_name = ",".join(country_name.split(";")[:-1])
        tick.set_color(
            "green"
            if full_name_to_a2.get(country_name, "N/A") in developed
            else "black"
        )


costs_per_ae_full = pd.read_csv(
    "./plots/cop30/costs_per_ae_dollar_per_tCO2e_2024-2035_combined.csv",
    index_col="Unnamed: 0",
)[::-1]["cost_per_ae"].to_dict()
costs_per_ae_full = {k: v for k, v in costs_per_ae_full.items() if k in ae_dict}
public_costs_per_ae_full = pd.read_csv(
    "./plots/cop30/public_costs_per_ae_dollar_per_tCO2e_2024-2035_combined.csv",
    index_col="Unnamed: 0",
)[::-1]["cost_per_ae"].to_dict()
public_costs_per_ae_full = {
    k: v for k, v in public_costs_per_ae_full.items() if k in ae_dict
}

PLOT_AVOIDED_EMISSIONS = True
ENABLE_TWINX = False
for mode in ["all", "developing", "developing_without_bot50cn"]:
    match mode:
        case "developing":
            costs_per_ae = {
                k: v for k, v in costs_per_ae_full.items() if k in developing
            }
        case "developing_without_bot50cn":
            costs_per_ae = {
                k: v
                for k, v in costs_per_ae_full.items()
                if k in developing and k != "CN" and k not in sorted_ae_dict_bot50
            }
        case _:
            costs_per_ae = costs_per_ae_full
    # Filter at breakpoint: South Sudan
    # costs_per_ae = filter_n_elements(costs_per_ae, 33)
    # costs_per_ae = filter_n_elements(costs_per_ae, len(costs_per_ae) - 33, last=True)

    xs = list(range(len(costs_per_ae)))

    # 1
    plt.figure(figsize=(35, 9))
    if PLOT_AVOIDED_EMISSIONS:
        ys = [ae_dict[c] for c in costs_per_ae]
        plt.ylabel("Avoided emissions (GtCO2)")
    else:
        ys = [public_costs_per_ae_full.get(c, 0) for c in costs_per_ae]
        plt.ylabel("Public costs / avoided emissions ($/tCO2)")
    plt.bar(
        xs, ys, color="#CBD5E1" if ENABLE_TWINX else "tab:blue"
    )  # color is slate-300
    labels = []
    for e in costs_per_ae:
        c = str(int(costs_per_ae.get(e, 0)))
        if PLOT_AVOIDED_EMISSIONS and mode in [
            "developing",
            "developing_without_bot50cn",
        ]:
            c += f",{int(public_costs_per_ae_full.get(e, 0))}"
        label = f"{a2_to_full_name[e]};$\\mathbf{{{c}}}$"
        labels.append(label)
    labels[0] = labels[0] + "$\\mathbf{\\$/tCO_2}$"
    common_xticks(xs, labels)

    if ENABLE_TWINX and mode in ["developing", "developing_without_bot50cn"]:
        ax2 = plt.gca().twinx()
        financiers = {
            "Developed countries": 47.99,
            "G7+EU (incl. Norway, Switzerland, Australia, South Korea excl. USA)": 21.0,
            "G7+EU (incl. China, Norway, Switzerland, Australia, South Korea excl. USA)": 32.19,
        }
        for financier, scc_share in financiers.items():
            global_sccs = [
                public_costs_per_ae_full.get(c, 0) / (scc_share / 100)
                for c in costs_per_ae
            ]
            plt.scatter(xs, global_sccs, label=financier, s=100)
        plt.ylabel(r"Global SCC ($/tCO2)")
        plt.legend()

    plt.grid(which="major", color="dimgray", linewidth=0.8)
    plt.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(f"plots/cop30/supply_curve_{mode}_{last_year}.png")
    plt.close()

    # 2. cumulative sum
    fig = plt.figure(figsize=(35, 9))
    cumsum_costs = np.cumsum([v * ae_dict[k] for k, v in costs_per_ae.items()]) / 1e3
    cumsum_public_costs = (
        np.cumsum([public_costs_per_ae_full[k] * ae_dict[k] for k in costs_per_ae])
        / 1e3
    )
    plt.plot(
        xs,
        cumsum_costs,
        label="Costs",
    )
    common_xticks(
        xs, [a2_to_full_name[e] for e in costs_per_ae], remove_marginal_number=False
    )
    plt.ylabel("Costs (trillion dollars)")
    ax2 = plt.gca().twinx()
    cumsum_ae = np.cumsum([ae_dict[c] for c in costs_per_ae])
    plt.plot(
        xs,
        cumsum_ae,
        color="tab:orange",
        label="Avoided emissions",
    )
    ax2.set_ylabel(r"Avoided emissions ($GtCO_2$)")
    fig.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(f"plots/cop30/supply_curve_cumsum_{mode}_{last_year}.png")
    plt.close()

    # yet another cumulative sum
    for name, measure in {
        "public costs": cumsum_public_costs,
        "costs": cumsum_costs,
        "costs_nocumsum": list(costs_per_ae.values()),
        "public_costs_nocumsum": [public_costs_per_ae_full[c] for c in costs_per_ae],
    }.items():
        plt.figure()
        plt.plot(cumsum_ae, measure, linestyle="dashed")
        color = ["green" if c in developed else "black" for c in costs_per_ae]
        plt.scatter(cumsum_ae, measure, color=color, facecolor="none")
        plt.xlabel(r"Cumulative avoided emissions ($GtCO_2$)")
        if name == "costs_nocumsum":
            ylabel = "Costs ($/tCO2)"
        elif name == "public_costs_nocumsum":
            ylabel = "Public costs ($/tCO2)"
        else:
            ylabel = f"Cumulative {name} (trillion dollars)"
        plt.ylabel(ylabel)
        countries = list(costs_per_ae)
        ax = plt.gca()
        texts = []
        for i in range(len(cumsum_ae)):
            c = countries[i]
            if c in sorted_ae_dict_top10:
                texts.append(
                    plt.text(
                        cumsum_ae[i],
                        measure[i],
                        a2_to_full_name[c],
                        color="green" if c in developed else "black",
                        fontsize=10,
                    )
                )
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color="black"))
        plt.savefig(f"plots/cop30/supply_curve_costsvsae_{mode}_{last_year}_{name}.png")
        plt.close()


# Supply curve but for developing altogether grouped
def supply_curve_grouped():
    plt.figure(figsize=(10, 10))
    xs = list(range(2))
    developing_minus_cn_petrol = [
        c for c in developing if c != "CN" and c not in with_learning.petrol_states
    ]
    ys = [
        sum(ae_dict.get(c, 0) for c in developing),
        # Developing minus CN minus petrol states
        sum(ae_dict.get(c, 0) for c in developing_minus_cn_petrol),
    ]

    plt.bar(xs, ys)
    labels = []
    public_vals = []
    for name, group in [
        ("Developing countries", developing),
        (
            "Developing countries\n(excl. China and petro states)",
            developing_minus_cn_petrol,
        ),
    ]:
        val = sum(costs_per_ae.get(c, 0) * ae_dict.get(c, 0) for c in group) / sum(
            ae_dict.get(c, 0) for c in group
        )
        # sum weighted public cost per ae
        public_val = sum(
            public_costs_per_ae_full.get(c, 0) * ae_dict.get(c, 0) for c in group
        ) / sum(ae_dict.get(c, 0) for c in group)
        val = f"{int(val)},{int(public_val)}"
        label = f"{name}\n$\\mathbf{{{val}\\,(\\$/tCO_2)}}$"
        labels.append(label)
        public_vals.append(public_val)

    plt.ylabel("Avoided emissions (GtCO2)")
    common_xticks(xs, labels)

    plt.gca().twinx()
    financiers = {
        "Developed countries": 47.99,
        "G7+EU (incl. Norway, Switzerland, Australia, South Korea excl. USA)": 21.0,
        "G7+EU (incl. China, Norway, Switzerland, Australia, South Korea excl. USA)": 32.19,
    }
    for financier, scc_share in financiers.items():
        global_sccs = [public_val / (scc_share / 100) for public_val in public_vals]
        plt.scatter(xs, global_sccs, label=financier, edgecolors="black", s=100)
    plt.ylabel(r"Global SCC ($/tCO2)")
    plt.legend()

    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/cop30/supply_curve_grouped_{last_year}.png")
    plt.close()


supply_curve_grouped()
