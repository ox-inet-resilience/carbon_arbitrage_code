import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import util

_, nonpower_coal, _ = util.read_masterdata()
products = {
    270111: "anthracite",
    270112: "bituminous",
    270119: "other",
    270120: "Briquettes",
}

content = {}
for trade_flow in ["import", "export"]:
    for product in products:
        fname = f"{trade_flow}_{product}"
        content[fname] = util.read_json(f"coal_export/aggregated/{fname}.json")

each_countries = {}
each_countries_split_product = {k: {} for k in products}
each_countries_split_product["domestic_consumption"] = {}
each_countries_split_product["import"] = {}
by_country = nonpower_coal.groupby("asset_country")._2019.sum()
# Drop XK
by_country = by_country[by_country.index != "XK"]
# Convert to giga tonnes of coal
by_country /= 1e9
for alpha2, row in by_country.items():
    production = row
    _export = 0.0
    _import = 0.0
    for product in products:
        try:
            if alpha2 in ["MZ", "PE"]:
                # Export from Mozambique is unrealistically high, so
                # we rescale so that the total doesn't exceed production.
                # While for Peru, the values are resonable, but they exceed Masterdata.
                # https://wits.worldbank.org/trade/comtrade/en/country/ALL/year/2019/tradeflow/Exports/partner/IND/product/270119
                # This total is a sum across 4 coal products.
                total = 3.554903679240 if alpha2 == "MZ" else 4.31897045e-04
                scale = production / total
                content[f"E_{product}"][alpha2] = {
                    k: v * scale for k, v in content[f"E_{product}"][alpha2].items()
                }

            e = sum(content[f"E_{product}"][alpha2].values())
            e /= 1e3  # Convert to tonne
            e /= 1e9  # Convert to giga tonnes of coal
            i = sum(content[f"I_{product}"][alpha2].values())
            i /= 1e3  # Convert to tonne
            i /= 1e9  # Convert to giga tonnes of coal
            _export += e
            _import += i
            each_countries_split_product[product][alpha2] = {
                "Export": e,
                "Import": i,
                "Product": products[product],
            }
        except KeyError as e:
            # print("KeyError", product, e)
            continue
    offset = production - _export + _import
    each_countries[alpha2] = {
        "Production": production,
        "Export": _export,
        "Import": _import,
        "P-E+I": offset,
    }
    each_countries_split_product["domestic_consumption"][alpha2] = {
        "Export": production - _export,  # This is very confusing, but is the only way for domestic consumption to be included the stacked plot
        "Import": np.nan,
        "Product": "domestic_consumption",
    }
    each_countries_split_product["import"][alpha2] = {
        "Export": _import,  # This is very confusing, but is the only way for import to be included in the stacked plot.
        "Import": np.nan,
        "Product": "import",
    }

if 1:
    from collections import defaultdict
    coal_export_content = {"import": defaultdict(dict), "export": defaultdict(dict)}
    for trade_flow in ["import", "export"]:
        for product in products:
            by_product = content[f"{trade_flow}_{product}"]
            for reporter, inner_dict in by_product.items():
                for partner, value in inner_dict.items():
                    if partner in coal_export_content[trade_flow][reporter]:
                        coal_export_content[trade_flow][reporter][partner] += value
                    else:
                        coal_export_content[trade_flow][reporter][partner] = value
    with open("coal_export/aggregated/combined_summed.json", "w") as f:
        json.dump(coal_export_content, f)
    exit()


def get_alpha2_to_full_name():
    (
        _,
        iso3166_df_alpha2,
        _,
        _,
        _,
    ) = util.prepare_from_climate_financing_data()
    return iso3166_df_alpha2["name"].to_dict()


alpha2_to_full_name = get_alpha2_to_full_name()
df = pd.DataFrame.from_dict(each_countries, orient="index")
df = df.reset_index()
df = df.rename(columns={"index": "Country"})
df["Country"] = df.Country.apply(lambda c: alpha2_to_full_name[c])

df = df.sort_values("P-E+I", ascending=False)

df = pd.melt(df, id_vars="Country")
df = df.rename(columns={"variable": "Measure"})

# df = df.sort_values("Country")
df = df.reset_index(drop=True)

print(df)


if 0:
    # Temporary analysis for sorted value
    df = df[df.Measure == "Export"]
    print(df.sort_values("value", ascending=0)[["Country", "value"]])
    exit()


df_split_product = {
    k: pd.DataFrame.from_dict(each_countries_split_product[k], orient="index")
    for k in list(products.keys()) + ["domestic_consumption", "import"]
}
for k, v in df_split_product.items():
    v.reset_index(inplace=True)
    v.rename(columns={"index": "Country"}, inplace=True)
df_split_product = pd.concat(list(df_split_product.values()))
df_split_product = df_split_product.reset_index(drop=True)
df_split_product["Country"] = df_split_product.Country.apply(
    lambda c: alpha2_to_full_name[c]
)

# c and d
if 0:
    df_split_product_export = df_split_product.pivot(
        index="Country", columns="Product", values="Export"
    ).reset_index()
    df_split_product_export.fillna(0, inplace=True)

    # Order by P-E+I of df
    custom_order = {}
    i = 0
    for c in df["Country"]:
        if c in custom_order:
            continue
        custom_order[c] = i
        i += 1
    df_split_product_export.sort_values("Country", key=lambda x: x.map(custom_order), inplace=True)

    plt.figure()
    product_names = ["anthracite", "bituminous", "other", "Briquettes", "domestic_consumption", "import"]
    df_split_product_export.plot(
        x="Country", y=product_names, figsize=(15, 12), kind="bar", stacked=True
    ).legend()
    plt.ylim(0, 1.8)
    plt.ylabel("Export or consumption or import\n(Giga tonnes of coal)")
    plt.savefig("plots/coal_export_c.png")

    plt.figure()
    # Exclude import
    summed = df_split_product_export[product_names[:-1]].sum(axis=1)
    _df = df_split_product_export.copy()
    for product_name in product_names:
        _df[product_name] = _df[product_name] * 100 / summed
    _df.plot(
        x="Country", y=product_names, figsize=(15, 12), kind="bar", stacked=True
    ).legend()
    plt.ylim(0, 800)
    plt.ylabel("Export or consumption or import\n(% of production)")
    plt.savefig("plots/coal_export_d.png")

    exit()

# e
if 1:
    combined = util.read_json("coal_export/aggregated/combined_summed.json")
    matrix = np.zeros((60, 60))
    # Order by P-E+I of df
    custom_order = {}
    i = 0
    full_name_to_alpha2 = {v: k for k, v in alpha2_to_full_name.items()}
    for c in df["Country"]:
        c = full_name_to_alpha2[c]
        if c in custom_order:
            continue
        custom_order[c] = i
        i += 1
    custom_order = list(custom_order.keys())
    custom_order_full_name = [alpha2_to_full_name[c] for c in custom_order]
    for i in range(60):
        for j in range(60):
            c_i = custom_order[i]
            c_j = custom_order[j]
            ii = combined["export"].get(c_i)
            if not ii:
                matrix[i, j] = 0
            else:
                e = ii.get(c_j, 0)
                e /= 1e3  # Convert to tonne
                e /= 1e9  # Convert to giga tonnes of coal
                matrix[i, j] = e
    fig = plt.figure(figsize=(18, 18))
    ax = plt.gca()
    im = ax.imshow(matrix, origin="lower")
    plt.colorbar(im).set_label("Giga tonnes of coal")
    plt.xlabel("Importer")
    plt.ylabel("Exporter")
    ax.set_xticks(range(60))
    ax.set_yticks(range(60))
    ax.set_xticklabels(custom_order_full_name, rotation=90)
    ax.set_yticklabels(custom_order_full_name)
    plt.savefig("plots/coal_export_e.png")
    exit()

# f
if 0:
    for product_name in ["anthracite", "bituminous", "other", "Briquettes"]:
        plt.figure()
        g = sns.barplot(
            x="Country",
            y="Export",
            color="steelblue",
            data=df_split_product[df_split_product.Product == product_name],
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        g.get_figure().savefig(f"plots/coal_export_f_{product_name}.png")
    exit()

# g
if 0:
    plt.figure()
    g = sns.catplot(
        x="Country",
        y="value",
        hue="Measure",
        hue_order=["Production", "Export", "Import", "P-E+I"],
        # data=df.loc[i * each_length : (i + 1) * each_length],
        data=df,
        kind="bar",
        height=5,
        aspect=3,
    )
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    g.fig.savefig("plots/coal_export_g.png")
