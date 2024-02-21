import csv
import os

# pip install GitPython
import git
import pandas as pd

years = [2030, 2035, 2050]
branches = ["coal_export", "main", "coal_export_over_battery", "battery"]

def run(year, branch):
    year_branch = f"{year}_{branch}"
    file_prefix = f"plots/bruegel/bruegel_4_{year_branch}_public_funding_"
    file_prefix_gov_cost = f"plots/bruegel/bruegel_4_gov_cost_{year_branch}_"
    data = {}
    data_gov_cost = {}
    public_funding_dict = {
        1: "Assumes public funding in full",
        0.5: "Assumes public funding covers half of the investment costs",
        0.2: "Assumes public funding covers 20% of the investment costs",
        0.1: "Assumes public funding covers 10% of the investment costs",
    }
    for public_funding, text in public_funding_dict.items():
        data[public_funding] = pd.read_csv(file_prefix + f"{public_funding}.csv")
        data_gov_cost[public_funding] = pd.read_csv(file_prefix_gov_cost + f"{public_funding}.csv")

    EMDEs = [c for c in data[1].country if c not in ["AU", "US", "CA", "DE", "GR"]]
    iso3166_df = pd.read_csv(
        "data/country_ISO-3166_with_region.csv",
        # Needs this because otherwise NA for Namibia is interpreted as NaN
        na_filter=False,
    )
    iso3166_df_alpha2 = iso3166_df.set_index("alpha-2")
    alpha2_to_full_name = iso3166_df_alpha2["name"].to_dict()

    with open(f"plots/bruegel/bruegel_4_combined_{year_branch}.csv", "w") as f:
        writer = csv.writer(f)
        columns_gov_cost = data_gov_cost[1].columns.tolist()[1:]
        columns_net_benefit = data[1].columns.tolist()[1:]
        row = [""]
        for name in public_funding_dict.values():
            row += [name] + [""] * (len(columns_gov_cost) + len(columns_net_benefit) - 1)
        writer.writerow(row)

        row = ["Recipient countries"] + (columns_gov_cost + columns_net_benefit) * 4
        writer.writerow(row)

        for country in EMDEs:
            row = [alpha2_to_full_name[country] if len(country) == 2 else country]
            for public_funding in public_funding_dict.keys():
                df = data[public_funding]
                df_gov_cost = data_gov_cost[public_funding]
                row_gov_cost = df_gov_cost[df_gov_cost["Country name"] == country].iloc[0].tolist()[1:]
                row_net_benefit = df[df.country == country].iloc[0].tolist()[1:]
                row += row_gov_cost + row_net_benefit
            writer.writerow(row)

        writer.writerow(["Memo item: share of world SCC (%)"])
        row = [""] + ["11.18%", "2.80%", "11.00%", "29.40%", "47.10%"] * 4


repo = git.Repo(os.path.dirname(os.path.dirname(__file__)))
assert not repo.bare
for branch in branches:
    repo.git.checkout(branch)
    for year in years:
        print(branch, year)
        run(year, branch)
repo.git.checkout("main")
