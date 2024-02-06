import csv

import pandas as pd

year = 2050
year_branch = f"{year}_coal_export"
year_branch = f"{year}_main"
year_branch = f"{year}_coal_export_over_battery"
year_branch = f"{year}_battery"

file_prefix = f"plots/bruegel/bruegel_4_{year_branch}_public_funding_"
data = {}
public_funding_dict = {
    1: "Assumes public funding in full",
    0.5: "Assumes public funding covers half of the investment costs",
    0.2: "Assumes public funding covers 20% of the investment costs",
    0.1: "Assumes public funding covers 10% of the investment costs",
}
for public_funding, text in public_funding_dict.items():
    data[public_funding] = pd.read_csv(file_prefix + f"{public_funding}.csv")

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
    row = [""]
    for name in public_funding_dict.values():
        row += [name, "", "", "", ""]
    writer.writerow(row)

    row = ["Recipient countries"] + data[1].columns.tolist()[1:] * 4
    writer.writerow(row)

    for country in EMDEs:
        row = [alpha2_to_full_name[country] if len(country) == 2 else country]
        for public_funding in public_funding_dict.keys():
            df = data[public_funding]
            row += df[df.country == country].iloc[0].tolist()[1:]
        writer.writerow(row)

    writer.writerow(["Memo item: share of world SCC (%)"])
    row = [""] + ["11.18%", "2.80%", "11.00%", "29.40%", "47.10%"] * 4
