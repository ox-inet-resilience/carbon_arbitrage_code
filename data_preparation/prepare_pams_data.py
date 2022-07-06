import pandas as pd

import util

# This script prepares the PAMS data so that it has the same column names as
# the masterdata.

df = pd.read_csv(
    "data_private/pams_ef_wide.csv.gz",
    compression="gzip",
)

# Index(['Company ID', 'Company Name', 'Asset Sector', 'Asset Technology',
#        'Asset Technology Type', 'Asset Country', 'Emissions Factor',
#        'Emissions Factor Unit', 'Activity Unit',
#        'direct_production_2020', ..., 'direct_production_2026',
#        'total_production_2020', ..., 'total_production_2026'],
#       dtype='object')

years = list(range(2020, 2026 + 1))

df.rename(
    columns={
        "Company ID": "company_id",
        "Company Name": "company_name",
        "Asset Sector": "sector",
        "Asset Technology": "technology",
        "Asset Technology Type": "technology_type",
        "Asset Country": "asset_country",
        "Emissions Factor": "emissions_factor",
        "Emissions Factor Unit": "emissions_factor_unit",
    },
    inplace=True,
)

for production_type in ["direct", "total"]:
    temp_df = df.copy()
    if production_type == "direct":
        to_delete = "total"
    else:
        to_delete = "direct"

    # Delete unused columns
    for y in years:
        del temp_df[f"{to_delete}_production_{y}"]

    temp_df.rename(
        columns={f"{production_type}_production_{y}": f"_{y}" for y in years},
        inplace=True,
    )

    # Convert MWh back to MW
    for y in years:
        temp_df[f"_{y}"] /= util.hours_in_1year

    # No need to multiply by capacity factor. The data has already been processed with this factored in.

    if production_type == "total":
        # Replace asset_country with the ultimate parent country
        df_info = pd.read_csv("data_private/Company Information.csv.gz", delimiter=";")
        df_info.rename(columns={"Company ID": "company_id"}, inplace=True)

        # Only ultimate parents
        df_info = df_info[df_info["Is Ultimate Parent"]]
        print("Before ultimate parent filter", len(temp_df))
        temp_df = temp_df[temp_df.company_id.isin(df_info.company_id)]
        print("After ultimate parent filter", len(temp_df))

        country_map = df_info.set_index("company_id")["Country of Domicile"].to_dict()
        temp_df["asset_country"] = temp_df.company_id.apply(lambda i: country_map[i])

    temp_df.to_csv(f"data_private/pams_{production_type}.csv.gz")
