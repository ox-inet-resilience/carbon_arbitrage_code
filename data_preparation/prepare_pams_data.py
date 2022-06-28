import pandas as pd

import util

# This script prepares the PAMS data so that it has the same column names as
# the masterdata.

df = pd.read_csv(
    "data_private/Company Indicator.csv.gz",
    compression="gzip",
    delimiter=";",
    decimal=",",
)

# Index(['Company ID', 'Company Name', 'Asset Sector', 'Asset Technology',
#        'Asset Technology Type', 'Asset Country', 'Emissions Factor',
#        'Emissions Factor Unit', 'Activity Unit',
#        'Total 2013', 'Total 2014', 'Total 2015', 'Total 2016', 'Total 2017',
#        'Total 2018', 'Total 2019', 'Total 2020', 'Total 2021', 'Total 2022',
#        'Total 2023', 'Total 2024', 'Total 2025', 'Total 2026'],
#       dtype='object')

years = list(range(2013, 2026 + 1))
# Delete unused columns
for y in years:
    del df[f"Direct {y}"]

df.rename(columns={f"Total {y}": f"_{y}" for y in years}, inplace=True)
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

if 1:
    print("Multiplying production by capacity factor...")
    # We always use SPS
    _, iea_sps = util.read_iea()
    for year in years:
        capacity_factor = util.get_capacity_factor(iea_sps, year)
        df.loc[df.sector == "Power", f"_{year}"] *= capacity_factor

df.to_csv("data_private/pams.csv.gz")
