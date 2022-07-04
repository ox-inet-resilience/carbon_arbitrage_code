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
#        'direct_production_2013', 'direct_production_2014', 'direct_production_2015', 'direct_production_2016', 'direct_production_2017',
#        'direct_production_2018', 'direct_production_2019', 'direct_production_2020', 'direct_production_2021', 'direct_production_2022',
#        'direct_production_2023', 'direct_production_2024', 'direct_production_2025', 'direct_production_2026'],
#       dtype='object')

years = list(range(2020, 2026 + 1))
# Delete unused columns
for y in years:
    del df[f"total_production_{y}"]

df.rename(columns={f"direct_production_{y}": f"_{y}" for y in years}, inplace=True)

# Convert MWh back to MW
for y in years:
    df[f"_{y}"] /= util.hours_in_1year

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

# No need to multiply by capacity factor. The data has already been processed with this factored in.
df.to_csv("data_private/pams.csv.gz")
