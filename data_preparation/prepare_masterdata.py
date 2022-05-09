import pandas as pd

import util

# This code
# - NO LONBER rescales the power production across all years by capacity factor
#   according to IEA SPS.

df = pd.read_csv(
    # "data_private/masterdata_ownership.csv.gz",  # 2020 version
    "data_private/masterdata_ownership_2021_version.csv.gz",
    compression="gzip",
    encoding="latin1",
    dtype={"company_name": str, "company_id": int, "asset_country": str},
)
df["technology_type"] = df["technology_type"].fillna("Not Known")

# All sectors are:
# {'Coal', 'Oil&Gas', 'Aviation', 'Shipping', 'HDV', 'Steel', 'Power',
# 'Automotive', 'Cement'}

# For capacity factor
if 1:
    print("Multiplying production by capacity factor...")
    # We always use SPS
    _, iea_sps = util.read_iea()
    years = list(range(2013, 2030 + 1))
    for year in years:
        capacity_factor = util.get_capacity_factor(iea_sps, year)
        df.loc[df.sector == "Power", f"_{year}"] *= capacity_factor

# Only keep the ultimate parents
df = df[df.is_ultimate_parent]

# Sanity check
if False:
    np = df[df.sector == "Coal"]
    print("Nonpower global production in 2020", np._2020.sum() / 1e9)
    p = df[df.sector == "Power"]
    p = p[p.technology == "CoalCap"]
    print("Power global production in 2020", util.MW2Gigatonnes_of_coal(p._2020.sum()))
    print(
        "Number of parent coal companies",
        len(set(np.company_id).union(set(p.company_id))),
    )
    exit()

# Full columns are
# ['Unnamed: 0', 'company_id', 'company_name', 'bloomberg_id',
#  'corporate_bond_ticker', 'is_ultimate_parent',
#  'is_ultimate_listed_parent', 'company_status', 'has_financial_data',
#  'sector', 'technology', 'technology_type', 'asset_country',
#  'emissions_factor', 'emissions_factor_unit', 'number_of_assets',
#  'p_eu_eligible_ownership', 'p_eu_green_ownership', 'metric', 'unit',
#  '_2013', '_2014', '_2015', '_2016', '_2017', '_2018', '_2019', '_2020',
#  '_2021', '_2022', '_2023', '_2024', '_2025', '_2026', '_2027', '_2028',
#  '_2029', '_2030', 'asset_level_timestamp']
df = df.drop(
    columns=[
        # "bloomberg_id",
        # "corporate_bond_ticker",
        "is_ultimate_parent",
        "is_ultimate_listed_parent",
        # "company_status",
        "has_financial_data",
        # "number_of_assets",  # needed to get the number of plants
        # "p_eu_eligible_ownership",
        # "p_eu_green_ownership",
        "metric",
        "asset_level_timestamp",
    ]
)

# Rename some columns
df = df.rename(columns={"emission_factor": "emissions_factor"})

df.to_csv(
    "data_private/masterdata_ownership_PROCESSED_capacity_factor.csv.gz",
    compression="gzip",
    encoding="latin1",
)
