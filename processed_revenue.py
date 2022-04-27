import os

import numpy as np
import pandas as pd
from scipy import interpolate

import util

revenue_data = pd.read_csv(
    "data_private/revenue_data_coal_companies_confidential.csv.gz",
    compression="gzip",
)

# "Location" is the same as "company_id" in masterdata!!
revenue_data = revenue_data.groupby("Location").first().reset_index()
# Note: this is only for coal companies.
revenue_data_companies = set(revenue_data.Location)

# We set the beta to be constant, based on the MM beta of aggregate_beta.py
# We simplify the model because the beta data is not good.
# The un_leveraged_beta is calculated from misc/aggregate_beta.py.
un_leveraged_beta = 0.9132710997126332
beta = un_leveraged_beta

# Because the masterdata is only available from 2013, we only do it from 2013
# to 2020 here.
NYEARS_REVENUE = 10 - 2
selected_years = range(2013, 2021)


def divide_or_zero(num, dem):
    if dem > 0.0:
        return num / dem
    return 0.0


def fill_nan(x):
    """
    interpolate to fill nan values for a 1D array.
    """
    num_nans = 0
    last_idx = 0
    for i in range(len(x)):
        if np.isnan(x[i]):
            num_nans += 1
        else:
            last_idx = i
    if num_nans == len(x):
        return x
    elif num_nans == 0:
        return x
    elif num_nans == (len(x) - 1):
        # If there is only 1 non-nan, return a constant value array
        return [x[last_idx] for i in range(len(x))]
    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    A = np.array(x)
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(
        inds[good], A[good], bounds_error=False, fill_value="extrapolate"
    )
    B = np.where(np.isfinite(A), A, f(inds))
    return B


def get_profit_margin_column_name(year):
    if year != 0:
        return f"Profit margin (%)\nYear - {year}"
    return "Profit margin (%)\nLast avail. yr"


def get_revenue_column_name(year):
    if year != 0:
        return f"Operating revenue (Turnover)\nth USD Year - {year}"
    return "Operating revenue (Turnover)\nth USD Last avail. yr"


def get_depreciation_column_name(year):
    if year != 0:
        return f"Depreciation & Amortization\nth USD Year - {year}"
    return "Depreciation & Amortization\nth USD Last avail. yr"


def get_interest_paid_column_name(year):
    if year != 0:
        return f"Interest paid\nth USD Year - {year}"
    return "Interest paid\nth USD Last avail. yr"


def get_taxation_column_name(year):
    if year != 0:
        return f"Taxation\nth USD Year - {year}"
    return "Taxation\nth USD Last avail. yr"


def parse_profit_margin_str(profit_margin):
    # Convert profit margin str (because sometime its value is a string 'n.s.'
    # to float.
    if profit_margin and profit_margin not in ["n.s.", "."]:
        profit_margin = float(profit_margin)
    else:
        profit_margin = np.nan
    return profit_margin


def toGJ(row, _year, coal_only=False):
    if row.sector == "Power":
        if coal_only and row.technology != "CoalCap":
            return 0.0
        return util.MW2GJ(row[f"_{_year}"])
    elif row.sector == "Coal":
        return util.coal2GJ(row[f"_{_year}"])
    elif row.sector == "Oil&Gas":
        if coal_only:
            return 0.0
        # The unit is already in GJ.
        return row[f"_{_year}"]
    # We ignore other sectors.
    # Other sectors are:
    # 'Cement', 'Shipping', 'HDV', 'Steel', 'Automotive', 'Aviation'
    return 0.0


def prepare_revenue_data(masterdata_df, sector):
    global revenue_data
    processed_revenue_filename = (
        f"data_private/revenue_data_companies_confidential_PROCESSED_{sector}.csv.gz"
    )
    if os.path.isfile(processed_revenue_filename):
        print("Processed revenue data file found. Reading...")
        revenue_data = pd.read_csv(
            processed_revenue_filename,
            compression="gzip",
        )
        revenue_data_whole_production_dict = {}
        for _year in selected_years:
            revenue_data_whole_production_dict[_year] = revenue_data.set_index(
                "Location"
            )[f"whole_production_GJ_{_year}"].to_dict()
        revenue_data_dict = revenue_data.set_index("Location").T.to_dict()
        print("Reading done")
        return revenue_data_whole_production_dict, revenue_data_dict

    # Else, generate processed revenue data from scratch
    revenue_column_names = [
        get_revenue_column_name(year) for year in range(NYEARS_REVENUE)
    ]
    profit_margin_column_names = [
        get_profit_margin_column_name(year) for year in range(NYEARS_REVENUE)
    ]

    def fill_nan_revenue(row):
        revenues = [row[column_name] for column_name in revenue_column_names]
        revenues = fill_nan(revenues)
        return pd.Series(revenues)

    # Fill nan revenue columns with interpolated-extrapolated values (only works
    # when not all of them are nans).
    revenue_data[revenue_column_names] = revenue_data.apply(fill_nan_revenue, axis=1)

    def fill_nan_profit_margin(row):
        profit_margins = [
            row[column_name] for column_name in profit_margin_column_names
        ]
        profit_margins = [parse_profit_margin_str(pm) for pm in profit_margins]
        return pd.Series(fill_nan(profit_margins))

    # Fill nan profit margin columns with interpolated-extrapolated values (only
    # works when not all of them are nans).
    revenue_data[profit_margin_column_names] = revenue_data.apply(
        fill_nan_profit_margin, axis=1
    )

    # If a company's profit margin is all nan, fill in from the average
    for profit_margin_column_name in profit_margin_column_names:
        util.fill_nan_with_mean(revenue_data, profit_margin_column_name)
    # For the remaining revenue nan, for the case when a company's revenue is all
    # nan, we fill in with the average of the non-nan rows.
    for revenue_column_name in revenue_column_names:
        util.fill_nan_with_mean(revenue_data, revenue_column_name)

    # Calculate the whole GJ production of each companies, for each years:
    def calculate_whole_production(_year):
        def _f(company_id):
            production_in_GJ = 0.0
            rows_for_one_id = masterdata_df[masterdata_df.company_id == company_id]
            for idx, row in rows_for_one_id.iterrows():
                production_in_GJ += toGJ(row, _year, coal_only=False)
            return production_in_GJ

        return _f

    revenue_data_whole_production_dict = {}
    for _year in selected_years:
        revenue_data[f"whole_production_GJ_{_year}"] = revenue_data.Location.apply(
            calculate_whole_production(_year)
        )
        revenue_data_whole_production_dict[_year] = revenue_data.set_index("Location")[
            f"whole_production_GJ_{_year}"
        ].to_dict()
    revenue_data_dict = revenue_data.set_index("Location").T.to_dict()
    revenue_data.to_csv(processed_revenue_filename)
    return revenue_data_whole_production_dict, revenue_data_dict


def prepare_average_unit_profit(masterdata_df):
    # We simplify the calculation by taking the median of the average unit
    # profit of those top 10 pure coal companies by their production in 2020
    # instead.
    # top10_pure_coal_companies = [
    #     "Siberian Coal Energy Co",
    #     "Peabody Energy Corp",
    #     "Arch Resources Inc",
    #     "Jinneng Holding Group Co Ltd",
    #     "Yanzhou Coal Mining Co Ltd",
    #     "Yankuang Group Co Ltd",
    #     "Shaanxi Coal And Chemical Industry Group Co Ltd",
    #     "Acnr Holdings Inc",
    #     "Exxaro Resources Ltd",
    #     "Adaro Energy Tbk Pt",
    # ]
    # To get the average unit profit of top 10 pure coal companies, go to the
    # pure coal section of analysis masterdata coal.
    top10_aup_median_pure_coal_pure_nonpower = 0.011480418063059742
    top_1hundred_aup_median_pure_coal_pure_nonpower = 0.01975720751720321
    top10_aup_median_pure_coal_pure_power = 0.23792516713950407
    top10_companies_aup_median = 0.0077046859026936015
    # top100_companies_aup_median = 0.020245097391400152
    masterdata_df[
        "energy_type_specific_average_unit_profit"
    ] = top10_aup_median_pure_coal_pure_nonpower
    return

    (
        revenue_data_whole_production_dict,
        revenue_data_dict,
    ) = prepare_revenue_data(masterdata_df, "coal")

    def zero_if_nan(x):
        return 0.0 if np.isnan(x) else x

    # Calculate the sector_specific_average_unit_profit now that the nans have been fixed.
    def calculate_sector_average_unit_profit(row):
        # This function uses revenue data, but is applied on df

        def get_profit(year_ago):
            col_profit_margin = get_profit_margin_column_name(year_ago)
            profit_margin = revenue_data_dict[row.company_id][col_profit_margin]
            col_revenue = get_revenue_column_name(year_ago)
            revenue = revenue_data_dict[row.company_id][col_revenue]
            depreciation = revenue_data_dict[row.company_id][
                get_depreciation_column_name(year_ago)
            ]
            interest_paid = revenue_data_dict[row.company_id][
                get_interest_paid_column_name(year_ago)
            ]
            taxation = revenue_data_dict[row.company_id][
                get_taxation_column_name(year_ago)
            ]

            profit = (
                profit_margin * revenue
                + zero_if_nan(depreciation)
                - zero_if_nan(taxation)
                - zero_if_nan(interest_paid)
            )
            # Truncate profit, because we are using the profit only for
            # calculating opportunity cost.
            profit = max(profit, 0)
            return profit

        average_unit_profit = 0.0
        for year_ago in range(NYEARS_REVENUE):
            profit = get_profit(year_ago)

            _year = 2020 - year_ago
            total_company_production_year_x = revenue_data_whole_production_dict[_year][
                row.company_id
            ]
            average_unit_profit += divide_or_zero(
                profit, total_company_production_year_x
            )
        average_unit_profit /= NYEARS_REVENUE
        return average_unit_profit

    # The following is used for the full version of average unit profit.
    # masterdata_df["energy_type_specific_average_unit_profit"] = masterdata_df.apply(
    #     calculate_sector_average_unit_profit, axis=1
    # )
