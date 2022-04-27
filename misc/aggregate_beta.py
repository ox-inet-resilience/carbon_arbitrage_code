import datetime

import numpy as np
import pandas as pd
from scipy import stats


rho_f = 0.0208

def calculate_return(series):
    series_without_nan = series.dropna()
    return np.array(series_without_nan[1:]) / np.array(series_without_nan[:-1]) - rho_f

# This is a timeseries data.
# CSV of date, MCSI World Index, and MSCI World/Metal & Mining Index
df = pd.read_csv("data_private/3-indices.csv")

# Filter the date to not exceed Jan 1st 2022
df["Date"] = pd.to_datetime(df.Date)
df = df[df.Date.dt.date <= datetime.date(year=2022, month=1, day=1)]

R_M = calculate_return(df.MXWO_Index)
var_R_M = np.var(R_M, ddof=1)
R_Metals_Mining = calculate_return(df.MXWO0MM_Index)

def calculate_beta(R):
    return np.cov(R_M, R, bias=True)[0][1] / var_R_M

beta_Metals_Mining = calculate_beta(R_Metals_Mining)
print("MM", beta_Metals_Mining)
slope, intercept, *_ = stats.linregress(R_M, R_Metals_Mining)
print("Slope", slope)
