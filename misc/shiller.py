import pandas as pd

df = pd.read_csv("data/ShillerData.csv", thousands=",")
df["AA"] = pd.to_datetime(df.AA.astype(str), format="%Y.%m")
df = df[df.AA.dt.month == 1]
df = df[df.AA.dt.year >= 1922]
print(df.AA)
print("PPP", df.PPP.str.rstrip("%").astype(float).mean())
