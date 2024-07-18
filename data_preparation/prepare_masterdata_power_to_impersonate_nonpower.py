import util


df = util.maybe_load_masterdata(None, use_pams=False, pams_mode="total")
power_companies = df[df.sector == "Power"]
power_coal = power_companies[power_companies.technology == "CoalCap"].copy()
efs = []
for y in range(2022, 2026 + 1):
    # tCO2
    emissions = power_coal[f"_{y}"] * util.hours_in_1year * power_coal.emissions_factor
    # Convert from MW to tonnes of coal
    power_coal[f"_{y}"] = power_coal[f"_{y}"].apply(
        lambda p: util.GJ2coal(util.MW2GJ(p))
    )
    # tCO2 over tonnes of coal
    ef = emissions / power_coal[f"_{y}"]
    efs.append(ef)
power_coal["emissions_factor"] = efs[0]
power_coal["sector"] = "Coal"
power_coal.to_csv(
    "./data_private/masterdata_power_but_impersonate_nonpower.csv.gz",
    compression="gzip",
    index=False,
)
