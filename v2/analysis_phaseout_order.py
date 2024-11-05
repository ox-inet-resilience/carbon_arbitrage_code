import pathlib
import sys

import pandas as pd

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import util  # noqa
import analysis_main  # noqa

last_year = 2030
util.CARBON_BUDGET_CONSISTENT = "15-50"

_, df_sector = util.read_forward_analytics_data(analysis_main.SECTOR_INCLUDED)
emissions_fa = util.get_emissions_by_country(df_sector)
iso3166_df = util.read_iso3166()
alpha2_to_alpha3 = iso3166_df.set_index("alpha-2")["alpha-3"].to_dict()
ngfs_df = util.read_ngfs()
maturity_dict = (
    pd.read_csv(
        "./data_private/v0_power_plant_maturity_dates.csv.zip", compression="zip"
    )
    .set_index("uniqueforwardassetid")["maturity_year"]
    .to_dict()
)


def get_emissions_projection(scenario):
    emissions_with_ngfs_projection, _, _ = util.calculate_ngfs_projection(
        "emissions",
        emissions_fa,
        ngfs_df,
        analysis_main.SECTOR_INCLUDED,
        scenario,
        analysis_main.NGFS_PEG_YEAR,
        last_year,
        alpha2_to_alpha3,
    )
    return emissions_with_ngfs_projection


def get_ngfs_scenario(scenario):
    ngfs = ngfs_df["emissions"]
    ngfs = ngfs[ngfs.Scenario == scenario]
    if scenario == "Net Zero 2050" and util.CARBON_BUDGET_CONSISTENT:
        ngfs = util.read_carbon_budget_consistent(util.CARBON_BUDGET_CONSISTENT)
    return ngfs


def get_activity_unit_multiplier(row):
    mul = None
    match row.sector:
        case "Extraction":
            mul = 1e-3  # From mega tonnes to giga tonnes
        case "Power":
            # From MWh to MJ to GJ to giga tonnes of coal
            mul = util.seconds_in_1hour / 1e3 * util.GJ2coal(1) / 1e9
        case _:
            raise Exception("Should never happen")
    return mul


emde6 = "IN ID VN TR PL KZ".split()

# emissions_projection_CPS = get_emissions_projection("Current Policies")
emissions_projection_NZ2050 = get_emissions_projection("Net Zero 2050")
years = list(range(analysis_main.NGFS_PEG_YEAR, last_year + 1))


def calculate_power_plant_phaseout_order(method_name, df, measure):
    rho = util.calculate_rho(util.beta, rho_mode=analysis_main.RHO_MODE)
    for country in emde6:
        ep_by_country = [ep.loc[country, :] for ep in emissions_projection_NZ2050]
        df_country = df[df.asset_country == country].sort_values(
            measure, ascending=False
        )
        # We clone df_country, because its value is going to be changed over the course of the years.
        df_country_mutated = df_country.copy()
        total_opportunity_cost_owner = 0
        total_opportunity_cost_owner_discounted = 0
        power_plant_index = 0
        power_plant_order = pd.DataFrame()
        for subsector in util.SUBSECTORS:
            for i, year in enumerate(years):
                if i == 0:
                    # There is nothing to phase out at the start
                    continue
                discount = util.calculate_discount(rho, i)
                try:
                    phaseout = (
                        ep_by_country[i - 1].loc[subsector]
                        - ep_by_country[i].loc[subsector]
                    )
                except KeyError:
                    # The country doesn't own this subsector
                    continue
                while phaseout > 0:
                    try:
                        row = df_country_mutated.iloc[power_plant_index]
                    except IndexError:
                        raise Exception("Should not happen")
                    # The division by 1e3 converts MtCO2 to GtCO2.
                    e = row[util.EMISSIONS_COLNAME] / 1e3
                    e_original = (
                        df_country.iloc[power_plant_index][util.EMISSIONS_COLNAME] / 1e3
                    )
                    phaseout_amount = min(e, phaseout)
                    fraction = min(1, phaseout_amount / e_original)
                    oc_owner = (
                        fraction * row.activity * get_activity_unit_multiplier(row)
                    )
                    total_opportunity_cost_owner += oc_owner
                    total_opportunity_cost_owner_discounted += oc_owner * discount
                    order = pd.DataFrame(
                        {
                            "uniqueforwardassetid": [row.uniqueforwardassetid],
                            "asset_name": [row.asset_name],
                            "subsector": [subsector],
                            "fraction": [fraction],
                            "amount_mtco2": [phaseout_amount * 1e3],
                            "oc_owner": [oc_owner],
                            "score": [row[measure]],
                            "year": [year],
                        }
                    )
                    # The order is important here.
                    if e < phaseout:
                        power_plant_index += 1
                    else:
                        df_country_mutated.loc[
                            df_country_mutated.index[power_plant_index],
                            util.EMISSIONS_COLNAME,
                        ] -= phaseout * 1e3  # converts phaseout from GtCO2 to MtCO2
                    phaseout -= phaseout_amount
                    power_plant_order = pd.concat(
                        [power_plant_order, order], ignore_index=True
                    )
        power_plant_order.to_csv(
            f"plots/v2_power_plant_phaseout_order_by_{method_name}_{country}_{last_year}.csv",
            index=False,
        )
        print(
            method_name,
            country,
            "owner OC",
            total_opportunity_cost_owner,
            total_opportunity_cost_owner_discounted,
        )


def prepare_by_emissions_per_oc(df):
    unit_profit_df = analysis_main.unit_profit_df

    total_production_fa = util.get_production_by_country(
        df, analysis_main.SECTOR_INCLUDED
    )
    scenario_CPS = "Current Policies"
    # Giga tonnes of coal
    (
        production_with_ngfs_projection,
        gigatonnes_coal_production,
        profit_ngfs_projection,
    ) = util.calculate_ngfs_projection(
        "production",
        total_production_fa,
        ngfs_df,
        analysis_main.SECTOR_INCLUDED,
        scenario_CPS,
        analysis_main.NGFS_PEG_YEAR,
        last_year,
        alpha2_to_alpha3,
        unit_profit_df=unit_profit_df,
    )
    # This is profit projection divided by total production for a given country and subsector.
    profit_projection_fraction = {
        subsector: sum(e) / total_production_fa.xs(subsector, level="subsector")
        for subsector, e in profit_ngfs_projection.items()
    }

    def func(row):
        if row.subsector == "Other":
            return 0

        opportunity_cost = (
            row.activity
            * get_activity_unit_multiplier(row)
            * profit_projection_fraction[row.subsector][row.asset_country]
        )
        if opportunity_cost <= 0:
            return 0
        # The division of emissions by 1e3 converts MtCO2 to GtCO2.
        return row[util.EMISSIONS_COLNAME] / 1e3 / opportunity_cost

    df["emissions/OC"] = df.apply(func, axis=1)


def prepare_by_maturity(df):
    def func(row):
        maturity = maturity_dict.get(row.uniqueforwardassetid, 0)
        if maturity < 1000:
            maturity = 3000
        return -maturity

    df["minus_maturity"] = df.apply(func, axis=1)


calculate_power_plant_phaseout_order("emission_factor", df_sector, "emission_factor")

prepare_by_emissions_per_oc(df_sector)
calculate_power_plant_phaseout_order(
    "emissions_per_opportunity_cost_projection", df_sector, "emissions/OC"
)

prepare_by_maturity(df_sector)
calculate_power_plant_phaseout_order("maturity", df_sector, "minus_maturity")
