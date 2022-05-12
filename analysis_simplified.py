# This file is a simplified version of analysis_main.py, which calculate the
# carbon arbitrage opportunity for the baseline parameters, and with restricted
# customization.
# - (Important) the residual benefit is not calculated, for simplicity purpose.
#   Hence our default result here is 58.01 trillion dollars instead of 77.89
#   trillion dollars.
# - The coal phase out scenario is restricted to "Net Zero 2050"
# - The energy type specific average unit profit is restricted to the median of
#   top 10 pure coal nonpower

import math

import util
import processed_revenue

# The default value is 75 here.
social_cost_of_carbon = util.social_cost_of_carbon
# This is the coal phase out scenario
scenario = "Net Zero 2050"
sector = "Coal"
# The year where the NGFS value is pegged/rescaled to be the same as
# Masterdata/AR-2DII global production value.
NGFS_PEG_YEAR = 2023
# Time horizon -- the last year of the arbitrage
last_year = 2100
# In $/GJ
top10_aup_median_pure_coal_pure_nonpower = 0.011480418063059742
energy_type_specific_average_unit_profit = top10_aup_median_pure_coal_pure_nonpower
# The weight / percentage of the replacement being solar
RENEWABLE_SOLAR_STATIC_WEIGHT = 0.5
# The following 4 variables should be self-explanatory
ENABLE_WRIGHTS_LAW = 1
ENABLE_RENEWABLE_GRADUAL_DEGRADATION = 1
ENABLE_RENEWABLE_30Y_LIFESPAN = 1
RENEWABLE_LIFESPAN = 30  # years


# rho is the discount rate. Its value is computed to be 0.028.
# We set the  processed_revenue.beta to be constant, based on the MM beta of
# aggregate_beta.py We simplify the model because the beta data is not good.
# It is basically the unleveraged beta, calculated from misc/aggregate_beta.py.
rho = util.calculate_rho(processed_revenue.beta)
# The NGFS coal production data (public). You can see the content for coal
# (nonpower) in data/ngfs_scenario_production_fossil.csv.
ngfss = util.read_ngfs_coal_and_power()
# The Masterdata, i.e. the AR-2DII data (private)
# This is what a row in the DataFrame is like
# ```
# {
#     "company_name": "Tazenda",
#     "sector": "Coal",
#     "technology": "Lignite",
#     "technology_type": "Surface",
#     "asset_country": "US",
#     "emissions_factor": 1.60,
#     "emission_factor_unit": "tonnes of CO2 per tonnes of coal",
#     "number_of_assets": 1.0,
#     "unit": "tonnes of coal",
#     # This is the coal production in 2013, and so on.
#     "_2013": 299792.0,
#     #... until 2026
#     "_2026": 105457.0,
# }
# ```
_, nonpower_coal, _ = util.read_masterdata()

# NGFS energy production over the years, rescaled so that its value during the
# peg year is 1.
fraction_increase_after_peg_year = util.calculate_ngfs_fractional_increase(
    ngfss, sector, scenario, start_year=NGFS_PEG_YEAR
)
fraction_increase_after_peg_year_CPS = util.calculate_ngfs_fractional_increase(
    ngfss, sector, "Current Policies ", start_year=NGFS_PEG_YEAR
)
# IRENA data (public)
# It's a short JSON file that you can read to get an idea what it contains.
irena = util.read_json("data/irena.json")


def get_cost_including_ngfs(
    fraction_increase_after_peg_year,
    rev_ren,  # the value is either "revenue" or "renewable"
):
    # This function calculates the present values of missed free cash flows of
    # coal companies resulting from phasing out coal -- all from 2022 to
    # last_year.

    # Since the scenario is NZ2050, the 2022-NGFS_PEG_YEAR costs are
    # force-set to 0, because the difference of s2-s1 in
    # 2022-NGFS_PEG_YEAR is 0.
    total_cost_discounted = 0.0
    # We simplify the code here. In the full version, you can choose rev_ren to
    # be "renewable", which calculates the investment cost of renewable using
    # the global lcoe average.
    assert rev_ren == "revenue"

    def calculate_cost(year):
        # Calculate cost for a given year.
        _df = nonpower_coal
        # This function converts coal production unit from tce to GJ.
        _convert2GJ = util.coal2GJ

        grouped = _df.groupby("asset_country")
        coal_production_by_country = grouped[f"_{year}"].sum()
        # An example of a coal_production_by_country value (as pandas.Series):
        # ```
        # asset_country
        # AR    2.639000e+06
        # AU    9.028651e+08
        #           ...
        # ZM    3.162331e+06
        # ZW    2.294319e+07
        # ```
        in_gj = _convert2GJ(coal_production_by_country)
        cost = energy_type_specific_average_unit_profit * in_gj
        return cost

    cost_peg_year = calculate_cost(NGFS_PEG_YEAR)
    for y, fraction_increase_np in fraction_increase_after_peg_year.items():
        discount = util.calculate_discount(rho, y - 2022)
        if y <= 2026:
            # If the year is <= 2026, we use masterdata for CPS.
            # It's slightly more complicated to calculate DeltaP in
            # this case, because we have to use the masterdata value
            # (obtained via calculate_cost) instead of the NGFS
            # fractional increase from NGFS_PEG_YEAR..
            cost_masterdata_y = calculate_cost(y)
            cost = cost_masterdata_y.subtract(
                cost_peg_year * fraction_increase_np, fill_value=0.0
            )
        else:
            fraction_delta = (
                fraction_increase_after_peg_year_CPS[y] - fraction_increase_np
            )
            cost = cost_peg_year * fraction_delta

        # cost is the left hand side of equation 6 in the paper.
        total_cost_discounted += cost * discount

    # The sum is over the countries, because remember, we use groupby on the
    # asset_country when calculating the cost.
    return sum(total_cost_discounted)


class InvestmentCostNewMethod:
    techs = ["solar", "onshore_wind", "offshore_wind"]
    # Mentioned in the carbon arbitrage paper page 21, which is from Staffell
    # and Green 2014.
    degradation_rate = {
        "solar": 0.5 / 100,
        "onshore_wind": 0.48 / 100,
        "offshore_wind": 0.48 / 100,
    }
    # Wright's law learning rate
    # See equation 15 in the carbon arbitrage paper on how these numbers are
    # calculated.
    gammas = {"solar": 0.32, "onshore_wind": 0.07, "offshore_wind": 0.04}

    def __init__(self):
        self.capacity_factors = {}
        self.installed_costs = {}
        self.global_installed_capacities_kW_2020 = {}
        self.alphas = {}
        for tech in self.techs:
            # The [-1] is needed to get the value in 2020.
            self.capacity_factors[tech] = (
                irena[f"capacity_factor_{tech}_2010_2020_percent"][-1] / 100
            )
            # Same as investment cost
            installed_cost = irena[f"installed_cost_{tech}_2010_2020_$/kW"][-1]
            self.installed_costs[tech] = installed_cost
            global_installed_capacity_kW = (
                irena[f"{tech}_MW_world_cumulative_total_installed_capacity_2011_2020"][
                    -1
                ]
                * 1e3
            )
            self.global_installed_capacities_kW_2020[
                tech
            ] = global_installed_capacity_kW
            # See the paragraph after equation 15 in the carbon arbitrage paper.
            alpha = installed_cost / (
                global_installed_capacity_kW ** -self.gammas[tech]
            )
            self.alphas[tech] = alpha
        self.stocks_kW = {tech: {} for tech in self.techs}

        # To be used in the full cost1 table calculation
        self.cost_non_discounted = []
        self.cost_discounted = []

        self.cached_wrights_law_investment_costs = {
            "solar": {},
            "onshore_wind": {},
            "offshore_wind": {},
        }

    def get_static_weight(self, tech):
        return {
            "solar": RENEWABLE_SOLAR_STATIC_WEIGHT,
            "onshore_wind": (1.0 - RENEWABLE_SOLAR_STATIC_WEIGHT) * 0.5,
            "offshore_wind": (1.0 - RENEWABLE_SOLAR_STATIC_WEIGHT) * 0.5,
        }[tech]

    def GJ2kW(self, x):
        # MW
        mw = util.GJ2MW(x)
        # kW
        return mw * 1e3

    def kW2GJ(self, x):
        # MW
        mw = x / 1e3
        return util.MW2GJ(mw)

    def calculate_total_R(self, country_name, year):
        total_R = 0.0
        for tech in self.techs:
            # S is calculated according to equation 13 in the carbon arbitrage
            # paper.
            S = self.get_stock(country_name, tech, year)
            R = self.kW2GJ(S) * self.capacity_factors[tech]
            total_R += R
        return total_R

    def _calculate_wrights_law(self, tech, year, cumulative_G):
        # This is equation 14 in the carbon arbitrage paper.
        return self.alphas[tech] * (cumulative_G ** -self.gammas[tech])

    def calculate_wrights_law_investment_cost(self, tech, year):
        if year in self.cached_wrights_law_investment_costs[tech]:
            return self.cached_wrights_law_investment_costs[tech][year]
        cumulative_G = self.global_installed_capacities_kW_2020[
            tech
        ] + self.get_stock_without_degradation(tech, year)
        ic = self._calculate_wrights_law(tech, year, cumulative_G)
        self.cached_wrights_law_investment_costs[tech][year] = ic
        return ic

    def calculate_investment_cost_one_country(
        self, country_name, DeltaP, year, discount
    ):
        # in GJ
        # total_R is calculated according to equation 12 in the carbon
        # arbitrage paper.
        total_R = self.calculate_total_R(country_name, year)
        # in GJ
        # D is calculated according to equation 11 in the carbon arbitrage paper.
        D = max(0, DeltaP - total_R)
        if math.isclose(D, 0):
            self.cost_non_discounted[-1][country_name] = 0.0
            self.cost_discounted[-1][country_name] = 0.0
            for tech in self.techs:
                if year in self.stocks_kW[tech]:
                    self.stocks_kW[tech][year][country_name] = 0.0
                else:
                    self.stocks_kW[tech][year] = {country_name: 0.0}
            return
        # In kW instead of GW because installed_costs unit is in $/kW.
        D_kW = self.GJ2kW(D)

        investment_cost = 0.0
        for tech in self.techs:
            # in kW
            # This is equivalent to NGFS_RENEWABLE_WEIGHT to be static_50% of
            # the full code version.
            weight = self.get_static_weight(tech)
            # G is the term you see in equation 9 in the paper.
            # G is calculated according to equation 10 in the carbon arbitrage paper.
            # weight is omega in the carbon arbitrage paper.
            # self.capacity_factors[tech] in the carbon arbitrage paper.
            G = weight * D_kW / self.capacity_factors[tech]
            # installed_cost is the second term of the rhs of equation 9 in the paper.
            installed_cost = self.installed_costs[tech]
            if ENABLE_WRIGHTS_LAW:
                installed_cost = self.calculate_wrights_law_investment_cost(tech, year)
            investment_cost += G * installed_cost
            if year in self.stocks_kW[tech]:
                self.stocks_kW[tech][year][country_name] = G
            else:
                self.stocks_kW[tech][year] = {country_name: G}
        self.cost_non_discounted[-1][country_name] = investment_cost
        self.cost_discounted[-1][country_name] = investment_cost * discount

    def calculate_investment_cost(self, DeltaP, year, discount):
        if isinstance(DeltaP, float):
            assert math.isclose(DeltaP, 0)
            self.cost_non_discounted.append(0.0)
            self.cost_discounted.append(0.0)
            return
        self.cost_non_discounted.append({})
        self.cost_discounted.append({})
        for country_name, dp in DeltaP.items():
            self.calculate_investment_cost_one_country(country_name, dp, year, discount)

    def get_stock(self, country_name, tech, year):
        # This is equation 13 in the carbon arbitrage paper.
        out = 0.0
        if len(self.stocks_kW[tech]) == 0:
            return out
        for stock_year, stock_amount in self.stocks_kW[tech].items():
            if stock_year >= year:
                break
            age = year - stock_year
            s = stock_amount[country_name]
            if ENABLE_RENEWABLE_GRADUAL_DEGRADATION:
                s *= (1 - self.degradation_rate[tech]) ** age

            if ENABLE_RENEWABLE_30Y_LIFESPAN:
                if age <= RENEWABLE_LIFESPAN:
                    out += s
            else:
                # No lifespan checking is needed.
                out += s
        return out

    def get_stock_without_degradation(self, tech, year):
        out = 0.0
        if len(self.stocks_kW[tech]) == 0:
            return out
        for stock_year, stock_amount in self.stocks_kW[tech].items():
            if stock_year >= year:
                break
            out += sum(stock_amount.values())
        return out


def get_cost_including_ngfs_renewable_new_method(
    fraction_increase_after_peg_year,
):
    # We keep the old name so that it is easier to compare with the full
    # version.
    temp_cost_new_method = InvestmentCostNewMethod()

    def calculate_gj(year):
        _df = nonpower_coal
        _convert2GJ = util.coal2GJ

        grouped = _df.groupby("asset_country")
        coal_production_by_country = grouped[f"_{year}"].sum()
        in_gj = _convert2GJ(coal_production_by_country)
        return in_gj

    gj_peg_year = calculate_gj(NGFS_PEG_YEAR)
    for y, fraction_increase_np in fraction_increase_after_peg_year.items():
        discount = util.calculate_discount(rho, y - 2022)
        if y <= 2026:
            # If the year is <= 2026, we use masterdata for CPS.
            # It's slightly more complicated to calculate DeltaP in
            # this case, because we have to use the masterdata value
            # (obtained via calculate_gj_and_c) instead of the NGFS
            # fractional increase from NGFS_PEG_YEAR..
            gj_sum_nonpower_y = calculate_gj(y)
            DeltaP = gj_sum_nonpower_y.subtract(
                gj_peg_year * fraction_increase_np, fill_value=0.0
            )
        else:
            fraction_delta = (
                fraction_increase_after_peg_year_CPS[y] - fraction_increase_np
            )
            DeltaP = gj_peg_year * fraction_delta

        temp_cost_new_method.calculate_investment_cost(DeltaP, y, discount)
    # The sum is over the countries
    out_discounted = sum(sum(e.values()) for e in temp_cost_new_method.cost_discounted)
    return out_discounted


# The opportunity cost of missed free cash flows of coal companies resulting
# from phasing out coal.
cost_discounted_revenue = get_cost_including_ngfs(
    fraction_increase_after_peg_year,
    "revenue",
)

# Cost of investing in green energy.
cost_discounted_investment = get_cost_including_ngfs_renewable_new_method(
    fraction_increase_after_peg_year,
)

# This is equation 4 in the carbon arbitrage paper.
cost_discounted = cost_discounted_revenue + cost_discounted_investment
# Convert to trillion USD
cost_discounted /= 1e12

# Calculating benefit
# util.years_masterdata is a list of years that goes from 2022 to 2026
# inclusive.
total_emissions_by_year = util.get_coal_nonpower_global_emissions_across_years(
    nonpower_coal, util.years_masterdata
)
total_emissions_peg_year_non_discounted = total_emissions_by_year[NGFS_PEG_YEAR - 2022]
# For CPS, the peg year is 2026
cps_peg_year = 2026
total_emissions_peg_year_non_discounted_CPS = total_emissions_by_year[
    cps_peg_year - 2022
]
total_emissions_non_discounted_after_peg_year = sum(
    total_emissions_peg_year_non_discounted * v_np
    for k, v_np in fraction_increase_after_peg_year.items()
)
fraction_increase_after_peg_year_CPS_different_peg_year = (
    util.calculate_ngfs_fractional_increase(
        ngfss, sector, "Current Policies ", start_year=cps_peg_year
    )
)
total_emissions_non_discounted_after_peg_year_CPS = sum(
    total_emissions_peg_year_non_discounted_CPS * v_np
    for k, v_np in fraction_increase_after_peg_year_CPS_different_peg_year.items()
)
total_emissions = (
    sum(total_emissions_by_year[: NGFS_PEG_YEAR + 1 - 2022])
    + total_emissions_non_discounted_after_peg_year
)
total_emissions_CPS = (
    sum(total_emissions_by_year[: cps_peg_year + 1 - 2022])
    + total_emissions_non_discounted_after_peg_year_CPS
)

# We multiply by 1e9 to go from GtCO2 to tCO2.
# We divide by 1e12 to get trilllion USD.
# We assume the social cost of carbon to be constant throughout the years.
# I.e. We essentially assume that the discount rate is equal to the growth rate
# in the SCC.
# The following code is equation 2 in the carbon arbitrage paper.
benefit_non_discounted = (
    (total_emissions_CPS - total_emissions) * 1e9 / 1e12 * social_cost_of_carbon
)

# `net_benefit` is calculated according to equation 1 in the carbon arbitrage
# paper.
net_benefit = benefit_non_discounted - cost_discounted
print(benefit_non_discounted, cost_discounted)
# Sanity check
assert math.isclose(net_benefit, 58.01459793600306)
print("Carbon arbitrage opportunity (in trillion dollars)", net_benefit)
