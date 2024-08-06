import json
import math


def read_json(filename):
    with open(filename) as f:
        obj = json.load(f)
    return obj


irena = read_json("data/irena.json")
hours_in_1year = 24 * 365.25
seconds_in_1hour = 3600  # seconds

ENABLE_WRIGHTS_LAW = True
RENEWABLE_WEIGHTS = {
    "solar": 0.5,
    "onshore_wind": 0.25,
    "offshore_wind": 0.25,
}
ENABLE_RENEWABLE_GRADUAL_DEGRADATION = 1
ENABLE_RENEWABLE_30Y_LIFESPAN = 1
RENEWABLE_LIFESPAN = 30  # years


def calculate_discount(rho, deltat):
    return (1 + rho) ** -deltat


def GJ2MW(x):
    # GJ to MJ
    mj = x * 1e3
    # MJ to MW
    return mj / (hours_in_1year * seconds_in_1hour)


def MW2GJ(x):
    # MW to GW
    gw = x / 1e3
    # GW to GJ
    return gw * hours_in_1year * seconds_in_1hour


def GJ2MWh(x):
    # GJ to J
    joule = x * 1e9
    # J to Wh
    wh = joule / 3600
    # Wh to MWh
    return wh / 1e6


def GJ2coal(x):
    return x / 29.3076


def EJ2Mcoal(x):
    coal = x * 1e9 / 29.3076
    return coal / 1e6  # million tonnes of coal


class InvestmentCostWithLearning:
    techs = ["solar", "onshore_wind", "offshore_wind"]
    weights_static_NGFS = {
        "solar": 55.98399919148438 / 100,
        "onshore_wind": 42.000406987439874 / 100,
        "offshore_wind": 2.0155938210757327 / 100,
    }
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
        return RENEWABLE_WEIGHTS[tech]

    def GJ2kW(self, x):
        # MW
        mw = GJ2MW(x)
        # kW
        return mw * 1e3

    def kW2GJ(self, x):
        # MW
        mw = x / 1e3
        return MW2GJ(mw)

    def calculate_total_R(self, country_name, year):
        total_R = 0.0
        for tech in self.techs:
            S = self.get_stock(country_name, tech, year)
            R = self.kW2GJ(S) * self.capacity_factors[tech]
            total_R += R
        return total_R

    def _calculate_wrights_law(self, tech, year, cumulative_G):
        # Equation WrightsLaw, i.e. 15
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

    def get_weight(self, tech, year):
        # Static 50%
        weight = self.get_static_weight(tech)
        return weight

    def calculate_investment_cost_one_country(
        self, country_name, DeltaP, year, discount
    ):
        # in GJ
        total_R = self.calculate_total_R(country_name, year)
        # in GJ
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
        # in kW because installed_costs is in $/kW
        D_kW = self.GJ2kW(D)

        investment_cost = 0.0
        for tech in self.techs:
            # in kW
            weight = self.get_weight(tech, year)
            G = weight * D_kW / self.capacity_factors[tech]
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

    def calculate_residual_one_year(self, year, weighted_emissions_factor_by_country):
        equivalent_emissions = 0.0
        equivalent_production = 0.0
        for (
            country_name,
            emissions_factor,
        ) in weighted_emissions_factor_by_country.items():
            # in GJ
            total_R = self.calculate_total_R(country_name, year)
            tonnes_of_coal_equivalent = GJ2coal(total_R)
            equivalent_emissions += tonnes_of_coal_equivalent * emissions_factor
            equivalent_production += tonnes_of_coal_equivalent
        return equivalent_emissions, equivalent_production

    def calculate_residual(
        self, year_start, year_end, weighted_emissions_factor_by_country
    ):
        residual_emissions = 0.0
        residual_production = 0.0
        for year in range(year_start, year_end + 1):
            (
                equivalent_emissions,
                equivalent_production,
            ) = self.calculate_residual_one_year(
                year, weighted_emissions_factor_by_country
            )
            residual_emissions += equivalent_emissions
            residual_production += equivalent_production
        return residual_emissions, residual_production
