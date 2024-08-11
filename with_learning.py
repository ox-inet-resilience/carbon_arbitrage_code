import math
from collections import defaultdict

import util

ENABLE_WRIGHTS_LAW = 1
ENABLE_BATTERY_SHORT = True
ENABLE_BATTERY_LONG = True
ENABLE_BATTERY_GRID = True
ENABLE_RESIDUAL_BENEFIT = 1
NGFS_RENEWABLE_WEIGHT = "static_50%"
assert NGFS_RENEWABLE_WEIGHT in ["static_50%", "static_NGFS", "dynamic_NGFS"]
ENABLE_RENEWABLE_GRADUAL_DEGRADATION = 1
ENABLE_RENEWABLE_30Y_LIFESPAN = 1
assert ENABLE_RENEWABLE_GRADUAL_DEGRADATION or ENABLE_RENEWABLE_30Y_LIFESPAN
# Lifespan of the renewable energy
RENEWABLE_LIFESPAN = 30  # years

RENEWABLE_WEIGHTS = {
    "solar": 0.5,
    "onshore_wind": 0.25,
    "offshore_wind": 0.25,
}

# To generate this data, make sure to generate
# NGFS_renewable_additional_capacity_MODEL.json by running
# data_preparation/prepare_world_wright_learning.py.
# And then run misc/NGFS_renewable_additional_capacity.py
if NGFS_RENEWABLE_WEIGHT == "dynamic_NGFS":
    NGFS_dynamic_weight = util.read_json(
        f"data/NGFS_renewable_dynamic_weight_{util.NGFS_MODEL}.json"
    )
else:
    NGFS_dynamic_weight = None

irena = util.read_json("data/irena.json")


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
            self.global_installed_capacities_kW_2020[tech] = (
                global_installed_capacity_kW
            )
            alpha = installed_cost / (
                global_installed_capacity_kW ** -self.gammas[tech]
            )
            self.alphas[tech] = alpha
        # 1173 is in GWh
        self.G_battery_short_2020 = util.MWh2GJ(1173 * 1e3)  # GJ
        self.gamma_battery_short = -math.log2(1 - 0.253)
        # The 310 is in $/kWh
        self.alpha_2020_short_per_GJ = 310 / util.MWh2GJ(0.001)
        self.alpha_battery_short = self.alpha_2020_short_per_GJ / (
            self.G_battery_short_2020**-self.gamma_battery_short
        )
        self.G_battery_long_2020 = 80.24 * 1e3  # kW
        self.gamma_battery_long = -math.log2(1 - 0.086)
        # The 1313 is in $/kW
        self.alpha_2020_long_per_kW = 1313
        self.alpha_battery_long = self.alpha_2020_long_per_kW / (
            self.G_battery_long_2020**-self.gamma_battery_long
        )
        self.sigma_battery_long = 1 / 12
        self.stocks_kW = {tech: {} for tech in self.techs}
        self.stocks_GJ_battery_short = defaultdict(dict)
        self.stocks_kW_battery_long = defaultdict(dict)
        self.stocks_kW_battery_pe = {tech: defaultdict(dict) for tech in self.techs}

        # To be used in the full table1 calculation
        self.cost_non_discounted = []
        self.cost_discounted = []
        self.cost_non_discounted_battery_short = []
        self.cost_non_discounted_battery_long = []
        self.cost_non_discounted_battery_pe = []
        self.cost_non_discounted_battery_grid = []
        self.cost_non_discounted_battery_short_by_country = []
        self.cost_non_discounted_battery_long_by_country = []
        self.cost_non_discounted_battery_pe_by_country = []
        self.cost_non_discounted_battery_grid_by_country = []

        self.battery_unit_ic = {
            "short": {},
            "long": {},
        }

        self.cached_wrights_law_investment_costs = {
            "solar": {},
            "onshore_wind": {},
            "offshore_wind": {},
        }

        self.cached_cumulative_G = {
            "solar": {},
            "onshore_wind": {},
            "offshore_wind": {},
            "short": {},
            "long": {},
        }

    def get_static_weight(self, tech):
        return RENEWABLE_WEIGHTS[tech]

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
        self.cached_cumulative_G[tech][year] = cumulative_G
        return ic

    def get_weight(self, tech, year):
        if NGFS_RENEWABLE_WEIGHT == "static_50%":
            weight = self.get_static_weight(tech)
        elif NGFS_RENEWABLE_WEIGHT == "static_NGFS":
            weight = self.weights_static_NGFS[tech]
        else:
            # Needs division by 100, because the unit is still in percent.
            weight = NGFS_dynamic_weight[tech][str(year)] / 100
        return weight

    def calculate_ic_1country_battery_short(self, year, country_name, total_R):
        multiplier_alpha_battery_short = 0.2 / 365
        # GJ
        R_battery_short = self.get_stock_battery_short(year, country_name)
        D_battery_short = max(
            0, total_R * multiplier_alpha_battery_short - R_battery_short
        )

        # Calculating unit_ic
        stock_without_degradation = 0.0
        for stock_year, stock_amount in self.stocks_GJ_battery_short.items():
            if stock_year >= year:
                break
            stock_without_degradation += sum(stock_amount.values())
        # GJ
        cumulative_G = self.G_battery_short_2020 + stock_without_degradation
        # $/GJ
        if ENABLE_WRIGHTS_LAW:
            unit_ic = self.alpha_battery_short * (
                cumulative_G**-self.gamma_battery_short
            )
        else:
            unit_ic = self.alpha_2020_short_per_GJ
        # End of calculating unit_ic
        self.battery_unit_ic["short"][year] = unit_ic
        self.cached_cumulative_G["short"][year] = cumulative_G

        investment_cost_battery_short = D_battery_short * unit_ic
        self.stocks_GJ_battery_short[year][country_name] = D_battery_short
        return investment_cost_battery_short

    def calculate_ic_1country_battery_long(self, year, country_name, total_R):
        psi = 0.7
        fe = 0.5
        # kW
        S_battery_long = self.get_stock_battery_long(year, country_name)
        # kW
        G_long = max(
            0, self.GJ2kW(total_R) * self.sigma_battery_long / psi / fe - S_battery_long
        )

        # Calculating unit_ic
        stock_without_degradation = 0.0
        for stock_year, stock_amount in self.stocks_kW_battery_long.items():
            if stock_year >= year:
                break
            stock_without_degradation += sum(stock_amount.values())
        # kW
        cumulative_G = self.G_battery_long_2020 + stock_without_degradation
        # $/kW
        if ENABLE_WRIGHTS_LAW:
            unit_ic = self.alpha_battery_long * (cumulative_G**-self.gamma_battery_long)
        else:
            unit_ic = self.alpha_2020_long_per_kW
        # End of calculating unit_ic
        self.battery_unit_ic["long"][year] = unit_ic
        self.cached_cumulative_G["long"][year] = cumulative_G

        investment_cost_battery_long = G_long * unit_ic
        self.stocks_kW_battery_long[year][country_name] = G_long
        return investment_cost_battery_long

    def calculate_ic_1country_battery_pe(self, year, country_name, total_R):
        # Based on calculate_total_R
        R_pe = 0.0
        for tech in self.techs:
            S = self.get_stock_battery_pe(country_name, tech, year)
            R = self.kW2GJ(S) * self.capacity_factors[tech]
            R_pe += R
        # End of based on calculate_total_R

        psi = 0.7
        coefficient = self.sigma_battery_long * (1 / psi - 1)
        # GJ
        D = max(0, total_R * coefficient - R_pe)
        # kW
        D_kW = self.GJ2kW(D)

        ic = 0.0
        for tech in self.techs:
            weight = self.get_weight(tech, year)
            # kW
            G = weight * D_kW / self.capacity_factors[tech]
            if ENABLE_WRIGHTS_LAW:
                installed_cost = self.calculate_wrights_law_investment_cost(tech, year)
            else:
                installed_cost = self.installed_costs[tech]
            ic += G * installed_cost
            self.stocks_kW_battery_pe[tech][year][country_name] = G
        return ic

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
            self.stocks_GJ_battery_short[year][country_name] = 0.0
            self.stocks_kW_battery_long[year][country_name] = 0.0
            for tech in self.techs:
                self.stocks_kW_battery_pe[tech][year][country_name] = 0.0
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
        if ENABLE_BATTERY_SHORT:
            investment_cost_battery_short = self.calculate_ic_1country_battery_short(
                year, country_name, total_R
            )
            investment_cost += investment_cost_battery_short
        else:
            investment_cost_battery_short = 0
        if ENABLE_BATTERY_LONG:
            investment_cost_battery_long = self.calculate_ic_1country_battery_long(
                year, country_name, total_R
            )
            investment_cost += investment_cost_battery_long
            ic_battery_pe = self.calculate_ic_1country_battery_pe(
                year, country_name, total_R
            )
            investment_cost += ic_battery_pe
        else:
            investment_cost_battery_long = 0
            ic_battery_pe = 0
        if ENABLE_BATTERY_GRID:
            # 4.14 is in $/GJ
            _cgrid = 4.14
            i_grid_s2 = total_R * _cgrid
            grid_scenario = "pessimistic"
            # grid_scenario = "bau"
            # grid_scenario = "phaseout"
            if grid_scenario == "pessimistic":
                ic_battery_grid = i_grid_s2
            elif grid_scenario == "bau":
                i_grid_s1 = 280e9
                ic_battery_grid = max(0, i_grid_s2 - i_grid_s1)
            else:
                assert grid_scenario == "phaseout"
                raise Exception("TODO")
                ic_battery_grid = max(0, i_grid_s2 - _cgrid * f_coal * S_coal)

            investment_cost += ic_battery_grid
        else:
            ic_battery_grid = 0
        self.cost_non_discounted[-1][country_name] = investment_cost
        self.cost_discounted[-1][country_name] = investment_cost * discount
        self.cost_non_discounted_battery_short[-1] += investment_cost_battery_short
        self.cost_non_discounted_battery_long[-1] += investment_cost_battery_long
        self.cost_non_discounted_battery_pe[-1] += ic_battery_pe
        self.cost_non_discounted_battery_grid[-1] += ic_battery_grid
        self.cost_non_discounted_battery_short_by_country[-1][country_name] = (
            investment_cost_battery_short
        )
        self.cost_non_discounted_battery_long_by_country[-1][country_name] = (
            investment_cost_battery_long
        )
        self.cost_non_discounted_battery_pe_by_country[-1][country_name] = ic_battery_pe
        self.cost_non_discounted_battery_grid_by_country[-1][country_name] = (
            ic_battery_grid
        )

    def calculate_investment_cost(self, DeltaP, year, discount):
        self.cost_non_discounted_battery_short.append(0.0)
        self.cost_non_discounted_battery_long.append(0.0)
        self.cost_non_discounted_battery_pe.append(0.0)
        self.cost_non_discounted_battery_grid.append(0.0)
        self.cost_non_discounted_battery_short_by_country.append({})
        self.cost_non_discounted_battery_long_by_country.append({})
        self.cost_non_discounted_battery_pe_by_country.append({})
        self.cost_non_discounted_battery_grid_by_country.append({})
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

    def get_stock_battery_short(self, year, country_name):
        out = 0.0
        if len(self.stocks_GJ_battery_short) == 0:
            return out
        for stock_year, stock_amount in self.stocks_GJ_battery_short.items():
            if stock_year >= year:
                break
            age = year - stock_year
            s = stock_amount[country_name]
            if ENABLE_RENEWABLE_30Y_LIFESPAN:
                if age <= 12:
                    out += s
            else:
                out += s
        return out

    def get_stock_battery_long(self, year, country_name):
        out = 0.0
        if len(self.stocks_kW_battery_long) == 0:
            return out
        for stock_year, stock_amount in self.stocks_kW_battery_long.items():
            if stock_year >= year:
                break
            age = year - stock_year
            s = stock_amount[country_name]
            if ENABLE_RENEWABLE_30Y_LIFESPAN:
                if age <= 16:
                    out += s
            else:
                out += s
        return out

    def get_stock_battery_pe(self, country_name, tech, year):
        # This method is identitical to get_stock except that it uses stocks_kW_battery_pe
        out = 0.0
        if len(self.stocks_kW_battery_pe[tech]) == 0:
            return out
        for stock_year, stock_amount in self.stocks_kW_battery_pe[tech].items():
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
            stock_battery_pe = sum(self.stocks_kW_battery_pe[tech][stock_year].values())
            out += sum(stock_amount.values()) + stock_battery_pe
        return out

    def zero_out_costs_and_stocks(self, peg_year):
        raise Exception("Don't use this!")
        # We zero out the values up to peg_year.
        peg_year_index = peg_year - 2022
        for i in range(peg_year_index + 1):
            self.cost_non_discounted[i] = 0.0
            self.cost_discounted[i] = 0.0
        for tech in self.techs:
            # Zero out all the stocks up to peg_year.
            self.stocks_kW[tech] = {
                y: (0.0 if y <= peg_year else v)
                for y, v in self.stocks_kW[tech].items()
            }

    def calculate_residual_one_year(self, year, weighted_emissions_factor_by_country):
        equivalent_emissions = {}
        equivalent_production = 0.0
        for (
            country_name,
            emissions_factor,
        ) in weighted_emissions_factor_by_country.items():
            # in GJ
            total_R = self.calculate_total_R(country_name, year)
            tonnes_of_coal_equivalent = util.GJ2coal(total_R)
            equivalent_emissions[country_name] = (
                tonnes_of_coal_equivalent * emissions_factor
            )
            equivalent_production += tonnes_of_coal_equivalent
        return equivalent_emissions, equivalent_production

    def calculate_residual(
        self, year_start, year_end, weighted_emissions_factor_by_country
    ):
        if not ENABLE_RESIDUAL_BENEFIT:
            return 0.0, 0.0
        residual_emissions = defaultdict(float)
        residual_production = 0.0
        for year in range(year_start, year_end + 1):
            (
                equivalent_emissions,
                equivalent_production,
            ) = self.calculate_residual_one_year(
                year, weighted_emissions_factor_by_country
            )
            for k, v in equivalent_emissions.items():
                residual_emissions[k] += v
            residual_production += equivalent_production
        return residual_emissions, residual_production
