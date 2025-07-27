import math
from collections import defaultdict
from functools import lru_cache

import pandas as pd

import util

VERBOSE_ANALYSIS = False
VERBOSE_ANALYSIS_COUNTRY = "PL"
ENABLE_WRIGHTS_LAW = 1
ENABLE_BATTERY_SHORT = True
ENABLE_BATTERY_LONG = True
ENABLE_BATTERY_GRID = True
ENABLE_RESIDUAL_BENEFIT = 1
ENABLE_RENEWABLE_GRADUAL_DEGRADATION = 1
ENABLE_RENEWABLE_30Y_LIFESPAN = 1
assert ENABLE_RENEWABLE_GRADUAL_DEGRADATION or ENABLE_RENEWABLE_30Y_LIFESPAN
# Lifespan of the renewable energy
RENEWABLE_LIFESPAN = 30  # years
BATTERY_LONG_LIFESPAN = 16  # years
BATTERY_SHORT_LIFESPAN = 12  # years
CAPACITY_FACTOR_SOURCE = "FA"
# CAPACITY_FACTOR_SOURCE = "IRENA"
RENEWABLE_WEIGHT_SOURCE = "FA"
# RENEWABLE_WEIGHT_SOURCE = "GCA1"

TECHS_WITH_LEARNING = ["solar", "onshore_wind", "offshore_wind"]
TECHS_NO_LEARNING = ["geothermal", "hydropower"]
TECHS = TECHS_WITH_LEARNING + TECHS_NO_LEARNING
if RENEWABLE_WEIGHT_SOURCE == "GCA1":
    TECHS = TECHS_WITH_LEARNING
irena = util.read_json("data/irena.json")
petro_states = "CN KR RU AE SA QA BH BN KW".split()


def get_emde_developing_developed():
    emerging_country_shortnames = util.get_emerging_countries()
    developING_country_shortnames = util.get_developing_countries()
    (
        _,
        _,
        _,
        _,
        developed_country_shortnames,
    ) = util.prepare_from_climate_financing_data()
    return (
        emerging_country_shortnames + developING_country_shortnames,
        developING_country_shortnames,
        developed_country_shortnames,
    )


EMDE, DEVELOPING, DEVELOPED = get_emde_developing_developed()
DEVELOPED_UNFCCC, DEVELOPING_UNFCCC = util.get_countries_unfccc()
# Modification for COP30
DEVELOPED_UNFCCC += petro_states
DEVELOPING_UNFCCC = [e for e in DEVELOPING_UNFCCC if e not in petro_states]


def prepare_fa_capacity_factor_data():
    fa_capacity_factor = pd.read_csv(
        "./data_private/v3_capacity_weighted_average_capacity_factor.csv"
    )
    fa_capacity_factor = fa_capacity_factor[
        [
            "region",
            "asset_location",
            "Solar",
            "Wind_Offshore",
            "Wind_Onshore",
            "Geothermal",
            "Hydropower",
        ]
    ].rename(
        columns={
            "Solar": "solar",
            "Wind_Offshore": "offshore_wind",
            "Wind_Onshore": "onshore_wind",
            "Geothermal": "geothermal",
            "Hydropower": "hydropower",
        }
    )
    fa_capacity_factor_world = fa_capacity_factor[
        fa_capacity_factor.region == "Global"
    ].iloc[0]
    fa_capacity_factor = fa_capacity_factor[
        fa_capacity_factor.asset_location.notna()
    ].set_index("asset_location")
    return fa_capacity_factor, fa_capacity_factor_world


def prepare_fa_renewable_weights_data():
    fa_energy_mix = pd.read_csv(
        "./data_private/v3_renewable_energy_mix_Forward_analytics.csv"
    )
    fa_energy_mix = fa_energy_mix[
        [
            "Unnamed: 0",
            "asset_location",
            "Solar_Capacity (%)",
            "Wind_Offshore_Capacity (%)",
            "Wind_Onshore_Capacity (%)",
            "Geothermal_Capacity (%)",
            "Hydropower_Capacity (%)",
        ]
    ].rename(
        columns={
            "Solar_Capacity (%)": "solar",
            "Wind_Offshore_Capacity (%)": "offshore_wind",
            "Wind_Onshore_Capacity (%)": "onshore_wind",
            "Geothermal_Capacity (%)": "geothermal",
            "Hydropower_Capacity (%)": "hydropower",
        }
    )
    # Normalize solar and wind
    normalization_factor = fa_energy_mix[TECHS].sum(axis=1)
    fa_energy_mix[TECHS] = fa_energy_mix[TECHS].div(normalization_factor, axis=0)

    fa_energy_mix_world = fa_energy_mix[fa_energy_mix["Unnamed: 0"] == "Global"].iloc[0]
    # This is where energy mix deviates from capacity factor.
    # In there, if there is NaN value, we fallback to world capacity factor,
    # but here, we fallback to 0.
    fa_energy_mix = fa_energy_mix.fillna(0.0)

    fa_energy_mix = fa_energy_mix[fa_energy_mix.asset_location.notna()].set_index(
        "asset_location"
    )
    return fa_energy_mix, fa_energy_mix_world


fa_capacity_factor, fa_capacity_factor_world = prepare_fa_capacity_factor_data()
fa_energy_mix, fa_energy_mix_world = prepare_fa_renewable_weights_data()
irena_capacity_factor = {
    tech: irena[f"capacity_factor_{tech}_2010_2020_percent"][-1] / 100
    for tech in TECHS_WITH_LEARNING
}


# Beware of the LRU cache!! If we modify CAPACITY_FACTOR_SOURCE programmatically,
# this will cause bugs.
@lru_cache(maxsize=1024)
def get_capacity_factor(tech, country_name):
    if CAPACITY_FACTOR_SOURCE == "IRENA":
        return irena_capacity_factor[tech]
    assert CAPACITY_FACTOR_SOURCE == "FA"
    cf = fa_capacity_factor[tech].get(country_name, pd.NA)
    if pd.isna(cf):
        cf = fa_capacity_factor_world[tech]
    return cf


# Beware of the LRU cache!! If we modify RENEWABLE_WEIGHT_SOURCE programmatically,
# this will cause bugs.
@lru_cache(maxsize=1024)
def get_renewable_weight(tech, country_name):
    if RENEWABLE_WEIGHT_SOURCE == "GCA1":
        return {
            "solar": 0.5,
            "onshore_wind": 0.25,
            "offshore_wind": 0.25,
        }[tech]
    assert RENEWABLE_WEIGHT_SOURCE == "FA"
    weight = fa_energy_mix[tech].get(country_name, pd.NA)
    if pd.isna(weight):
        weight = fa_energy_mix_world[tech]
    return weight


def per_GJ2per_kWh(x):
    # Convert from $/GJ to $/kWh
    return x / (util.GJ2MWh(1) * 1e3)


class InvestmentCostWithLearning:
    # Mentioned in the carbon arbitrage paper page 21, which is from Staffell
    # and Green 2014.
    degradation_rate = {
        "solar": 0.5 / 100,
        "onshore_wind": 0.48 / 100,
        "offshore_wind": 0.48 / 100,
        "geothermal": 0.5 / 100,
        "hydropower": 0.5 / 100,
        "short": 2 / 100,
        "long": 0.5 / 100,
    }
    # Wright's law learning rate
    # See equation 15 in the carbon arbitrage paper on how these numbers are
    # calculated.
    # From Samadi 2018
    gammas = {"solar": 0.32, "onshore_wind": 0.07, "offshore_wind": 0.04}

    # Source: Table H.1. IRENA (2023), Renewable power generation costs in 2022
    # Same as investment cost
    # $/kW
    installed_costs = {
        "solar": 876,
        "onshore_wind": 1274,
        "offshore_wind": 3461,
        "geothermal": 3478,
        "hydropower": 2881,
    }

    # IRENA 2023
    global_installed_capacities_kW = {
        "solar": 1412093 * 1e3,
        "onshore_wind": 944205 * 1e3,
        "offshore_wind": 73185 * 1e3,
        "geothermal": 14846 * 1e3,
        "hydropower": 1406863 * 1e3,
    }

    def __init__(self):
        self.alphas = {
            tech: self.installed_costs[tech]
            / (self.global_installed_capacities_kW[tech] ** -self.gammas[tech])
            for tech in TECHS_WITH_LEARNING
        }

        # IEA 2023
        # 2400 is in GWh
        self.G_battery_short = util.MWh2GJ(2400 * 1e3)  # GJ
        # 0.42
        self.gamma_battery_short = -math.log2(1 - 0.253)
        # The 315 is in $/kWh, in 2022
        self.alpha_short_per_GJ = 315 / util.MWh2GJ(0.001)
        self.alpha_battery_short = self.alpha_short_per_GJ / (
            self.G_battery_short**-self.gamma_battery_short
        )
        self.G_battery_long = 217 * 1e3  # kW
        self.gamma_battery_long = -math.log2(1 - 0.086)
        # The 1355 is in $/kW, in 2022
        self.alpha_long_per_kW = 1355
        self.alpha_battery_long = self.alpha_long_per_kW / (
            self.G_battery_long**-self.gamma_battery_long
        )
        self.sigma_battery_long = 1 / 12
        self.stocks_kW = {tech: defaultdict(dict) for tech in TECHS + ["long"]}
        self.stocks_GJ_battery_short = defaultdict(dict)
        self.stocks_kW_battery_pe = {
            tech: defaultdict(dict) for tech in TECHS_WITH_LEARNING
        }

        # To be used in the full table1 calculation
        self.cost_non_discounted = []
        self.cost_discounted = []
        self.cost_non_discounted_battery_short_by_country = []
        self.cost_non_discounted_battery_long_by_country = []
        self.cost_non_discounted_battery_pe_by_country = []
        self.cost_non_discounted_battery_grid_by_country = []
        self.green_energy_produced_by_country = []

        self.battery_unit_ic = {
            "short": {},
            "long": {},
        }

        self.cached_investment_costs = {}
        self.cached_cumulative_G = {}
        for obj in [
            "solar",
            "onshore_wind",
            "offshore_wind",
            "short",
            "long",
            "geothermal",
            "hydropower",
        ]:
            self.cached_investment_costs[obj] = {}
            self.cached_cumulative_G[obj] = {}
        self.cached_stock_without_degradation = defaultdict(dict)
        self.cached_stock = defaultdict(dict)
        self.cached_stock_without_degradation_fast = {}
        self.cached_total_R = {}
        self.cached_get_stock = {}

    def GJ2kW(self, x):
        # MW
        mw = util.GJ2MW(x)
        # kW
        return mw * 1e3

    def kW2GJ(self, x):
        # MW
        mw = x / 1e3
        return util.MW2GJ(mw)

    def _calculate_R(self, country_name, tech, year):
        S = self.get_stock(country_name, tech, year)
        R = self.kW2GJ(S) * get_capacity_factor(tech, country_name)
        # Sanity check that it must be non-negative
        assert R >= 0, (R, country_name, tech, year)
        return R

    def calculate_total_R(self, country_name, year):
        # Check cache first
        cache_key = (country_name, year)
        if cache_key in self.cached_total_R:
            return self.cached_total_R[cache_key]

        total_R = 0.0
        for tech in TECHS:
            total_R += self._calculate_R(country_name, tech, year)

        # Cache the result
        self.cached_total_R[cache_key] = total_R
        return total_R

    def _calculate_wrights_law(self, tech, year, cumulative_G):
        # Equation WrightsLaw, i.e. 15
        return self.alphas[tech] * (cumulative_G ** -self.gammas[tech])

    def calculate_installed_cost_maybe_with_learning(self, tech, year):
        if ENABLE_WRIGHTS_LAW and tech not in TECHS_NO_LEARNING:
            if year in self.cached_investment_costs[tech]:
                return self.cached_investment_costs[tech][year]

        cumulative_G = self.global_installed_capacities_kW[
            tech
        ] + self.get_stock_without_degradation(tech, year)

        if ENABLE_WRIGHTS_LAW and tech not in TECHS_NO_LEARNING:
            ic = self._calculate_wrights_law(tech, year, cumulative_G)
        else:
            ic = self.installed_costs[tech]

        self.cached_investment_costs[tech][year] = ic
        self.cached_cumulative_G[tech][year] = cumulative_G
        return ic

    def get_weight(self, tech, country_name, year):
        weight = get_renewable_weight(tech, country_name)
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
        cumulative_G = self.G_battery_short + stock_without_degradation
        # $/GJ
        if ENABLE_WRIGHTS_LAW:
            unit_ic = self.alpha_battery_short * (
                cumulative_G**-self.gamma_battery_short
            )
        else:
            unit_ic = self.alpha_short_per_GJ
        # End of calculating unit_ic
        # Need to convert $/GJ to $/kWh, then to $/kW, by multiplying by number of hours in 1 day
        # This is because the short term battery is fully discharged by each day.
        self.battery_unit_ic["short"][year] = per_GJ2per_kWh(unit_ic) * 24
        self.cached_cumulative_G["short"][year] = cumulative_G

        investment_cost_battery_short = D_battery_short * unit_ic
        self.stocks_GJ_battery_short[year][country_name] = D_battery_short
        return investment_cost_battery_short

    def calculate_ic_1country_battery_long(self, year, country_name, total_R):
        psi = 0.7
        fe = 0.5
        # kW
        S_battery_long = self.get_stock(country_name, "long", year)
        # kW
        G_long = max(
            0, self.GJ2kW(total_R) * self.sigma_battery_long / psi / fe - S_battery_long
        )

        # Calculating unit_ic
        stock_without_degradation = 0.0
        for stock_year, stock_amount in self.stocks_kW["long"].items():
            if stock_year >= year:
                break
            stock_without_degradation += sum(stock_amount.values())
        # kW
        cumulative_G = self.G_battery_long + stock_without_degradation
        # $/kW
        if ENABLE_WRIGHTS_LAW:
            unit_ic = self.alpha_battery_long * (cumulative_G**-self.gamma_battery_long)
        else:
            unit_ic = self.alpha_long_per_kW
        # End of calculating unit_ic
        self.battery_unit_ic["long"][year] = unit_ic
        self.cached_cumulative_G["long"][year] = cumulative_G

        investment_cost_battery_long = G_long * unit_ic
        self.stocks_kW["long"][year][country_name] = G_long
        return investment_cost_battery_long

    def calculate_ic_1country_battery_pe(self, year, country_name, total_R):
        # Based on calculate_total_R
        R_pe = 0.0
        for tech in TECHS_WITH_LEARNING:
            S = self.get_stock_battery_pe(country_name, tech, year)
            R = self.kW2GJ(S) * get_capacity_factor(tech, country_name)
            R_pe += R
        # End of based on calculate_total_R

        psi = 0.7
        coefficient = self.sigma_battery_long * (1 / psi - 1)
        # GJ
        D = max(0, total_R * coefficient - R_pe)
        # kW
        D_kW = self.GJ2kW(D)

        ic = 0.0
        for tech in TECHS_WITH_LEARNING:
            weight = self.get_weight(tech, country_name, year)
            capacity_factor = get_capacity_factor(tech, country_name)
            # kW
            G = weight * D_kW / capacity_factor if capacity_factor > 0 else 0
            installed_cost = self.calculate_installed_cost_maybe_with_learning(
                tech, year
            )

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
            for tech in TECHS + ["long"]:
                self.stocks_kW[tech][year][country_name] = 0.0
            self.stocks_GJ_battery_short[year][country_name] = 0.0
            for tech in TECHS_WITH_LEARNING:
                self.stocks_kW_battery_pe[tech][year][country_name] = 0.0
            return
        # in kW because installed_costs is in $/kW
        D_kW = self.GJ2kW(D)

        investment_cost = 0.0
        for tech in TECHS:
            # in kW
            weight = self.get_weight(tech, country_name, year)
            capacity_factor = get_capacity_factor(tech, country_name)
            G = weight * D_kW / capacity_factor if capacity_factor > 0 else 0
            installed_cost = self.calculate_installed_cost_maybe_with_learning(
                tech, year
            )
            investment_cost += G * installed_cost
            self.stocks_kW[tech][year][country_name] = G
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
        if VERBOSE_ANALYSIS:
            self.green_energy_produced_by_country[-1][country_name] = {
                tech: self._calculate_R(country_name, tech, year) for tech in TECHS
            }

    def initialize_verbose_analysis(self, DeltaP, year):
        # This method is not essential for understanding how with_learning.py works.
        if not VERBOSE_ANALYSIS:
            return
        for tech in TECHS + ["short", "long"]:
            stock_battery_pe = 0
            if tech in TECHS_WITH_LEARNING:
                match VERBOSE_ANALYSIS_COUNTRY:
                    case "WORLD":
                        stock_battery_pe = sum(
                            self.stocks_kW_battery_pe[tech][year].values()
                        )
                    case "EMDE":
                        stock_battery_pe = sum(
                            self.stocks_kW_battery_pe[tech][year].get(c, 0)
                            for c in EMDE
                        )
                    case "Developing_UNFCCC":
                        stock_battery_pe = sum(
                            self.stocks_kW_battery_pe[tech][year].get(c, 0)
                            for c in DEVELOPING_UNFCCC
                        )
                    case "Developed_UNFCCC":
                        stock_battery_pe = sum(
                            self.stocks_kW_battery_pe[tech][year].get(c, 0)
                            for c in DEVELOPED_UNFCCC
                        )
                    case _:
                        stock_battery_pe = self.stocks_kW_battery_pe[tech][year].get(
                            VERBOSE_ANALYSIS_COUNTRY, 0
                        )
            if tech == "short":
                _stocks = self.stocks_GJ_battery_short
            else:
                _stocks = self.stocks_kW[tech]
            stock = _stocks.get(year, 0)
            if isinstance(stock, dict):
                match VERBOSE_ANALYSIS_COUNTRY:
                    case "WORLD":
                        stock = sum(stock.values())
                    case "EMDE":
                        stock = sum(stock.get(c, 0) for c in EMDE)
                    case "Developing_UNFCCC":
                        stock = sum(stock.get(c, 0) for c in DEVELOPING_UNFCCC)
                    case "Developed_UNFCCC":
                        stock = sum(stock.get(c, 0) for c in DEVELOPED_UNFCCC)
                    case _:
                        stock = stock.get(VERBOSE_ANALYSIS_COUNTRY, 0)
            else:
                stock = 0
            if tech == "short":
                stock = self.GJ2kW(stock)
            self.cached_stock_without_degradation[tech][year] = stock + stock_battery_pe

            # With degradation
            stock_battery_pe = 0
            if tech in TECHS_WITH_LEARNING:
                match VERBOSE_ANALYSIS_COUNTRY:
                    case "WORLD":
                        stock_battery_pe = sum(
                            self.get_stock_battery_pe(c, tech, year)
                            for c in DeltaP.keys()
                        )
                    case "EMDE":
                        stock_battery_pe = sum(
                            self.get_stock_battery_pe(c, tech, year)
                            for c in DeltaP.keys()
                            if c in EMDE
                        )
                    case "Developing_UNFCCC":
                        stock_battery_pe = sum(
                            self.get_stock_battery_pe(c, tech, year)
                            for c in DeltaP.keys()
                            if c in DEVELOPING_UNFCCC
                        )
                    case "Developed_UNFCCC":
                        stock_battery_pe = sum(
                            self.get_stock_battery_pe(c, tech, year)
                            for c in DeltaP.keys()
                            if c in DEVELOPED_UNFCCC
                        )
                    case _:
                        stock_battery_pe = self.get_stock_battery_pe(
                            VERBOSE_ANALYSIS_COUNTRY, tech, year
                        )

            match tech:
                case "short":
                    _fn = lambda c, y: self.GJ2kW(self.get_stock_battery_short(y, c))  # noqa
                case _:
                    _fn = lambda c, y: self.get_stock(c, tech, y)  # noqa
            cs = {
                "WORLD": None,
                "EMDE": EMDE,
                "Developing_UNFCCC": DEVELOPING_UNFCCC,
                "Developed_UNFCCC": DEVELOPED_UNFCCC,
            }.get(VERBOSE_ANALYSIS_COUNTRY, VERBOSE_ANALYSIS_COUNTRY)
            if cs == VERBOSE_ANALYSIS_COUNTRY:
                self.cached_stock[tech][year] = (
                    _fn(VERBOSE_ANALYSIS_COUNTRY, year) + stock_battery_pe
                )
            else:
                self.cached_stock[tech][year] = (
                    sum(_fn(c, year) for c in DeltaP.keys() if cs is None or c in cs)
                    + stock_battery_pe
                )

    def calculate_investment_cost(self, DeltaP, year, discount):
        self.cost_non_discounted_battery_short_by_country.append({})
        self.cost_non_discounted_battery_long_by_country.append({})
        self.cost_non_discounted_battery_pe_by_country.append({})
        self.cost_non_discounted_battery_grid_by_country.append({})
        self.green_energy_produced_by_country.append({})
        if isinstance(DeltaP, float):
            assert math.isclose(DeltaP, 0)
            self.cost_non_discounted.append(0.0)
            self.cost_discounted.append(0.0)
            return
        self.cost_non_discounted.append({})
        self.cost_discounted.append({})
        for country_name, dp in DeltaP.items():
            self.calculate_investment_cost_one_country(country_name, dp, year, discount)
        self.initialize_verbose_analysis(DeltaP, year)

    def get_stock(self, country_name, tech, year):
        # Check cache first
        cache_key = (country_name, tech, year)
        if cache_key in self.cached_get_stock:
            return self.cached_get_stock[cache_key]

        lifespan = RENEWABLE_LIFESPAN
        if tech == "long":
            lifespan = BATTERY_LONG_LIFESPAN
        out = 0.0
        if len(self.stocks_kW[tech]) == 0:
            self.cached_get_stock[cache_key] = out
            return out
        for stock_year, stock_amount in self.stocks_kW[tech].items():
            # This means the current year is also excluded
            if stock_year >= year:
                break
            age = year - stock_year
            s = stock_amount.get(country_name, 0.0)
            if ENABLE_RENEWABLE_GRADUAL_DEGRADATION:
                s *= (1 - self.degradation_rate[tech]) ** age

            if ENABLE_RENEWABLE_30Y_LIFESPAN:
                if age <= lifespan:
                    out += s
            else:
                # No lifespan checking is needed.
                out += s

        # Cache the result
        self.cached_get_stock[cache_key] = out
        return out

    def get_stock_battery_short(self, year, country_name):
        out = 0.0
        if len(self.stocks_GJ_battery_short) == 0:
            return out
        for stock_year, stock_amount in self.stocks_GJ_battery_short.items():
            # This means the current year is also excluded
            if stock_year >= year:
                break
            age = year - stock_year
            s = stock_amount.get(country_name, 0.0)
            if ENABLE_RENEWABLE_GRADUAL_DEGRADATION:
                s *= (1 - self.degradation_rate["short"]) ** age

            if ENABLE_RENEWABLE_30Y_LIFESPAN:
                if age <= BATTERY_SHORT_LIFESPAN:
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
            # This means the current year is also excluded
            if stock_year >= year:
                break
            age = year - stock_year
            s = stock_amount.get(country_name, 0.0)
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
        # Check cache first
        if (tech, year) in self.cached_stock_without_degradation_fast:
            return self.cached_stock_without_degradation_fast[(tech, year)]

        out = 0.0
        if len(self.stocks_kW[tech]) == 0:
            self.cached_stock_without_degradation_fast[(tech, year)] = out
            return out
        for stock_year, stock_amount in self.stocks_kW[tech].items():
            if stock_year >= year:
                break
            # Geothermal, hydropower doesn't use battery PE
            stock_battery_pe = 0
            if tech in TECHS_WITH_LEARNING:
                stock_battery_pe = sum(
                    self.stocks_kW_battery_pe[tech][stock_year].values()
                )
            out += sum(stock_amount.values()) + stock_battery_pe

        # Cache the result
        self.cached_stock_without_degradation_fast[(tech, year)] = out
        return out

    def calculate_residual(
        self, year_start, year_end, weighted_emissions_factor_by_country
    ):
        if not ENABLE_RESIDUAL_BENEFIT:
            return 0.0, 0.0
        residual_emissions = defaultdict(float)
        residual_production = defaultdict(float)
        
        for year in range(year_start, year_end + 1):
            for country_name, emissions_factor in weighted_emissions_factor_by_country.items():
                # in GJ
                total_R = self.calculate_total_R(country_name, year)
                tonnes_of_coal_equivalent = util.GJ2coal(total_R)
                residual_emissions[country_name] += tonnes_of_coal_equivalent * emissions_factor
                residual_production[country_name] += tonnes_of_coal_equivalent
        
        return pd.Series(residual_emissions), pd.Series(residual_production)
