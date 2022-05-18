# This file is for sensitivity analysis purpose, where we calculate the
# discounted storage cost of renewable energy. For the data source for
# storage_lower_list and storage_upper_list.
# https://www.mckinsey.com/business-functions/sustainability/our-insights/net-zero-power-long-duration-energy-storage-for-a-renewable-grid

import util
import processed_revenue

start_year = 2022
RHO_MODE = "default"
# The value of rho is 0.02795381840850683
rho = util.calculate_rho(processed_revenue.beta, rho_mode=RHO_MODE)

years = [2025, 2030, 2035, 2040]
# In billion dollars
storage_lower_list = [50, 200, 1100, 1500]
storage_upper_list = [50, 500, 1800, 3000]

storage_lower_cummulative_discounted = 0.0
storage_upper_cummulative_discounted = 0.0
for i in range(len(years)):
    year = years[i]
    lower = storage_lower_list[i]
    upper = storage_upper_list[i]

    discount = util.calculate_discount(rho, year - start_year)
    storage_lower_cummulative_discounted += discount * lower
    storage_upper_cummulative_discounted += discount * upper

# This prints
# "Discounted storage cost 1888.31 to 3531.28 billion dollars"
print(
    "Discounted storage cost",
    f"{storage_lower_cummulative_discounted:.2f}",
    "to",
    f"{storage_upper_cummulative_discounted:.2f}",
    "billion dollars",
)
