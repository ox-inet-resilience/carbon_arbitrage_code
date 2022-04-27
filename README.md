This repository contains the code for the great carbon arbitrage paper. The
data required for the analysis is confidential, but you can run it with your
own data.

## main code
- analysis_main.py: contains the code for identifying the net social gain from phasing out coal, as well as the requisite climate financing to phase out coal
- analysis_data_section.py: contains analysis for the data section
- processed_revenue: computes the median coal free cash flow per unit of coal production
- util.py: common code used by other files

## data_preparation
- gdp/: contains data downloaded from https://data.worldbank.org/indicator/NY.GDP.PCAP.CD and https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
- gdp_2020.py
- LearningSolarWind.csv.gz
- prepare_coal_price.py
- prepare_masterdata.py
- prepare_world_wright_learning.py
- Solar-Wind-Capacity.csv

## increasing_renewables
In increasing_renewables.py output, we show that solar and wind are the
dominant technology.

## misc
- aggregate_beta.py: to calculate un_leveraged_beta in processed_revenue.py
- experience_curve.py
- hq_country_analysis.py
- NGFS_renewable_additional_capacity.py
- region_vs_world.py
- shiller.py: to calculate the 100 year CARP used in 100year rho_mode in util.calculate_rho
- unused_analysis.py

## To reproduce the plots
### Figure 2

## data
The public data can be found in the data/ folder in this Git repo.
But the private data is confidential.

### private
```
3-indices.csv
Arial.ttf
IEA-Scenarios.csv
masterdata_country_of_domicile.csv.gz
masterdata_ownership.csv.gz
masterdata_ownership_2021_version.csv.gz
masterdata_ownership_PROCESSED_capacity_factor.csv.gz
masterdata_ownership_PROCESSED_capacity_factor_domicile.csv.gz
revenue_data_coal_companies_confidential.csv.gz
revenue_data_companies_confidential_PROCESSED_coal.csv.gz
```

### public
```
2DII-country-name-shortcuts.csv
all_countries_gdp_marketcap_2020.json
all_countries_gdp_per_capita_2020.json
country_ISO-3166_with_region.csv
developing.txt
developing_shortnames.csv
emerging.txt
emerging_shortnames.csv
GDP-Developed-World.csv
irena.json
NGFS-Power-Sector-Scenarios.csv.gz
ngfs_energy_demand_interpolated.csv
NGFS_region.json
NGFS_renewable_additional_capacity_GCAM5.3_NGFS.json
NGFS_renewable_dynamic_weight_GCAM5.3_NGFS.json
ngfs_scenario_production_fossil.csv
ShillerData.csv
TRISK-Data.xlsx
TRISK_data_NGFS_scenarios.csv
world_bank_group_carbon_tax.json
```
