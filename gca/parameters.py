ENABLE_WORKER = 1
# Possible values: "default", "100year", "5%", "8%", "0%"
RHO_MODE = "default"
LAST_YEAR = 2050
# The year where the NGFS value is pegged/rescaled to be the same as Masterdata
# global production value.
NGFS_PEG_YEAR = 2024
# SECTOR_INCLUDED = "Coal"
SECTOR_INCLUDED = "Power"
assert SECTOR_INCLUDED in ["Power", "Coal"]
ENABLE_COAL_EXPORT = 0

global_cost_with_learning = None
MEASURE_GLOBAL_VARS = False
# Only change this to current policies if you want to see the result
# for halt to fossil fuel production scenario
MEASURE_GLOBAL_VARS_SCENARIO = "Net Zero 2050"
