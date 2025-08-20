import pathlib
import sys

import pandas as pd

parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import analysis_main  # noqa
import util  # noqa
import with_learning  # noqa

developed, developing = util.get_countries_unfccc()

countries = [c for c in developing if c != "CN" and c not in with_learning.petrol_states]
out = analysis_main.run_table2(included_countries=countries)
out_2035 = out[0].to_dict()
ic = out_2035["Investment costs (in trillion dollars)"]
oc = out_2035["Opportunity costs (in trillion dollars)"]
ae = out_2035["Avoided emissions (GtCO2e)"]
print("ic", ic, "oc", oc, "ae", ae)

G7 = "US JP CA GB DE IT NO".split()
EU = "AT BE BG CY CZ DK EE FI FR DE GR HU HR IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split()
financiers = list(set(G7 + EU + ["NO", "CH", "AU", "KR"]))
financiers = [c for c in financiers if c != "US"]
country_sccs = pd.Series(util.read_country_specific_scc_filtered())
scc_share_percent = (
    100
    * sum(country_sccs.get(c, 0) for c in financiers)
    / country_sccs.sum()
)
print("scc share", scc_share_percent, "scc", util.scc_bilal)
print("freeloader benefit", scc_share_percent / 100 * ae * util.scc_bilal / 1000, "trillion dollars")
