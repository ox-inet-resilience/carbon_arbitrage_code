import json
import subprocess
import time

import numpy as np

import util

# These are countries that are found in Masterdata, but for nonpower coal only.
# As such, the number of countries is fewer than if it were sampled from the
# whole Masterdata.
countries = "AR AU BA BD BG BR BW CA CD CL CN CO CZ DE ES ET GB GE GR HU ID IN IR JP KG KZ LA ME MG MK MM MN MW MX MZ NE NG NO NZ PE PH PK PL RO RS RU SI SK TH TJ TR TZ UA US UZ VE VN XK ZA ZM ZW".split()
countries.remove("XK")


def fn(last_year):
    print(last_year)
    tic = time.time()
    benefits_yearly = {}
    benefits = {}
    for country in countries:
        out = (
            subprocess.check_output(
                [
                    "python",
                    "analysis_main.py",
                    country,
                    str(last_year),
                ]
            )
            .decode("utf-8")
            .strip()
        )
        outsplit = out.split("\n")
        out = outsplit[-1]  # total
        if out == "NA":
            # The country is not listed in country-specific SCC
            continue
        out_yearly = outsplit[-2]
        out_yearly = out_yearly.replace("OUTPUT1 ", "")
        out_yearly = json.loads(out_yearly)
        if np.allclose(out_yearly, 0.0):
            # Skip zeros
            continue
        benefits_yearly[country] = out_yearly
        out = out.replace("OUTPUT2 ", "")
        benefits[country] = float(out)

    assert len(benefits_yearly) == 60
    assert len(benefits) == 60
    with open(
        f"cache/unilateral_benefit_yearly_avoided_emissions_GtCO2_{last_year}.json",
        "w",
    ) as f:
        json.dump(benefits_yearly, f)
    with open(
        f"cache/unilateral_benefit_total_trillion_{last_year}.json",
        "w",
    ) as f:
        json.dump(benefits, f)

    print("Elapsed", time.time() - tic)


#util.run_parallel(fn, [2030, 2035, 2050, 2070, 2100], ())
util.run_parallel(fn, [2100], ())
