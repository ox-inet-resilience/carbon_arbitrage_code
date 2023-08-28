import csv
import json

import util

# This is obtained from hacking analysis_country_specific.py, by printing out the value of
# unilateral_emissions += ub / scc
avoided_emissions_upto_2100 = {
    "CN": 324.31864198167125,
    "AU": 210.88392255442193,
    "US": 169.76007984197312,
    "IN": 151.62685162688658,
    "RU": 134.41053934436312,
    "ID": 91.08973829998874,
    "ZA": 88.31023529194344,
    "CA": 26.35656351707315,
    "PL": 23.138517157332764,
    "KZ": 22.11787208149155,
    "CO": 20.71410638221643,
    "DE": 17.716209938697503,
    "MZ": 17.69996535412617,
    "MN": 17.173148952818888,
    "UA": 11.191927795456502,
    "TR": 10.194717861180987,
    "VN": 10.152055694546538,
    "BW": 8.604314544061468,
    "GR": 7.573094423320425,
    "BR": 6.217290295422924,
    "CZ": 4.358417431326265,
    "BG": 4.1167546903089915,
    "RO": 3.8021364308579666,
    "TH": 3.5351996529100034,
    "RS": 3.502196239391694,
    "GB": 2.9816379175094783,
    "UZ": 2.9765779338010225,
    "PH": 2.9422508136490686,
    "ZW": 2.841921792053882,
    "NZ": 2.8324118700906484,
    "MX": 2.7321817378678874,
    "BD": 2.351176776282867,
    "BA": 1.8081467019809279,
    "LA": 1.7020376971081606,
    "IR": 1.607164480354883,
    "CL": 1.4227111251730238,
    "ES": 1.0544365328652607,
    "PK": 0.9966186815117757,
    "VE": 0.9130698572881613,
    "TZ": 0.8708146176678915,
    "HU": 0.8350768544942265,
    "ME": 0.6350296331043865,
    "SK": 0.4920182078251383,
    "ZM": 0.4834945171203027,
    "SI": 0.4809886062409861,
    "MG": 0.4205952819616051,
    "TJ": 0.4063008471678118,
    "MK": 0.30028583347475696,
    "GE": 0.299913882220746,
    "AR": 0.2783001174679414,
    "MM": 0.20621830932783047,
    "JP": 0.20024838735900657,
    "KG": 0.1855434374567874,
    "MW": 0.08492276037340747,
    "NG": 0.07525189031299664,
    "NE": 0.06315729833376413,
    "PE": 0.05601648436133825,
    "NO": 0.039792934993789636,
    "ET": 0.007854922444545226,
    "CD": 0.0008262877393210473,
}
avoided_emissions_upto_2050 = {
    "BD": 0.85747548565268,
    "CN": 105.4241229269197,
    "GE": 0.09785306888627565,
    "IN": 49.20602202151846,
    "ID": 29.820977952752823,
    "IR": 0.5238723208268574,
    "JP": 0.06516730455240091,
    "KZ": 7.214093777751692,
    "KG": 0.06039891112435658,
    "LA": 0.5546337193398853,
    "MN": 5.311008255658245,
    "MM": 0.06688859653406569,
    "PK": 0.30677145054975685,
    "PH": 0.9759550603813135,
    "TJ": 0.13246434791583392,
    "TH": 1.1545194173636397,
    "TR": 3.3169878043211973,
    "UZ": 0.9694163272082861,
    "VN": 3.3057722717733617,
    "BW": 2.6976003015217507,
    "CD": 0.00026935675552020293,
    "ET": 0.002556002376511532,
    "MG": 0.12149374424423218,
    "MW": 0.027661837768978157,
    "MZ": 4.906958937950898,
    "NE": 0.02059723231053174,
    "NG": 0.0241995906419701,
    "ZA": 28.79986468734664,
    "TZ": 0.25096898980370697,
    "ZM": 0.15681260807096328,
    "ZW": 0.9238563201454053,
    "CA": 7.761473878778931,
    "US": 55.1728729993887,
    "AR": 0.09064060941275257,
    "BR": 2.029366993893424,
    "CL": 0.44556319150503987,
    "CO": 6.618697088971391,
    "MX": 0.8922701771912159,
    "PE": 0.01785007711671892,
    "VE": 0.28186098545961735,
    "BA": 0.5950770212311324,
    "BG": 1.3458628518691949,
    "CZ": 1.4405011506941334,
    "DE": 5.798593983421332,
    "GR": 2.469977282164791,
    "HU": 0.2719567368281863,
    "ME": 0.20654191369933378,
    "MK": 0.10103670490068346,
    "NO": 0.013097370780043261,
    "PL": 7.347023801521803,
    "RO": 1.243936731902158,
    "RU": 43.66300446455017,
    "RS": 1.1483052005904955,
    "SK": 0.16042704630772975,
    "SI": 0.15683965874020064,
    "ES": 0.3431898912302651,
    "UA": 3.649299356454965,
    "GB": 0.9663000331243272,
    "AU": 66.63556231958266,
    "NZ": 1.0318414772739655,
}
avoided_emissions_upto_2030 = {
    "BD": 0.7911807638582696,
    "CN": 37.936892460921186,
    "GE": 0.03614455056989303,
    "IN": 17.36149323869804,
    "ID": 11.171327418854224,
    "IR": 0.19207461171898363,
    "JP": 0.02358191442792097,
    "KZ": 2.650213936422168,
    "KG": 0.021906402891332024,
    "LA": 0.20256073947450276,
    "MN": 2.5126780008162624,
    "MM": 0.023714578979592556,
    "PK": 0.1588102731331615,
    "PH": 0.4165437120357029,
    "TJ": 0.04868896323169053,
    "TH": 0.42946548907580334,
    "TR": 1.1995367867710702,
    "UZ": 0.35300040826247786,
    "VN": 1.207547544364803,
    "BW": 1.824842913134175,
    "CD": 9.879156679277508e-05,
    "ET": 0.0009242457999030312,
    "MG": 0.02337120014511478,
    "MW": 0.00999972455627085,
    "MZ": 3.4752272943781275,
    "NE": 0.007580185254038957,
    "NG": 0.005403609530779089,
    "ZA": 10.590657903153431,
    "TZ": 0.15281002284660808,
    "ZM": 0.05508954616927289,
    "ZW": 0.32877760058944944,
    "CA": 4.229404809661235,
    "US": 20.046814389618064,
    "AR": 0.03301181471400249,
    "BR": 0.751596702108363,
    "CL": 0.29990829186811135,
    "CO": 2.2106070582745776,
    "MX": 0.3319182762842518,
    "PE": 0.005902096065169914,
    "VE": 0.1304888558530353,
    "BA": 0.23453454433655754,
    "BG": 0.49435613147772,
    "CZ": 0.6268612659842455,
    "DE": 2.1940051692559437,
    "GR": 0.9095514249039124,
    "HU": 0.09898025791101984,
    "ME": 0.07440398663938393,
    "MK": 0.051015830520497164,
    "NO": 0.005262481894115598,
    "PL": 2.5993770763700326,
    "RO": 0.46951920888815246,
    "RU": 15.72562502900281,
    "RS": 0.44052546529477093,
    "SK": 0.059090024741950346,
    "SI": 0.05765253215099304,
    "ES": 0.12436345984482809,
    "UA": 1.3410075063924254,
    "GB": 0.3385886387878579,
    "AU": 26.108281433370074,
    "NZ": 1.2269081440891314,
}

_, nonpower_coal, _ = util.read_masterdata()
coal_export_content = util.read_json("coal_export/aggregated/combined_summed.json")
production_2019 = nonpower_coal.groupby("asset_country")._2019.sum()


def get_export_fraction(country):
    if country not in coal_export_content["E"]:
        fraction = 0
    else:
        # Exclude self export
        export = sum(
            v for k, v in coal_export_content["E"][country].items() if k != country
        )
        export /= 1e3  # Convert kg to tonnes of coal
        production = production_2019[country]
        fraction = export / production if production > 0 else 0
        if country == "PE" and fraction > 1:
            fraction = 1
    assert 0 <= fraction <= 1, (country, fraction)
    return fraction


def get_import_fraction(e, i):
    if e not in coal_export_content["E"]:
        fraction = 0
    else:
        _import = coal_export_content["E"][e].get(i, 0.0)
        _import /= 1e3  # Convert kg to tonnes of coal
        production = production_2019[e]
        fraction = _import / production if production > 0 else 0
    assert 0 <= fraction <= 1
    return fraction


emissions_modified_by_export = {2100: {}, 2050: {}, 2030: {}}

for last_year in (2100, 2050, 2030):
    ae = {
        2100: avoided_emissions_upto_2100,
        2050: avoided_emissions_upto_2050,
        2030: avoided_emissions_upto_2030,
    }[last_year]
    for country, emissions in ae.items():
        modified_e = emissions * (1 - get_export_fraction(country)) + sum(
            v * get_import_fraction(k, country)
            for k, v in avoided_emissions_upto_2100.items()
            if k != country  # avoid self
        )
        emissions_modified_by_export[last_year][country] = modified_e

# Sort from largest value to smallest
emissions_modified_by_export[2100] = dict(
    sorted(
        emissions_modified_by_export[2100].items(),
        key=lambda item: item[1],
        reverse=True,
    )
)

(
    _,
    iso3166_df_alpha2,
    _,
    _,
    _,
) = util.prepare_from_climate_financing_data()
alpha2_to_full_name = iso3166_df_alpha2["name"].to_dict()
print("sanity check should add up to ~1424", sum(emissions_modified_by_export[2100].values()))
with open("plots/avoided_emissions_modified_by_coal_export.csv", "w") as f:
    for k, v in emissions_modified_by_export[2100].items():
        v_2050 = emissions_modified_by_export[2050][k]
        v_2030 = emissions_modified_by_export[2030][k]
        name = alpha2_to_full_name[k]
        f.write(f"{name},{v},{v_2050},{v_2030}\n")
# print(json.dumps(emissions_full_name_version, indent=2))
with open("plots/avoided_emissions_nonadjusted.csv", "w") as f:
    for k, v in avoided_emissions_upto_2100.items():
        v_2050 = avoided_emissions_upto_2050[k]
        v_2030 = avoided_emissions_upto_2030[k]
        name = alpha2_to_full_name[k]
        f.write(f"{name},{v},{v_2050},{v_2030}\n")
