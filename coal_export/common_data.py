import util


(
    iso3166_df,
    _,
    _,
    _,
    _,
) = util.prepare_from_climate_financing_data()
a2_to_a3 = iso3166_df.set_index("alpha-2")["alpha-3"].to_dict()
a2_to_a3["NA"] = "NAM"

# Excluding XK
masterdata_alpha2 = "AR AU BA BD BG BR BW CA CD CL CN CO CZ DE ES ET GB GE GR HU ID IN IR JP KG KZ LA ME MG MK MM MN MW MX MZ NE NG NO NZ PE PH PK PL RO RS RU SI SK TH TJ TR TZ UA US UZ VE VN ZA ZM ZW".split()
non_masterdata_alpha2 = "AE FR DK AW AF UY BO RW FI ML KH CG LB SZ CH HK BZ EC SG MY CI GH NL GT BI IT PS JM EG HN UG BH GD FJ NA OM SE MA KE BJ SA MT AM TN PY PF AT MS LC CR BM LV BB TG AZ PT BE DO BS YE MD NI BF IE MU BY PA NP LY KR AG LU SC LS LT SV TT SN IL HR KW QA GY VC CY IS EE LK MV AO JO KM BN".split()
# Excluding XK
masterdata_alpha3 = "ARG AUS BIH BGD BGR BRA BWA CAN COD CHL CHN COL CZE DEU ESP ETH GBR GEO GRC HUN IDN IND IRN JPN KGZ KAZ LAO MNE MDG MKD MMR MNG MWI MEX MOZ NER NGA NOR NZL PER PHL PAK POL ROU SRB RUS SVN SVK THA TJK TUR TZA UKR USA UZB VEN VNM ZAF ZMB ZWE".split()
non_masterdata_alpha3 = [a2_to_a3[i] for i in non_masterdata_alpha2]
