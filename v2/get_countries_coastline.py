# Partially generated by GPT-4o as of Dec 2024 right after o3 announcement.
import geopandas as gpd
import polars as pl
import pycountry
from shapely import to_wkb


# Create a mapping of country names to alpha-2 codes
def get_alpha2(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_2
    except LookupError:
        match country_name:
            case "Kosovo":
                return "XK"
            case "Vatican City":
                return "VA"
        return None  # Handle cases where the name isn't matched


# Load the country boundaries and water bodies data (replace file paths with actual data)
country_boundaries_gdf = gpd.read_file("./data/World_24NM_v4_20231025/eez_24nm_v4.shp")
water_bodies_gdf = gpd.read_file(
    "./data/simplified-water-polygons-split-3857/simplified_water_polygons.shp"
)

# Ensure both datasets use the same CRS
common_crs = "EPSG:3395"  # Example projected CRS in meters

country_boundaries_gdf = country_boundaries_gdf.to_crs(common_crs)
water_bodies_gdf = water_bodies_gdf.to_crs(common_crs)

country_boundaries_gdf["alpha2"] = country_boundaries_gdf["SOVEREIGN1"].apply(
    get_alpha2
)


# Validate and fix invalid geometries
def fix_invalid_geometries(geometry):
    if not geometry.is_valid:
        return geometry.buffer(0)  # Attempt to fix invalid geometries
    return geometry


country_boundaries_gdf["geometry"] = country_boundaries_gdf["geometry"].apply(
    fix_invalid_geometries
)
water_bodies_gdf["geometry"] = water_bodies_gdf["geometry"].apply(
    fix_invalid_geometries
)

# Convert geometries to WKB format for Polars compatibility
country_boundaries_gdf["geometry_wkb"] = country_boundaries_gdf["geometry"].apply(
    to_wkb
)
water_bodies_gdf["geometry_wkb"] = water_bodies_gdf["geometry"].apply(to_wkb)

# Convert GeoDataFrames to Polars DataFrames
country_boundaries_df = pl.from_pandas(
    country_boundaries_gdf[["geometry_wkb", "alpha2", "SOVEREIGN1"]]
)
water_bodies_df = pl.from_pandas(water_bodies_gdf[["geometry_wkb"]])

# Perform intersection computation
coastline_lengths = []

i = 0
for country in country_boundaries_df.iter_rows(named=True):
    i += 1
    country_geom = gpd.GeoSeries.from_wkb([country["geometry_wkb"]])[0]
    alpha2 = country["alpha2"]
    full_name = country["SOVEREIGN1"]
    print(i, alpha2, full_name)

    total_length_km = 0

    for water_body in water_bodies_df.iter_rows(named=True):
        water_geom = gpd.GeoSeries.from_wkb([water_body["geometry_wkb"]])[0]

        # Check and compute intersection
        if country_geom.intersects(water_geom):
            intersection_geom = country_geom.intersection(water_geom)
            if intersection_geom.is_empty:
                continue

            # Calculate length in kilometers (convert from meters)
            length_km = intersection_geom.length / 1000
            total_length_km += length_km

    coastline_lengths.append(
        {
            "full_name": full_name,
            "alpha2": alpha2,
            "coastline_length_km": total_length_km,
        }
    )

# Convert results to a Polars DataFrame
coastline_lengths_df = pl.DataFrame(coastline_lengths)

# Save the results to a CSV file
coastline_lengths_df.write_csv("plots/country_coastlines.csv")
