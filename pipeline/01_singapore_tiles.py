"""
01_singapore_tiles.py
=====================
Downloads Singapore data from OpenStreetMap, cuts it into 500m tiles,
and computes per-tile metrics for:
  - density          (building footprint coverage)
  - walkability      (intersection density, amenity density, land-use mix)
  - flood resilience (permeable/green surface, water coverage, building setback from water)

Output: singapore_tiles.csv  (one row per tile, all metrics)
        singapore_tiles.gpkg (same, but with geometry so you can view it in QGIS)

HOW TO RUN (Google Colab):
    !pip install osmnx geopandas shapely --quiet
    then run this script top to bottom. Expect 15-45 min total,
    most of it in the OSM download step.

NOTE: downloads are cached by osmnx, so re-running is much faster.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box

# ----------------------------------------------------------------------
# 0. SETTINGS
# ----------------------------------------------------------------------
CRS_METRIC = "EPSG:3414"   # SVY21 / Singapore TM projection. Units = meters.
                           # We need a metric CRS so "500" means 500 meters.
TILE_SIZE = 500            # tile edge length in meters
MIN_LAND_FRACTION = 0.5    # keep a tile only if >=50% of it is on land
SETBACK_CAP = 1000.0       # cap building->water distance at 1km (tiles far from
                           # any water would otherwise dominate the metric)

ox.settings.log_console = True   # show download progress

# ----------------------------------------------------------------------
# 1. GET THE SINGAPORE BOUNDARY
# ----------------------------------------------------------------------
print("Step 1: fetching Singapore boundary...")
boundary = ox.geocode_to_gdf("Singapore")
boundary = boundary.to_crs(CRS_METRIC)
land = boundary.geometry.iloc[0]   # a single polygon of Singapore's land area

# ----------------------------------------------------------------------
# 2. DOWNLOAD THE OSM LAYERS
#    Each call fetches every matching feature in Singapore.
#    This is the slow part (buildings especially).
# ----------------------------------------------------------------------
print("Step 2: downloading OSM layers (this is the slow part)...")

# -- buildings: every building footprint
buildings = ox.features_from_place("Singapore", tags={"building": True})
buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
buildings = buildings.to_crs(CRS_METRIC)[["geometry"]]
print(f"  buildings: {len(buildings):,}")

# -- green space: parks, gardens, grass, forest, etc. (permeable surfaces)
green_tags = {
    "leisure": ["park", "garden", "nature_reserve", "pitch", "golf_course"],
    "landuse": ["grass", "forest", "meadow", "recreation_ground", "cemetery",
                "farmland", "orchard"],
    "natural": ["wood", "scrub", "grassland", "wetland"],
}
green = ox.features_from_place("Singapore", tags=green_tags)
green = green[green.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
green = green.to_crs(CRS_METRIC)[["geometry"]]
print(f"  green polygons: {len(green):,}")

# -- water: rivers, reservoirs, canals, coastal water inside the boundary
water_tags = {"natural": ["water"], "landuse": ["reservoir", "basin"],
              "waterway": ["riverbank", "canal", "river"]}
water = ox.features_from_place("Singapore", tags=water_tags)
water = water[water.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
water = water.to_crs(CRS_METRIC)[["geometry"]]
print(f"  water polygons: {len(water):,}")

# -- land use polygons for the mix/entropy metric
lu_tags = {"landuse": ["residential", "commercial", "industrial", "retail"]}
landuse = ox.features_from_place("Singapore", tags=lu_tags)
landuse = landuse[landuse.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
landuse = landuse.to_crs(CRS_METRIC)[["geometry", "landuse"]]
print(f"  landuse polygons: {len(landuse):,}")

# -- amenities: daily-life destinations for the walkability metric (points)
amenity_tags = {"amenity": ["school", "kindergarten", "clinic", "hospital",
                            "restaurant", "cafe", "food_court", "marketplace",
                            "library", "community_centre", "place_of_worship",
                            "pharmacy", "bank"],
                "shop": True}
amenities = ox.features_from_place("Singapore", tags=amenity_tags)
amenities = amenities.to_crs(CRS_METRIC)
amenities["geometry"] = amenities.geometry.centroid  # reduce everything to points
amenities = amenities[["geometry"]]
print(f"  amenities: {len(amenities):,}")

# -- road network: for intersection density
print("  downloading road network...")
G = ox.graph_from_place("Singapore", network_type="drive")
nodes, _ = ox.graph_to_gdfs(G)
nodes = nodes.to_crs(CRS_METRIC)
# an "intersection" = a node where 3+ street segments meet
intersections = nodes[nodes["street_count"] >= 3][["geometry"]]
print(f"  intersections: {len(intersections):,}")

# ----------------------------------------------------------------------
# 3. BUILD THE 500m TILE GRID
# ----------------------------------------------------------------------
print("Step 3: building tile grid...")
minx, miny, maxx, maxy = boundary.total_bounds
xs = np.arange(minx, maxx, TILE_SIZE)
ys = np.arange(miny, maxy, TILE_SIZE)

tiles = []
for x in xs:
    for y in ys:
        tiles.append(box(x, y, x + TILE_SIZE, y + TILE_SIZE))

tiles = gpd.GeoDataFrame({"geometry": tiles}, crs=CRS_METRIC)
print(f"  raw grid tiles: {len(tiles):,}")

# keep only tiles that are mostly on land
tile_area = TILE_SIZE * TILE_SIZE
land_overlap = tiles.geometry.intersection(land).area / tile_area
tiles = tiles[land_overlap >= MIN_LAND_FRACTION].reset_index(drop=True)
tiles["tile_id"] = tiles.index
print(f"  tiles on land (>= {MIN_LAND_FRACTION:.0%}): {len(tiles):,}")

# ----------------------------------------------------------------------
# 4. PER-TILE METRICS
#    Pattern used throughout: overlay a layer with the tiles, then
#    group the clipped pieces by tile_id and sum their areas.
# ----------------------------------------------------------------------
print("Step 4: computing per-tile metrics (takes a few minutes)...")

def area_per_tile(layer, tiles, colname):
    """Clip `layer` to tiles and return total clipped area per tile."""
    if len(layer) == 0:
        return pd.Series(0.0, index=tiles["tile_id"], name=colname)
    clipped = gpd.overlay(layer, tiles[["tile_id", "geometry"]],
                          how="intersection", keep_geom_type=True)
    out = clipped.assign(a=clipped.geometry.area).groupby("tile_id")["a"].sum()
    return out.reindex(tiles["tile_id"]).fillna(0.0).rename(colname)

def count_per_tile(points, tiles, colname):
    """Count point features falling inside each tile."""
    if len(points) == 0:
        return pd.Series(0, index=tiles["tile_id"], name=colname)
    joined = gpd.sjoin(points, tiles[["tile_id", "geometry"]], predicate="within")
    out = joined.groupby("tile_id").size()
    return out.reindex(tiles["tile_id"]).fillna(0).rename(colname)

df = tiles[["tile_id"]].copy()
df["building_area"]  = area_per_tile(buildings, tiles, "building_area").values
df["green_area"]     = area_per_tile(green, tiles, "green_area").values
df["water_area"]     = area_per_tile(water, tiles, "water_area").values
df["n_amenities"]    = count_per_tile(amenities, tiles, "n_amenities").values
df["n_intersections"]= count_per_tile(intersections, tiles, "n_intersections").values

# land-use entropy: how mixed are residential/commercial/industrial/retail?
lu_areas = {}
for cat in ["residential", "commercial", "industrial", "retail"]:
    layer = landuse[landuse["landuse"] == cat][["geometry"]]
    lu_areas[cat] = area_per_tile(layer, tiles, cat).values
lu = pd.DataFrame(lu_areas)
lu_total = lu.sum(axis=1)
p = lu.div(lu_total.replace(0, np.nan), axis=0)      # proportions per category
entropy = -(p * np.log(p)).sum(axis=1, skipna=True)  # Shannon entropy
entropy = entropy / np.log(lu.shape[1])              # normalize to 0..1
df["landuse_entropy"] = entropy.fillna(0.0).values

# building setback from water: mean distance from building centroids to the
# nearest water feature, capped at SETBACK_CAP. Larger = more set back.
print("  computing setbacks (slowest metric)...")
water_union = water.union_all()   # one big geometry to measure distance against
b_cent = buildings.copy()
b_cent["geometry"] = b_cent.geometry.centroid
b_cent["setback"] = b_cent.geometry.distance(water_union).clip(upper=SETBACK_CAP)
joined = gpd.sjoin(b_cent, tiles[["tile_id", "geometry"]], predicate="within")
mean_setback = joined.groupby("tile_id")["setback"].mean()
df["mean_setback_m"] = mean_setback.reindex(df["tile_id"]).fillna(SETBACK_CAP).values

# ----------------------------------------------------------------------
# 5. RAW METRICS -> THE THREE STUDY VARIABLES
# ----------------------------------------------------------------------
df["density"] = df["building_area"] / tile_area          # 0..1 coverage

df["intersection_density"] = df["n_intersections"] / (tile_area / 1e6)  # per km2
df["amenity_density"]      = df["n_amenities"]     / (tile_area / 1e6)  # per km2

df["permeable_ratio"] = (df["green_area"] + df["water_area"]) / tile_area
df["permeable_ratio"] = df["permeable_ratio"].clip(upper=1.0)

def zscore(s):
    """Standardize a column (mean 0, std 1) so components can be averaged."""
    return (s - s.mean()) / s.std(ddof=0)

# composite walkability: connectivity + destinations + mix (equal weights)
df["walkability"] = (
    zscore(df["intersection_density"])
    + zscore(df["amenity_density"])
    + zscore(df["landuse_entropy"])
) / 3.0

# composite flood resilience: permeable surface + setback (equal weights)
df["flood_resilience"] = (
    zscore(df["permeable_ratio"])
    + zscore(df["mean_setback_m"])
) / 2.0

# drop tiles with essentially nothing built on them (no signal to learn)
before = len(df)
keep = df["building_area"] > 0
df = df[keep].reset_index(drop=True)
tiles_out = tiles[keep.values].reset_index(drop=True)
print(f"Step 5: dropped {before - len(df)} empty tiles, kept {len(df):,}")

# ----------------------------------------------------------------------
# 6. SAVE
# ----------------------------------------------------------------------
df.to_csv("singapore_tiles.csv", index=False)
tiles_out.merge(df, on="tile_id").to_file("singapore_tiles.gpkg", driver="GPKG")
print("Saved singapore_tiles.csv and singapore_tiles.gpkg (session storage)")

# --- auto-save permanent copies to Google Drive ---------------------------
# Requires Drive to be mounted first (run this in a cell BEFORE this script):
#     from google.colab import drive
#     drive.mount('/content/drive')
import os, shutil
DRIVE_DIR = "/content/drive/MyDrive"
if os.path.isdir(DRIVE_DIR):
    shutil.copy("singapore_tiles.csv",  os.path.join(DRIVE_DIR, "singapore_tiles.csv"))
    shutil.copy("singapore_tiles.gpkg", os.path.join(DRIVE_DIR, "singapore_tiles.gpkg"))
    print("Copied both files to Google Drive (MyDrive). They are now permanent.")
else:
    print("WARNING: Google Drive is not mounted, files were NOT copied to Drive.")
    print("Mount it and copy manually:")
    print("    from google.colab import drive; drive.mount('/content/drive')")
    print("    !cp singapore_tiles.csv singapore_tiles.gpkg /content/drive/MyDrive/")
# ---------------------------------------------------------------------------

print(df[["density", "walkability", "flood_resilience"]].describe())
