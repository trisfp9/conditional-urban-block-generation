"""
04_build_training_pairs.py
==========================
Builds the image dataset the model trains on. For each of the 1,754 filtered
tiles it renders TWO 256x256 images:

  INPUT canvas  = fixed geography the model designs around: roads + water
  TARGET layout = the real layout: green, water, buildings colored by
                  land-use class, roads

and does the SPATIAL train/test split (contiguous 2km superblocks held out,
~15% of tiles) BEFORE any augmentation. Augmentation itself happens
on-the-fly in the training script (05), applied to train tiles only.

Inputs : singapore_tiles_filtered.csv (script 03), singapore_tiles.gpkg (script 01)
Outputs: dataset/{train,test}/{input,target}/tile_<id>.png
         dataset/manifest.csv   (tile_id, split, labels, scores)
         dataset/preview.png    (6 sample pairs for visual verification)
         dataset.zip            (copied to Drive)

Colab: mount Drive first; pull singapore_tiles_filtered.csv, singapore_tiles.gpkg
from Drive into the working dir; then run. Expect 30-60 min (OSM re-download
is the slow part; osmnx caches it).
"""

import os, shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from rasterio import features
from rasterio.transform import from_bounds
from PIL import Image

# ---------------------------------------------------------------- settings
CRS = "EPSG:3414"
IMG = 256                    # image size in pixels (500m tile -> ~2m/pixel)
ROAD_BUFFER_M = 4.0          # half-width to make road lines visible
SUPERBLOCK = 4               # split unit = 4x4 tiles (2km x 2km chunks)
TEST_FRACTION = 0.15
SEED = 42

# class palettes: index 0 is background/black
INPUT_CLASSES  = {"water": 1, "road": 2}
INPUT_PALETTE  = {0: (0,0,0), 1: (0,90,255), 2: (140,140,140)}
TARGET_CLASSES = {"green":1, "water":2, "residential":3, "commercial":4,
                  "industrial":5, "other_building":6, "road":7}
TARGET_PALETTE = {0:(0,0,0), 1:(0,170,0), 2:(0,90,255), 3:(255,190,0),
                  4:(230,30,30), 5:(150,60,200), 6:(240,240,240),
                  7:(140,140,140)}

# ---------------------------------------------------------------- load tiles
print("Loading tile list...")
meta = pd.read_csv("singapore_tiles_filtered.csv")
geoms = gpd.read_file("singapore_tiles.gpkg")[["tile_id", "geometry"]]
tiles = geoms.merge(meta, on="tile_id", how="inner")
tiles = gpd.GeoDataFrame(tiles, geometry="geometry", crs=CRS)
print(f"  {len(tiles)} filtered tiles with geometry")

# ---------------------------------------------------------------- OSM layers
print("Downloading OSM layers (cached if run before)...")
def polys(gdf):
    g = gdf[gdf.geometry.geom_type.isin(["Polygon","MultiPolygon"])]
    return g.to_crs(CRS)[["geometry"]] if len(g) else g

buildings = ox.features_from_place("Singapore", tags={"building": True})
buildings = buildings[buildings.geometry.geom_type.isin(["Polygon","MultiPolygon"])]
buildings = buildings.to_crs(CRS)[["geometry"]].reset_index(drop=True)

green = polys(ox.features_from_place("Singapore", tags={
    "leisure": ["park","garden","nature_reserve","pitch","golf_course"],
    "landuse": ["grass","forest","meadow","recreation_ground","cemetery",
                "farmland","orchard"],
    "natural": ["wood","scrub","grassland","wetland"]}))
water = polys(ox.features_from_place("Singapore", tags={
    "natural":["water"], "landuse":["reservoir","basin"],
    "waterway":["riverbank","canal","river"]}))
landuse = ox.features_from_place("Singapore", tags={
    "landuse":["residential","commercial","industrial","retail"]})
landuse = landuse[landuse.geometry.geom_type.isin(["Polygon","MultiPolygon"])]
landuse = landuse.to_crs(CRS)[["geometry","landuse"]].reset_index(drop=True)

G = ox.graph_from_place("Singapore", network_type="drive")
edges = ox.graph_to_gdfs(G, nodes=False).to_crs(CRS)[["geometry"]]
roads = edges.copy()
roads["geometry"] = roads.geometry.buffer(ROAD_BUFFER_M)
print(f"  buildings {len(buildings):,} | green {len(green):,} | "
      f"water {len(water):,} | roads {len(roads):,}")

# ------------------------------------------------- classify buildings by use
print("Classifying buildings by land use (spatial join)...")
landuse["lu"] = landuse["landuse"].replace({"retail": "commercial"})
bj = gpd.sjoin(buildings, landuse[["geometry","lu"]],
               how="left", predicate="intersects")
bj = bj[~bj.index.duplicated(keep="first")]           # one class per building
buildings["cls"] = bj["lu"].fillna("other_building").map(
    {"residential":"residential", "commercial":"commercial",
     "industrial":"industrial", "other_building":"other_building"})
print(buildings["cls"].value_counts().to_string())

# ---------------------------------------------------------------- rendering
def render_tile(bounds, layers):
    """layers = list of (geoms, class_index), drawn in order (later on top).
    Returns a uint8 class-index raster of shape (IMG, IMG)."""
    transform = from_bounds(*bounds, IMG, IMG)
    out = np.zeros((IMG, IMG), dtype=np.uint8)
    for geoms, idx in layers:
        if len(geoms) == 0:
            continue
        features.rasterize(((g, idx) for g in geoms), out=out,
                           transform=transform, default_value=idx)
    return out

def to_png(arr, palette, path):
    rgb = np.zeros((IMG, IMG, 3), dtype=np.uint8)
    for idx, col in palette.items():
        rgb[arr == idx] = col
    Image.fromarray(rgb).save(path)

def subset(layer, tile_geom):
    idx = layer.sindex.query(tile_geom, predicate="intersects")
    return layer.geometry.iloc[idx]

# --------------------------------------------------------- spatial split
# superblock id from tile grid position; whole 2km blocks go to test.
minx, miny = tiles.total_bounds[0], tiles.total_bounds[1]
cx = ((tiles.geometry.centroid.x - minx) // 500).astype(int)
cy = ((tiles.geometry.centroid.y - miny) // 500).astype(int)
tiles["sblock"] = (cx // SUPERBLOCK).astype(str) + "_" + (cy // SUPERBLOCK).astype(str)

rng = np.random.default_rng(SEED)
blocks = tiles["sblock"].unique()
rng.shuffle(blocks)
test_blocks, count, target_n = set(), 0, int(len(tiles) * TEST_FRACTION)
for b in blocks:
    if count >= target_n:
        break
    test_blocks.add(b)
    count += (tiles["sblock"] == b).sum()
tiles["split"] = np.where(tiles["sblock"].isin(test_blocks), "test", "train")
print(f"\nsplit: {(tiles['split']=='train').sum()} train / "
      f"{(tiles['split']=='test').sum()} test "
      f"({(tiles['split']=='test').mean():.1%})")
print("test-set label coverage:")
print(pd.crosstab(tiles.loc[tiles.split=='test','walkability_label'],
                  tiles.loc[tiles.split=='test','flood_resilience_label']))

# ---------------------------------------------------------------- main loop
for sp in ["train", "test"]:
    for kind in ["input", "target"]:
        os.makedirs(f"dataset/{sp}/{kind}", exist_ok=True)

b_by_cls = {c: buildings[buildings["cls"] == c] for c in
            ["residential", "commercial", "industrial", "other_building"]}

print("\nRendering tiles...")
for i, row in tiles.iterrows():
    tg, bounds = row.geometry, row.geometry.bounds
    w  = subset(water, tg);  gr = subset(green, tg);  rd = subset(roads, tg)

    inp = render_tile(bounds, [(w, INPUT_CLASSES["water"]),
                               (rd, INPUT_CLASSES["road"])])
    layers = [(gr, TARGET_CLASSES["green"]), (w, TARGET_CLASSES["water"])]
    for c in ["residential", "commercial", "industrial", "other_building"]:
        layers.append((subset(b_by_cls[c], tg), TARGET_CLASSES[c]))
    layers.append((rd, TARGET_CLASSES["road"]))
    tgt = render_tile(bounds, layers)

    tid = int(row["tile_id"])
    to_png(inp, INPUT_PALETTE,  f"dataset/{row['split']}/input/tile_{tid}.png")
    to_png(tgt, TARGET_PALETTE, f"dataset/{row['split']}/target/tile_{tid}.png")
    if i % 200 == 0:
        print(f"  {i}/{len(tiles)}")

# ---------------------------------------------------------------- manifest
lab = {"low": 0, "med": 1, "high": 2}
man = tiles[["tile_id","split","walkability","flood_resilience","density",
             "walkability_label","flood_resilience_label"]].copy()
man["walk_lab"]  = man["walkability_label"].map(lab)
man["flood_lab"] = man["flood_resilience_label"].map(lab)
man.drop(columns="geometry", errors="ignore").to_csv("dataset/manifest.csv", index=False)

# ---------------------------------------------------------------- preview
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sample = tiles[tiles.split=="train"].sample(6, random_state=1)
fig, axes = plt.subplots(2, 6, figsize=(20, 7))
for j, (_, r) in enumerate(sample.iterrows()):
    tid = int(r["tile_id"])
    axes[0,j].imshow(Image.open(f"dataset/train/input/tile_{tid}.png"))
    axes[0,j].set_title(f"input {tid}", fontsize=9)
    axes[1,j].imshow(Image.open(f"dataset/train/target/tile_{tid}.png"))
    axes[1,j].set_title(f"target w={r['walkability_label']} "
                        f"f={r['flood_resilience_label']}", fontsize=9)
for ax in axes.flat: ax.set_axis_off()
plt.tight_layout(); plt.savefig("dataset/preview.png", dpi=110)
print("Saved dataset/, manifest.csv, preview.png")

# ---------------------------------------------------------------- zip+Drive
shutil.make_archive("dataset", "zip", "dataset")
DRIVE = "/content/drive/MyDrive"
if os.path.isdir(DRIVE):
    shutil.copy("dataset.zip", os.path.join(DRIVE, "dataset.zip"))
    print("dataset.zip copied to Drive.")
else:
    print("WARNING: Drive not mounted — copy dataset.zip manually.")
