"""
03_filter_and_check.py
======================
Applies the exclusion decision officially and re-validates the design on the
filtered dataset. Produces the tile list that ALL downstream scripts use.

Exclusion criterion (mentor-approved): tiles with zero amenities AND zero
intersections AND zero land-use entropy — non-habitable land (industrial
estates, military training areas, water catchment), verified by map
inspection on 16 Jul.

Inputs : singapore_tiles.csv          (from script 01)
Outputs: singapore_tiles_filtered.csv (official training tile list, with labels)
         filtered_stats.txt           (numbers for the paper's Data section)

Run AFTER mounting Drive. Copies outputs to Drive automatically.
"""

import os, shutil
import numpy as np
import pandas as pd

df = pd.read_csv("singapore_tiles.csv")
n_full = len(df)

# ---- exclusion ----------------------------------------------------------
floor = (df["n_amenities"] == 0) & (df["n_intersections"] == 0) & \
        (df["landuse_entropy"] == 0)
f = df[~floor].copy().reset_index(drop=True)
print(f"full: {n_full} tiles | excluded (floor): {floor.sum()} | kept: {len(f)}")

# ---- re-standardize composites on the filtered population ---------------
for c in ["walkability", "flood_resilience"]:
    f[c] = (f[c] - f[c].mean()) / f[c].std(ddof=0)

# ---- correlations -------------------------------------------------------
cols = ["walkability", "flood_resilience", "density"]
pearson = f[cols].corr()
spearman = f[cols].corr(method="spearman")
r = pearson.loc["walkability", "flood_resilience"]

# ---- labels (tercile, tie-safe) -----------------------------------------
for c in ["walkability", "flood_resilience"]:
    f[c + "_label"] = pd.qcut(f[c].rank(method="first"), q=3,
                              labels=["low", "med", "high"])
ct = pd.crosstab(f["walkability_label"], f["flood_resilience_label"])

# ---- report -------------------------------------------------------------
lines = [
    f"Full dataset: {n_full} tiles",
    f"Excluded (zero amenities+intersections+entropy): {int(floor.sum())}",
    f"Filtered training dataset: {len(f)} tiles",
    "",
    "Pearson:",  pearson.round(3).to_string(),
    "Spearman:", spearman.round(3).to_string(),
    "",
    f"MAIN CHECK walkability vs flood_resilience: r = {r:.3f}",
    "",
    "Label crosstab:", ct.to_string(),
    f"smallest cell: {ct.values.min()}",
]
report = "\n".join(lines)
print("\n" + report)

f.to_csv("singapore_tiles_filtered.csv", index=False)
with open("filtered_stats.txt", "w") as fh:
    fh.write(report + "\n")
print("\nSaved singapore_tiles_filtered.csv and filtered_stats.txt")

DRIVE = "/content/drive/MyDrive"
if os.path.isdir(DRIVE):
    for p in ["singapore_tiles_filtered.csv", "filtered_stats.txt"]:
        shutil.copy(p, os.path.join(DRIVE, p))
    print("Copied to Drive.")
else:
    print("WARNING: Drive not mounted — copy the outputs manually.")
