"""
R4. Map of the spatial train/test partition and the excluded tiles (reviewer comments 3 and 4).

Input: the same tiles.csv as R1, plus tile centroid coordinates (SVY21 metres) and superblock ids.
Output: partition_map.png (300 dpi) and partition_summary.json.

Run
  python r4_partition_map.py tiles.csv --out results_r4
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

COLS = {"x": "x", "y": "y", "excluded": "excluded", "split": "split", "superblock": "superblock_id"}
TILE = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles_csv")
    ap.add_argument("--out", default="results_r4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = pd.read_csv(a.tiles_csv)
    exc = t[COLS["excluded"]].astype(str).str.lower().isin(["1", "true", "yes"])
    split = t[COLS["split"]].astype(str).str.lower()
    cat = pd.Series("train", index=t.index)
    cat[split == "test"] = "test"
    cat[exc] = "excluded"
    colors = {"train": "#9aa3ad", "test": "#1f6fb2", "excluded": "#d1495b"}
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)
    for c, col in colors.items():
        m = cat == c
        ax.scatter(t.loc[m, COLS["x"]], t.loc[m, COLS["y"]], s=4.2, marker="s", c=col, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting, SVY21 (m)")
    ax.set_ylabel("Northing, SVY21 (m)")
    ax.legend(handles=[Patch(color=colors["train"], label=f"Training ({(cat=='train').sum()})"),
                       Patch(color=colors["test"], label=f"Held-out test ({(cat=='test').sum()})"),
                       Patch(color=colors["excluded"], label=f"Excluded ({(cat=='excluded').sum()})")],
              loc="lower right", frameon=False, fontsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "partition_map.png"))
    sb = t.loc[cat == "test", COLS["superblock"]]
    summary = {"counts": cat.value_counts().to_dict(), "n_test_superblocks": int(sb.nunique()),
               "test_superblock_ids": sorted(map(str, sb.unique())),
               "tiles_per_test_superblock": sb.value_counts().describe().to_dict()}
    with open(os.path.join(a.out, "partition_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
