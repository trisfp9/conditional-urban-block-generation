"""
R1. Preprocessing audit and exclusion sensitivity (reviewer comments 3 and 4).

CPU only. No retraining.

What it does
  1. Recomputes the walkability (W) and flood resilient design (F) composites three ways:
       a. "as published": using the standardization the paper reports (edit PUBLISHED below)
       b. "training only": mean and SD estimated on the 1,485 training tiles only
       c. "all developed": mean and SD estimated on all 2,551 developed tiles (797 floor tiles retained)
     and prints the resulting equations (coefficients and intercepts), so Equation 1 and 2 can be
     checked and, if needed, corrected.
  2. Recomputes tercile thresholds on the training set only and counts how many training and
     test tiles would change label (with Cohen's kappa). If few labels change, the leakage from
     the test set into preprocessing is shown to be negligible; if many change, retraining is needed.
  3. Exclusion sensitivity: terciles and the 3x3 label table when the 797 floor tiles are retained,
     plus how many of the 1,754 retained tiles would change label.
  4. Correlations (W vs F, density vs W, density vs F) under each population.

Input
  A single CSV with one row per developed tile (all 2,551, including the 797 excluded ones).
  Edit COLS below to match your column names.

Run
  python r1_preprocessing_audit.py tiles.csv --out results_r1
Send back
  the whole results_r1 folder.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- CONFIG
COLS = {
    "tile_id": "tile_id",
    "intersections": "intersection_density",   # nodes with 3+ segments per km2
    "amenities": "amenity_density",            # amenities per km2
    "entropy": "landuse_entropy",              # normalized 0..1
    "permeable": "permeable_ratio",            # (green + water) / tile area
    "setback": "setback_m",                    # mean building-to-water distance, capped at 1000
    "coverage": "building_coverage",           # building footprint / tile area
    "excluded": "excluded",                    # True for the 797 floor tiles
    "split": "split",                          # "train" / "test" (blank for excluded tiles)
    "walk_label": "walk_label",                # labels actually used in training (low/med/high)
    "walk_score": "walk_score",                # stored walkability composite (optional)
    "flood_score": "flood_score",              # stored flood composite (optional)
    "flood_label": "flood_label",
}
# Standardization the paper reports. Fill in from your code if these differ.
PUBLISHED = {
    "intersections": {"sd": 29.97, "mean": None},
    "amenities": {"sd": 61.09, "mean": None},
    "entropy": {"sd": 0.112, "mean": None},
    "permeable": {"sd": 0.327, "mean": None},
    "setback": {"sd": 323.1, "mean": None},
}
PUBLISHED_EQ = {
    "W": {"intersections": 0.0111231, "amenities": 0.00545628, "entropy": 2.98158, "intercept": -0.30415},
    "F": {"permeable": 1.53122, "setback": 0.00154735, "intercept": -1.52288},
}
LABELS = ["low", "med", "high"]
W_PARTS = ["intersections", "amenities", "entropy"]
F_PARTS = ["permeable", "setback"]
# ------------------------------------------------------------------------


def to_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y", "t"])


def fit_standardization(df, parts):
    return {p: {"mean": float(df[p].mean()), "sd": float(df[p].std(ddof=1))} for p in parts}


def composite(df, parts, stdz):
    z = [(df[p] - stdz[p]["mean"]) / stdz[p]["sd"] for p in parts]
    return sum(z) / len(parts)


def equation(parts, stdz):
    k = len(parts)
    coef = {p: 1.0 / (k * stdz[p]["sd"]) for p in parts}
    intercept = -sum(stdz[p]["mean"] / stdz[p]["sd"] for p in parts) / k
    return {**{p: round(c, 8) for p, c in coef.items()}, "intercept": round(intercept, 5)}


def terciles(x):
    return [float(np.quantile(x, 1 / 3)), float(np.quantile(x, 2 / 3))]


def label(x, cuts):
    return pd.cut(x, [-np.inf, cuts[0], cuts[1], np.inf], labels=LABELS).astype(str)


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in LABELS)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def crosstab(w, f):
    t = pd.crosstab(pd.Categorical(w, LABELS), pd.Categorical(f, LABELS))
    t.index, t.columns = [f"walk_{c}" for c in LABELS], [f"flood_{c}" for c in LABELS]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles_csv")
    ap.add_argument("--out", default="results_r1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = pd.read_csv(args.tiles_csv)
    df = pd.DataFrame({k: raw[v] for k, v in COLS.items() if v in raw.columns})
    missing = [v for k, v in COLS.items() if v not in raw.columns]
    if missing:
        print("WARNING: columns not found (edit COLS):", missing)
    global F_PARTS
    F_PARTS = [p for p in F_PARTS if p in df.columns]
    if len(F_PARTS) < 2:
        print("NOTE: flood components not both present; flood standardization checks use the stored score only")
    df["excluded"] = to_bool(df["excluded"])
    kept = df[~df["excluded"]].copy()
    train = kept[kept["split"].astype(str).str.lower() == "train"]
    test = kept[kept["split"].astype(str).str.lower() == "test"]
    report = {"n_all_developed": len(df), "n_kept": len(kept), "n_train": len(train), "n_test": len(test)}

    # 0. Recover the composite actually used, by exact linear regression of the stored score
    rec = {}
    for score, parts in [("walk_score", W_PARTS), ("flood_score", F_PARTS)]:
        if score in raw.columns and parts:
            X = np.column_stack([df[p].values for p in parts] + [np.ones(len(df))])
            beta, res, *_ = np.linalg.lstsq(X, raw[score].values, rcond=None)
            fit = X @ beta
            rec[score] = {**{p: float(b) for p, b in zip(parts, beta[:-1])}, "intercept": float(beta[-1]),
                          "max_abs_residual": float(np.max(np.abs(fit - raw[score].values))),
                          "implied_sd": {p: float(1 / (len(parts) * b)) if b else None for p, b in zip(parts, beta[:-1])}}
    report["recovered_composites"] = rec

    # 1. Equations under each standardization population
    pops = {"training_only": train, "kept_1754": kept, "all_developed_2551": df}
    eqs = {}
    for name, pop in pops.items():
        sw, sf = fit_standardization(pop, W_PARTS), fit_standardization(pop, F_PARTS)
        eqs[name] = {"W_stats": sw, "F_stats": sf, "W_eq": equation(W_PARTS, sw),
                     "F_eq": equation(F_PARTS, sf) if F_PARTS else None}
    # Check which population reproduces the published equation
    pub_w = sum(PUBLISHED_EQ["W"][p] * kept[p] for p in W_PARTS) + PUBLISHED_EQ["W"]["intercept"]
    pub_f = (sum(PUBLISHED_EQ["F"][p] * kept[p] for p in F_PARTS) + PUBLISHED_EQ["F"]["intercept"]) if len(F_PARTS) == 2 else pd.Series([np.nan])
    eqs["published_equation_check"] = {
        "W_published_mean_over_kept": float(pub_w.mean()),
        "F_published_mean_over_kept": float(pub_f.mean()),
        "note": "If W_eq/F_eq of one population above matches PUBLISHED_EQ, that population was used.",
    }
    report["equations"] = eqs

    # 2. Training-only terciles vs labels used
    sw_tr, sf_tr = eqs["training_only"]["W_stats"], eqs["training_only"]["F_stats"]
    kept["W_tr"] = composite(kept, W_PARTS, sw_tr)
    kept["F_tr"] = composite(kept, F_PARTS, sf_tr) if len(F_PARTS) == 2 else raw.loc[kept.index, "flood_score"]
    tr_mask = kept["split"].astype(str).str.lower() == "train"
    cw, cf = terciles(kept.loc[tr_mask, "W_tr"]), terciles(kept.loc[tr_mask, "F_tr"])
    kept["walk_tr"], kept["flood_tr"] = label(kept["W_tr"], cw), label(kept["F_tr"], cf)
    have_labels = "walk_label" in kept and "flood_label" in kept
    leak = {"train_only_cuts_W": cw, "train_only_cuts_F": cf}
    if have_labels:
        for part, m in [("train", tr_mask), ("test", ~tr_mask)]:
            sub = kept[m]
            w_change = int((sub["walk_tr"] != sub["walk_label"].astype(str)).sum())
            f_change = int((sub["flood_tr"] != sub["flood_label"].astype(str)).sum())
            either = int(((sub["walk_tr"] != sub["walk_label"].astype(str)) |
                          (sub["flood_tr"] != sub["flood_label"].astype(str))).sum())
            leak[part] = {
                "n": int(len(sub)),
                "walk_label_changes": w_change, "flood_label_changes": f_change,
                "either_label_changes": either, "pct_either": round(100 * either / max(len(sub), 1), 2),
                "kappa_walk": round(kappa(sub["walk_tr"], sub["walk_label"].astype(str)), 4),
                "kappa_flood": round(kappa(sub["flood_tr"], sub["flood_label"].astype(str)), 4),
            }
        crosstab(kept.loc[~tr_mask, "walk_tr"], kept.loc[~tr_mask, "flood_tr"]).to_csv(
            os.path.join(args.out, "test_label_table_train_only_thresholds.csv"))
    report["leakage_check"] = leak

    # 3. Exclusion sensitivity: retain the 797 floor tiles
    sw_all, sf_all = eqs["all_developed_2551"]["W_stats"], eqs["all_developed_2551"]["F_stats"]
    df["W_all"] = composite(df, W_PARTS, sw_all)
    df["F_all"] = composite(df, F_PARTS, sf_all) if len(F_PARTS) == 2 else raw["flood_score"]
    cw_all, cf_all = terciles(df["W_all"]), terciles(df["F_all"])
    df["walk_all"], df["flood_all"] = label(df["W_all"], cw_all), label(df["F_all"], cf_all)
    tab_all = crosstab(df["walk_all"], df["flood_all"])
    tab_all.to_csv(os.path.join(args.out, "label_table_retaining_797.csv"))
    floor_W = df.loc[df["excluded"], "W_all"]
    sens = {
        "cuts_W_retaining_797": cw_all, "cuts_F_retaining_797": cf_all,
        "share_of_all_tiles_tied_at_W_minimum": round(float((df["W_all"] <= df["W_all"].min() + 1e-9).mean()), 4),
        "floor_tiles_W_unique_values": int(floor_W.round(9).nunique()),
        "floor_tiles_labeled": df.loc[df["excluded"], "walk_all"].value_counts().to_dict(),
        "high_high_cell_retaining_797": int(tab_all.loc["walk_high", "flood_high"]),
        "smallest_cell_retaining_797": int(tab_all.values.min()),
    }
    if have_labels:
        merged = df.loc[~df["excluded"]]  # same row order as `kept`
        same_w = (merged["walk_all"].values == kept["walk_label"].astype(str).values)
        same_f = (merged["flood_all"].values == kept["flood_label"].astype(str).values)
        sens["kept_tiles_changing_walk_label"] = int((~same_w).sum())
        sens["kept_tiles_changing_flood_label"] = int((~same_f).sum())
    report["exclusion_sensitivity"] = sens

    # 4. Correlations under each population
    corr = {}
    for name, d, wcol, fcol in [("kept_train_only_std", kept, "W_tr", "F_tr"),
                                ("all_2551", df, "W_all", "F_all")]:
        corr[name] = {
            "pearson_W_F": round(float(stats.pearsonr(d[wcol], d[fcol])[0]), 4),
            "spearman_W_F": round(float(stats.spearmanr(d[wcol], d[fcol])[0]), 4),
            "pearson_cov_W": round(float(stats.pearsonr(d["coverage"], d[wcol])[0]), 4),
            "pearson_cov_F": round(float(stats.pearsonr(d["coverage"], d[fcol])[0]), 4),
        }
    report["correlations"] = corr
    if "setback" in kept:
        report["setback_cap_share_kept"] = round(float((kept["setback"] >= 1000 - 1e-6).mean()), 4)

    with open(os.path.join(args.out, "r1_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
