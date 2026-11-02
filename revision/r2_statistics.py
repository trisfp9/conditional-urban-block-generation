"""
R2. Repeated-measures statistics, spatial block-bootstrap uncertainty, effect sizes, and
multiple-comparison control (reviewer comment 7, and supporting comments 2 and 8).

CPU only. Uses the existing sweep outputs; no retraining.

What it does
  1. Linear mixed model per outcome (green, mix, building):
        outcome ~ C(walk) * C(flood) + (1 | canvas)
     with likelihood-ratio tests for the walkability effect, the flood effect, and their
     interaction, plus partial eta-squared from a repeated-measures ANOVA (balanced design).
  2. Spatial block bootstrap: whole 2 km superblocks of held-out canvases are resampled with
     replacement (default 5,000 draws), so spatial autocorrelation between neighbouring canvases
     is respected. Gives 95% CIs for every cell mean (Table 7), every marginal mean (Table 6),
     adjacent differences, the green fold change, and the high-walk corner cost (absolute and %).
  3. Paired adjacent-step tests on canvas-level marginal means (Wilcoxon signed-rank), Holm
     correction across the whole family, and paired effect sizes (Cohen's dz).
  4. Operational criteria used in the revised text:
        monotonic        = both adjacent marginal steps have Holm-adjusted p < 0.05 and
                           block-bootstrap 95% CIs that exclude zero, in the same direction
        indistinguishable= adjacent step 95% CI includes zero AND |step| < EQUIV_MARGIN (TOST-style)
        largely independent = the cross factor explains < 1% of within-canvas variance
                           (partial eta-squared) while the own factor explains more than 10x that
  5. Optional: unconditional-vs-conditional paired comparison and the ambiguity analysis with
     block-bootstrap CIs, if baseline.csv is supplied.

Inputs
  sweep.csv     long format, one row per (canvas, walk label, flood label): 269 x 9 = 2,421 rows
  canvases.csv  canvas_id -> superblock_id (and optionally ambiguity score)
  baseline.csv  (optional) canvas_id, green, mix, building for the unconditional model
Edit COLS to match your column names.

Run
  python r2_statistics.py sweep.csv canvases.csv [--baseline baseline.csv] --out results_r2
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- CONFIG
COLS = {
    "canvas": "canvas_id",
    "walk": "walk_label",
    "flood": "flood_label",
    "green": "green",
    "mix": "mix",
    "building": "building",
    "superblock": "superblock_id",
    "ambiguity": "ambiguity",
}
LABELS = ["low", "med", "high"]
OUTCOMES = ["green", "mix", "building"]
N_BOOT = 5000
SEED = 20260713
EQUIV_MARGIN = {"green": 0.02, "mix": 0.02, "building": 0.01}
NEAR_EMPTY = 0.02
# ------------------------------------------------------------------------


def norm_label(s):
    m = {"low": "low", "l": "low", "0": "low", "med": "med", "medium": "med", "m": "med", "1": "med",
         "high": "high", "h": "high", "2": "high"}
    return s.astype(str).str.strip().str.lower().map(m)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def cell_table(d, y):
    return d.pivot_table(index="walk", columns="flood", values=y, aggfunc="mean").reindex(index=LABELS, columns=LABELS)


def summaries(d):
    """All quantities that get bootstrap CIs, from a long-format frame."""
    out = {}
    for y in OUTCOMES:
        cells = cell_table(d, y)
        for w in LABELS:
            for f in LABELS:
                out[f"{y}|cell|walk={w}|flood={f}"] = cells.loc[w, f]
        for fac in ["walk", "flood"]:
            marg = d.groupby(fac)[y].mean().reindex(LABELS)
            for lab in LABELS:
                out[f"{y}|marg|{fac}={lab}"] = marg[lab]
            out[f"{y}|step|{fac}:low->med"] = marg["med"] - marg["low"]
            out[f"{y}|step|{fac}:med->high"] = marg["high"] - marg["med"]
            out[f"{y}|step|{fac}:low->high"] = marg["high"] - marg["low"]
    g = d.groupby("flood")["green"].mean().reindex(LABELS)
    out["green|fold|flood high/low"] = g["high"] / g["low"]
    b = cell_table(d, "building")
    out["building|corner_cost_abs|(high,low)-(high,high)"] = b.loc["high", "low"] - b.loc["high", "high"]
    out["building|corner_cost_pct"] = 100 * (b.loc["high", "low"] - b.loc["high", "high"]) / b.loc["high", "low"]
    m = cell_table(d, "mix")
    out["mix|corner_minus_max"] = m.loc["high", "high"] - np.nanmax(m.values)
    return out


def block_bootstrap(d, blocks, rng, n_boot):
    groups = {b: g for b, g in d.groupby("superblock")}
    keys = list(groups)
    point = summaries(d)
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        parts = []
        for j, i in enumerate(pick):
            g = groups[keys[i]].copy()
            g["canvas"] = g["canvas"].astype(str) + f"_b{j}"
            parts.append(g)
        s = summaries(pd.concat(parts, ignore_index=True))
        for k, v in s.items():
            draws[k].append(v)
    rows = []
    for k, v in point.items():
        arr = np.asarray(draws[k], float)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        rows.append({"quantity": k, "estimate": v, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def mixed_models(d):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import AnovaRM
    res = {}
    for y in OUTCOMES:
        full = smf.mixedlm(f"{y} ~ C(walk, Treatment('low')) * C(flood, Treatment('low'))", d,
                           groups=d["canvas"]).fit(reml=False)
        add = smf.mixedlm(f"{y} ~ C(walk, Treatment('low')) + C(flood, Treatment('low'))", d,
                          groups=d["canvas"]).fit(reml=False)
        no_w = smf.mixedlm(f"{y} ~ C(flood, Treatment('low'))", d, groups=d["canvas"]).fit(reml=False)
        no_f = smf.mixedlm(f"{y} ~ C(walk, Treatment('low'))", d, groups=d["canvas"]).fit(reml=False)

        def lrt(big, small, df):
            stat = 2 * (big.llf - small.llf)
            return {"chi2": round(float(stat), 3), "df": df, "p": float(stats.chi2.sf(stat, df))}

        aov = AnovaRM(d, depvar=y, subject="canvas", within=["walk", "flood"]).fit().anova_table
        eta = {}
        # partial eta^2 = F*df1 / (F*df1 + df2)
        for term in aov.index:
            F, df1, df2 = aov.loc[term, "F Value"], aov.loc[term, "Num DF"], aov.loc[term, "Den DF"]
            eta[term] = round(float(F * df1 / (F * df1 + df2)), 5)
        res[y] = {
            "LRT_walk": lrt(add, no_w, 2),
            "LRT_flood": lrt(add, no_f, 2),
            "LRT_interaction": lrt(full, add, 4),
            "partial_eta_sq": eta,
            "rmANOVA": {t: {"F": round(float(aov.loc[t, "F Value"]), 3), "p": float(aov.loc[t, "Pr > F"])}
                        for t in aov.index},
            "canvas_random_intercept_var": float(full.cov_re.iloc[0, 0]),
            "residual_var": float(full.scale),
        }
    return res


def paired_steps(d):
    rows = []
    for y in OUTCOMES:
        for fac in ["walk", "flood"]:
            cm = d.groupby(["canvas", fac])[y].mean().unstack().reindex(columns=LABELS)
            for a, b in [("low", "med"), ("med", "high"), ("low", "high")]:
                diff = cm[b] - cm[a]
                p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
                rows.append({"outcome": y, "factor": fac, "step": f"{a}->{b}", "mean_diff": diff.mean(),
                             "cohens_dz": dz, "wilcoxon_p": p, "n_canvases": int(diff.notna().sum())})
    t = pd.DataFrame(rows)
    t["holm_p"] = holm(t["wilcoxon_p"])
    return t


def criteria(steps, boot, mm):
    crit = {}
    b = boot.set_index("quantity")
    for y in OUTCOMES:
        for fac in ["walk", "flood"]:
            s = steps[(steps.outcome == y) & (steps.factor == fac)].set_index("step")
            ci = [(b.loc[f"{y}|step|{fac}:{k}", "ci_low"], b.loc[f"{y}|step|{fac}:{k}", "ci_high"])
                  for k in ["low->med", "med->high"]]
            sig = [s.loc[k, "holm_p"] < 0.05 for k in ["low->med", "med->high"]]
            excl = [lo > 0 or hi < 0 for lo, hi in ci]
            signs = [np.sign(s.loc[k, "mean_diff"]) for k in ["low->med", "med->high"]]
            mono = all(sig) and all(excl) and signs[0] == signs[1]
            indist = {k: bool((lo <= 0 <= hi) and abs(s.loc[k, "mean_diff"]) < EQUIV_MARGIN[y])
                      for k, (lo, hi) in zip(["low->med", "med->high"], ci)}
            crit[f"{y} by {fac}"] = {"monotonic": bool(mono), "indistinguishable_steps": indist}
    for y, own, cross in [("green", "flood", "walk"), ("mix", "walk", "flood")]:
        e = mm[y]["partial_eta_sq"]
        crit[f"{y} independence"] = {"own_partial_eta_sq": e.get(own), "cross_partial_eta_sq": e.get(cross),
                                     "interaction_partial_eta_sq": e.get("walk:flood"),
                                     "largely_independent": bool(e.get(cross, 1) < 0.01 and
                                                                 e.get(own, 0) > 10 * e.get(cross, 1))}
    return crit


def baseline_analysis(d, base, canv, rng, n_boot):
    cond = d.groupby("canvas")[OUTCOMES].mean()
    base = base.set_index("canvas")[OUTCOMES]
    j = cond.join(base, lsuffix="_cond", rsuffix="_unc", how="inner").join(canv.set_index("canvas"))
    j["diff_building"] = j["building_unc"] - j["building_cond"]
    j["diff_green"] = j["green_unc"] - j["green_cond"]
    out = {
        "n": int(len(j)),
        "share_unc_builds_less": float((j["diff_building"] < 0).mean()),
        "mean_diff_building": float(j["diff_building"].mean()),
        "wilcoxon_building_p": float(stats.wilcoxon(j["diff_building"]).pvalue),
        "cohens_dz_building": float(j["diff_building"].mean() / j["diff_building"].std(ddof=1)),
        "mean_diff_green": float(j["diff_green"].mean()),
        "wilcoxon_green_p": float(stats.wilcoxon(j["diff_green"]).pvalue),
        "near_empty_unc": int((j["building_unc"] < NEAR_EMPTY).sum()),
        "near_empty_cond_mean": int((j["building_cond"] < NEAR_EMPTY).sum()),
    }
    out["holm_p"] = dict(zip(["building", "green"], holm([out["wilcoxon_building_p"], out["wilcoxon_green_p"]])))
    has_amb = "ambiguity" in j and j["ambiguity"].notna().any()
    if has_amb:
        rho, p = stats.spearmanr(j["ambiguity"], j["diff_building"])
        out["spearman_ambiguity_vs_deficit"] = {"rho": float(rho), "p": float(p)}
    # block bootstrap over superblocks
    groups = {b: g for b, g in j.groupby("superblock")}
    keys = list(groups)
    draws = {"mean_diff_building": [], "mean_diff_green": [], "share_less": [], "rho": []}
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        s = pd.concat([groups[keys[i]] for i in pick])
        draws["mean_diff_building"].append(s["diff_building"].mean())
        draws["mean_diff_green"].append(s["diff_green"].mean())
        draws["share_less"].append((s["diff_building"] < 0).mean())
        if has_amb:
            draws["rho"].append(stats.spearmanr(s["ambiguity"], s["diff_building"])[0])
    out["block_bootstrap_95ci"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                                   for k, v in draws.items() if len(v)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_csv")
    ap.add_argument("canvases_csv")
    ap.add_argument("--baseline")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="results_r2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    sw = pd.read_csv(args.sweep_csv)
    d = pd.DataFrame({k: sw[COLS[k]] for k in ["canvas", "walk", "flood"] + OUTCOMES})
    d["walk"], d["flood"] = norm_label(d["walk"]), norm_label(d["flood"])
    cv = pd.read_csv(args.canvases_csv)
    canv = pd.DataFrame({"canvas": cv[COLS["canvas"]], "superblock": cv[COLS["superblock"]]})
    if COLS["ambiguity"] in cv:
        canv["ambiguity"] = cv[COLS["ambiguity"]]
    d = d.merge(canv[["canvas", "superblock"]], on="canvas", how="left")
    if d["superblock"].isna().any():
        raise SystemExit("Some canvases have no superblock_id; check canvases.csv")
    counts = d.groupby("canvas").size()
    print(f"{d['canvas'].nunique()} canvases, {d['superblock'].nunique()} superblocks, "
          f"rows per canvas: {counts.min()}-{counts.max()}")

    mm = mixed_models(d)
    steps = paired_steps(d)
    boot = block_bootstrap(d, "superblock", rng, args.boot)
    crit = criteria(steps, boot, mm)

    steps.to_csv(os.path.join(args.out, "paired_steps_holm.csv"), index=False)
    boot.to_csv(os.path.join(args.out, "block_bootstrap_ci.csv"), index=False)
    report = {"n_canvases": int(d["canvas"].nunique()), "n_superblocks": int(d["superblock"].nunique()),
              "n_boot": args.boot, "seed": SEED, "mixed_models": mm, "criteria": crit}
    if args.baseline:
        bs = pd.read_csv(args.baseline)
        base = pd.DataFrame({"canvas": bs[COLS["canvas"]], **{y: bs[COLS[y]] for y in OUTCOMES}})
        report["unconditional_vs_conditional"] = baseline_analysis(d, base, canv, rng, args.boot)
    with open(os.path.join(args.out, "r2_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))
    print(boot.to_string())


if __name__ == "__main__":
    main()
