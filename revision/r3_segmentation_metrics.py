"""
R3. Domain-relevant evaluation of generated layouts (reviewer comment 6).

CPU only. Works on PNG files the model has already produced (or that you regenerate on CPU).

What it does
  1. Decodes every image to class indices by nearest palette colour (Table 3 palette), and
     reports how far decoded pixels were from their nearest palette colour (decoding quality).
  2. Per-class intersection-over-union and Dice, mean IoU over the generated classes, pixel
     accuracy, and a confusion matrix (real class by generated class), for each variant:
       true labels, shuffled labels, unconditional model.
  3. A categorical per-tile disagreement rate (share of pixels whose class differs from the real
     tile). This replaces the RGB-based mean absolute error the reviewer questioned.
  4. Building boundary F-score with a 2 pixel tolerance (about 4 m).
  5. Building topology: number of connected building components per tile, generated vs real.
  6. Paired comparison of true vs shuffled labels per tile (Wilcoxon), so the label effect is
     shown on categorical metrics, not only on RGB error.

Roads and water are copied from the input canvas, so metrics are reported both over all
classes and over the generated classes only (background, green, four building classes).

Inputs (folders of PNGs named <canvas_id>.png, all 256x256)
  --real      real target layouts of the 269 held-out tiles
  --true      generations under each tile's true labels
  --shuffled  generations under shuffled labels (optional)
  --uncond    unconditional model generations (optional)

Run
  python r3_segmentation_metrics.py --real real/ --true gen_true/ --shuffled gen_shuf/ --uncond gen_unc/ --out results_r3
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage, stats

PALETTE = {  # Table 3
    "background": (0, 0, 0),
    "road": (140, 140, 140),
    "water": (0, 90, 255),
    "green": (0, 170, 0),
    "residential": (255, 190, 0),
    "commercial": (230, 30, 30),
    "industrial": (150, 60, 200),
    "unclassified": (240, 240, 240),
}
NAMES = list(PALETTE)
PAL = np.array([PALETTE[n] for n in NAMES], dtype=np.float32)
GENERATED = ["background", "green", "residential", "commercial", "industrial", "unclassified"]
BUILDING = ["residential", "commercial", "industrial", "unclassified"]
BUILDING_IDX = [NAMES.index(n) for n in BUILDING]


def decode(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    dist = ((rgb[..., None, :] - PAL[None, None]) ** 2).sum(-1)
    cls = dist.argmin(-1)
    return cls, np.sqrt(dist.min(-1))


def confusion(real, gen, k):
    return np.bincount(real.ravel() * k + gen.ravel(), minlength=k * k).reshape(k, k)


def iou_dice(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
    return iou, dice


def boundary(mask):
    return mask & ~ndimage.binary_erosion(mask)


def boundary_f(real_b, gen_b, tol=2):
    rb, gb = boundary(real_b), boundary(gen_b)
    if rb.sum() == 0 and gb.sum() == 0:
        return np.nan
    dr = ndimage.distance_transform_edt(~rb) if rb.any() else np.full(rb.shape, np.inf)
    dg = ndimage.distance_transform_edt(~gb) if gb.any() else np.full(gb.shape, np.inf)
    prec = (dr[gb] <= tol).mean() if gb.any() else 0.0
    rec = (dg[rb] <= tol).mean() if rb.any() else 0.0
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def evaluate(real_dir, gen_dir):
    k = len(NAMES)
    cm = np.zeros((k, k), np.int64)
    per_tile, dec_err = {}, []
    for rp in sorted(glob.glob(os.path.join(real_dir, "*.png"))):
        cid = os.path.splitext(os.path.basename(rp))[0]
        gp = os.path.join(gen_dir, cid + ".png")
        if not os.path.exists(gp):
            continue
        r, _ = decode(rp)
        g, err = decode(gp)
        dec_err.append(err.ravel())
        c = confusion(r, g, k)
        cm += c
        rb, gb = np.isin(r, BUILDING_IDX), np.isin(g, BUILDING_IDX)
        gen_mask = np.isin(r, [NAMES.index(n) for n in GENERATED])
        per_tile[cid] = {
            "disagreement_all": float((r != g).mean()),
            "disagreement_generated": float((r[gen_mask] != g[gen_mask]).mean()) if gen_mask.any() else np.nan,
            "building_iou": float((rb & gb).sum() / max((rb | gb).sum(), 1)),
            "boundary_f_2px": boundary_f(rb, gb),
            "components_real": int(ndimage.label(rb)[1]),
            "components_gen": int(ndimage.label(gb)[1]),
        }
    iou, dice = iou_dice(cm)
    gi = [NAMES.index(n) for n in GENERATED]
    dec = np.concatenate(dec_err) if dec_err else np.array([0.0])
    summary = {
        "n_tiles": len(per_tile),
        "pixel_accuracy_all": float(np.trace(cm) / cm.sum()),
        "pixel_accuracy_generated": float(cm[np.ix_(gi, gi)].trace() / cm[np.ix_(gi, gi)].sum()),
        "iou": {n: float(iou[i]) for i, n in enumerate(NAMES)},
        "dice": {n: float(dice[i]) for i, n in enumerate(NAMES)},
        "mean_iou_generated_classes": float(np.nanmean(iou[gi])),
        "mean_dice_generated_classes": float(np.nanmean(dice[gi])),
        "per_tile_mean": {m: float(np.nanmean([t[m] for t in per_tile.values()]))
                          for m in next(iter(per_tile.values()))} if per_tile else {},
        "decoding_distance_rgb": {"median": float(np.median(dec)), "p95": float(np.percentile(dec, 95)),
                                  "share_over_40": float((dec > 40).mean())},
        "confusion_matrix_rows_real_cols_generated": {"classes": NAMES, "counts": cm.tolist()},
    }
    comp = [(t["components_real"], t["components_gen"]) for t in per_tile.values()]
    if len(comp) > 2:
        cr, cg = np.array(comp).T
        summary["building_components"] = {"real_mean": float(cr.mean()), "gen_mean": float(cg.mean()),
                                          "spearman": float(stats.spearmanr(cr, cg)[0])}
    return summary, per_tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--true", required=True)
    ap.add_argument("--shuffled")
    ap.add_argument("--uncond")
    ap.add_argument("--out", default="results_r3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report, tiles = {}, {}
    for name, folder in [("true_labels", a.true), ("shuffled_labels", a.shuffled), ("unconditional", a.uncond)]:
        if folder:
            report[name], tiles[name] = evaluate(a.real, folder)
            print(name, "mIoU(generated)=%.4f" % report[name]["mean_iou_generated_classes"])
    for other in ["shuffled_labels", "unconditional"]:
        if other in tiles:
            ids = sorted(set(tiles["true_labels"]) & set(tiles[other]))
            comp = {}
            for m in ["disagreement_generated", "building_iou", "boundary_f_2px"]:
                x = np.array([tiles["true_labels"][i][m] for i in ids], float)
                y = np.array([tiles[other][i][m] for i in ids], float)
                ok = ~np.isnan(x) & ~np.isnan(y)
                diff = x[ok] - y[ok]
                comp[m] = {"true_mean": float(x[ok].mean()), f"{other}_mean": float(y[ok].mean()),
                           "mean_diff": float(diff.mean()),
                           "wilcoxon_p": float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0}
            report[f"paired_true_vs_{other}"] = comp
    with open(os.path.join(a.out, "r3_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(a.out, "r3_per_tile.json"), "w") as fh:
        json.dump(tiles, fh)
    print(json.dumps({k: v for k, v in report.items()}, indent=2)[:6000])


if __name__ == "__main__":
    main()
