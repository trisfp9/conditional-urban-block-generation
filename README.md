# Conditional Generation of Urban Block Layouts Under Competing Walkability and Flood Resilience Proxies in Singapore

Code for the paper of the same title (Tristan Firmansyah Park, Sinarmas World Academy). A conditional Pix2Pix GAN is trained on 500 m OpenStreetMap tiles of Singapore. It generates land use layouts from a fixed canvas of roads and water, under a walkability label and a flood resilient design label, and building footprint coverage is measured as an outcome the model never controls.

## Pipeline

All scripts were run in Google Colab with Google Drive mounted. Paths assume the Drive layout below.

| Step | Script | Output |
|---|---|---|
| 1 | `pipeline/01_singapore_tiles.py` | `singapore_tiles.csv`, `singapore_tiles.gpkg`: 500 m grid, per-tile metrics, walkability and flood composites (2,551 developed tiles) |
| 3 | `pipeline/03_filter_and_check.py` | `singapore_tiles_filtered.csv`: excludes the 797 floor tiles, assigns tercile labels (1,754 tiles) |
| 4 | `pipeline/04_build_training_pairs.py` | `dataset.zip`: 256 x 256 input and target PNGs, spatial superblock split (seed 42), `manifest.csv` |
| 5 | `pipeline/05_train_pix2pix.py` | `conditional_final.pt`. Set `BASELINE = False` for the conditional model |
| 5b | `pipeline/05b_train_baseline_v2.py` | `baseline_final.pt`: unconditional model, discriminator learning rate / 4 |
| 6a | `pipeline/06a_controllability_test.py` | single-canvas 3 x 3 sweep (Figure 6) |
| 6 | `pipeline/06_evaluate.py` | `evaluation/sweep_metrics.csv`, tradeoff maps, FID (pytorch-fid) |
| 6b | `pipeline/06b_fix_baseline_eval.py` | corrected FID for the unconditional model (native zero-label mode), shuffled-label ablation, per-tile L1 |
| 6c | `pipeline/06c_baseline_native_sweep.py` | `evaluation/baseline_native_metrics.csv` |
| fig | `pipeline/make_figure3_losses.py` | Figure 3 |

There is no step 2. The numbering follows the original project notebook, `notebooks/AI_UrbanPlanning.ipynb`, which has its outputs stripped.

## Revision analyses (peer review)

`revision/AJSR_revision_analyses.ipynb` reproduces every statistic added during peer review: the preprocessing audit and exclusion sensitivity (R1), the mixed models with spatial block-bootstrap intervals and Holm correction (R2), the categorical metrics IoU, Dice, confusion matrix, boundary F and topology (R3), the partition map (R4), and the environment report (R5). It runs on CPU and needs no retraining. The standalone scripts are in `revision/`, and `revision/Figure_fix.ipynb` regenerates Figures 4 and 7 from the evaluation outputs. Outputs are in `results/`.

## Drive layout expected by the scripts

```
MyDrive/
  AI-UrbanPlanning/singapore_tiles.csv, singapore_tiles.gpkg   (step 1)
  singapore_tiles_filtered.csv                                 (step 3)
  dataset.zip                                                  (step 4)
  pix2pix_runs/conditional_final.pt, baseline_final.pt, *_losses.csv
  pix2pix_runs/evaluation/sweep_metrics.csv, baseline_native_metrics.csv, fid_real/, fid_cond/, fid_cond_shuffled/, fid_base_zero/
```

## Data and models

The tile data are derived from OpenStreetMap (downloaded 13 July 2026), © OpenStreetMap contributors, available under the Open Database License (ODbL). They are not included here, but step 1 regenerates them. OpenStreetMap is edited continuously, so a fresh download will differ slightly from the one used in the paper. Model checkpoints (about 200 MB each) are not included because of their size and are available from the author on request.

## Environment

Training: Google Colab, NVIDIA T4 GPU, PyTorch (Colab default runtime, July and August 2026), seed 42. Revision analyses: see `results/` and `revision/r5_environment_report.py`. Install with `pip install -r requirements.txt`.

## License

MIT, see `LICENSE`.
