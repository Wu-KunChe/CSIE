# B0 / E1 runner — updated report generation

This package replaces ONLY `main.py` and `experiments/E1_region.py` in the existing project. Do not overwrite `configs`, `data`, or `src`.

## What changed

- `python main.py --experiment baseline` trains **Full30 only** (B0 = common bandpass + resampling + CNN1D protocol applied to all 30 scalp channels).
- `python main.py --experiment E1` compares **Full30 and six 2-channel groups**. It reuses B0 Full30 folds when their run signatures match.
- `--summarize-only` regenerates all reports, per-group merged predictions, pooled out-of-fold metrics, confusion matrix and labeled comparison chart from EXISTING completed folds. No CNN retraining or EEG loading.
- E1 and B0 plots print **mean BA as a percent above every bar** and show error bars for fold standard deviation.
- Per-group `summary.csv`, `fold_summary.csv`, `combined_predictions.csv`, `combined_metrics.json`, `confusion_matrix.csv`, `report.txt` are generated without changing any saved model.

## Recommended for an existing 5-fold result set

Open PowerShell in `F:\CSIE\EEG_Attention_Project`:

```powershell
cd 'F:\CSIE\EEG_Attention_Project'
python main.py --experiment baseline --summarize-only
python main.py --experiment E1 --summarize-only
```

This automatically overwrites aggregate `summary.csv`, `fold_summary.csv`, `comparison.png` and creates group-level merged reports. It does NOT overwrite or delete `fold_1` ... `fold_5` checkpoints. The baseline root now reports ONLY Full30. To remove the old, unused B0 Forehead2 duplicate, first confirm `results\E1\Forehead2\fold_1` ... `fold_5` exist; then optionally:

```powershell
Remove-Item '.\results\baseline\Forehead2' -Recurse -Force
```

Do not delete `results\E1\Forehead2`. Baseline's stale `per_subject_comparison.csv` (if any) is automatically removed.

`--summarize-only` validates that every saved test trial matches the CURRENT label table, is assigned to the correct held-out subject, and has consistent metrics. It cannot independently prove all hyperparameters used by legacy training files. Use it only when `configs`, `data/labels`, `data/splits` and the training modules have not been modified since those folds were trained.

## If you deliberately want to retrain EVERY model

Before the commands below, back up the existing `results` folder. Move the two old experiment folders aside to avoid incompatible old run hashes:

```powershell
cd 'F:\CSIE\EEG_Attention_Project'
Rename-Item '.\results\baseline' 'baseline_old'
Rename-Item '.\results\E1' 'E1_old'
python main.py --experiment baseline --check
python main.py --experiment baseline
python main.py --experiment E1
```

This performs 5 Full30 trainings and 30 extra 2-channel trainings (E1 reuses Full30 when the protocol hash matches). Do NOT delete raw EEG, `nm000275_epochs`, labels, folds, `configs`, or `src`.

If only changing plotting/reporting, retraining wastes time: use `--summarize-only`.

## Group-level outputs

Example `results/E1/Central2/`:

```
fold_1/ ... fold_5/        # unchanged checkpoints and test predictions
summary.csv                # ONE row: mean/std BA and F1 + pooled BA/F1 + confusion counts
fold_summary.csv           # five individual fold metrics in one file
combined_predictions.csv   # all held-out predictions across all five folds
combined_metrics.json      # means vs pooled metrics clearly separated
confusion_matrix.csv       # pooled 2-by-2 confusion counts
report.txt                 # readable overview
```

Experiment-level outputs:

```
results/E1/summary.csv             # one row per group, now including pooled metrics
results/E1/fold_summary.csv        # one row per group and fold
results/E1/subject_summary.csv
results/E1/per_subject_comparison.csv
results/E1/comparison.png          # percentage value labels + fold SD
results/E1/report.txt
```

`mean_BA` = arithmetic mean of the five test-fold BAs; `std_BA` = sample SD of these five scores. `pooled_BA` = independently recomputed BA from all out-of-fold test samples (16k predictions). They need not be numerically identical. **The bar chart displays mean_BA, NOT pooled_BA.**

`baseline/` has the same report structure but ONLY for Full30. If a selected experiment has fewer than five finished folds, reports are labeled PARTIAL and never pretend it is a complete five-fold evaluation.

## Normal execution / testing

```powershell
python main.py --experiment baseline --check
python main.py --experiment baseline --smoke-test
python main.py --experiment baseline
python main.py --experiment E1
```

Smoke tests are isolated in `results/smoke_test`; B0 smoke now uses Full30. Running regular training on old legacy folders without `--summarize-only` may require moving old results aside because the run signature includes source code hashes.

## Requirements

Uses existing dependencies: Python 3.10+, numpy, pandas, scipy, scikit-learn, PyYAML, torch, matplotlib.
