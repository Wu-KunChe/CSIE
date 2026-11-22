# E2 — Frequency-Band Input Analysis (4 sec / original 1D-CNN)

## Goal and experimental conditions

E2 changes **only the input EEG frequency band**, keeping the signal as a 4-second
waveform (250 Hz, 1000 samples). It does NOT compare different model families.

- B0, 1–30 Hz: use the **existing original 4sec 1D-CNN E1 results** as a read-only baseline.
  It is not retrained, and the old results are not modified.
- Theta: 4–8 Hz; Alpha: 8–13 Hz; Beta: 13–30 Hz.
- Channel groups: `Full30` and `Forehead2` (`FP1`, `FP2`), separately trained.
- Five **unchanged subject-wise train/val/test folds** from `data/splits/subject_folds.csv`.
- Exact unchanged `data/labels/window_length/4sec/rt30_labels.csv`:
  13,604 labeled events, 27 subjects, 62 sessions; per-session RT fastest 30%
  vs slowest 30%, with overlap rejection **before** RT ranking.
- 1D-CNN architecture and hyperparameters read from the original `configs/baseline.yaml`.
  Uses the original `src/model.py`, `src/trainer.py`, `src/preprocessing.py`, etc.
  **No Random Search parameters.**
- Train-only channelwise Z-score, class weight calculation only on train,
  early stopping on val, balanced accuracy and Macro-F1 from held-out test.
- One band/group/fold is one independently initialized and trained CNN. 
  **3 frequency bands × 2 channel groups × 5 folds = 30 new trainings.**
  B0 already has its five-fold original results, no re-training.

## Install

Extract the ZIP into:

`F:\CSIE\EEG_Attention_Project\`

The ZIP adds only:

```
configs/e2_frequency.yaml
experiments/E2_frequency/__init__.py
experiments/E2_frequency/filtering.py
experiments/E2_frequency/runner.py
scripts/run_e2_frequency.py
README_E2_Frequency_4sec.md
```

It never overwrites existing E1/Random Search/window-length scripts or outputs.
Existing Python packages should already include `numpy`, `pandas`, `torch`,
`scikit-learn`, `scipy`, `pyyaml`, `matplotlib` (matplotlib needed for PNG only).

Paths in `configs/e2_frequency.yaml` are all relative to project root, and match
`資料結構 目前(4).txt`:

- `epochs/4sec/*_unlabeled.npz`: already-preprocessed `X_B0_uV`, `[N,30,1000]`.
- `data/labels/window_length/4sec/rt30_labels.csv`.
- `data/splits/subject_folds.csv`.
- B0 baseline: `results/diagnostics/window_length/4sec/{Full30,Forehead2}/`.

## Run

In PowerShell:

```powershell
cd F:\CSIE\EEG_Attention_Project
python scripts\run_e2_frequency.py --check
python scripts\run_e2_frequency.py --smoke-test
python scripts\run_e2_frequency.py
```

`--check`: verifies paths, labeling method, subject folds, first NPZ shape and
original E1 channels. Does not train. It only inspects the first NPZ; other NPZ
files are checked for matching trial IDs during staging.

`--smoke-test`: isolated tiny sample, **2 epochs**, Theta/Forehead2/Fold 1 only.
Writes to `results/diagnostics/E2_frequency_4sec/_smoke_test/`. Results do not
contribute to the formal experiment.

`python scripts\run_e2_frequency.py`: run all 30 training folds with no additional
command. Normal auto device uses CUDA if available. Explicit device:

```powershell
python scripts\run_e2_frequency.py --device cuda
```

Only selected band/group/fold, with automatic resume:

```powershell
python scripts\run_e2_frequency.py --band theta --group Full30 --fold 1
python scripts\run_e2_frequency.py --band alpha
python scripts\run_e2_frequency.py --summarize-only
```

Repeat the same command to **resume**. Fully finished folds are skipped.
An interrupted training fold starts again from epoch 1; all other completed
folds are retained. `--force` redoes only selected E2 folds; it cannot touch E1.
Do not edit YAML, source code, CSV labels, folds or NPZ during a resumed run,
because the resume fingerprint intentionally detects changed experiments.

## Filtering details (important limitation)

The input NPZ already contains **causally filtered 1–30Hz B0** samples. To form
Theta/Alpha/Beta without changing labels, window count, or CNN input shape,
we apply **one additional 4th-order Butterworth bandpass** (SciPy second-order
sections, `sosfiltfilt`) to each 4-second epoch. Reflection padding adds 250
samples *from within the epoch* to both edges; those 250 padding samples are
removed after filtering. No data from other epochs or subjects are used.

**This is an offline diagnostic, not a deployment-ready causal filter.**
Filtering a 4-second window separately can still create edge artifacts despite
padding, especially near event onset. `sosfiltfilt` is zero-phase and
non-causal within that window. For stronger physiological claims or real-time
implementation, regenerate the frequency-band versions from each original
continuous EEG recording **before epoching**, preserving the exact original
window timestamps, trial IDs, labels and folds; then compare with this first
screening. Do not interpret measured filtered-band BA as an isolated causal
contribution of a brain rhythm. Beta near 30Hz is also influenced by the B0
1–30Hz roll-off.

## Disk and runtime

Full30 13,604 × 30 × 1,000 × 4 bytes ≈ **1.63 GB** temporary `.npy`.
This is stored under `results/diagnostics/E2_frequency_4sec/.scratch` and
removed on normal completion. Ensure free disk space (at least several GB),
particularly for Full30; Python/NPZ loading also needs RAM for one session.

`filter_timing.json` reports filter computation and temporary staging/IO time
for each band/group when processed. Fold `metrics.json` records model training
and test inference wall-clock time and parameter count; `inference_ms_per_epoch`
includes data loading/normalization and is **not a pure neural-network latency**.
Timing comparisons are machine-dependent and are not equivalent to end-to-end
real-time signal acquisition or streaming filtering latency.

## Results

All outputs only under `results/diagnostics/E2_frequency_4sec/`:

```
E2_frequency_4sec/
  summary.csv
  fold_summary.csv
  subject_summary.csv
  e2_comparison.csv       # includes original B0 rows if E1 reference is present
  comparison.png
  report.txt
  theta/Full30/fold_1/...   # best_model.pt, metrics.json, predictions.csv,
  theta/Forehead2/fold_1/...# scaler.json, training_history.csv, etc.
  alpha/{Full30,Forehead2}/...
  beta/{Full30,Forehead2}/...
```

- `summary.csv`: five-fold mean BA (primary), fold std, Macro-F1, pooled OOF BA.
- `fold_summary.csv`: test scores, best epoch, train/test speed for each fold.
- `subject_summary.csv`: held-out performance of individual subjects.
- `{band}/{group}/combined_predictions.csv`: all held-out test predictions;
  use to analyze Fast/Slow confusion and consistency across models.
- `e2_comparison.csv`: B0 1–30Hz reference plus completed E2 results and BA
  difference from the *same-group* B0 result.
- The program verifies the original B0 E1 `combined_predictions.csv`
  identities, labels and held-out fold assignments before using it.
  If the existing E1 results are missing it still produces E2 reports and warns.
- `all_folds_complete` indicates whether a result is the full five-fold experiment.
  Do not compare incomplete/smoke results as if they were final.

## Interpretation

The goal is to determine the extent to which band-limited EEG contains useful
prediction information for the **same fixed original CNN**. It is not a new
architecture, separate cross-validated hyperparameter search, or E3 artifact
removal experiment. Correlation with RT labels does not prove a causal
neurophysiological mechanism. Because the chosen 4s window and channel groups
came from earlier experiments on this dataset, final external generalization
needs independent participant data for robust confirmation.
