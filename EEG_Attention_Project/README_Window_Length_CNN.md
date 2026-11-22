# Window Length Comparison — 1D-CNN (2–5 seconds)

**Place these two program files into the existing project, without replacing the original E1 scripts:**

```text
F:/CSIE/EEG_Attention_Project/
  scripts/run_window_length.py                 <- NEW
  experiments/window_length/__init__.py         <- NEW
  experiments/window_length/runner.py           <- NEW
  configs/window_length.yaml                   <- EXISTING, unchanged
  configs/baseline.yaml                        <- EXISTING, unchanged
  configs/channels.yaml                        <- EXISTING, unchanged
  data/splits/subject_folds.csv                <- EXISTING, unchanged
  data/labels/window_length/{2,3,4,5}sec/rt30_labels.csv
  epochs/{2,3,4,5}sec/*_unlabeled.npz
```

The launcher automatically finds the project root from `scripts/`. It does NOT
use the legacy root-level RT5-35 label CSV. It reuses the exact CNN1D class,
training loop, train-only channelwise standardization, and 5-fold subject
assignments from E1, but it **re-trains 2 seconds from scratch too**.

## Installation

1. Extract the provided ZIP into `F:/CSIE/EEG_Attention_Project/`.
2. Ensure PyTorch, NumPy, pandas, PyYAML, scikit-learn and matplotlib are
   installed in the same Python environment you used for the original E1.
3. In PowerShell (project root):

```powershell
cd F:\CSIE\EEG_Attention_Project
python scripts\run_window_length.py --check
python scripts\run_window_length.py --smoke-test
python scripts\run_window_length.py
```

The final command runs **2, 3, 4, 5 seconds × seven E1 groups × five folds =
140 distinct fits**, sequentially. `--check` and `--smoke-test` are optional.
No original E1 result, checkpoint, path setting, label file, or epoch is modified.

## How to control the run

```powershell
# Default: all 140 models, all windows and groups
python scripts\run_window_length.py

# Automatically skip folds that already completed under identical inputs+config
python scripts\run_window_length.py

# Rebuild summaries/graphs only (no CNN training)
python scripts\run_window_length.py --summarize-only

# Example of a specific fold and group
python scripts\run_window_length.py --window 4 --group Forehead2 --fold 2

# Force RE-TRAIN for only one selected fold (overwrites only this experiment)
python scripts\run_window_length.py --window 4 --group Forehead2 --fold 2 --force

# Force CPU even if a GPU is available
python scripts\run_window_length.py --device cpu
```

CAUTION: Avoid `--force` with the default 140-model selection unless you truly
want to discard all previously completed window-length results.

## Where results appear

```text
results/diagnostics/window_length/
  summary.csv                    # 28 rows (4 windows x 7 groups), final
  fold_summary.csv               # 140 held-out folds, final
  subject_summary.csv            # all available per-subject held-out results
  window_length_comparison.csv   # group x 2s/3s/4s/5s mean BA pivot
  comparison.png                 # cross-window grouped bar chart
  report.txt                     # interpretation notes
  2sec/
    summary.csv, fold_summary.csv, subject_summary.csv, comparison.png, ...
    Full30/fold_1/best_model.pt, metrics.json, predictions.csv, ...
    Forehead2/fold_1/... etc
  3sec/... same structure
  4sec/... same structure
  5sec/... same structure
  _smoke_test/                    # optional, strictly separate
  .scratch/                       # temporary disk-backed batches, cleaned automatically
```

Each model fold saves `best_model.pt`, `run_config.json`, `metrics.json`,
`predictions.csv`, `per_subject_metrics.csv`, `training_history.csv`, and
`scaler.json`, as in the original E1.

## Experimental protocol

- Inputs: `X_B0_uV`, 250 Hz, 30 original channels in canonical order, before the current deviation only.
- Window lengths: 2 s = 500; 3 s = 750; 4 s = 1,000; 5 s = 1,250 samples.
- Every window uses its **own** labels from `data/labels/window_length/{T}sec/rt30_labels.csv`, generated after previous-event exclusion and RT ranking per session (fastest 30% = label 0, slowest 30% = label 1).
- Same seven `channels.yaml` E1 groups: Full30, Forehead2, Frontal2, Central2, Temporal2, Parietal2, Occipital2.
- Same original `data/splits/subject_folds.csv`, 5-fold split, with mutually exclusive train/validation/test subjects.
- Channelwise standardization is fitted **only** on train-subject EEG within each fold.
- CNN architecture and optimizer, early stopping, threshold, and random seed are loaded unchanged from `configs/baseline.yaml`; only the input sample count and window length are overridden per duration.
- Each test fold contributes held-out-subject Balanced Accuracy and Macro F1. `mean_BA` is the arithmetic average of the 5 per-fold BAs; `pooled_BA` pools all held-out epoch predictions, and is generally different.
- Different windows discard a different set of trials and independently re-rank RT, so changes in test BA reflect both duration and sample/label composition. Treat this as the planned dataset-specific comparison, not a paired same-trial causal estimate of input duration alone.
- `--smoke-test` trains on a small subset for two epochs only and is *not* a research result.

## Execution safety and restart

A completed fold with matching model, labels, split, source-code hash and NPZ file
size/modified-time metadata is skipped. Partial folds from the same protocol
can be retried; incompatible existing results raise an error rather than being
silently overwritten. `--force` is restricted to the selected output folds.
A group is staged into a temporary disk-backed `.npy` array to avoid loading
an entire Full30 dataset into RAM. One window and one channel group at a time;
there may be a few GB of temporary disk activity. Temporary arrays are removed
when a group finishes. All 140 CNN fits must run on **your** PC; no actual EEG
training has been run in this development environment.
