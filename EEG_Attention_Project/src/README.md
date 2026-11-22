# src/ — shared, reusable code (E1/E2/E3 runners NOT included yet)

- `config.py`: read YAML, project-relative paths.
- `labeling.py`: turn session-specific fastest/slowest 30% RT into a label CSV.
- `splits.py`: fixed 5-fold subject-exclusive test and inner subject validation.
- `data_loader.py`: match labeled events to `.npz` rows using `x_row`; select channels.
- `preprocessing.py`: TRAIN-fold-only channel-wise z-score. **Does not rerun bandpass.**
- `model.py`: 1D-CNN learned temporal features + binary prediction head.
- `trainer.py`: AdamW, BCEWithLogitsLoss, early stopping, checkpoint.
- `evaluation.py`: balanced accuracy, macro-F1, confusion counts, per-subject metrics.

## Requirements (Python 3.10+)

    python -m pip install numpy pandas scipy scikit-learn PyYAML torch

`mne` is NOT needed here because the original EEG files have already been
converted to B0 NPZ epochs. E3 will require new dependencies later.

## Regenerate metadata from the real local file (no EEG data read)

From the project ROOT (i.e. directory containing `configs`, `src`, `data`):

    python -m src.labeling --project-root .
    python -m src.splits --project-root .

## Future E1 experiment runner responsibilities

1. Read `data/labels/rt30_labels.csv` and `data/splits/subject_folds.csv`.
2. For each fold and channel group, select train/val/test by **subject**.
3. Use `data_loader.iter_session_arrays` to load specified channel data.
4. Fit `ChannelwiseStandardizer` on TRAIN data only, apply to all roles.
5. Build `CNN1D`, call `trainer.train_binary_classifier` with TRAIN and VAL.
6. Evaluate once on TEST using `evaluation.py`, save per-subject records.

The .npz source is already filtered (causal 1-30 Hz Butterworth) and sampled
at 250 Hz. No additional EEG filtering is done by these modules.

**Known caution:** RT30 is a relative behavioral label, and the current config
flags but does NOT remove extreme RTs (<0.1 s or >10 s). This is deliberate
until a defensible, agreed quality policy is established. In particular, the
current label table is NOT proof of physiological focus or of online capability.
