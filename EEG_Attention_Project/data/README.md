# data/

This directory contains **small metadata only**, not EEG waves.

- `labels/rt30_labels.csv`: 27,192 event rows. For an included EEG epoch:
  - `label=0`: the fastest 30% RT within its own session.
  - `label=1`: the slowest 30% RT within its own session.
  - `label=blank`: middle 40%, or event without usable EEG/RT.
  - `x_row` is the ZERO-based row of `X_B0_uV` in `npz_file`.
  - `rt_quality_flag` is informational ONLY. Suspicious short/long RTs remain
    in the current 30% labels until a documented quality policy is approved.
- `labels/rt30_summary.csv`: counts and flags by session.
- `splits/subject_folds.csv`: rows `(fold, subject)` with `train`, `val`, `test`
  role. Every subject appears in TEST exactly once, no overlap within a fold.
- `splits/fold_summary.csv`: class counts and subject counts for each fold/role.

RT30 labels are **relative within-session behavioral proxies**, not absolute
attention labels. Inner validation is grouped by subject, and held-out test
subjects must never contribute to normalization, early stopping, or model fitting.

Regenerate from local raw metadata (no need to read NPZ EEG):

    python -m src.labeling --project-root .
    python -m src.splits --project-root .

Do not regenerate just one CSV with a new labeling config and continue using
old splits without checking their class/sample balances.
