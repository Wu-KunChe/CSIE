# RT labels: fastest 5-35% / slowest 65-95% (2026-11-03)

## What changed (only one Python file)

Replace `src/labeling.py` with the supplied file. `main.py`, `experiments/E1_region.py`, all other `src/` files and `configs/` remain unchanged. The original config `fraction_per_class: 0.30` still controls the 30% per-class amount. The modified script sets `DEFAULT_TAIL_EXCLUSION_FRACTION = 0.05` and also exposes `--tail-fraction 0.05` (optional; default 0.05).

The CSV file names remain `data/labels/rt30_labels.csv` and `data/labels/rt30_summary.csv` deliberately: existing scripts look up these two paths from `configs/paths.yaml`, and no paths need changing. The name `rt30` describes 30% *per class*, even though the selected rank intervals have changed.

## Rank-based RT label selection (separately within each session)

Sort eligible EEG trials by reaction time ascending, with `trial_index` to resolve equal RTs. For `n` eligible trials in a session:

- `tail = floor(0.05*n)`; exclude exactly `tail` trials from EACH end;
- `k = floor(0.30*n)`; after the short tail, take `k` as label 0;
- immediately before the long tail, take `k` as label 1;
- leave the remaining middle trials without class labels.

Because of flooring, exact percentages vary slightly at the session level. Only metadata is recalculated; no EEG `.npz` is modified. Anomalous RT flags below 0.1 s and above 10 s are retained for reporting and are not separately excluded.

## Report observed with the provided 27,192 event rows

- EEG epochs saved: 26,131, 62 sessions, 27 subjects.
- Label 0 (fast 5%-35%): 7,814.
- Label 1 (slow 65%-95%): 7,814.
- RT tails excluded: 1,280 short + 1,280 long.
- Middle unlabeled: 7,943.
- Without saved EEG: 1,061.
- Among *labeled* samples, RT < 0.1s: 230; RT >10s: 234; combined 464 (previously 1,268).
- Same `subject_folds.csv` and `fold_summary.csv` are valid: exact matching output when regenerating with unchanged split parameters.

## Recommended Windows PowerShell procedure

```powershell
cd F:\CSIE\EEG_Attention_Project

# Backup earlier label tables and trained results, to compare RT30 vs RT5-35 later
Copy-Item .\data\labels\rt30_labels.csv .\data\labels\rt30_labels_BEFORE_RT0535.csv
Copy-Item .\data\labels\rt30_summary.csv .\data\labels\rt30_summary_BEFORE_RT0535.csv
Rename-Item .\results\baseline baseline_before_RT0535
Rename-Item .\results\E1 E1_before_RT0535

# Copy the supplied src/labeling.py over the existing one, then regenerate labels
python -m src.labeling --project-root .

# Sanity check (does not train)
python main.py --experiment baseline --check

# Train models with new labels
python main.py --experiment baseline
python main.py --experiment E1
```

If you extracted the provided replacement `data/labels/*.csv` to the project, the `python -m src.labeling` step is optional (the supplied CSVs were already generated from the event table you uploaded). **Back up old CSVs BEFORE extracting the replacement CSVs.** Do not run `--summarize-only` on old results after changing labels; the old predictions used different truth labels and different selected trials.

Caveat: RT flags / labeling scheme are not absolute attentional ground truth. Changing labels changes the training target; you must retrain. Do not infer the new accuracy without completing the new run.
