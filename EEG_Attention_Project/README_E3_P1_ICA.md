# E3 P1 — Self-ICA (4 sec, original CNN)

## Purpose
Compare P0 (basic preprocessing) and P1 (offline 30-channel ICA + ICLabel), holding RT labels, trial identities, baseline filter and CNN protocol fixed. P1 Full30 and Forehead2 are two independently trained CNNs using **the same cleaned 30-channel source**. This is not 2-channel ICA, and it is not a deployable 2-electrode denoiser. P2 stays untouched.

## Install / execute (PowerShell, existing Python environment)
Extract this archive directly into `F:\CSIE\EEG_Attention_Project` (not a new nested project). New files only. Keep the existing `src/`, baseline YAML and raw/P0/P2 files.

```powershell
cd F:\CSIE\EEG_Attention_Project
python -m pip install -r requirements_e3_p1.txt
python scripts\prepare_e3_p1_ica.py --check
python scripts\prepare_e3_p1_ica.py --pilot
python scripts\prepare_e3_p1_ica.py --run
python scripts\run_e3_p1_train.py --check
python scripts\run_e3_p1_train.py --smoke-test
python scripts\run_e3_p1_train.py
```

Do these separately. Stop if a command fails. The Pilot runs the **whole first real Session** (not a short fragment) through the exact ICA configuration, checking P0 reproduction, classification and epoch export; it writes `epochs/_P1_ica_pilot/4sec/` and `results/E3/P1_ica_PILOT/`. It is not published as P1 and the production pass recomputes that session. `--check` audits all 62 sessions' event metadata and existing B0 NPZ contracts; it does not fit ICA. `--run` validates again, generates all P1 data, resumes already audited staging sessions, and publishes only after all 62 sessions pass. No training starts automatically. No source files overwritten. Do not run multiple preparations against the same destination simultaneously.

If the raw dataset is in a different directory, pass `--raw-root` to every preparation command:

```powershell
python scripts\prepare_e3_p1_ica.py --pilot --raw-root 'F:\CSIE\nm000275_v1.0.0'
```

Use the NEMAR BIDS `.set` (the same used for P0), not Figshare P2. The P1 package does not contain EEG data.

## ICA algorithm / processing

1. Read canonical 30 scalp channels from each original continuous 500-Hz EEG. Validate full 4-second original label table and BIDS event samples/codes, valid x_row ordering and P0 NPZ metadata. The source file's original electrical reference is kept for the final output.
2. On a fitting copy, assign the standard_1005 *template* positions (T3/T4/T5/T6 mapped to T7/T8/P7/P8), average-reference, FIR zero-phase filter 1–100 Hz, resample to 250 Hz. Template locations are not measured individual positions. Rank is estimated from the covariance eigenvalues with fixed relative threshold 1e-10; normally 29 after CAR. No channels or epochs are silently dropped.
3. Fit separate Extended Infomax using Picard (ortho=False, extended=True, tol=1e-7, max_iter=500, seed=42). Whole-session data used; including unlabeled test sessions independently. No RT labels used. Fit time includes preparation of fitting copy. Failure/nonconvergence halts.
4. ICLabel with torch backend; use all 7 probabilities; remove only Eye >=0.8 OR Muscle >=0.8. Other classes retained. Save probabilities, fit warnings, ICA solution, removal indices. No threshold tuning and no automatic fallback solver. Record installed package versions. ICLabel scores are model estimates, not known biological ground truth.
5. Obtain affine artifact contribution D(x) from MNE ICA reconstruction with the selected components excluded, evaluated on CAR input. Apply `x_clean = x_original - D(x_original)` where D already includes CAR projection. This preserves the original reference common mode. It is a custom reference-preserving protocol rather than standard ICA CAR output; components lost in CAR cannot be removed by this projection. Numerically check it against MNE's direct reconstruction on CAR samples. If no components removed, exact identity transformation.
6. For **every** original valid trial, calculate the event-restricted warmup (max 4 s; never include a previous event), reconstruct original raw epoch first and compare it to existing B0 NPZ (rtol=2e-5, atol=0.001 uV), then subtract artifact on original 500-Hz segment and run the original causal 4th-order Butterworth 1–30 Hz filter and `resample_poly(1,2,padtype='line')` on final 4 seconds. Keep identical 30 × 1000 samples. No RT reranking. The spatial artifact removal affine operator is derived at fit time, therefore applying it to 500-Hz signal is well-defined; not a claim that IC separation is optimal outside fit band. Final band limits match B0.
7. Save P1 arrays and quality reports, including average forehead PSD, band powers, first eligible waveform, removed topographies when plotting available, fit times, rank, convergence and B0 numeric error. QA is descriptive; no rejection based on whether BA or band power improves.

## Outputs

```
epochs/P1_ica/4sec/
    sub-XX_ses-YY_unlabeled.npz  # X_P1_uV, all 30 channels; trial_index; onset_s; sfreq_P1
    sub-XX_ses-YY_manifest.csv
    all_trials_UNLABELED.csv    # original, if exists
    P1_provenance.json
    extraction_report.txt
data/labels/E3/P1_ica/4sec/
    rt30_labels.csv            # byte-for-byte original 4s table
    rt30_summary.csv           # copy, if available
    P1_provenance.json
results/E3/P1_preflight/
    preflight.json
    session_check.csv
results/E3/P1_ica/
    quality/sub-XX_ses-YY/
        components.csv
        solution-ica.fif
        artifact_operator.npz
        fit_warnings.txt
        band_power.csv
        forehead_waveform_psd.png
        removed_topography_*.png  # only if excluded; plot errors recorded separately
        done.json
    ica_session_summary.csv
    Full30/fold_1/ ... fold_5/
    Forehead2/fold_1/ ... fold_5/
    Full30/{fold_summary.csv,combined_predictions.csv,subject_summary.csv,summary.json}
    Forehead2/{fold_summary.csv,combined_predictions.csv,subject_summary.csv,summary.json}
    comparison.csv             # P0 and P1 (P0 only when original predictions found/verified)
```

If interrupted, rerun same command; complete sessions/folds with matching signatures skipped, interrupted one recomputed. Config/source/version change may invalidate signatures. Move obsolete staging/quality or fold directories aside manually after inspecting; never discard the published P0/P2. Pilot and smoke results isolated. Expect ~2.73 GB of uncompressed P1 arrays (22,777 × 30 × 1000 float32), plus ICA reports, models and training scratch (~1.63 GB for labeled Full30). Allow additional free space/RAM for one entire 30ch continuous session and filtering/PCA/ICLabel. Hard interruption can leave temporary P1 training directories; remove only when no P1 training running. No need to create separate Forehead2 NPZ (same all-channel array).

## CNN
Original `src` modules and `configs/baseline.yaml` read unchanged. 4 seconds overrides older 2-second input field via loader.expected_samples=1000. Train-only channelwise zscore, train-only class weights, AdamW, original fixed hyperparameters; subject-wise folds unchanged, random_seed + fold. Defaults both groups, 5 folds each, 10 new trainings. Smoke: Forehead2, fold 1, subset of data, 2 epochs, writes `results/E3/P1_ica_SMOKE`. Device automatically selected. Use same device as E1 (CPU) if desired:

```powershell
python scripts\run_e3_p1_train.py --device cpu
# alternatively select only one group or fold, no additional fits
python scripts\run_e3_p1_train.py --group Forehead2 --device cpu
python scripts\run_e3_p1_train.py --group Full30 --fold 1 --device cpu
```

If you use --device cpu for part of the formal experiment, keep device the same for resume. P0 is the original `results/diagnostics/window_length/4sec/{group}/fold_*/predictions.csv`, checked against 13,604 labels + subject folds and run config, not Random Search. P0 not retrained. Existing P2 label file checked for matching hash if P2 provenance present; this is not a new audit of the P2 signal, or a P2 training run. Earlier P2 runner supports Forehead2; if Full30 P2 desired, use full saved P2 arrays and adapt its own runner separately. Comparison only claims P0 vs P1 until P2 trained and audited.

## Interpretation / constraints
E3 measures offline artifact handling utility. ICA uses the same test session's continuous unlabeled EEG (may include later samples), hence this is not a fully prospective test. Different session-specific ICA matrices share rules, not weights. Not possible to transfer 30ch matrix directly to 2ch. The cleaned forehead pair depends on all 30 electrodes during preprocessing. Accuracy and power changes are observations, not evidence of artifact removal alone; no guarantee of improved BA or signal validity. Runtime logged is not real-time end-to-end latency. P2 uses the author's distinct full processing pipeline.

## Local tests / verification
`python tests/test_e3_p1.py` runs a small synthetic test with actual MNE Extended Infomax (not the production Picard package), verifies the affine projection and no-exclusion identity, synthetic ICLabel scores, matched B0 filter and event logic, data loader, 2-epoch Full30 and Forehead2 CNN fitting, and prediction alignment. Passed locally. The full Picard+ICLabel stack was not installed in the creation environment; actual model inference, long session convergence and all 62 raw recordings have not been executed here. Pilot on your real data is required. Existing E1 4-sec prediction tables and labels/folds were independently matched and original BA re-derived (68.7955%, 66.0898%).
