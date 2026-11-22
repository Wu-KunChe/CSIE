"""Create per-session reaction-time labels using the 5-35 / 65-95 rank bands.

This is a label-only change. Existing EEG epochs, channel selection, CNN,
subject-based folds, and all project file paths remain unchanged.

Within EACH session, sort eligible trials by RT (then trial_index for ties):
  - shortest 0-5%:   unlabelled (extreme tail)
  - fastest 5-35%:   label 0, relatively fast RT
  - middle 35-65%:  unlabelled
  - slowest 65-95%:  label 1, relatively slow RT
  - longest 95-100%: unlabelled (extreme tail)

Counts use floor(0.05*N) for each tail and floor(0.30*N) for each
labeled class. Percentile slices are therefore approximate for small sessions.
This is a rank-based exclusion, NOT an absolute physiological RT threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_project_config, resolve_project_path

REQUIRED_COLUMNS = {
    'subject', 'session', 'trial_index', 'rt_s', 'x_row',
    'included_in_npz', 'complete_triplet',
}

# The one intentional change from the previous fastest/slowest 30% scheme.
DEFAULT_TAIL_EXCLUSION_FRACTION = 0.05
LABELING_METHOD = 'session_RT_rank_5to35_vs_65to95'


def make_rt30_labels(
    trials: pd.DataFrame,
    fraction: float = 0.30,
    min_rt_s: float | None = None,
    max_rt_s: float | None = None,
    flag_rt_below_s: float = 0.1,
    flag_rt_above_s: float = 10.0,
    tail_exclusion_fraction: float = DEFAULT_TAIL_EXCLUSION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all-trial labels and a per-session report, in original row order.

    Keep the previous public function name so no other module must be updated.
    Fraction is the proportion PER CLASS (normally 0.30), after skipping
    a fraction from EACH tail (normally 0.05). Ranking is done independently
    per session, using a deterministic tie-breaker.
    """
    missing = REQUIRED_COLUMNS - set(trials.columns)
    if missing:
        raise ValueError(f'Missing required metadata fields: {sorted(missing)}')
    if not 0.0 < fraction < 0.5:
        raise ValueError('fraction must be in (0, 0.5)')
    if not 0 <= tail_exclusion_fraction < 0.5:
        raise ValueError('tail_exclusion_fraction must be in [0, 0.5)')
    if fraction + tail_exclusion_fraction >= 0.5:
        raise ValueError('fraction + tail_exclusion_fraction must be < 0.5')
    if min_rt_s is not None and max_rt_s is not None and min_rt_s >= max_rt_s:
        raise ValueError('min_rt_s must be below max_rt_s')
    if trials.duplicated(['subject', 'session', 'trial_index']).any():
        raise ValueError('Duplicate subject/session/trial_index keys')

    df = trials.copy().reset_index(drop=True)
    df['label'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    df['label_status'] = 'not_in_npz'
    df['npz_file'] = (df['subject'].astype(str) + '_' +
                      df['session'].astype(str) + '_unlabeled.npz')
    df['labeling_method'] = LABELING_METHOD
    df['rt_rank_within_session'] = pd.Series(pd.NA, index=df.index, dtype='Int64')

    rt = pd.to_numeric(df['rt_s'], errors='coerce')
    eligible = (df['included_in_npz'].eq(1) & df['complete_triplet'].eq(1) &
                df['x_row'].ge(0) & rt.notna() & np.isfinite(rt) & rt.gt(0))
    saved = df['included_in_npz'].eq(1)
    df.loc[saved & ~eligible, 'label_status'] = 'no_positive_finite_rt'

    # Quality flags remain descriptive, never hard RT exclusion by default.
    df['rt_quality_flag'] = 'ok'
    df.loc[rt.notna() & rt.lt(flag_rt_below_s), 'rt_quality_flag'] = 'suspicious_short'
    df.loc[rt.notna() & rt.gt(flag_rt_above_s), 'rt_quality_flag'] = 'suspicious_long'
    df.loc[rt.isna() | ~np.isfinite(rt), 'rt_quality_flag'] = 'nonfinite'

    if min_rt_s is not None:
        excluded = eligible & rt.lt(min_rt_s)
        df.loc[excluded, 'label_status'] = 'excluded_rt_below_limit'
        eligible &= ~excluded
    if max_rt_s is not None:
        excluded = eligible & rt.gt(max_rt_s)
        df.loc[excluded, 'label_status'] = 'excluded_rt_above_limit'
        eligible &= ~excluded
    df.loc[eligible, 'label_status'] = 'unlabeled_middle'

    report_rows = []
    for (subject, session), group in df.groupby(['subject', 'session'], sort=True):
        idx = group.index[eligible.loc[group.index]]
        ordered = df.loc[idx].sort_values(['rt_s', 'trial_index'], kind='mergesort')
        n = len(ordered)
        tail = int(np.floor(tail_exclusion_fraction * n))
        k = int(np.floor(fraction * n))
        if 2 * (k + tail) > n:
            raise ValueError(f'{subject}/{session}: class bands overlap')

        df.loc[ordered.index, 'rt_rank_within_session'] = np.arange(1, n + 1)
        if tail:
            fastest_extreme = ordered.index[:tail]
            slowest_extreme = ordered.index[n-tail:]
            df.loc[fastest_extreme, 'label_status'] = 'excluded_fastest_tail'
            df.loc[slowest_extreme, 'label_status'] = 'excluded_slowest_tail'
        if k:
            # When tail==0, slicing via explicit n-tail avoids [-0:] == [:].
            fast = ordered.index[tail:tail+k]
            slow = ordered.index[n-tail-k:n-tail]
            if len(fast) != k or len(slow) != k or set(fast) & set(slow):
                raise AssertionError(f'{subject}/{session}: label slices incorrect')
            df.loc[fast, 'label'] = 0
            df.loc[fast, 'label_status'] = 'labeled_fast'
            df.loc[slow, 'label'] = 1
            df.loc[slow, 'label_status'] = 'labeled_slow'

        session_labeled = df.loc[group.index]
        report_rows.append({
            'subject': subject,
            'session': session,
            'labeling_method': LABELING_METHOD,
            'epochs_saved': int(group['included_in_npz'].eq(1).sum()),
            'rt_eligible_for_ranking': n,
            'excluded_fastest_tail': tail,
            'excluded_slowest_tail': tail,
            'label_0': k,
            'label_1': k,
            'unlabeled_middle': n - 2 * (k + tail),
            'suspicious_short_saved': int((session_labeled['rt_quality_flag'].eq('suspicious_short') &
                                          session_labeled['included_in_npz'].eq(1)).sum()),
            'suspicious_long_saved': int((session_labeled['rt_quality_flag'].eq('suspicious_long') &
                                         session_labeled['included_in_npz'].eq(1)).sum()),
            'suspicious_short_labeled': int((session_labeled['rt_quality_flag'].eq('suspicious_short') &
                                           session_labeled['label'].notna()).sum()),
            'suspicious_long_labeled': int((session_labeled['rt_quality_flag'].eq('suspicious_long') &
                                          session_labeled['label'].notna()).sum()),
            'min_rt_s': float(ordered['rt_s'].min()) if n else np.nan,
            'median_rt_s': float(ordered['rt_s'].median()) if n else np.nan,
            'max_rt_s': float(ordered['rt_s'].max()) if n else np.nan,
        })

    if (df['label'].notna() & ~eligible).any():
        raise AssertionError('Labels assigned to ineligible events')
    selected = df.loc[df.label.notna()]
    if selected.duplicated(['subject', 'session', 'x_row']).any():
        raise ValueError('Duplicate x_row within a session')
    if not set(df['label'].dropna().unique()).issubset({0, 1}):
        raise AssertionError('Unexpected class values')
    return df, pd.DataFrame(report_rows)


def main() -> None:
    p = argparse.ArgumentParser(description='Generate within-session RT rank 5-35 / 65-95 labels')
    p.add_argument('--project-root', default='.', type=Path)
    p.add_argument('--input-csv', type=Path, default=None)
    p.add_argument('--tail-fraction', type=float,
                   default=DEFAULT_TAIL_EXCLUSION_FRACTION,
                   help='Excluded RT fraction at EACH extreme (default 0.05)')
    args = p.parse_args()
    cfg = load_project_config(args.project_root)
    settings = cfg['baseline']['labeling']
    paths = cfg['paths']
    input_path = args.input_csv or resolve_project_path(cfg['project_root'], paths['trials_csv'])
    out_labels = resolve_project_path(cfg['project_root'], paths['labels_csv'])
    out_summary = resolve_project_path(cfg['project_root'], paths['label_summary_csv'])
    if not input_path.exists():
        p.error(f'Metadata not found: {input_path}')

    labels, summary = make_rt30_labels(
        pd.read_csv(input_path),
        fraction=float(settings['fraction_per_class']),
        min_rt_s=settings.get('min_rt_s'),
        max_rt_s=settings.get('max_rt_s'),
        flag_rt_below_s=float(settings['flag_rt_below_s']),
        flag_rt_above_s=float(settings['flag_rt_above_s']),
        tail_exclusion_fraction=args.tail_fraction,
    )
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_labels, index=False, encoding='utf-8-sig')
    summary.to_csv(out_summary, index=False, encoding='utf-8-sig')
    labeled = labels.loc[labels.label.notna()]
    print(f'Method: {LABELING_METHOD}; tail fraction: {args.tail_fraction:.1%} each')
    print(f'Wrote {out_labels} and {out_summary}')
    print(f'Sessions: {len(summary)}, epochs saved: {int(labels.included_in_npz.sum())}')
    print(f'Labeled: {len(labeled)} | 0: {(labeled.label==0).sum()} | 1: {(labeled.label==1).sum()}')
    print(f'Excluded tail trials: {int(summary.excluded_fastest_tail.sum())} shortest + '
          f'{int(summary.excluded_slowest_tail.sum())} longest')
    print(f'Suspicious RT STILL labeled: {(labeled.rt_quality_flag != "ok").sum()} '
          '(tail exclusion is rank-based, not a guaranteed numerical cutoff)')


if __name__ == '__main__':
    main()
