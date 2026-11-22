"""Generate fixed grouped outer test folds and subject-disjoint inner validation."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from .config import load_project_config, resolve_project_path


def make_subject_splits(
    labels: pd.DataFrame,
    n_folds: int = 5,
    validation_subject_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = labels.loc[labels['label'].notna()].copy()
    if df.empty:
        raise ValueError('No labeled EEG trials')
    if df.subject.nunique() < n_folds + 2:
        raise ValueError('Not enough independent subjects for outer and inner folds')
    df['label'] = df['label'].astype('int64')
    subjects = sorted(df.subject.unique())
    counts = (df.groupby(['subject', 'label']).size().unstack(fill_value=0)
              .reindex(subjects, fill_value=0))
    splitter = GroupKFold(n_splits=n_folds)
    roles, summaries = [], []
    # GroupKFold on samples balances approximately by sample count.
    for fold, (train_val_idx, test_idx) in enumerate(
        splitter.split(df, y=df['label'], groups=df['subject']), 1
    ):
        train_val_subjects = np.array(sorted(df.iloc[train_val_idx].subject.unique()))
        test_subjects = set(df.iloc[test_idx].subject.unique())
        inner = GroupShuffleSplit(n_splits=1, test_size=validation_subject_fraction,
                                  random_state=seed + fold)
        a, b = next(inner.split(train_val_subjects, groups=train_val_subjects))
        train_subjects = set(train_val_subjects[a])
        val_subjects = set(train_val_subjects[b])
        assert not (train_subjects & val_subjects or train_subjects & test_subjects or val_subjects & test_subjects)
        for subject in subjects:
            role = ('test' if subject in test_subjects else
                    'val' if subject in val_subjects else 'train')
            c0 = int(counts.loc[subject, 0]) if 0 in counts.columns else 0
            c1 = int(counts.loc[subject, 1]) if 1 in counts.columns else 0
            roles.append({'fold': fold, 'subject': subject, 'role': role,
                          'n_labeled': c0+c1, 'n_label0': c0, 'n_label1': c1})
        fold_rows = [row for row in roles if row['fold'] == fold]
        for role in ['train', 'val', 'test']:
            chosen = [r for r in fold_rows if r['role'] == role]
            summaries.append({
                'fold': fold, 'role': role, 'n_subjects': len(chosen),
                'n_labeled': sum(r['n_labeled'] for r in chosen),
                'n_label0': sum(r['n_label0'] for r in chosen),
                'n_label1': sum(r['n_label1'] for r in chosen),
            })
    assignments = pd.DataFrame(roles)
    # Every subject is a test subject in EXACTLY one outer fold.
    assert assignments[assignments.role.eq('test')].groupby('subject').size().eq(1).all()
    assert len(assignments) == len(subjects)*n_folds
    return assignments, pd.DataFrame(summaries)


def main() -> None:
    p = argparse.ArgumentParser(description='Generate grouped subject fold assignments')
    p.add_argument('--project-root', default='.', type=Path)
    args = p.parse_args()
    cfg = load_project_config(args.project_root)
    settings = cfg['baseline']['splits']
    paths = cfg['paths']
    labels_path = resolve_project_path(cfg['project_root'], paths['labels_csv'])
    if not labels_path.exists():
        p.error(f'Labels not found: {labels_path}; run python -m src.labeling first')
    labels = pd.read_csv(labels_path)
    assignments, summaries = make_subject_splits(
        labels, int(settings['outer_folds']),
        float(settings['validation_subject_fraction']),
        int(settings['random_seed']),
    )
    out_roles = resolve_project_path(cfg['project_root'], paths['subject_folds_csv'])
    out_summary = resolve_project_path(cfg['project_root'], paths['fold_summary_csv'])
    out_roles.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(out_roles, index=False, encoding='utf-8-sig')
    summaries.to_csv(out_summary, index=False, encoding='utf-8-sig')
    print(f'Wrote {out_roles} ({len(assignments)} fold-subject roles)')
    print(summaries.to_string(index=False))


if __name__ == '__main__':
    main()
