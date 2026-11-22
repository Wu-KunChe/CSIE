#!/usr/bin/env python3
"""Resume the ORIGINAL 50-candidate study, but STOP SEARCH after candidate 20.

No original project files are modified.  The original experiment fingerprint and
fold signatures are retained, so trial_001 through trial_019 are recognized as
completed, and an interrupted trial_020 is resumed from its first incomplete fold.
Then the unchanged 7 channel groups x 5 folds are evaluated.

From EEG_Attention_Project root:
    python scripts/run_hyperparameter_search_20.py --check
    python scripts/run_hyperparameter_search_20.py

IMPORTANT: Keep the original run_hyperparameter_search.py, runner.py and
hyperparameter_search.yaml unchanged; they are included in the resume hash.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hyperparameter_search import runner as engine

# Change ONLY the number of search candidates at runtime.  Files already
# fingerprinted by the first run MUST stay byte-for-byte unchanged.
SEARCH_CANDIDATES_TO_RUN = 20
ORIGINAL_CANDIDATE_COUNT = 50
_original_load_setup = engine.load_setup
_original_build_fingerprint = engine.build_fingerprint


def _load_setup_first_20(root):
    ctx = _original_load_setup(root)
    original_count = int(ctx['tuning']['study']['n_random_candidates'])
    if original_count != ORIGINAL_CANDIDATE_COUNT:
        raise RuntimeError(
            f'Original YAML should still say n_random_candidates: '
            f'{ORIGINAL_CANDIDATE_COUNT}; it currently says {original_count}. '
            'Do not change the original YAML: restore it to preserve Resume.'
        )
    ctx['tuning']['study']['n_random_candidates'] = SEARCH_CANDIDATES_TO_RUN
    return ctx


def _fingerprint_with_original_50(ctx, first_20_candidates, inner_roles):
    # The smoke-test output has its own independent signature, so its single
    # candidate should still be fingerprinted in the original way.
    if ctx['out'].name.endswith('_SMOKE'):
        return _original_build_fingerprint(ctx, first_20_candidates, inner_roles)

    original_tune = deepcopy(ctx['tuning'])
    original_tune['study']['n_random_candidates'] = ORIGINAL_CANDIDATE_COUNT
    original_50 = engine.sample_candidates(
        original_tune, ORIGINAL_CANDIDATE_COUNT, ctx['baseline']
    )
    if len(first_20_candidates) != SEARCH_CANDIDATES_TO_RUN or first_20_candidates != original_50[:SEARCH_CANDIDATES_TO_RUN]:
        raise RuntimeError('First 20 candidates do not match the original 50-candidate sequence.')

    # Keep the precise original run fingerprint (and thus ALL existing
    # per-fold done.json signatures).  Only the selected search budget changes.
    return _original_build_fingerprint(ctx, original_50, inner_roles)


engine.load_setup = _load_setup_first_20
engine.build_fingerprint = _fingerprint_with_original_50


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--check', action='store_true', help='Check labels, NPZ and folds; print plan; do not train')
    parser.add_argument('--smoke-test', action='store_true', help='Separate tiny 2-epoch pipeline test (not research output)')
    parser.add_argument('--phase', choices=['all', 'search', 'final'], default='all',
                        help='All=search then final; final requires complete saved search results')
    args = parser.parse_args()
    engine.run(args.project_root, device=args.device, check=args.check,
               smoke=args.smoke_test, phase=args.phase)


if __name__ == '__main__':
    main()
