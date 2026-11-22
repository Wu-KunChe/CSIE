#!/usr/bin/env python3
"""One-command 4s CNN Random Search + seven-group evaluation.

From EEG_Attention_Project root:
  python scripts/run_hyperparameter_search.py --check
  python scripts/run_hyperparameter_search.py --smoke-test
  python scripts/run_hyperparameter_search.py

Default: 50 random hyperparameter sets * 3 subject-wise scoring folds,
then 7 channel groups * 5 original held-out folds = 185 independent fits.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hyperparameter_search.runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--check', action='store_true', help='Check labels, NPZ and folds; print plan; do not train')
    parser.add_argument('--smoke-test', action='store_true', help='Separate tiny 2-epoch pipeline test (not research output)')
    parser.add_argument('--phase', choices=['all', 'search', 'final'], default='all',
                        help='All=search then final; final requires complete saved search results')
    args = parser.parse_args()
    run(args.project_root, device=args.device, check=args.check,
        smoke=args.smoke_test, phase=args.phase)


if __name__ == '__main__':
    main()
