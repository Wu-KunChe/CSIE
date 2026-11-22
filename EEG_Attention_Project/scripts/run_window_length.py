#!/usr/bin/env python3
"""Run 2/3/4/5-second input-window 1D-CNN experiments in one command.

From EEG_Attention_Project root:
    python scripts/run_window_length.py --check
    python scripts/run_window_length.py --smoke-test
    python scripts/run_window_length.py
    python scripts/run_window_length.py --summarize-only

Use --window/--group/--fold only for focused runs or resuming individual cases.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.window_length.runner import run_window_length, SEVEN_GROUPS


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT,
                        help='Root folder containing configs, epochs, data and src')
    parser.add_argument('--window', action='append', type=int, choices=(2,3,4,5),
                        help='Select one second-length; repeat for several (default: all 2-5)')
    parser.add_argument('--group', action='append', choices=SEVEN_GROUPS,
                        help='Limit to one E1 group; repeatable (default: all seven)')
    parser.add_argument('--fold', action='append', type=int, choices=(1,2,3,4,5),
                        help='Limit to one outer fold; repeatable (default: all five)')
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    parser.add_argument('--check', action='store_true',
                        help='Check paths, labels, subject splits and one NPZ; no training')
    parser.add_argument('--smoke-test', action='store_true',
                        help='2-epoch tiny-data check (default Forehead2/fold1 per window)')
    parser.add_argument('--summarize-only', action='store_true',
                        help='Rebuild all available summary tables/figures; no model fitting')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite ONLY selected window-length folds (never E1)')
    args = parser.parse_args()
    run_window_length(args.project_root, windows=args.window, groups=args.group,
                      selected_folds=args.fold, device=args.device, force=args.force,
                      check=args.check, smoke_test=args.smoke_test,
                      summarize_only=args.summarize_only)


if __name__ == '__main__':
    main()
