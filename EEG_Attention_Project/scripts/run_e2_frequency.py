#!/usr/bin/env python3
"""E2 frequency-band study: 4sec original 1D-CNN, Full30 + Forehead2.

  python scripts/run_e2_frequency.py --check
  python scripts/run_e2_frequency.py --smoke-test
  python scripts/run_e2_frequency.py
  python scripts/run_e2_frequency.py --summarize-only
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.E2_frequency.runner import run_e2, GROUPS, BANDS


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--band", choices=BANDS, action="append",
                        help="Specify one or more frequency bands; default all three")
    parser.add_argument("--group", choices=GROUPS, action="append",
                        help="Specify a group; default both Full30 and Forehead2")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4, 5), action="append",
                        help="Specify folds; default five")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--check", action="store_true", help="Verify paths/splits/labels/first NPZ without training")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny two-epoch isolated flow test")
    parser.add_argument("--summarize-only", action="store_true", help="Regenerate reports without training")
    parser.add_argument("--force", action="store_true", help="Overwrite ONLY selected E2 band/group/folds")
    a = parser.parse_args()
    run_e2(a.project_root, bands=a.band, groups=a.group, selected_folds=a.fold,
           device=a.device, check=a.check, smoke=a.smoke_test,
           summarize_only=a.summarize_only, force=a.force)


if __name__ == "__main__":
    main()
