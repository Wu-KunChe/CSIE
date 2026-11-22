#!/usr/bin/env python3
"""Launch the shared B0 baseline or the full E1 regional EEG experiment.

Examples (PowerShell, from EEG_Attention_Project):
    python scripts/main.py --experiment baseline --check
    python scripts/main.py --experiment baseline --smoke-test
    python scripts/main.py --experiment baseline --summarize-only
    python scripts/main.py --experiment baseline
    python scripts/main.py --experiment E1
    python scripts/main.py --experiment E1 --summarize-only
    python scripts/main.py --experiment E1 --groups Frontal2 Central2 --fold 1
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Entrypoint location: EEG_Attention_Project/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.E1_region import run_experiment


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="baseline", choices=["baseline", "E1"],
                   help="baseline: Full30 only; E1: Full30 plus six two-channel groups")
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT,
                   help="Project directory, containing configs/data/src (default: script's parent)")
    select = p.add_mutually_exclusive_group()
    select.add_argument("--group", action="append", metavar="NAME",
                        help="One channel group (repeat the option to select multiple)")
    select.add_argument("--groups", nargs="+", metavar="NAME",
                        help="Several named groups")
    p.add_argument("--fold", action="append", type=int, metavar="N",
                   help="Run only this outer test fold (repeatable); default: all five")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="auto selects CUDA if available, else CPU")
    p.add_argument("--check", action="store_true",
                   help="Check files, labels and splits WITHOUT model training")
    p.add_argument("--smoke-test", action="store_true",
                   help="Train 2 epochs on tiny subset into results/smoke_test; NOT research results")
    p.add_argument("--summarize-only", action="store_true",
                   help="Rebuild combined five-fold group reports and labeled plots "
                        "from existing fold predictions, WITHOUT retraining")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing results for requested groups/folds")
    p.add_argument("--no-reuse-baseline", action="store_true",
                   help="For E1, train all groups instead of reusing compatible baseline folds")
    args = p.parse_args()
    groups = args.groups if args.groups is not None else args.group
    run_experiment(
        args.project_root, experiment=args.experiment,
        requested_groups=groups, requested_folds=args.fold,
        device=args.device, force=args.force, check_only=args.check,
        smoke_test=args.smoke_test, reuse_baseline=not args.no_reuse_baseline,
        summarize_only=args.summarize_only)


if __name__ == "__main__":
    main()
