"""Independent XGBoost + Spectral Features comparison entrypoint.

Examples:
  python scripts/run_xgboost_comparison.py --check
  python scripts/run_xgboost_comparison.py --smoke-test
  python run_xgboost_comparison.py
  python scripts/run_xgboost_comparison.py --group Forehead2 --fold 1
  python scripts/run_xgboost_comparison.py --summarize-only
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Entrypoint location: EEG_Attention_Project/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.model_comparison.xgboost.runner import run_xgboost, GROUPS


def main() -> None:
    parser=argparse.ArgumentParser(description='Standalone seven-group XGBoost EEG model comparison')
    parser.add_argument('--project-root',type=Path,default=PROJECT_ROOT)
    parser.add_argument('--group',action='append',choices=GROUPS,
                        help='Optional selection; may be repeated. Default: all seven groups.')
    parser.add_argument('--fold',action='append',type=int,
                        help='Optional outer test fold; may be repeated. Default: all five.')
    parser.add_argument('--check',action='store_true',help='Verify data, labels, splits and features')
    parser.add_argument('--smoke-test',action='store_true',help='Fast 1-fold test (Forehead2 by default)')
    parser.add_argument('--summarize-only',action='store_true',help='Rebuild reports without training')
    parser.add_argument('--force',action='store_true',help='Overwrite ONLY selected XGBoost result folds')
    args=parser.parse_args()
    run_xgboost(args.project_root,groups=args.group,selected_folds=args.fold,
                check=args.check,smoke_test=args.smoke_test,
                summarize_only=args.summarize_only,force=args.force)

if __name__=='__main__':
    main()
