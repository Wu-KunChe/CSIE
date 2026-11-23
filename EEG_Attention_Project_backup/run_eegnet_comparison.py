"""Independent EEGNet model comparison launcher. Existing main.py is untouched."""
from __future__ import annotations
import argparse
from pathlib import Path
from experiments.model_comparison.eegnet.runner import run_eegnet


def main() -> None:
    parser=argparse.ArgumentParser(description='EEGNet vs existing 1D-CNN E1 (unchanged B0 data)')
    parser.add_argument('--group',action='append',default=None,
                        help='One group; repeat for multiple groups. Default: all 3')
    parser.add_argument('--fold',action='append',type=int,default=None,
                        help='One fold; repeat for multiple. Default: all 5')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--check',action='store_true',help='Verify data/config without training')
    parser.add_argument('--smoke-test',action='store_true',help='Small 2-epoch sanity check in separate output')
    parser.add_argument('--summarize-only',action='store_true',
                        help='Rebuild reports from completed EEGNet folds without training')
    parser.add_argument('--force',action='store_true',
                        help='Replace ONLY selected EEGNet folds (never old E1/B0)')
    args=parser.parse_args()
    run_eegnet(Path(__file__).resolve().parent,groups=args.group,
               selected_folds=args.fold,device=args.device,
               check=args.check,smoke_test=args.smoke_test,
               summarize_only=args.summarize_only,force=args.force)

if __name__=='__main__':
    main()
