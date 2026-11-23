#!/usr/bin/env python3
"""E3 P2 official CNN: Full30 / Forehead2 with fixed five-fold protocol.

python scripts/run_e3_p2_train.py --check
python scripts/run_e3_p2_train.py --group Full30 --smoke-test
python scripts/run_e3_p2_train.py --group Full30
python scripts/run_e3_p2_train.py  # both groups; old Forehead2 is reused
"""
from pathlib import Path
import argparse, sys
PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path: sys.path.insert(0,str(PROJECT))
from experiments.E3_P2.train import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=PROJECT)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--smoke-test', action='store_true')
    p.add_argument('--group', action='append', choices=('Full30','Forehead2'),
                   help='Repeatable; default both, or Forehead2 for smoke')
    p.add_argument('--fold', type=int, action='append', choices=range(1,6))
    p.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    p.add_argument('--force', action='store_true',
                   help='Explicitly overwrite selected existing P2 fold results')
    a=p.parse_args()
    run(a.project_root, folds_select=a.fold, device=a.device, check=a.check,
        smoke=a.smoke_test, force=a.force, groups=a.group)

if __name__=='__main__': main()
