#!/usr/bin/env python3
"""Check official P2 data, then optionally create matched 4-sec epochs.

python scripts/prepare_e3_p2_official.py --check
python scripts/prepare_e3_p2_official.py --run
python scripts/prepare_e3_p2_official.py --preproc-root "F:\\CSIE\\official_preprocessed" --run
"""
from pathlib import Path
import argparse,sys
PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from experiments.E3_P2.prepare import run

def main():
    a=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--project-root',type=Path,default=PROJECT)
    a.add_argument('--raw-root',type=Path,help='Default: sibling nm000275_v1.0.0')
    a.add_argument('--preproc-root',type=Path,help='Official, separately downloaded preprocessed .set/.set.zip. Default sibling nm000275_preprocessed')
    a.add_argument('--forehead-only',action='store_true',help='Save only FP1/FP2 (~180 MB vs ~2.7 GB for all epochs). Preflight still checks 30ch.')
    mode=a.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true',help='Check all sessions; do not create P2 EEG')
    mode.add_argument('--run',action='store_true',help='Check all sessions, then generate/publish P2 EEG and labels')
    args=a.parse_args()
    root=args.project_root.resolve()
    raw=args.raw_root if args.raw_root else root.parent/'nm000275_v1.0.0'
    preproc=args.preproc_root if args.preproc_root else root.parent/'nm000275_preprocessed'
    run(root,raw,preproc,check_only=not args.run,forehead_only=args.forehead_only)

if __name__=='__main__':main()
