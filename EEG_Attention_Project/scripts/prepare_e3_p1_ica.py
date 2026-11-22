#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import read_yaml
from experiments.E3_P1.prepare import run

def main():
    p=argparse.ArgumentParser(description='E3 P1: full30 ICA and aligned 4-sec data.')
    p.add_argument('--project-root',type=Path,default=ROOT);p.add_argument('--raw-root',type=Path)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--check',action='store_true');mode.add_argument('--pilot',action='store_true');mode.add_argument('--run',action='store_true')
    a=p.parse_args();root=a.project_root.resolve();cfg=read_yaml(root/'configs/e3_p1_ica.yaml')
    raw=(a.raw_root or root/cfg['raw_root']).resolve()
    run(root,raw,cfg,'run' if a.run else 'pilot' if a.pilot else 'check')
if __name__=='__main__':main()
