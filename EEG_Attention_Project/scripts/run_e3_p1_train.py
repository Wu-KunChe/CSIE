#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import read_yaml
from experiments.E3_P1.train import run

def main():
    p=argparse.ArgumentParser(description='E3 P1, fixed CNN training: Forehead2 and Full30.')
    p.add_argument('--project-root',type=Path,default=ROOT)
    m=p.add_mutually_exclusive_group();m.add_argument('--check',action='store_true');m.add_argument('--smoke-test',action='store_true')
    p.add_argument('--group',action='append',choices=['Full30','Forehead2']);p.add_argument('--fold',type=int,action='append',choices=range(1,6));p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args();root=a.project_root.resolve();cfg=read_yaml(root/'configs/e3_p1_ica.yaml')
    run(root,cfg,a.group or (['Forehead2'] if a.smoke_test else ['Full30','Forehead2']),a.fold,a.device,a.check,a.smoke_test)
if __name__=='__main__':main()
