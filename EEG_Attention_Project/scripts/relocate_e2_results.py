#!/usr/bin/env python3
"""Safely relocate FINISHED E2 reports to results/E2; no model retraining.

python scripts/relocate_e2_results.py               # dry-run
python scripts/relocate_e2_results.py --apply       # move results and update path in E2 YAML

WARNING: Changing E2 YAML changes its resume fingerprint. Do not restart finished
E2 runs without reviewing the changed experiment signatures. This does NOT
change E1, E2 statistics, CNN weights or E3 data.
"""
from pathlib import Path
import argparse

ROOT=Path(__file__).resolve().parents[1]
OLD='results/diagnostics/E2_frequency_4sec'
NEW='results/E2/E2_frequency_4sec'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--project-root',type=Path,default=ROOT)
    a=p.parse_args()
    root=a.project_root.resolve()
    src=root/OLD;dst=root/NEW
    yaml=root/'configs'/'e2_frequency.yaml'
    if not src.is_dir(): raise FileNotFoundError(f'Original E2 results not found: {src}')
    if dst.exists(): raise FileExistsError(f'Refusing to overwrite destination: {dst}')
    if not yaml.is_file(): raise FileNotFoundError(yaml)
    txt=yaml.read_text(encoding='utf-8')
    if OLD not in txt: raise ValueError('Cannot safely patch E2 YAML; expected old path missing')
    if not any((src/name).is_file() for name in ('e2_comparison.csv','summary.csv')):
        raise ValueError('Source E2 results do not look complete')
    print('From:',src,'\nTo:  ',dst,'\nConfig:',yaml)
    if not a.apply:
        print('DRY RUN. Add --apply to move existing results and update the E2 config path.')
        return
    dst.parent.mkdir(parents=True,exist_ok=True)
    src.rename(dst)
    try:
        yaml.write_text(txt.replace(OLD,NEW),encoding='utf-8')
    except Exception:
        dst.rename(src)
        raise
    print('DONE. Historical E2 results moved, E2 config path updated.')
    print('NOTE: The original E2 resume hashes reference the old YAML; do not relaunch completed E2 jobs.')

if __name__=='__main__':main()
