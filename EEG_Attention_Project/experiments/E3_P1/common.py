"""P1 contracts adapted from the checked P2 baseline validator."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

SCALP = ('FP1','FP2','F7','F3','FZ','F4','F8','FT7','FC3','FCZ','FC4',
         'FT8','T3','C3','CZ','C4','T4','TP7','CP3','CPZ','CP4','TP8',
         'T5','P3','PZ','P4','T6','O1','OZ','O2')
LABEL_METHOD = 'session_RT_rank_0to30_vs_70to100'
GROUPS = {'Full30': SCALP, 'Forehead2': SCALP[:2]}
KEY='X_P1_uV'
IDS=['subject','session','trial_index']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def json_save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
    tmp.replace(path)

def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()

def stat(path):
    s=Path(path).stat();return [str(Path(path).resolve()),s.st_size,s.st_mtime_ns]

def validate_baseline(project):
    epochs = project/'epochs'/'4sec'
    label_file = project/'data'/'labels'/'window_length'/'4sec'/'rt30_labels.csv'
    fold_file = project/'data'/'splits'/'subject_folds.csv'
    if not label_file.is_file(): raise FileNotFoundError(label_file)
    if not fold_file.is_file(): raise FileNotFoundError(fold_file)
    labels = pd.read_csv(label_file, low_memory=False)
    req = {'subject','session','trial_index','deviation_code','onset_s',
           'rt_s','included_in_npz','x_row','window_seconds','label','npz_file'}
    if not req.issubset(labels.columns):
        raise ValueError('4sec label CSV missing '+str(sorted(req-set(labels))))
    if not labels.window_seconds.eq(4).all():
        raise ValueError('Use the 4-second RT30 label table, not root-level legacy labels')
    if 'labeling_method' in labels and not labels.labeling_method.eq(LABEL_METHOD).all():
        raise ValueError('Unexpected RT30 labeling method')
    if labels.duplicated(['subject','session','trial_index']).any():
        raise ValueError('Duplicate trial identities in baseline labels')
    active = labels.loc[labels.included_in_npz.eq(1)]
    labeled = labels.loc[labels.label.notna()]
    if labeled.empty or not labeled.label.isin([0,1]).all(): raise ValueError('Bad baseline labels')
    if (labeled.included_in_npz.ne(1)|labeled.x_row.lt(0)).any():
        raise ValueError('Labeled rows must be saved valid B0 epochs')
    for (sub,ses),g in labels.groupby(['subject','session']):
        kept = g.loc[g.included_in_npz.eq(1)].sort_values('x_row')
        if not np.array_equal(kept.x_row.to_numpy(dtype=int), np.arange(len(kept))):
            raise ValueError(f'{sub}/{ses}: non-contiguous B0 x_row')
        filename=f'{sub}_{ses}_unlabeled.npz'
        if not (epochs/filename).is_file(): raise FileNotFoundError(epochs/filename)
        if not g.npz_file.eq(filename).all(): raise ValueError(f'{sub}/{ses}: npz_file mismatch')
        mf=epochs/f'{sub}_{ses}_manifest.csv'
        if not mf.is_file(): raise FileNotFoundError(mf)
        man=pd.read_csv(mf)
        if len(man)!=len(g) or not np.array_equal(man.trial_index.to_numpy(),g.trial_index.to_numpy()):
            raise ValueError(f'{sub}/{ses}: original manifest/labels differ')
    folds=pd.read_csv(fold_file)
    if not {'fold','subject','role'}.issubset(folds): raise ValueError('Bad subject_folds.csv')
    if set(folds.fold.unique())!=set(range(1,6)):
        raise ValueError('Expected five subject-wise folds')
    subjects=set(labels.subject.unique())
    for f,g in folds.groupby('fold'):
        if g.duplicated('subject').any() or set(g.subject)!=subjects:
            raise ValueError(f'Fold {f}: missing/duplicated subjects')
        if set(g.role.unique())!={'train','val','test'}:
            raise ValueError(f'Fold {f}: expected train/val/test')
    return labels, label_file, folds, fold_file



def protocol(root):
    from src.config import read_yaml
    base=read_yaml(root/'configs/baseline.yaml')
    model={'name':'cnn1d','conv_filters':[32,64],'kernels':[51,15],'dropout':.3,'num_classes':2}
    train={'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'batch_size':64,'max_epochs':60,'early_stopping_patience':8,'class_weighting':'train_only','decision_threshold':.5,'random_seed':42}
    for section,expected in [('model',model),('training',train)]:
        for k,v in expected.items():
            if base[section].get(k)!=v:raise ValueError(f'Original CNN changed: {section}.{k}')
    return base

def contract(root):
    labels,lf,folds,ff=validate_baseline(root)
    active=labels[labels.included_in_npz.eq(1)]
    labeled=labels[labels.label.notna()]
    if (len(labels),len(active),len(labeled),labels.subject.nunique(),labels.groupby(['subject','session']).ngroups)!=(27192,22777,13604,27,62):
        raise ValueError('Expected original 4s: 27192 events, 22777 valid, 13604 labeled, 27 subjects, 62 sessions')
    if labeled.label.value_counts().to_dict()!={0.0:6802,1.0:6802}:raise ValueError('Unexpected RT class counts')
    test=folds[folds.role.eq('test')].groupby('subject').size()
    if set(test.index)!=set(labels.subject) or not test.eq(1).all():raise ValueError('Subjects must appear in exactly one test fold')
    return labels,lf,folds,ff
