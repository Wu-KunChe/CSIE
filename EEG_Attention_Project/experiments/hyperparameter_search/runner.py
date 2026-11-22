"""4 s Full30 randomized hyperparameter search and fixed-parameter 7-group evaluation.

Uses unchanged src.model/src.trainer/src.data_loader/src.preprocessing/src.evaluation.
All splits are SUBJECT-disjoint. Selection folds are separate from the
subjects used for early-stopping within each of the three search fits.

Search across all 27 subjects then reporting original 5-fold scores is a
DEVELOPMENT analysis, NOT a fully independent nested-CV estimate: the held-out
subjects did contribute to the global hyperparameter choice via the search.
"""
from __future__ import annotations

import gc
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from src.config import load_project_config, read_yaml, resolve_project_path
from src.data_loader import iter_session_arrays, read_labeled_table
from src.evaluation import binary_metrics, per_subject_metrics, predict_probabilities, predictions_table, save_evaluation
from src.model import build_model
from src.preprocessing import ChannelwiseStandardizer
from src.trainer import train_binary_classifier, set_random_seed

GROUP_NAMES = ("Full30", "Forehead2", "Frontal2", "Central2", "Temporal2", "Parietal2", "Occipital2")
EXPECTED_SUBJECTS = 27
EXPECTED_SESSIONS = 62


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True), encoding='utf-8')


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def hash_file(path: Path) -> str:
    h = sha256()
    with path.open('rb') as f:
        for piece in iter(lambda: f.read(1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def hash_object(x: Any) -> str:
    return sha256(json.dumps(x, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def load_setup(root: Path) -> dict:
    shared = load_project_config(root)
    tune = read_yaml(root / 'configs' / 'hyperparameter_search.yaml')
    if int(tune['study']['window_seconds']) != 4 or int(tune['study']['sampling_rate_hz']) != 250:
        raise ValueError('This runner is deliberately limited to 4 seconds, 250 Hz')
    if tune['study']['source_key'] != 'X_B0_uV':
        raise ValueError('Source key must be preprocessed B0 X_B0_uV')
    if int(tune['study']['search_folds']) != 3:
        raise ValueError('Expected 3 search folds')
    if int(tune['final_evaluation']['subject_folds']) != 5:
        raise ValueError('Expected 5 original outer folds')
    if tune['study']['selection_metric'] != 'mean_score_balanced_accuracy':
        raise ValueError('Unsupported search metric')
    groups_conf = shared['channels']['E1']
    canonical = [str(c).upper() for c in shared['channels']['canonical_30']]
    groups: dict[str, list[str]] = {}
    for name in tune['final_evaluation']['channel_groups']:
        if name not in groups_conf:
            raise ValueError(f'Missing channel group in channels.yaml: {name}')
        entry = groups_conf[name]
        channels = canonical if entry == 'all' else [str(c).upper() for c in entry]
        if not channels or len(set(channels)) != len(channels) or not set(channels).issubset(canonical):
            raise ValueError(f'Invalid channel list for {name}: {channels}')
        groups[name] = channels
    if tuple(groups) != GROUP_NAMES or len(groups['Full30']) != 30:
        raise ValueError('Final evaluation must use the original seven groups including canonical Full30')
    if shared['baseline']['model']['name'].lower() != 'cnn1d':
        raise ValueError('Expected existing CNN1D baseline')
    return dict(root=root, baseline=shared['baseline'], tuning=tune, groups=groups,
                epochs=resolve_project_path(root, tune['paths']['epochs_dir']),
                labels_csv=resolve_project_path(root, tune['paths']['labels_csv']),
                folds_csv=resolve_project_path(root, tune['paths']['subject_folds_csv']),
                out=resolve_project_path(root, tune['paths']['results_dir']))


def verify_inputs(ctx: dict, smoke: bool=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    for key in ('labels_csv', 'folds_csv'):
        if not ctx[key].is_file():
            raise FileNotFoundError(f'Cannot find {key}: {ctx[key]}')
    if not ctx['epochs'].is_dir():
        raise FileNotFoundError(f'Cannot find epochs directory: {ctx["epochs"]}')
    labeled = read_labeled_table(ctx['labels_csv'])
    all_labels = pd.read_csv(ctx['labels_csv'], usecols=['window_seconds', 'labeling_method'])
    if (not smoke and len(all_labels) != 27192) or not all_labels['window_seconds'].eq(4).all():
        raise ValueError('Expected 27,192 total 4-second trial records in current study labels')
    if not all_labels['labeling_method'].eq('session_RT_rank_0to30_vs_70to100').all():
        raise ValueError('Expected NEW window_length/4sec RT0-30 labels, not legacy RT5-35')
    if labeled.subject.nunique() != EXPECTED_SUBJECTS or labeled.groupby(['subject', 'session']).ngroups != EXPECTED_SESSIONS:
        raise ValueError('Current 4s dataset expected 27 subjects, 62 sessions')
    if (not smoke and len(labeled) != 13604) or set(labeled.label) != {0, 1}:
        raise ValueError('Expected 13,604 labeled epochs with both classes in this dataset')
    folds = pd.read_csv(ctx['folds_csv'])
    required = {'fold','subject','role'}
    if not required.issubset(folds):
        raise ValueError(f'Folds file missing columns: {required-set(folds)}')
    if folds.duplicated(['fold','subject']).any():
        raise ValueError('Duplicate fold/subject in subject_folds.csv')
    if set(folds.fold.unique()) != {1,2,3,4,5}:
        raise ValueError('Original folds file must contain folds 1 through 5')
    if not folds.role.isin(['train','val','test']).all():
        raise ValueError('Unexpected subject fold role')
    all_subjects = set(labeled.subject.unique())
    for i in range(1, 6):
        f = folds.loc[folds.fold == i]
        if set(f.subject) != all_subjects or set(f.role) != {'train','val','test'}:
            raise ValueError(f'Original outer fold {i} missing subjects or role')
        for role in ('train','val','test'):
            ids = labeled[labeled.subject.isin(f.loc[f.role == role,'subject'])]
            if set(ids.label) != {0,1}:
                raise ValueError(f'Outer fold {i} {role}: missing classes')
    test_counts = folds.loc[folds.role == 'test'].groupby('subject').size()
    if not set(test_counts.index) == all_subjects or not test_counts.eq(1).all():
        raise ValueError('Every subject must be outer test exactly once')
    files = sorted(labeled.npz_file.unique())
    if len(files) != 62:
        raise ValueError('Expected 62 unique NPZ files')
    missing = [f for f in files if not (ctx['epochs']/f).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} session NPZs: {missing[:3]}')
    with np.load(ctx['epochs']/files[0], allow_pickle=False) as npz:
        if 'channel_names' not in npz or 'X_B0_uV' not in npz or 'trial_index' not in npz:
            raise ValueError('First NPZ missing required arrays')
        names = [str(x).strip().upper() for x in npz['channel_names'].tolist()]
        if names != ctx['groups']['Full30']:
            raise ValueError('First NPZ canonical 30-channel order differs from channels.yaml')
        if npz['X_B0_uV'].ndim != 3 or npz['X_B0_uV'].shape[1:] != (30,1000):
            raise ValueError(f'First NPZ B0 shape mismatch: {npz["X_B0_uV"].shape}')
    return labeled, folds


def _cast_sample(value: Any, typ: str, rng: np.random.Generator):
    if typ == 'log_uniform':
        a,b = float(value['low']), float(value['high'])
        if a <= 0 or b <= a:
            raise ValueError('Invalid log_uniform range')
        return float(np.exp(rng.uniform(np.log(a), np.log(b))))
    if typ == 'uniform':
        a,b = float(value['low']), float(value['high'])
        if b <= a:
            raise ValueError('Invalid uniform range')
        return float(rng.uniform(a,b))
    if typ == 'categorical':
        return value['choices'][int(rng.integers(0, len(value['choices'])))]
    if typ == 'randint_inclusive':
        return int(rng.integers(int(value['low']), int(value['high'])+1))
    raise ValueError(f'Unexpected sampling type: {typ}')


def sample_candidates(tune: dict, n: int, baseline: dict) -> list[dict]:
    rng = np.random.default_rng(int(tune['study']['search_seed']))
    space = tune['search_space']
    names = ('learning_rate','weight_decay','dropout','conv_filters','kernel_1',
             'kernel_2','batch_size','early_stopping_patience')
    if set(names) != set(space):
        raise ValueError(f'Wrong search space names: {set(space)^set(names)}')
    candidates = []
    seen = set()
    if bool(tune['study'].get('include_baseline_as_one_candidate', True)):
        # Count the original baseline as ONE of 50, not three additional fits.
        b_model=baseline['model']; b_train=baseline['training']
        reference=dict(id='trial_001',
                       learning_rate=float(b_train['learning_rate']),
                       weight_decay=float(b_train['weight_decay']),
                       dropout=float(b_model['dropout']),
                       conv_filters=[int(x) for x in b_model['conv_filters']],
                       kernels=[int(x) for x in b_model['kernels']],
                       batch_size=int(b_train['batch_size']),
                       early_stopping_patience=int(b_train['early_stopping_patience']))
        candidates.append(reference)
        seen.add(hash_object({k:v for k,v in reference.items() if k!='id'}))
    while len(candidates) < n:
        x = {k: _cast_sample(space[k],space[k]['type'],rng) for k in names}
        x['conv_filters'] = [int(v) for v in x['conv_filters']]
        x['kernels'] = [int(x.pop('kernel_1')),int(x.pop('kernel_2'))]
        x['batch_size'] = int(x['batch_size'])
        x['early_stopping_patience'] = int(x['early_stopping_patience'])
        x['id'] = f"trial_{len(candidates)+1:03d}"
        sig = hash_object({k:v for k,v in x.items() if k != 'id'})
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(x)
    return candidates


def make_inner_roles(labeled: pd.DataFrame, seed: int, val_fraction: float) -> pd.DataFrame:
    if not (0 < val_fraction < 0.5):
        raise ValueError('Expected 0 < early_stop_subject_fraction < .5')
    subjects = sorted(labeled.subject.unique())
    # GroupKFold approximately balances 4s *trial counts*, not just subject counts.
    splitter = GroupKFold(n_splits=3)
    records = []
    for i,(train_idx,score_idx) in enumerate(splitter.split(labeled, groups=labeled.subject),1):
        pool_sub = np.asarray(sorted(labeled.iloc[train_idx].subject.unique()))
        score_sub = set(labeled.iloc[score_idx].subject.unique())
        inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed+i)
        a,b = next(inner.split(pool_sub, groups=pool_sub))
        fit_sub = set(pool_sub[a]); val_sub = set(pool_sub[b])
        if fit_sub & val_sub or fit_sub & score_sub or val_sub & score_sub:
            raise RuntimeError('Subject leakage in search fold')
        for subject in subjects:
            role = 'score' if subject in score_sub else ('early_stop' if subject in val_sub else 'train')
            records.append({'search_fold':i, 'subject':subject, 'role':role,
                            'n_labeled':int(labeled.subject.eq(subject).sum())})
    result = pd.DataFrame(records)
    for i in [1,2,3]:
        subset = result[result.search_fold == i]
        if len(subset) != len(subjects) or set(subset.role) != {'train','early_stop','score'}:
            raise RuntimeError('Invalid search split')
        for role in ('train','early_stop','score'):
            subs=set(subset.loc[subset.role==role,'subject'])
            if set(labeled.loc[labeled.subject.isin(subs),'label']) != {0,1}:
                raise RuntimeError(f'Search fold {i}/{role} has one class')
    if not result.loc[result.role=='score'].groupby('subject').size().eq(1).all():
        raise RuntimeError('Every subject must be scored exactly once in 3-fold CV')
    return result


def build_fingerprint(ctx: dict, candidates: list[dict], inner_roles: pd.DataFrame) -> dict:
    root = ctx['root']
    file_paths = [root/'configs'/'hyperparameter_search.yaml',root/'configs'/'baseline.yaml',
                  root/'configs'/'channels.yaml',root/'experiments'/'hyperparameter_search'/'runner.py',
                  root/'scripts'/'run_hyperparameter_search.py',
                  *sorted((root/'src').glob('*.py')),
                  ctx['labels_csv'],ctx['folds_csv']]
    source = {str(p.relative_to(root)): hash_file(p) for p in file_paths}
    npz_files = sorted(ctx['epochs'].glob('*_unlabeled.npz'))
    return dict(protocol='4sec-full30-random-search-v1',
                files_sha256=source,
                epochs_dir=str(ctx['epochs'].resolve()),
                npz_file_stats=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in npz_files],
                candidates=candidates,
                inner_roles=inner_roles[['search_fold','subject','role']].to_dict(orient='records'))


def write_or_verify_manifest(out: Path, fingerprint: dict, mode: str) -> str:
    sig = hash_object(fingerprint)
    location = out / 'experiment_manifest.json'
    if location.is_file():
        prev = read_json(location)
        if prev.get('fingerprint_hash') != sig:
            raise RuntimeError(f'Experiment config/data/code changed since last run: {out}. '
                               'Restore original settings or archive/rename this output folder before rerunning.')
        if prev.get('run_mode') != mode:
            raise RuntimeError('Refusing to mix real and smoke runs')
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(location, {'fingerprint_hash':sig, 'run_mode':mode,
                              'fingerprint':fingerprint})
    return sig


class EEGMemmapDataset(Dataset):
    def __init__(self, X: np.ndarray, indices: np.ndarray, labels: np.ndarray, scaler: ChannelwiseStandardizer):
        self.X = X
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = labels[self.indices].astype(np.int64, copy=False)
        self.mu = scaler.mean_.reshape(-1,1)
        self.sd = scaler.std_.reshape(-1,1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x = self.X[int(self.indices[index])]
        x = np.asarray((x-self.mu)/self.sd,dtype=np.float32)
        return torch.from_numpy(x.copy()), torch.tensor(float(self.y[index]),dtype=torch.float32)


def stage_data(labeled: pd.DataFrame, directory: Path, channels: list[str], epochs: Path) -> tuple[Path,pd.DataFrame,np.ndarray]:
    directory.mkdir(parents=True,exist_ok=True)
    out = directory/'data.npy'
    n = len(labeled)
    store = np.lib.format.open_memmap(out,mode='w+',dtype='float32',shape=(n,len(channels),1000))
    meta_parts=[]; ys=[]; pos=0
    try:
        for X,y,meta in iter_session_arrays(labeled,epochs,channels,source_key='X_B0_uV',expected_samples=1000):
            nxt = pos+len(y)
            store[pos:nxt]=X
            pos=nxt
            meta_parts.append(meta)
            ys.append(y)
        store.flush()
    finally:
        del store
    if pos != n or not ys:
        raise RuntimeError(f'Memmap staging mismatch {pos} vs {n}')
    meta = pd.concat(meta_parts,ignore_index=True)
    targets = np.concatenate(ys).astype(np.int64)
    if not np.array_equal(meta.label.to_numpy(dtype=int),targets):
        raise ValueError('Staged labels and metadata mismatch')
    return out,meta,targets


def indices_for(meta: pd.DataFrame, roles: pd.DataFrame, fold: int, role: str, phase: str) -> np.ndarray:
    key = 'search_fold' if phase=='search' else 'fold'
    subs = set(roles.loc[(roles[key]==fold)&(roles.role==role),'subject'])
    if not subs:
        raise ValueError(f'Missing {phase} fold{fold}/{role}')
    return np.flatnonzero(meta.subject.isin(subs).to_numpy())


def device_name(which: str) -> str:
    if which=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return 'cuda' if which=='cuda' or (which=='auto' and torch.cuda.is_available()) else 'cpu'


def completed(folder: Path, sig: str, phase: str) -> bool:
    required = ['done.json','metrics.json','training_history.csv','scaler.json','run_config.json']
    if phase=='final':
        required += ['best_model.pt','predictions.csv','per_subject_metrics.csv']
    if not all((folder/f).is_file() for f in required):
        return False
    try:
        return read_json(folder/'done.json').get('signature')==sig
    except Exception:
        return False


def write_done(folder: Path, signature: str) -> None:
    tmp = folder/'done.json.tmp'
    write_json(tmp, {'signature':signature})
    os.replace(tmp,folder/'done.json')


def train_one(stage: Path, meta: pd.DataFrame, targets: np.ndarray, roles: pd.DataFrame,
              fold: int, phase: str, group: str, channels: list[str], base: dict,
              params: dict, folder: Path, signature: str, device: str, smoke: bool) -> dict:
    if completed(folder, signature, phase):
        print(f'  [SKIP] {folder} already complete',flush=True)
        return read_json(folder/'metrics.json')
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'run_intent.json',{'signature':signature})

    X = np.load(stage,mmap_mode='r',allow_pickle=False)
    fit_role='train'; val_role='early_stop' if phase=='search' else 'val'
    eval_role='score' if phase=='search' else 'test'
    fit_ids=indices_for(meta,roles,fold,fit_role,phase)
    stop_ids=indices_for(meta,roles,fold,val_role,phase)
    eval_ids=indices_for(meta,roles,fold,eval_role,phase)
    if min(len(fit_ids),len(stop_ids),len(eval_ids)) == 0:
        raise ValueError('Empty train/validation/evaluation split')
    for ix in (fit_ids,stop_ids,eval_ids):
        if set(targets[ix].tolist())!={0,1}:
            raise ValueError('Fold has only one label')
    if set(fit_ids)&set(stop_ids) or set(fit_ids)&set(eval_ids) or set(stop_ids)&set(eval_ids):
        raise RuntimeError('Row overlap in subject-wise split')
    rng_seed = int(base['training'].get('random_seed',42)) + fold
    set_random_seed(rng_seed)
    scaler=ChannelwiseStandardizer(float(base['preprocessing']['std_epsilon']))
    scaler.fit_from_batches(X[fit_ids[i:i+64]] for i in range(0,len(fit_ids),64))
    ds_fit=EEGMemmapDataset(X,fit_ids,targets,scaler)
    ds_val=EEGMemmapDataset(X,stop_ids,targets,scaler)
    ds_eval=EEGMemmapDataset(X,eval_ids,targets,scaler)
    is_cuda=device=='cuda'
    b=int(params['batch_size'])
    loader_args={'num_workers':0,'pin_memory':is_cuda}
    fit_loader=DataLoader(ds_fit,batch_size=b,shuffle=True,
                          generator=torch.Generator().manual_seed(rng_seed),**loader_args)
    val_loader=DataLoader(ds_val,batch_size=b,shuffle=False,**loader_args)
    eval_loader=DataLoader(ds_eval,batch_size=b,shuffle=False,**loader_args)
    model_cfg=dict(base['model'])
    model_cfg.update(conv_filters=params['conv_filters'],kernels=params['kernels'],dropout=params['dropout'])
    train_cfg=dict(base['training'])
    train_cfg.update(learning_rate=params['learning_rate'],weight_decay=params['weight_decay'],
                     batch_size=b,early_stopping_patience=params['early_stopping_patience'],
                     random_seed=rng_seed)
    if smoke:
        train_cfg['max_epochs']=2
        train_cfg['early_stopping_patience']=2
    net=build_model(model_cfg,n_channels=len(channels),seed=rng_seed)
    checkpoint_path=folder/'best_model.pt' if phase=='final' else None
    result=train_binary_classifier(net,fit_loader,val_loader,train_cfg,
                                    checkpoint_path=checkpoint_path,device=device)
    proba=predict_probabilities(result['model'],eval_loader,device=device)
    threshold=float(train_cfg.get('decision_threshold',0.5))
    metrics=binary_metrics(targets[eval_ids],proba,threshold=threshold)
    metrics.update(phase=phase,fold=fold,group=group,
                   n_train=len(fit_ids),n_early_stop=len(stop_ids),n_evaluation=len(eval_ids),
                   best_epoch=int(result['best_epoch']),best_val_loss=float(result['best_val_loss']),
                   train_positive_weight=float(result['train_positive_weight']),
                   smoke_test=smoke,random_seed=rng_seed)
    # Save train/validation history for every fit; no redundant 150 search checkpoints.
    pd.DataFrame(result['history']).to_csv(folder/'training_history.csv',index=False)
    write_json(folder/'scaler.json',scaler.to_metadata())
    write_json(folder/'run_config.json',{'signature':signature,'phase':phase,'fold':fold,
              'channels':channels,'hyperparameters':params,'model_config':model_cfg,
              'training_config':train_cfg,'source':'X_B0_uV','window_seconds':4,
              'selection_role':eval_role,'early_stop_role':val_role,
              'train_only_normalization':True})
    if phase=='final':
        predictions=predictions_table(meta.iloc[eval_ids],targets[eval_ids],proba,threshold)
        save_evaluation(folder,metrics,predictions,per_subject_metrics(predictions))
    else:
        write_json(folder/'metrics.json',metrics)
    write_done(folder,signature)
    print(f'  [{phase.upper()} fold={fold}] train={len(fit_ids)} early_stop={len(stop_ids)} '
          f'{eval_role}={len(eval_ids)} BA={metrics["balanced_accuracy"]:.4f} '
          f'F1={metrics["macro_f1"]:.4f} epoch={metrics["best_epoch"]}',flush=True)
    del result,net,fit_loader,val_loader,eval_loader,ds_fit,ds_val,ds_eval,X
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()
    return metrics


def summarize_search(out: Path, candidates: list[dict], signature: str,
                     required_folds: tuple[int,...]) -> tuple[pd.DataFrame,dict|None]:
    rows=[]
    for c in candidates:
        id=c['id']
        f_metrics=[]
        for fold in required_folds:
            path=out/'search'/id/f'fold_{fold}'
            sig=hash_object((signature,'search',id,fold,c))
            if completed(path,sig,'search'):
                f_metrics.append(read_json(path/'metrics.json'))
        row={'candidate_id':id,'completed_folds':len(f_metrics),
             'expected_folds':len(required_folds),'learning_rate':c['learning_rate'],
             'weight_decay':c['weight_decay'],'dropout':c['dropout'],
             'filters_1':c['conv_filters'][0],'filters_2':c['conv_filters'][1],
             'kernel_1':c['kernels'][0],'kernel_2':c['kernels'][1],
             'batch_size':c['batch_size'],'patience':c['early_stopping_patience'],
             'mean_score_BA':float(np.mean([m['balanced_accuracy'] for m in f_metrics])) if f_metrics else np.nan,
             'std_score_BA':float(np.std([m['balanced_accuracy'] for m in f_metrics],ddof=1)) if len(f_metrics)>1 else np.nan,
             'mean_score_macro_F1':float(np.mean([m['macro_f1'] for m in f_metrics])) if f_metrics else np.nan}
        rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'search_summary.csv',index=False,encoding='utf-8-sig')
    if not frame.completed_folds.eq(len(required_folds)).all():
        return frame,None
    ranked=frame.sort_values(['mean_score_BA','mean_score_macro_F1','candidate_id'],ascending=[False,False,True]).copy()
    ranked.insert(0, 'rank', np.arange(1, len(ranked)+1))
    ranked.to_csv(out/'search_ranked.csv',index=False,encoding='utf-8-sig')
    winner=next(c for c in candidates if c['id']==ranked.iloc[0]['candidate_id'])
    best={'candidate_id':winner['id'],'hyperparameters':{k:v for k,v in winner.items() if k!='id'},
          'mean_search_score_BA':float(ranked.iloc[0]['mean_score_BA']),
          'std_search_score_BA':float(ranked.iloc[0]['std_score_BA']),
          'mean_search_score_macro_F1':float(ranked.iloc[0]['mean_score_macro_F1']),
          'n_candidates':len(candidates),'n_search_fits':len(candidates)*len(required_folds),
          'search_protocol_signature':signature,
          'selection_rule':'highest unweighted mean of three subject-heldout BA scores, tie-break macroF1',
          'data_reuse_warning':'Subjects used in 3-fold search also appear in final 5-fold: development comparison, not independent generalization estimate.'}
    write_json(out/'best_params.json',best)
    return frame,best


def summarize_final(out: Path, signature: str, groups: dict, roles: pd.DataFrame,
                    labeled: pd.DataFrame,best: dict,folds: tuple[int,...]) -> bool:
    records=[]; subject_records=[]; fold_records=[]
    all_complete=True
    for group,ch in groups.items():
        each=[]; predictions=[]
        for fold in folds:
            folder=out/'final'/group/f'fold_{fold}'
            sig=hash_object((signature,'final',best['candidate_id'],group,fold,best['hyperparameters']))
            if not completed(folder,sig,'final'):
                all_complete=False
                continue
            m=read_json(folder/'metrics.json')
            pred=pd.read_csv(folder/'predictions.csv',encoding='utf-8-sig')
            # Check predictions against exact held-out events and their binary labels.
            held_subs=set(roles.loc[(roles.fold==fold)&(roles.role=='test'),'subject'])
            expected=labeled.loc[labeled.subject.isin(held_subs)]
            keys=['npz_file','x_row']
            if len(pred)!=len(expected) or pred.duplicated(keys).any():
                raise ValueError(f'Unexpected test prediction coverage {folder}')
            merged=pred[keys+['y_true']].merge(expected[keys+['label']],on=keys,
                                                how='left',indicator=True,validate='one_to_one')
            if not merged._merge.eq('both').all() or not merged.y_true.eq(merged.label).all():
                raise ValueError(f'Predictions/labels mismatched at {folder}')
            if not pred.subject.isin(held_subs).all():
                raise ValueError(f'Subject leakage in test predictions at {folder}')
            re_score=binary_metrics(pred.y_true,pred.prob_label1,float(m['threshold']))
            if not math.isclose(re_score['balanced_accuracy'],float(m['balanced_accuracy']),abs_tol=1e-10):
                raise ValueError(f'Metrics mismatch at {folder}')
            each.append(m)
            predictions.append(pred.assign(fold=fold))
            fold_records.append({'group':group,**m})
        if not each:
            continue
        all_pred=pd.concat(predictions,ignore_index=True)
        pooled=binary_metrics(all_pred.y_true,all_pred.prob_label1,0.5)
        summary=dict(group=group,completed_folds=len(each),expected_folds=len(folds),
                     mean_BA=float(np.mean([m['balanced_accuracy'] for m in each])),
                     std_BA=float(np.std([m['balanced_accuracy'] for m in each],ddof=1)) if len(each)>1 else np.nan,
                     pooled_BA=pooled['balanced_accuracy'],
                     mean_macro_F1=float(np.mean([m['macro_f1'] for m in each])),
                     n_predictions=len(all_pred))
        records.append(summary)
        folder=out/'final'/group
        pd.DataFrame(each).to_csv(folder/'fold_summary.csv',index=False,encoding='utf-8-sig')
        all_pred.to_csv(folder/'combined_predictions.csv',index=False,encoding='utf-8-sig')
        write_json(folder/'combined_metrics.json',pooled)
        sub=per_subject_metrics(all_pred)
        sub.insert(0,'group',group)
        sub.to_csv(folder/'per_subject_summary.csv',index=False,encoding='utf-8-sig')
        subject_records.append(sub)
    if records:
        final=out/'final'; final.mkdir(parents=True,exist_ok=True)
        table=pd.DataFrame(records)
        table.to_csv(final/'summary.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(fold_records).to_csv(final/'fold_summary.csv',index=False,encoding='utf-8-sig')
        if subject_records:
            pd.concat(subject_records,ignore_index=True).to_csv(final/'subject_summary.csv',index=False,encoding='utf-8-sig')
        if 'Full30' in set(table.group):
            full=float(table.loc[table.group=='Full30','mean_BA'].iloc[0])
            comparison=table.copy()
            comparison['BA_delta_vs_Full30_pp']=(comparison.mean_BA-full)*100
            comparison['BA_retained_vs_Full30_pct']=100*comparison.mean_BA/full if full>0 else np.nan
            # Retention of above-chance discrimination: avoid interpreting 50% as useful signal.
            comparison['chance_corrected_retention_pct']=(
                100*(comparison.mean_BA-0.5)/(full-0.5) if full>0.5 else np.nan)
            comparison.to_csv(final/'channel_comparison.csv',index=False,encoding='utf-8-sig')
            if all_complete:
                _plot_final(comparison,final/'channel_comparison.png')
    return all_complete


def _plot_final(frame: pd.DataFrame,path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('[PLOT] matplotlib absent, metrics still saved')
        return
    ordered=frame.set_index('group').reindex([g for g in GROUP_NAMES if g in frame.group.values])
    fig,ax=plt.subplots(figsize=(10,5.5))
    x=np.arange(len(ordered))
    mean=ordered.mean_BA.to_numpy()*100
    sd=ordered.std_BA.fillna(0).to_numpy()*100
    bars=ax.bar(x,mean,yerr=sd,capsize=4)
    ax.set_ylabel('Held-out subject-wise BA (%)')
    ax.set_xticks(x,ordered.index,rotation=20,ha='right')
    ax.axhline(50,ls='--',linewidth=1,alpha=.55)
    ax.set_ylim(0,max(100,float(np.max(mean+sd))+8))
    ax.set_title('4-second tuned CNN: Full30 versus 2-channel groups')
    for bar,v,e in zip(bars,mean,sd):
        ax.annotate(f'{v:.2f}%',(bar.get_x()+bar.get_width()/2,v+e),xytext=(0,5),
                    textcoords='offset points',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def limited_smoke_rows(labeled: pd.DataFrame) -> pd.DataFrame:
    return labeled.groupby(['subject','label'],sort=True).head(3).reset_index(drop=True)


def run(project_root: str|Path, *, device: str='auto',check: bool=False,
        smoke: bool=False,phase: str='all') -> None:
    root=Path(project_root).expanduser().resolve()
    ctx=load_setup(root)
    labeled,outer=verify_inputs(ctx,smoke=smoke)
    tune=ctx['tuning']
    count=int(tune['study']['n_random_candidates'])
    if count!=50 and not smoke:
        print(f'[NOTE] Config currently specifies {count} candidates, not standard 50')
    candidates=sample_candidates(tune, count, ctx['baseline'])
    inner=make_inner_roles(labeled,int(tune['study']['search_seed']),
                           float(tune['study']['early_stop_subject_fraction']))
    print(f'4sec EEG / {len(labeled):,} labeled trials / {labeled.subject.nunique()} subjects')
    reference = int(bool(tune['study'].get('include_baseline_as_one_candidate',True)))
    print(f'Search: {count} candidates ({reference} original baseline + {count-reference} random) x 3 scoring folds = {count*3} fits')
    print(f'Final: {len(ctx["groups"])} groups x 5 outer folds = {len(ctx["groups"])*5} CNN fits')
    print(f'Planned total: {count*3+len(ctx["groups"])*5} fits, training 4s B0 EEG')
    print(f'Search scorer is not used for early stopping, each scorer subject appears exactly once.')
    print(f'Fixed test subjects in each final fold follow EXISTING data/splits/subject_folds.csv')
    print(f'Output: {ctx["out"]}')
    if check:
        print('[CHECK OK] No training, no files modified')
        return
    if phase not in ('all','search','final'):
        raise ValueError('Bad phase')
    if device not in ('auto','cuda','cpu'):
        raise ValueError('Bad device')
    device=device_name(device)
    if smoke:
        if phase!='all':
            raise ValueError('Smoke-test supports --phase all only')
        labeled=limited_smoke_rows(labeled)
        candidates=candidates[:1]
        ctx['out']=ctx['out'].with_name(ctx['out'].name+'_SMOKE')
        # All 3 search folds, then two outer groups, one fold, for end-to-end testing.
        final_groups={k:ctx['groups'][k] for k in ('Full30','Forehead2')}
        final_folds=(1,)
        search_folds=(1,2,3)
        print('[SMOKE] Using 2 training epochs, 1 candidate x 3 search folds + 2 final fits')
    else:
        final_groups=ctx['groups'];final_folds=(1,2,3,4,5);search_folds=(1,2,3)

    fingerprint=build_fingerprint(ctx,candidates,inner)
    fingerprint['mode']='smoke' if smoke else 'formal'
    signature=write_or_verify_manifest(ctx['out'],fingerprint,fingerprint['mode'])
    ctx['out'].mkdir(parents=True,exist_ok=True)
    inner.to_csv(ctx['out']/'search_subject_roles.csv',index=False,encoding='utf-8-sig')
    candidate_table=pd.DataFrame(candidates)
    candidate_table.insert(1,'source', ['original_baseline' if i==0 and bool(tune['study'].get('include_baseline_as_one_candidate',True)) else 'random' for i in range(len(candidates))])
    candidate_table.to_csv(ctx['out']/'search_candidates.csv',index=False,encoding='utf-8-sig')
    params=None
    if phase in ('all','search'):
        with tempfile.TemporaryDirectory(prefix='eeg_search_full30_',
                                          dir=ctx['out']) as tmp:
            stage,meta,targets=stage_data(labeled,Path(tmp),ctx['groups']['Full30'],ctx['epochs'])
            for ci,c in enumerate(candidates,1):
                print(f'\n[SEARCH] candidate {ci}/{len(candidates)}: {c["id"]} '
                      f'filters={c["conv_filters"]} kernels={c["kernels"]} '
                      f'LR={c["learning_rate"]:.3g} WD={c["weight_decay"]:.3g} '
                      f'dropout={c["dropout"]:.3f}',flush=True)
                for fold in search_folds:
                    folder=ctx['out']/'search'/c['id']/f'fold_{fold}'
                    sig=hash_object((signature,'search',c['id'],fold,c))
                    train_one(stage,meta,targets,inner,fold,'search','Full30',ctx['groups']['Full30'],
                              ctx['baseline'],c,folder,sig,device,smoke)
                # Save interim state after each candidate to make progress inspectable.
                summarize_search(ctx['out'],candidates,signature,search_folds)
            del stage,meta,targets
            gc.collect()
    frame,best=summarize_search(ctx['out'],candidates,signature,search_folds)
    if best is None:
        if phase=='final':
            raise RuntimeError('Search not complete; run --phase search first')
        print('[INCOMPLETE] Search is incomplete; final evaluation was not started')
        return
    if phase=='search':
        print(f'[SEARCH DONE] Best candidate: {best["candidate_id"]}, '
              f'inner score BA={best["mean_search_score_BA"]:.4f}')
        return
    print(f'\n[BEST] {best["candidate_id"]}: search BA={best["mean_search_score_BA"]:.4f}, '
          f'params={best["hyperparameters"]}',flush=True)
    for name,channels in final_groups.items():
        print(f'\n[FINAL] {name}: 5-fold original outer test split',flush=True)
        with tempfile.TemporaryDirectory(prefix=f'eeg_final_{name}_',dir=ctx['out']) as tmp:
            stage,meta,targets=stage_data(labeled,Path(tmp),channels,ctx['epochs'])
            for fold in final_folds:
                folder=ctx['out']/'final'/name/f'fold_{fold}'
                sig=hash_object((signature,'final',best['candidate_id'],name,fold,best['hyperparameters']))
                train_one(stage,meta,targets,outer,fold,'final',name,channels,
                          ctx['baseline'],best['hyperparameters'],folder,sig,device,smoke)
            del stage,meta,targets
            gc.collect()
        complete=summarize_final(ctx['out'],signature,final_groups,outer,
                                 labeled,best,final_folds)
        print(f'  [FINAL PROGRESS] {name} done; summary partial={not complete}',flush=True)
    complete=summarize_final(ctx['out'],signature,final_groups,outer,labeled,best,final_folds)
    report=[f'4 SECOND FULL30 RANDOM SEARCH + FIXED CNN CHANNEL ABLATION',
            f'Random candidates: {len(candidates)}, search folds: {len(search_folds)}',
            f'Search fits: {len(candidates)*len(search_folds)}',
            f'Final fits: {len(final_groups)*len(final_folds)}',
            f'Best candidate: {best["candidate_id"]}',
            f'Search scoring BA (subject-wise, epoch selection separate): {best["mean_search_score_BA"]:.6f}',
            f'Search best parameters: {json.dumps(best["hyperparameters"])}',
            f'Final all combinations complete: {complete}',
            f'WARNING: Search across all subjects and final test on the same cohort is development-phase model selection,',
            f'NOT independent test generalization; use independent data or nested CV for final unbiased claim.']
    (ctx['out']/'report.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'\n[DONE] Results in {ctx["out"]}. Completed={complete}.')
