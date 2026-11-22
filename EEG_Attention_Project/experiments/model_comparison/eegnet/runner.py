"""Independent EEGNet comparison runner; never modifies B0, E1, E2, E3.

Data, labels, subject splits and metrics are borrowed from existing src modules.
Neural architecture and fitting loop are EEGNet-specific, intentionally separate.
Outputs live ONLY in results/model_comparison/EEGNet/.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import gc
import json
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.config import load_project_config, read_yaml, resolve_project_path
from src.data_loader import read_labeled_table, select_fold_rows, iter_session_arrays
from src.preprocessing import ChannelwiseStandardizer
from src.evaluation import (binary_metrics, predict_probabilities,
                            predictions_table, per_subject_metrics, save_evaluation)
from src.trainer import set_random_seed
from .model import build_eegnet


def _json_save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _hash_file(path: Path) -> str:
    h = sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _channels(cfg: dict, group: str) -> list[str]:
    mapping = cfg['channels']['E1']
    if group not in mapping:
        raise ValueError(f'Unknown E1 channel group {group}')
    x = mapping[group]
    names = ([str(ch).upper() for ch in cfg['channels']['canonical_30']]
             if x == 'all' else [str(ch).upper() for ch in x])
    if not names or len(names) != len(set(names)):
        raise ValueError(f'Invalid channel group {group}')
    return names


def _protocol_signature(root: Path, cfg: dict, eegnet_cfg: dict,
                        labels_csv: Path, folds_csv: Path, smoke: bool) -> str:
    sources = [root/'src'/'data_loader.py', root/'src'/'preprocessing.py',
               root/'src'/'evaluation.py', root/'src'/'trainer.py',
               root/'experiments'/'model_comparison'/'eegnet'/'model.py',
               root/'experiments'/'model_comparison'/'eegnet'/'runner.py']
    payload = {
        'labels_sha256': _hash_file(labels_csv),
        'folds_sha256': _hash_file(folds_csv),
        'baseline_training': cfg['baseline']['training'],
        'baseline_input': cfg['baseline']['input'],
        'baseline_preprocessing': cfg['baseline']['preprocessing'],
        'eegnet_settings': eegnet_cfg,
        'channels': cfg['channels']['E1'],
        'source_hashes': {p.relative_to(root).as_posix():_hash_file(p) for p in sources},
        'smoke': smoke,
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _fold_signature(protocol: str, group: str, fold: int, channels: list[str]) -> str:
    return sha256(json.dumps({'protocol':protocol, 'group':group, 'fold':fold,
                              'channels':channels}, sort_keys=True).encode()).hexdigest()


def _verify_folds(labels: pd.DataFrame, folds: pd.DataFrame,
                  n_folds: int) -> list[int]:
    if not {'subject', 'fold', 'role'}.issubset(folds.columns):
        raise ValueError('Missing fold/role/subject column from subject_folds.csv')
    if folds.duplicated(['subject','fold']).any():
        raise ValueError('Duplicate subject/fold assignments')
    if not folds.role.isin(['train','val','test']).all():
        raise ValueError('Unknown fold role')
    found = sorted(int(v) for v in folds.fold.unique())
    if len(found) != n_folds:
        raise ValueError(f'Expected {n_folds} folds; found {found}')
    subjects = set(labels.subject.unique())
    for fold in found:
        f = folds.loc[folds.fold == fold]
        if set(f.subject) != subjects or set(f.role) != {'train','val','test'}:
            raise ValueError(f'Fold {fold}: missing subjects/roles')
        for role in ('train','val','test'):
            part = select_fold_rows(labels, folds, fold, role)
            if not len(part) or set(part.label.unique()) != {0,1}:
                raise ValueError(f'Fold {fold}/{role}: empty or single-class')
    if not folds[folds.role.eq('test')].groupby('subject').size().eq(1).all():
        raise ValueError('Each subject must belong to exactly one outer TEST fold')
    return found


def _validate_reference(labels: pd.DataFrame, folds: pd.DataFrame,
                        ref_dir: Path, groups: list[str], warn_absent: bool = True) -> None:
    """Check E1 reference predictions have identical epoch identities and labels."""
    for group in groups:
        ref_file = ref_dir/group/'combined_predictions.csv'
        if not ref_file.exists():
            if warn_absent:
                print(f'[REFERENCE] {group}: E1 combined_predictions.csv not found; '
                      'will report EEGNet alone.', flush=True)
            continue
        ref = pd.read_csv(ref_file, encoding='utf-8-sig')
        keys = ['npz_file','x_row']
        required = keys + ['fold','y_true','subject','session','trial_index']
        if not set(required).issubset(ref.columns) or ref.duplicated(keys).any():
            raise ValueError(f'E1 reference invalid: {ref_file}')
        expected = labels[keys + ['label','subject','session','trial_index']]
        matched = ref[keys + ['fold','y_true','subject','session','trial_index']].merge(
            expected, on=keys, how='left', suffixes=('', '_expected'),
            validate='one_to_one', indicator=True)
        if (len(matched) != len(expected) or not matched._merge.eq('both').all()
                or not matched.y_true.eq(matched.label).all()
                or any(not matched[col].eq(matched[f'{col}_expected']).all()
                       for col in ['subject','session','trial_index'])):
            raise ValueError(f'E1 reference uses different RT labels/epochs: {ref_file}. '
                             'Re-run E1 with matching labels or omit the old comparison.')
        test_folds = folds.loc[folds.role.eq('test'), ['subject','fold']]
        checked = matched.merge(test_folds, on='subject', how='left',
                                validate='many_to_one', suffixes=('', '_expected'))
        if not checked.fold.eq(checked.fold_expected).all():
            raise ValueError(f'E1 reference has different test-subject folds: {ref_file}')


@dataclass
class Staging:
    file: Path
    labels: np.ndarray
    meta: pd.DataFrame
    shape: tuple[int,int,int]


def _stage_epochs(labels: pd.DataFrame, epochs_dir: Path,
                  channels: list[str], output: Path, samples: int) -> Staging:
    """Write one on-disk float32 NPY array. Never load 30ch epochs at once."""
    count = len(labels)
    dst = np.lib.format.open_memmap(output, mode='w+', dtype=np.float32,
                                    shape=(count, len(channels), samples))
    parts, all_y, offset = [], [], 0
    for X, y, metadata in iter_session_arrays(
            labels, epochs_dir, channels, source_key='X_B0_uV',
            expected_samples=samples):
        dst[offset:offset+len(y)] = X
        offset += len(y)
        parts.append(metadata)
        all_y.append(y)
    dst.flush()
    del dst
    if count != offset:
        raise RuntimeError(f'Expected {count} epochs, staged {offset}')
    meta = pd.concat(parts, ignore_index=True)
    y = np.concatenate(all_y).astype(np.int64)
    if not np.array_equal(meta.label.to_numpy(), y):
        raise ValueError('Staging EEG/metadata row mismatch')
    return Staging(output, y, meta, (count, len(channels), samples))


class EEGMemmapDataset(Dataset):
    def __init__(self, memmap: np.ndarray, indices: np.ndarray,
                 all_y: np.ndarray, scaler: ChannelwiseStandardizer):
        self.memmap = memmap
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = np.asarray(all_y[self.indices], dtype=np.int64)
        self.mean = scaler.mean_.reshape(-1, 1)
        self.std = scaler.std_.reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        x = self.memmap[int(self.indices[index])]
        # Standardization parameters fitted ONLY on train subjects.
        scaled = (x - self.mean) / self.std
        return torch.from_numpy(np.array(scaled, dtype=np.float32, copy=True)), \
            torch.tensor(float(self.y[index]), dtype=torch.float32)


def _indices(meta: pd.DataFrame, folds: pd.DataFrame, fold: int, role: str) -> np.ndarray:
    subjects = set(folds.loc[folds.fold.eq(fold) & folds.role.eq(role), 'subject'])
    return np.flatnonzero(meta.subject.isin(subjects).to_numpy())


def _smoke_labels(labels: pd.DataFrame, folds: pd.DataFrame, fold: int) -> pd.DataFrame:
    subjects = set(folds.loc[folds.fold.eq(fold), 'subject'])
    subset = labels.loc[labels.subject.isin(subjects)].copy()
    # Keep a few samples per class/subject for a quick pipeline test.
    subset = subset.groupby(['subject','label'], sort=True).head(2)
    return subset.reset_index(drop=True)


def _is_complete(path: Path, signature: str) -> bool:
    required = ['run_config.json','metrics.json','predictions.csv',
                'per_subject_metrics.csv','best_model.pt','training_history.csv',
                'scaler.json']
    if not all((path/file).is_file() for file in required):
        return False
    try:
        return _json_load(path/'run_config.json').get('signature') == signature
    except (OSError, ValueError):
        return False


def _prepare_fold(path: Path, signature: str, force: bool) -> bool:
    if path.exists() and _is_complete(path, signature) and not force:
        print(f'  [SKIP] {path.parent.name}/{path.name}: completed', flush=True)
        return False
    if path.exists() and any(path.iterdir()) and not force:
        intent = path/'run_intent.json'
        same = intent.exists() and _json_load(intent).get('signature') == signature
        if not same:
            raise RuntimeError(f'Existing EEGNet result has different settings: {path}. '
                               'Back it up and run with --force to overwrite EEGNet only.')
        print(f'  [RETRY] Interrupted EEGNet fold {path}', flush=True)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    _json_save(path/'run_intent.json', {'signature': signature})
    return True


def _fit(model: nn.Module, train: DataLoader, valid: DataLoader,
         opts: dict, device: torch.device, output: Path) -> dict:
    model.to(device)
    optimizer_name = str(opts.get('optimizer','AdamW')).lower()
    if optimizer_name != 'adamw':
        raise ValueError('Only AdamW supported in this EEGNet comparison')
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=float(opts['learning_rate']), weight_decay=float(opts.get('weight_decay',1e-4)))
    y_train = np.asarray(train.dataset.y)
    pos, neg = int((y_train==1).sum()), int((y_train==0).sum())
    if min(pos,neg) < 1:
        raise ValueError('Training subjects missing a class')
    weight = neg/pos if opts.get('class_weighting','train_only') == 'train_only' else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    epochs = int(opts['max_epochs'])
    patience = int(opts['early_stopping_patience'])
    if epochs < 1 or patience < 1:
        raise ValueError('Epoch count and patience must be positive')
    history = []
    best_loss, best_epoch, best_state, stale = float('inf'), 0, None, 0
    for epoch in range(1,epochs+1):
        model.train()
        train_sum, n_train = 0.0, 0
        for X, y in train:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X).reshape(-1), y.reshape(-1))
            loss.backward()
            optimizer.step()
            model.constrain_weights()
            train_sum += float(loss.item())*len(y)
            n_train += len(y)
        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for X, y in valid:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                loss = criterion(model(X).reshape(-1), y.reshape(-1))
                val_sum += float(loss.item())*len(y)
                n_val += len(y)
        tl, vl = train_sum/n_train, val_sum/n_val
        history.append({'epoch':epoch,'train_loss':tl,'val_loss':vl})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f'    Epoch {epoch:2d}/{epochs}: train {tl:.4f} | val {vl:.4f}',flush=True)
        if vl < best_loss-1e-7:
            best_loss, best_epoch, best_state, stale = vl, epoch, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                print(f'    Early stopping at epoch {epoch}, best={best_epoch}',flush=True)
                break
    model.load_state_dict(best_state)
    torch.save({'model_state_dict':best_state,'best_epoch':best_epoch,
                'best_val_loss':best_loss,'n_channels':model.n_channels,
                'n_times':model.n_times}, output/'best_model.pt')
    pd.DataFrame(history).to_csv(output/'training_history.csv', index=False)
    return {'best_epoch':best_epoch,'best_val_loss':best_loss,
            'positive_weight':float(weight),'device':str(device),'history':history}


def _train_fold(staging: Staging, folds: pd.DataFrame, fold: int,
                group: str, channel_names: list[str],
                eegnet_cfg: dict, train_cfg: dict, output: Path,
                signature: str, device: str) -> None:
    arr = np.load(staging.file, mmap_mode='r',allow_pickle=False)
    train_ix = _indices(staging.meta, folds,fold,'train')
    val_ix = _indices(staging.meta,folds,fold,'val')
    test_ix = _indices(staging.meta,folds,fold,'test')
    for role, ix in [('train',train_ix),('val',val_ix),('test',test_ix)]:
        if len(ix)==0 or set(staging.labels[ix]) != {0,1}:
            raise ValueError(f'Fold {fold}/{role} has no samples or a missing class')
    print(f'  [FOLD {fold}] train={len(train_ix)} val={len(val_ix)} test={len(test_ix)}',flush=True)
    scaler = ChannelwiseStandardizer(float(eegnet_cfg['_std_epsilon']))
    scaler.fit_from_batches(arr[train_ix[i:i+128]] for i in range(0,len(train_ix),128))
    datasets = [EEGMemmapDataset(arr, idx, staging.labels, scaler)
                for idx in (train_ix,val_ix,test_ix)]
    n_batch = int(train_cfg['batch_size'])
    if n_batch < 2:
        raise ValueError('Batch size must be at least 2 for EEGNet BatchNorm')
    is_cuda = device=='cuda' or (device=='auto' and torch.cuda.is_available())
    device_t = torch.device('cuda' if is_cuda else 'cpu')
    seed = int(train_cfg['random_seed']) + fold
    set_random_seed(seed)
    args={'num_workers':0,'pin_memory':is_cuda}
    loaders = [DataLoader(d,batch_size=n_batch,shuffle=(j==0),
                          generator=torch.Generator().manual_seed(seed+j),**args)
               for j,d in enumerate(datasets)]
    opts=dict(train_cfg)
    opts['random_seed']=seed
    model_cfg={k:v for k,v in eegnet_cfg['model'].items() if k!='variant'}
    model = build_eegnet(model_cfg, len(channel_names), staging.shape[2])
    fit = _fit(model,loaders[0],loaders[1],opts,device_t,output)
    prob = predict_probabilities(model,loaders[2],device=device_t)
    thr = float(eegnet_cfg['report']['decision_threshold'])
    metrics = binary_metrics(staging.labels[test_ix],prob,thr)
    metrics.update({'group':group,'fold':fold,'n_train':len(train_ix),'n_val':len(val_ix),
                    'best_epoch':int(fit['best_epoch']),
                    'best_val_loss':float(fit['best_val_loss']),
                    'train_positive_weight':fit['positive_weight'],'model':'EEGNet'})
    preds = predictions_table(staging.meta.iloc[test_ix],staging.labels[test_ix],prob,thr)
    preds.insert(0,'fold',fold)
    save_evaluation(output,metrics,preds,per_subject_metrics(preds,thr))
    _json_save(output/'scaler.json',scaler.to_metadata())
    _json_save(output/'run_config.json',{'signature':signature,'group':group,'fold':fold,
          'model':'EEGNet','channels':channel_names,'source':'X_B0_uV',
          'train_only_scaling':True,'best_epoch':fit['best_epoch'],
          'best_val_loss':fit['best_val_loss'],'device':str(device_t)})
    print(f'  [DONE] {group} fold {fold}: BA={metrics["balanced_accuracy"]:.4f} '
          f'Macro-F1={metrics["macro_f1"]:.4f}',flush=True)
    del model,fit,loaders,datasets,arr
    gc.collect()
    if is_cuda: torch.cuda.empty_cache()


def _check_prediction_against_labels(pred: pd.DataFrame, labels: pd.DataFrame,
                                     folds: pd.DataFrame, fold: int,
                                     partial: bool = False) -> None:
    keys=['npz_file','x_row']
    required=keys+['subject','session','trial_index','y_true','prob_label1','fold']
    if not set(required).issubset(pred) or pred.duplicated(keys).any():
        raise ValueError('Invalid/duplicated prediction records')
    expected=select_fold_rows(labels,folds,fold,'test')
    expected=expected[keys+['subject','session','trial_index','label']]
    checked=pred[required].merge(expected,on=keys,how='left',
                                  indicator=True,validate='one_to_one',
                                  suffixes=('','_expected'))
    if (not partial and len(pred)!=len(expected)) or not checked._merge.eq('both').all():
        raise ValueError(f'Fold {fold}: saved predictions do not match selected test set')
    if (not checked.y_true.eq(checked.label).all() or
            not checked.fold.eq(fold).all() or
            any(not checked[col].eq(checked[f'{col}_expected']).all()
                for col in ('subject','session','trial_index'))):
        raise ValueError(f'Fold {fold}: saved labels or subjects differ from current data')


def _read_existing_predictions(output: Path, labels: pd.DataFrame, folds: pd.DataFrame,
                               fold: int, expected_signature: str,
                               smoke: bool) -> tuple[dict,pd.DataFrame] | None:
    if not _is_complete(output,expected_signature):
        return None
    metrics=_json_load(output/'metrics.json')
    pred=pd.read_csv(output/'predictions.csv',encoding='utf-8-sig')
    _check_prediction_against_labels(pred,labels,folds,fold,partial=smoke)
    if metrics['fold'] != fold:
        raise ValueError(f'Fold number mismatch in {output}')
    return metrics,pred


def _confusion_file(path: Path, scores: dict) -> None:
    pd.DataFrame([[scores['tn'],scores['fp']],[scores['fn'],scores['tp']]],
                 index=['true_label0','true_label1'],
                 columns=['pred_label0','pred_label1']).to_csv(path)


def _read_old_e1_summary(ref_dir: Path, group: str, completed: list[int]) -> dict | None:
    f = ref_dir/group/'fold_summary.csv'
    if not f.is_file():
        return None
    d = pd.read_csv(f,encoding='utf-8-sig')
    if 'fold' not in d or not set(completed).issubset(set(d.fold.astype(int))):
        return None
    d=d.loc[d.fold.isin(completed)]
    for name in ('balanced_accuracy','macro_f1'):
        if name not in d:
            return None
    return {'mean_BA':float(d.balanced_accuracy.mean()),
            'mean_macro_F1':float(d.macro_f1.mean()),
            'std_BA':float(d.balanced_accuracy.std(ddof=1)) if len(d)>1 else 0.0}


def _make_plots(base: Path, group_rows: pd.DataFrame,
                comparisons: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not group_rows.empty:
        fig,ax=plt.subplots(figsize=(9.5,5.3))
        xs=np.arange(len(group_rows))
        means=group_rows.mean_BA.to_numpy()*100
        sds=group_rows.std_BA.fillna(0).to_numpy()*100
        bars=ax.bar(xs,means,yerr=sds,capsize=5,width=.58)
        ax.set_xticks(xs,group_rows.group.to_numpy())
        ax.set_ylim(0,max(75,float(np.max(means+sds)+11)))
        ax.set_ylabel('Mean Balanced Accuracy (%)')
        ax.set_title('EEGNet: five-fold BA by channel group')
        ax.axhline(50,ls='--',lw=1,color='gray')
        for bar,v,err in zip(bars,means,sds):
            ax.text(bar.get_x()+bar.get_width()/2, v+err+1.3,f'{v:.2f}%',
                    ha='center',va='bottom',fontsize=10)
        fig.tight_layout()
        fig.savefig(base/'eegnet_channels.png',dpi=180)
        plt.close(fig)

    if not comparisons.empty:
        fig,ax=plt.subplots(figsize=(10,5.7))
        x=np.arange(len(comparisons))
        width=.34
        ref=comparisons.cnn_mean_BA.to_numpy()*100
        net=comparisons.eegnet_mean_BA.to_numpy()*100
        ba=ax.bar(x-width/2,ref,width,label='1D-CNN (existing E1)',
                  yerr=comparisons.cnn_std_BA.to_numpy()*100,capsize=4)
        bb=ax.bar(x+width/2,net,width,label='EEGNet (new)',
                  yerr=comparisons.eegnet_std_BA.to_numpy()*100,capsize=4)
        ax.set_xticks(x,comparisons.group)
        ax.set_ylabel('Mean Balanced Accuracy (%)')
        ax.set_title('Matched RT labels and subject folds: 1D-CNN vs EEGNet')
        ax.set_ylim(0,max(75,float(max(np.max(ref),np.max(net))+12)))
        ax.axhline(50,ls='--',lw=1,color='gray')
        ax.legend(loc='upper left')
        for bar_set,values,errs in ((ba,ref,comparisons.cnn_std_BA*100),
                                    (bb,net,comparisons.eegnet_std_BA*100)):
            for bar,v,err in zip(bar_set,values,errs):
                ax.text(bar.get_x()+bar.get_width()/2, v+float(err)+1.1,
                        f'{v:.2f}%',ha='center',va='bottom',fontsize=9)
        fig.tight_layout()
        fig.savefig(base/'comparison.png',dpi=180)
        plt.close(fig)


def _aggregate(base: Path, ref_dir: Path, group_map: dict[str,list[str]],
               all_folds: list[int], signatures: dict[tuple[str,int],str],
               labels: pd.DataFrame, folds: pd.DataFrame, smoke: bool) -> None:
    base.mkdir(parents=True,exist_ok=True)
    overview=[]
    comparisons=[]
    for group,chs in group_map.items():
        group_dir=base/group
        loaded=[]
        for fold in all_folds:
            found=_read_existing_predictions(group_dir/f'fold_{fold}',labels,folds,
                                              fold,signatures[(group,fold)],smoke)
            if found is not None:
                loaded.append((fold,*found))
        if not loaded:
            print(f'[PENDING] {group}: no completed folds',flush=True)
            continue
        joined=pd.concat([pr for _,_,pr in loaded],ignore_index=True)
        folds_done=[fold for fold,_,_ in loaded]
        if joined.duplicated(['npz_file','x_row']).any():
            raise ValueError(f'{group}: repeated held-out test epoch across folds')
        if not smoke and len(loaded)==len(all_folds) and len(joined)!=len(labels):
            raise ValueError(f'{group}: complete folds do not cover all labeled epochs')
        thr=float(loaded[0][1]['threshold'])
        pooled=binary_metrics(joined.y_true,joined.prob_label1,thr)
        scores=pd.DataFrame([{'fold':fold, **metrics} for fold,metrics,_ in loaded])
        a=float(scores.balanced_accuracy.mean())
        sd=float(scores.balanced_accuracy.std(ddof=1)) if len(scores)>1 else 0.0
        row={'group':group,'channels':','.join(chs),
             'folds_completed':len(loaded),'folds_expected':len(all_folds),
             'all_folds_complete':len(loaded)==len(all_folds) and not smoke,
             'mean_BA':a,'std_BA':sd,
             'mean_macro_F1':float(scores.macro_f1.mean()),
             'std_macro_F1':float(scores.macro_f1.std(ddof=1)) if len(scores)>1 else 0.0,
             'pooled_BA':pooled['balanced_accuracy'],
             'pooled_macro_F1':pooled['macro_f1'],
             'pooled_specificity_label0':pooled['specificity_label0'],
             'pooled_sensitivity_label1':pooled['sensitivity_label1'],
             'tn':pooled['tn'],'fp':pooled['fp'],
             'fn':pooled['fn'],'tp':pooled['tp'],
             'total_test_predictions':len(joined),
             'n_test_subjects':joined.subject.nunique()}
        overview.append(row)
        group_dir.mkdir(parents=True,exist_ok=True)
        scores.to_csv(group_dir/'fold_summary.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([row]).to_csv(group_dir/'summary.csv',index=False,encoding='utf-8-sig')
        joined.to_csv(group_dir/'combined_predictions.csv',index=False,encoding='utf-8-sig')
        per_subject_metrics(joined,thr).to_csv(group_dir/'per_subject_metrics.csv',
                                               index=False,encoding='utf-8-sig')
        _json_save(group_dir/'combined_metrics.json',{'group':group, **row,
                                                         'pooled_metrics':pooled})
        _confusion_file(group_dir/'confusion_matrix.csv',pooled)
        (group_dir/'report.txt').write_text(
            f'EEGNet {group} | {len(loaded)}/{len(all_folds)} folds\n'
            f'Mean BA: {a*100:.2f}% +/- {sd*100:.2f} pp (fold SD)\n'
            f'Pooled BA: {pooled["balanced_accuracy"]*100:.2f}%\n'
            f'Mean Macro-F1: {row["mean_macro_F1"]*100:.2f}%\n'
            f'Pooled Macro-F1: {row["pooled_macro_F1"]*100:.2f}%\n'
            f'RT label is a relative fast/slow response proxy, not ground-truth attention.\n',
            encoding='utf-8')
        old=_read_old_e1_summary(ref_dir,group,folds_done)
        if old is not None and not smoke:
            comparisons.append({'group':group,'folds_compared':len(folds_done),
                'cnn_mean_BA':old['mean_BA'],'cnn_std_BA':old['std_BA'],
                'eegnet_mean_BA':a,'eegnet_std_BA':sd,
                'delta_BA_EEGNet_minus_CNN':a-old['mean_BA'],
                'cnn_mean_macro_F1':old['mean_macro_F1'],
                'eegnet_mean_macro_F1':row['mean_macro_F1']})
        print(f'[REPORT] {group}: mean BA={a*100:.2f}%  '
              f'pooled BA={pooled["balanced_accuracy"]*100:.2f}%  '
              f'folds={len(loaded)}/{len(all_folds)}',flush=True)
    summary=pd.DataFrame(overview)
    summary.to_csv(base/'summary.csv',index=False,encoding='utf-8-sig')
    if overview:
        pd.concat([pd.read_csv(base/group/'fold_summary.csv') for group in summary.group],
                  ignore_index=True).to_csv(base/'fold_summary.csv',index=False,
                                            encoding='utf-8-sig')
    comps=pd.DataFrame(comparisons)
    comps.to_csv(base/'model_comparison.csv',index=False,encoding='utf-8-sig')
    if not summary.empty:
        _make_plots(base,summary,comps)
    if (not comps.empty and {'Full30','Forehead2'}.issubset(comps.group)):
        a=comps.set_index('group')
        for small in ('Forehead2','Frontal2'):
            if small in a.index:
                delta_cnn=float(a.loc['Full30','cnn_mean_BA']-a.loc[small,'cnn_mean_BA'])
                delta_net=float(a.loc['Full30','eegnet_mean_BA']-a.loc[small,'eegnet_mean_BA'])
                print(f'[GAP] Full30 minus {small}: 1D-CNN {delta_cnn*100:.2f} pp; '
                      f'EEGNet {delta_net*100:.2f} pp',flush=True)
    (base/'report.txt').write_text(
        'EEGNet Model Comparison: independent from B0/E1/E2/E3.\n'
        'summary.csv: per-group five-fold means and pooled metrics.\n'
        'model_comparison.csv: paired fold-average difference vs pre-existing E1 1D-CNN.\n'
        'comparison.png: E1 1D-CNN vs EEGNet, when E1 reference exists.\n'
        'eegnet_channels.png: EEGNet-only channel comparison with numeric labels.\n'
        'All metrics refer to RT percentile proxy labels, not ground-truth attention.\n',
        encoding='utf-8')


def run_eegnet(project_root: str | Path, *, groups: Sequence[str] | None = None,
               selected_folds: Sequence[int] | None = None, device: str = 'auto',
               check: bool = False, smoke_test: bool = False,
               summarize_only: bool = False, force: bool = False) -> None:
    root=Path(project_root).expanduser().resolve()
    cfg=load_project_config(root)
    eegnet_cfg=read_yaml(root/'configs'/'eegnet.yaml')
    paths=cfg['paths']
    labels_csv=resolve_project_path(root,paths['labels_csv'])
    folds_csv=resolve_project_path(root,paths['subject_folds_csv'])
    epochs_dir=resolve_project_path(root,paths['epochs_dir'])
    output=resolve_project_path(root,eegnet_cfg['experiment']['output_dir'])
    ref_dir=resolve_project_path(root,eegnet_cfg['experiment']['reference_e1_dir'])
    if check and (summarize_only or smoke_test or force):
        raise ValueError('--check cannot be combined with --summarize-only/--smoke-test/--force')
    if summarize_only and (smoke_test or force):
        raise ValueError('--summarize-only cannot be combined with --smoke-test/--force')
    if device not in {'auto','cuda','cpu'}:
        raise ValueError('Unknown device: choose auto, cuda, cpu')
    if device=='cuda' and not check and not summarize_only and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')
    if eegnet_cfg['experiment']['source_key']!='X_B0_uV':
        raise ValueError('Only existing B0 data X_B0_uV supported')
    groups_defined=list(eegnet_cfg['experiment']['groups'])
    chosen=groups_defined if not groups else list(dict.fromkeys(groups))
    if not chosen or set(chosen)-set(groups_defined):
        raise ValueError(f'Invalid group selection {chosen}; allowed {groups_defined}')
    group_map={name:_channels(cfg,name) for name in chosen}
    labels=read_labeled_table(labels_csv)
    folds=pd.read_csv(folds_csv)
    all_folds=_verify_folds(labels,folds,int(cfg['baseline']['splits']['outer_folds']))
    requested=all_folds if not selected_folds else sorted(set(int(n) for n in selected_folds))
    if not set(requested).issubset(all_folds):
        raise ValueError(f'Invalid folds {requested}; allowed {all_folds}')
    if smoke_test:
        if groups is None:
            chosen=['Forehead2']
            group_map={name:_channels(cfg,name) for name in chosen}
        if selected_folds is None:
            requested=[all_folds[0]]
        if len(requested) != 1:
            raise ValueError('--smoke-test only supports one fold; choose --fold 1')
        output=output.parent/'_smoke_test'/'EEGNet'
    if not smoke_test:
        # Every configured E1 reference must match; prevent partial group runs
        # from reporting comparisons based on a stale labeling experiment.
        _validate_reference(labels,folds,ref_dir,groups_defined)
    elif check:
        raise ValueError('Smoke + check are mutually exclusive')
    model_params=eegnet_cfg['model']
    samples=int(cfg['baseline']['input']['expected_samples'])
    # Verify network can actually instantiate at both 2ch and 30ch widths.
    for group,channel_names in group_map.items():
        m=build_eegnet(model_params,len(channel_names),samples)
        with torch.no_grad():
            m.eval()
            if tuple(m(torch.zeros(2,len(channel_names),samples)).shape)!=(2,):
                raise ValueError(f'EEGNet forward shape failed for {group}')
        del m
    print(f'Project: {root}\nLabeled EEG epochs: {len(labels):,}  '
          f'Subjects: {labels.subject.nunique()}\n'
          f'Model: EEGNet | Groups: {chosen} | Folds: {requested}',flush=True)
    if 'labeling_method' in labels.columns:
        print('RT label method:',', '.join(sorted(labels.labeling_method.dropna().unique())),flush=True)
    if check or not summarize_only:
        if not epochs_dir.is_dir():
            raise FileNotFoundError(f'Missing EEG Epoch folder: {epochs_dir} '
                                    '(edit configs/paths.yaml on your machine)')
        missing=[f for f in labels.npz_file.unique() if not (epochs_dir/f).is_file()]
        if missing:
            raise FileNotFoundError(f'Missing {len(missing)} EEG archives; e.g. {missing[0]}')
        with np.load(epochs_dir/labels.npz_file.iloc[0],allow_pickle=False) as one:
            if 'channel_names' not in one or 'X_B0_uV' not in one:
                raise ValueError('B0 data not found in NPZ')
            available=set(str(x).upper() for x in one['channel_names'].tolist())
            needed=set(x for chs in group_map.values() for x in chs)
            if not needed.issubset(available) or one['X_B0_uV'].shape[2]!=samples:
                raise ValueError('Mismatch in channel names / expected time samples')
    if check:
        print('[CHECK OK] Existing NPZ, RT labels, subjects, five folds, '
              'model input shapes, and matching old E1 reference verified. '
              'No EEGNet training.',flush=True)
        return
    if not eegnet_cfg.get('training',{}).get('inherit_from_baseline',True):
        raise ValueError('This comparison must inherit the B0 training protocol')
    train_cfg=dict(cfg['baseline']['training'])
    train_cfg.update(eegnet_cfg['training'].get('overrides',{}) or {})
    model_options=dict(eegnet_cfg)
    model_options['_std_epsilon']=float(cfg['baseline']['preprocessing']['std_epsilon'])
    signature=_protocol_signature(root,cfg,eegnet_cfg,labels_csv,folds_csv,smoke_test)
    # Always summarize ALL configured groups, including earlier partial --group runs.
    # Only the selected groups are trained; existing EEGNet groups are read-only.
    full_map={name:_channels(cfg,name) for name in groups_defined}
    report_map=group_map if smoke_test else full_map
    signatures={(group,fold):_fold_signature(signature,group,fold,report_map[group])
                for group in report_map for fold in all_folds}
    if summarize_only:
        print('[REPORT ONLY] Reading completed EEGNet folds, no training.',flush=True)
        _aggregate(output,ref_dir,report_map,all_folds,signatures,labels,folds,smoke=False)
        return
    output.mkdir(parents=True,exist_ok=True)
    scratch=output/'.scratch'
    scratch.mkdir(exist_ok=True)
    for group in chosen:
        channels=group_map[group]
        todo=[]
        for fold in requested:
            out=output/group/f'fold_{fold}'
            sig=signatures[(group,fold)]
            if _is_complete(out,sig) and not force:
                print(f'  [SKIP] {group}/fold_{fold} already complete',flush=True)
                continue
            # Do not touch other results: only this EEGNet folder can be replaced.
            if _prepare_fold(out,sig,force):
                todo.append((fold,out,sig))
        if not todo:
            continue
        needed=_smoke_labels(labels,folds,todo[0][0]) if smoke_test else labels
        print(f'[LOAD] {group}: {len(needed):,} epochs, {len(channels)} channels',flush=True)
        with tempfile.TemporaryDirectory(prefix='eegnet_',dir=scratch) as temp:
            stage=_stage_epochs(needed,epochs_dir,channels,Path(temp)/'EEG.npy',samples)
            for fold,out,sig in todo:
                opts=dict(train_cfg)
                if smoke_test:
                    opts['max_epochs']=2
                    opts['early_stopping_patience']=2
                    opts['batch_size']=min(int(opts['batch_size']),32)
                _train_fold(stage,folds,fold,group,channels,model_options,opts,out,
                            sig,device)
            del stage
            gc.collect()
    _aggregate(output,ref_dir,report_map,all_folds,signatures,labels,folds,smoke=smoke_test)
    print(f'[COMPLETE] Results written to: {output}',flush=True)
