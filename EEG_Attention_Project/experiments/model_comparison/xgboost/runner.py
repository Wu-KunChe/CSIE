"""Standalone seven-group XGBoost/Spectral E1-style comparison.

Only ADDS results/model_comparison/XGBoost/; no B0/E1/E2/E3 modifications.
All training/validation/test subjects are read from the pre-existing split CSV.
Epoch-wise deterministic features are cached once for all seven groups;
XGBoost models are trained independently for each (group, fold).
"""
from __future__ import annotations

import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from xgboost import __version__ as xgb_version

from src.config import load_project_config, read_yaml, resolve_project_path
from src.data_loader import read_labeled_table, select_fold_rows, iter_session_arrays
from src.evaluation import binary_metrics, predictions_table, per_subject_metrics, save_evaluation
from .features import FEATURE_NAMES, compute_features, flatten_channel_features
from .model import build_xgboost


GROUPS = ('Full30','Forehead2','Frontal2','Central2','Temporal2','Parietal2','Occipital2')


def _json_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _hash_file(path: Path) -> str:
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def _hash_object(value: dict) -> str:
    return sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def _canonical_channels(cfg: dict) -> list[str]:
    result=[str(x).upper() for x in cfg['channels']['canonical_30']]
    if len(result)!=30 or len(set(result))!=30:
        raise ValueError('Expected exactly 30 distinct scalp EEG channels')
    return result


def _channel_groups(cfg: dict, desired: Sequence[str]) -> dict[str,list[str]]:
    canonical=_canonical_channels(cfg)
    definitions=cfg['channels']['E1']
    result={}
    for name in desired:
        if name not in definitions:
            raise ValueError(f'Missing E1 channel group {name} in channels.yaml')
        v=definitions[name]
        chs=canonical if v=='all' else [str(x).upper() for x in v]
        if not chs or len(chs)!=len(set(chs)) or not set(chs).issubset(canonical):
            raise ValueError(f'Invalid E1 channel group: {name} = {chs}')
        result[name]=chs
    return result


def _verify_folds(labels: pd.DataFrame, folds: pd.DataFrame, n_folds: int) -> list[int]:
    if not {'subject','fold','role'}.issubset(folds.columns):
        raise ValueError('subject_folds.csv must contain subject/fold/role')
    if folds.duplicated(['subject','fold']).any():
        raise ValueError('Duplicated subject/fold assignment')
    if not folds.role.isin(['train','val','test']).all():
        raise ValueError('Unexpected fold role')
    present=sorted(int(v) for v in folds.fold.unique())
    if len(present)!=n_folds:
        raise ValueError(f'Expected {n_folds} folds; found {present}')
    subjects=set(labels.subject.unique())
    for fold in present:
        x=folds.loc[folds.fold.eq(fold)]
        if set(x.subject)!=subjects or set(x.role)!={'train','val','test'}:
            raise ValueError(f'Fold {fold}: incomplete or overlapping subject roles')
        for role in ('train','val','test'):
            sub=select_fold_rows(labels,folds,fold,role)
            if not len(sub) or set(sub.label.unique())!={0,1}:
                raise ValueError(f'Fold {fold}/{role}: empty or single-class')
    if not folds[folds.role.eq('test')].groupby('subject').size().eq(1).all():
        raise ValueError('Every subject must be test exactly once')
    return present


def _selected_indices(meta: pd.DataFrame, folds: pd.DataFrame,
                      fold: int, role: str) -> np.ndarray:
    subjects=set(folds.loc[folds.fold.eq(fold)&folds.role.eq(role),'subject'])
    return np.flatnonzero(meta.subject.isin(subjects).to_numpy())


def _check_reference(labels: pd.DataFrame, folds: pd.DataFrame,
                     ref_dir: Path, groups: Sequence[str]) -> None:
    """Use reference E1 only when each group has exactly the same held-out trials."""
    keys=['npz_file','x_row']
    expected=labels[keys+['label','subject','session','trial_index']]
    test=folds.loc[folds.role.eq('test'),['subject','fold']]
    for group in groups:
        path=ref_dir/group/'combined_predictions.csv'
        if not path.exists():
            print(f'[REFERENCE] {group}: no E1 predictions found; standalone XGBoost still works.')
            continue
        ref=pd.read_csv(path,encoding='utf-8-sig')
        required=set(keys+['y_true','fold','subject','session','trial_index','prob_label1'])
        if not required.issubset(ref.columns) or ref.duplicated(keys).any():
            raise ValueError(f'Invalid E1 reference: {path}')
        checked=ref[list(required)].merge(expected,on=keys,validate='one_to_one',
                                          how='left',suffixes=('','_expected'),indicator=True)
        if (len(checked)!=len(expected) or not checked._merge.eq('both').all()
            or not checked.y_true.eq(checked.label).all()
            or any(not checked[k].eq(checked[f'{k}_expected']).all()
                   for k in ('subject','session','trial_index'))):
            raise ValueError(f'E1/{group} was made with a different RT label version. '
                             'Do not compare stale labels; keep reference E1 and XGBoost separate.')
        checked=checked.merge(test,on='subject',how='left',validate='many_to_one',
                              suffixes=('','_expected'))
        if not checked.fold.eq(checked.fold_expected).all():
            raise ValueError(f'E1/{group} uses different held-out subject folds')


def _feature_cache_signature(root: Path, labels_csv: Path, cfg: dict,
                             epochs_dir: Path, feature_cfg: dict) -> str:
    paths=sorted(set(str(s) for s in pd.read_csv(labels_csv,usecols=['npz_file']).npz_file.dropna()))
    stamps=[]
    for name in paths:
        if Path(name).name!=name:
            raise ValueError(f'Unexpected data filename {name}')
        p=epochs_dir/name
        if not p.is_file():
            raise FileNotFoundError(f'Missing EEG epoch file: {p}')
        stat=p.stat()
        stamps.append((name,stat.st_size,stat.st_mtime_ns))
    return _hash_object({'labels':_hash_file(labels_csv),
                         'features_py':_hash_file(root/'experiments'/'model_comparison'/'xgboost'/'features.py'),
                         'data_loader_py':_hash_file(root/'src'/'data_loader.py'),
                         'features_cfg':feature_cfg,'canonical':_canonical_channels(cfg),
                         'file_stamps':stamps})


def _prepare_cache(labels: pd.DataFrame, cfg: dict, features_cfg: dict,
                   epochs_dir: Path, cache_dir: Path, signature: str,
                   *, force: bool = False) -> tuple[np.ndarray,pd.DataFrame,np.ndarray]:
    """Compute 30ch deterministic spectral features ONE time, session by session."""
    manifest=cache_dir/'metadata.csv'
    feature_file=cache_dir/'features.npy'
    status=cache_dir/'cache_config.json'
    if (not force and all(p.is_file() for p in (manifest,feature_file,status))
        and _json_load(status).get('signature')==signature):
        print('[CACHE] Using previously verified 30ch feature matrix',flush=True)
        meta=pd.read_csv(manifest,encoding='utf-8-sig')
        X=np.load(feature_file,mmap_mode='r',allow_pickle=False)
        if X.shape==(len(labels),30,len(FEATURE_NAMES)) and len(meta)==len(labels):
            expected=labels[['npz_file','x_row','label']].sort_values(
                ['npz_file','x_row'],kind='stable').reset_index(drop=True)
            actual=meta[['npz_file','x_row','label']].sort_values(
                ['npz_file','x_row'],kind='stable').reset_index(drop=True)
            if expected.equals(actual):
                return X,meta,meta.label.to_numpy(np.int64)
        raise ValueError('Feature cache metadata does not match current RT labels; '
                         'move XGBoost/_cache elsewhere to rebuild it')

    cache_dir.mkdir(parents=True,exist_ok=True)
    tmp=cache_dir/'features.partial.npy'
    if tmp.exists():
        tmp.unlink()
    channels=_canonical_channels(cfg)
    samples=int(features_cfg['expected_samples'])
    n=len(labels)
    mapped=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,
                                    shape=(n,len(channels),len(FEATURE_NAMES)))
    parts=[]
    count=0
    try:
        for X,y,meta in iter_session_arrays(labels,epochs_dir,channels,
                  source_key='X_B0_uV',expected_samples=samples):
            batch=int(features_cfg.get('batch_size',128))
            if batch<1: raise ValueError('features.batch_size must be >= 1')
            for j in range(0,len(y),batch):
                chunk=compute_features(X[j:j+batch],features_cfg)
                mapped[count+j:count+j+len(chunk)] = chunk
            count+=len(y)
            parts.append(meta)
            print(f'[FEATURES] {meta.subject.iloc[0]} {meta.session.iloc[0]} '
                  f'({len(y)} epochs, total {count}/{n})',flush=True)
        if count!=n:
            raise ValueError(f'Feature count mismatch: {count} vs {n}')
        mapped.flush()
        del mapped
        complete_meta=pd.concat(parts,ignore_index=True)
        if len(complete_meta)!=n or not np.isfinite(np.load(tmp,mmap_mode='r')).all():
            raise ValueError('Invalid feature matrix')
        complete_meta.to_csv(cache_dir/'metadata.partial.csv',index=False,encoding='utf-8-sig')
        os.replace(tmp,feature_file)
        os.replace(cache_dir/'metadata.partial.csv',manifest)
        _json_save(status,{'signature':signature,'shape':[n,30,len(FEATURE_NAMES)],
                           'feature_names':list(FEATURE_NAMES),
                           'source':'B0 1-30Hz, 250Hz, 2s EEG',
                           'notes':'Independent per-epoch features, not globally standardized.'})
        return np.load(feature_file,mmap_mode='r',allow_pickle=False),complete_meta,complete_meta.label.to_numpy(np.int64)
    except Exception:
        del mapped
        if tmp.exists(): tmp.unlink()
        raise


def _signature(root:Path,labels_csv:Path,folds_csv:Path,
               cfg:dict,xgb_cfg:dict,cache_signature:str,smoke:bool) -> str:
    source_files=[root/'experiments'/'model_comparison'/'xgboost'/n
                  for n in ('model.py','features.py','runner.py')]
    source_files.extend([root/'src'/'data_loader.py',root/'src'/'evaluation.py'])
    return _hash_object({'label_hash':_hash_file(labels_csv),'fold_hash':_hash_file(folds_csv),
       'cache_signature':cache_signature,'model_config':xgb_cfg,
       'channels':cfg['channels']['E1'],
       'model_source_hashes':{p.name+str(i):_hash_file(p) for i,p in enumerate(source_files)},
       'xgboost_version':xgb_version,'smoke':smoke})


def _fold_signature(signature:str,group:str,fold:int,channels:list[str]) -> str:
    return _hash_object({'protocol':signature,'group':group,'fold':fold,'channels':channels})


def _is_complete(path:Path,signature:str)->bool:
    expected=('run_config.json','metrics.json','predictions.csv',
              'per_subject_metrics.csv','best_model.json','feature_importance.csv')
    if not all((path/n).is_file() for n in expected): return False
    try:
        return _json_load(path/'run_config.json').get('signature')==signature
    except (ValueError,OSError):
        return False


def _prepare_fold(path:Path,signature:str,force:bool)->bool:
    if path.is_dir() and _is_complete(path,signature) and not force:
        print(f'  [SKIP] {path.parent.name}/{path.name} already complete',flush=True)
        return False
    if path.exists() and any(path.iterdir()) and not force:
        intent=path/'run_intent.json'
        if not intent.exists() or _json_load(intent).get('signature')!=signature:
            raise RuntimeError(f'Existing XGBoost fold has different settings: {path}. '
                               'Back up XGBoost results and use --force if you want to overwrite it.')
        print(f'  [RETRY] Incomplete fold {path}',flush=True)
    if path.exists(): shutil.rmtree(path)
    path.mkdir(parents=True,exist_ok=True)
    _json_save(path/'run_intent.json',{'signature':signature})
    return True


def _train_fold(X:np.ndarray,meta:pd.DataFrame,labels:np.ndarray,folds:pd.DataFrame,
                fold:int,group:str,cols:list[str],model_cfg:dict,
                threshold:float,output:Path,signature:str,smoke:bool)->None:
    train_ix=_selected_indices(meta,folds,fold,'train')
    val_ix=_selected_indices(meta,folds,fold,'val')
    test_ix=_selected_indices(meta,folds,fold,'test')
    for role,ix in [('train',train_ix),('val',val_ix),('test',test_ix)]:
        if not len(ix) or set(labels[ix])!={0,1}:
            raise ValueError(f'Fold {fold}/{role}: no samples or one-class data')
    print(f'  [FOLD {fold}] train={len(train_ix)} val={len(val_ix)} test={len(test_ix)}; '
          f'{len(cols)} features',flush=True)
    model=build_xgboost(model_cfg,fold,smoke_test=smoke)
    # Early stopping is evaluated only on VALIDATION subjects, never TEST subjects.
    train_df=pd.DataFrame(X[train_ix],columns=cols)
    val_df=pd.DataFrame(X[val_ix],columns=cols)
    test_df=pd.DataFrame(X[test_ix],columns=cols)
    model.fit(train_df,labels[train_ix],eval_set=[(val_df,labels[val_ix])],verbose=False)
    y_prob=model.predict_proba(test_df)[:,1]
    metrics=binary_metrics(labels[test_ix],y_prob,threshold)
    best_iteration=getattr(model,'best_iteration',None)
    metrics.update({'model':'XGBoost','group':group,'fold':int(fold),
                    'n_train':len(train_ix),'n_val':len(val_ix),
                    'best_iteration':int(best_iteration) if best_iteration is not None else None,
                    'n_features':len(cols)})
    preds=predictions_table(meta.iloc[test_ix],labels[test_ix],y_prob,threshold)
    preds.insert(0,'fold',fold)
    save_evaluation(output,metrics,preds,per_subject_metrics(preds,threshold))
    model.save_model(str(output/'best_model.json'))
    booster=model.get_booster()
    # Consistent named feature columns for interpretability. Gain is a split-statistic,
    # not a causal contribution or evidence of a brain region's physiological role.
    gains=booster.get_score(importance_type='total_gain')
    importance=pd.DataFrame({'feature':cols,
        'total_gain':[float(gains.get(c,0.0)) for c in cols]})
    importance.sort_values('total_gain',ascending=False).to_csv(
        output/'feature_importance.csv',index=False,encoding='utf-8-sig')
    _json_save(output/'run_config.json',{'signature':signature,'fold':fold,
        'group':group,'model':'XGBoost + fixed spectral/time EEG features',
        'feature_names':cols,'input':'X_B0_uV','threshold':threshold,
        'best_iteration':int(best_iteration) if best_iteration is not None else None,
        'early_stop_subject_role':'val','scale_method':'none; tree-based estimator',
        'xgboost_version':xgb_version})
    print(f'  [DONE] {group} fold {fold}: BA={metrics["balanced_accuracy"]*100:.2f}%, '
          f'Macro-F1={metrics["macro_f1"]*100:.2f}%',flush=True)
    del model
    gc.collect()


def _load_fold(group_path:Path,fold:int,signature:str,
               labels:pd.DataFrame,folds:pd.DataFrame,smoke:bool)->tuple[dict,pd.DataFrame]|None:
    path=group_path/f'fold_{fold}'
    if not _is_complete(path,signature): return None
    result=_json_load(path/'metrics.json')
    pred=pd.read_csv(path/'predictions.csv',encoding='utf-8-sig')
    keys=['npz_file','x_row']
    required=keys+['fold','y_true','prob_label1','subject','session','trial_index']
    if not set(required).issubset(pred.columns) or pred.duplicated(keys).any():
        raise ValueError(f'Invalid saved XGBoost predictions {path}')
    expected=select_fold_rows(labels,folds,fold,'test')
    if smoke:
        # Subsampled smoke dataset is independently validated by train-time role checks.
        expected=expected.loc[expected.set_index(keys).index.isin(pred.set_index(keys).index)]
    matched=pred[required].merge(expected[keys+['label','subject','session','trial_index']],
                                  on=keys,how='left',indicator=True,
                                  suffixes=('','_expected'),validate='one_to_one')
    if len(matched)!=len(expected) or not matched._merge.eq('both').all() or not matched.y_true.eq(matched.label).all():
        raise ValueError(f'Saved fold {fold} does not match current labels')
    if not matched.fold.eq(fold).all() or any(
        not matched[k].eq(matched[f'{k}_expected']).all()
        for k in ['subject','session','trial_index']):
        raise ValueError(f'Saved fold {fold} does not match current IDs')
    return result,pred


def _reference_fold_metrics(ref_dir:Path,group:str,completed:list[int]) -> pd.DataFrame|None:
    file=ref_dir/group/'combined_predictions.csv'
    if not file.is_file(): return None
    ref=pd.read_csv(file,encoding='utf-8-sig')
    rows=[]
    for fold in completed:
        one=ref.loc[ref.fold.eq(fold)]
        if one.empty: return None
        m=binary_metrics(one.y_true,one.prob_label1)
        rows.append({'fold':fold,'ref_BA':m['balanced_accuracy'],
                     'ref_macro_F1':m['macro_f1']})
    return pd.DataFrame(rows)


def _plot_summary(base:Path,summary:pd.DataFrame,comparison:pd.DataFrame)->None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=list(summary.group)
    xpos=np.arange(len(groups))
    # 0-100% axis prevents emphasizing tiny differences from a truncated baseline.
    fig,ax=plt.subplots(figsize=(12.5,5.6))
    v=summary.mean_BA.to_numpy(float)*100
    e=summary.std_BA.to_numpy(float)*100
    bars=ax.bar(xpos,v,yerr=e,capsize=3,width=.65,label='XGBoost mean BA')
    for bar,value,err in zip(bars,v,e):
        ax.text(bar.get_x()+bar.get_width()/2,value+err+1.2,
                f'{value:.2f}%',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(xpos,groups,rotation=25,ha='right')
    ax.set_ylim(0,100)
    ax.axhline(50,linestyle='--',color='gray',linewidth=1)
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('XGBoost + Spectral Features: Seven E1 Channel Groups')
    ax.legend()
    fig.tight_layout()
    fig.savefig(base/'xgboost_channels.png',dpi=180)
    plt.close(fig)
    if comparison.empty: return
    comparison=comparison.set_index('group').reindex(groups).dropna(subset=['cnn_mean_BA'])
    if comparison.empty: return
    pos=np.arange(len(comparison))
    fig,ax=plt.subplots(figsize=(13,5.8))
    width=.38
    ref=comparison.cnn_mean_BA.to_numpy(float)*100
    new=comparison.xgboost_mean_BA.to_numpy(float)*100
    err_ref=comparison.cnn_std_BA.to_numpy(float)*100
    err_new=comparison.xgboost_std_BA.to_numpy(float)*100
    a=ax.bar(pos-width/2,ref,width,yerr=err_ref,capsize=3,label='1D-CNN (existing E1)')
    b=ax.bar(pos+width/2,new,width,yerr=err_new,capsize=3,label='XGBoost + Spectral')
    for bars,values,errs in [(a,ref,err_ref),(b,new,err_new)]:
        for bar,value,err in zip(bars,values,errs):
            ax.text(bar.get_x()+bar.get_width()/2,value+err+1.1,
                    f'{value:.2f}%',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(pos,list(comparison.index),rotation=25,ha='right')
    ax.set_ylim(0,100)
    ax.axhline(50,linestyle='--',color='gray',linewidth=1)
    ax.set_ylabel('Mean Balanced Accuracy (%)')
    ax.set_title('1D-CNN vs XGBoost: Same RT Labels and Subject Folds')
    ax.legend()
    fig.tight_layout()
    fig.savefig(base/'comparison.png',dpi=180)
    plt.close(fig)


def _write_confusion(path:Path,metrics:dict)->None:
    pd.DataFrame([[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]],
        index=['true_0_fast_RT','true_1_slow_RT'],
        columns=['pred_0_fast_RT','pred_1_slow_RT']).to_csv(path,encoding='utf-8-sig')


def _aggregate(output:Path,ref_dir:Path,groups:dict[str,list[str]],
               folds_all:list[int],signatures:dict[tuple[str,int],str],
               labels:pd.DataFrame,folds:pd.DataFrame,smoke:bool)->None:
    overview=[]
    compares=[]
    for group,chs in groups.items():
        group_dir=output/group
        existing=[]
        for fold in folds_all:
            loaded=_load_fold(group_dir,fold,signatures[(group,fold)],labels,folds,smoke)
            if loaded is not None:
                existing.append((fold,*loaded))
        if not existing:
            print(f'[PENDING] {group}: no completed folds',flush=True)
            continue
        scores=pd.DataFrame([m for _,m,_ in existing])
        merged=pd.concat([p for _,_,p in existing],ignore_index=True)
        if merged.duplicated(['npz_file','x_row']).any():
            raise ValueError(f'Overlapping held-out trials across folds in {group}')
        all_complete=len(existing)==len(folds_all) and not smoke
        if all_complete and len(merged)!=len(labels):
            raise ValueError(f'{group}: five folds do not cover all labeled data')
        pooled=binary_metrics(merged.y_true,merged.prob_label1,
                              threshold=float(scores.threshold.iloc[0]))
        mean=float(scores.balanced_accuracy.mean())
        sd=float(scores.balanced_accuracy.std(ddof=1)) if len(scores)>1 else 0.0
        row={'group':group,'channels':','.join(chs),
            'folds_completed':len(existing),'folds_expected':len(folds_all),
            'all_folds_complete':all_complete,
            'mean_BA':mean,'std_BA':sd,
            'mean_macro_F1':float(scores.macro_f1.mean()),
            'std_macro_F1':float(scores.macro_f1.std(ddof=1)) if len(scores)>1 else 0.0,
            'pooled_BA':pooled['balanced_accuracy'],
            'pooled_macro_F1':pooled['macro_f1'],
            'pooled_specificity_label0':pooled['specificity_label0'],
            'pooled_sensitivity_label1':pooled['sensitivity_label1'],
            'tn':pooled['tn'],'fp':pooled['fp'],'fn':pooled['fn'],'tp':pooled['tp'],
            'total_test_predictions':len(merged),
            'n_test_subjects':merged.subject.nunique(),
            'n_features':len(chs)*len(FEATURE_NAMES)}
        overview.append(row)
        group_dir.mkdir(parents=True,exist_ok=True)
        pd.DataFrame([row]).to_csv(group_dir/'summary.csv',index=False,encoding='utf-8-sig')
        scores.to_csv(group_dir/'fold_summary.csv',index=False,encoding='utf-8-sig')
        # Feature importance is descriptive only; average each fold's
        # normalized total gain to make different model sizes comparable.
        importance_rows=[]
        for fold,_,_ in existing:
            imp_path=group_dir/f'fold_{fold}'/'feature_importance.csv'
            imp=pd.read_csv(imp_path,encoding='utf-8-sig')
            total=float(imp.total_gain.sum())
            imp['normalized_gain']=imp.total_gain/total if total>0 else 0.0
            imp['fold']=fold
            importance_rows.append(imp)
        if importance_rows:
            by_feature=pd.concat(importance_rows,ignore_index=True).groupby('feature',as_index=False).agg(
                mean_normalized_gain=('normalized_gain','mean'),
                std_normalized_gain=('normalized_gain','std'),
                mean_total_gain=('total_gain','mean'))
            by_feature.sort_values('mean_normalized_gain',ascending=False).to_csv(
                group_dir/'feature_importance_summary.csv',index=False,encoding='utf-8-sig')
        merged.to_csv(group_dir/'combined_predictions.csv',index=False,encoding='utf-8-sig')
        per_subject_metrics(merged).to_csv(group_dir/'per_subject_metrics.csv',index=False,encoding='utf-8-sig')
        _json_save(group_dir/'combined_metrics.json',{'group':group,**row,'pooled_metrics':pooled})
        _write_confusion(group_dir/'confusion_matrix.csv',pooled)
        (group_dir/'report.txt').write_text(
            f'XGBoost + Spectral Features | {group} | {len(existing)}/{len(folds_all)} folds\n'
            f'Mean BA: {mean*100:.2f}% +/- {sd*100:.2f} pp (fold SD)\n'
            f'Pooled BA: {pooled["balanced_accuracy"]*100:.2f}%\n'
            f'Mean Macro-F1: {row["mean_macro_F1"]*100:.2f}%\n'
            f'Pooled Macro-F1: {row["pooled_macro_F1"]*100:.2f}%\n'
            'Labels represent within-session fast/slow RT quantiles, not absolute focus.\n',
            encoding='utf-8')
        if not smoke:
            completed=[k for k,_,_ in existing]
            r=_reference_fold_metrics(ref_dir,group,completed)
            if r is not None:
                compares.append({'group':group,'folds_compared':len(completed),
                    'cnn_mean_BA':float(r.ref_BA.mean()),
                    'cnn_std_BA':float(r.ref_BA.std(ddof=1)) if len(r)>1 else 0.0,
                    'xgboost_mean_BA':mean,'xgboost_std_BA':sd,
                    'delta_BA_XGB_minus_CNN':mean-float(r.ref_BA.mean()),
                    'cnn_mean_macro_F1':float(r.ref_macro_F1.mean()),
                    'xgboost_mean_macro_F1':row['mean_macro_F1']})
        print(f'[REPORT] {group}: mean BA={mean*100:.2f}% / pooled={pooled["balanced_accuracy"]*100:.2f}%; '
              f'{len(existing)}/{len(folds_all)} folds',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    summary=pd.DataFrame(overview)
    summary.to_csv(output/'summary.csv',index=False,encoding='utf-8-sig')
    if overview:
        pd.concat([pd.read_csv(output/r['group']/'fold_summary.csv') for r in overview],
                  ignore_index=True).to_csv(output/'fold_summary.csv',index=False,encoding='utf-8-sig')
    comp=pd.DataFrame(compares)
    comp.to_csv(output/'model_comparison.csv',index=False,encoding='utf-8-sig')
    if overview:
        _plot_summary(output,summary,comp)
    (output/'report.txt').write_text(
        'Independent XGBoost Spectral Model Comparison for ALL seven E1 groups.\n'
        'Mean BA: equal weighting of five test folds; Pooled BA: merge their predictions.\n'
        'model_comparison.csv uses EXISTING E1 1D-CNN predictions only when labels match.\n'
        'Predictions are relative RT proxy labels, not ground-truth alertness.\n',encoding='utf-8')
    if not comp.empty and {'Full30','Forehead2'}.issubset(set(comp.group)):
        c=comp.set_index('group')
        for small in ('Forehead2','Frontal2'):
            if small in c.index:
                a=float(c.loc['Full30','cnn_mean_BA']-c.loc[small,'cnn_mean_BA'])
                b=float(c.loc['Full30','xgboost_mean_BA']-c.loc[small,'xgboost_mean_BA'])
                print(f'[GAP] Full30-{small}: CNN={a*100:.2f} pp; XGBoost={b*100:.2f} pp')


def run_xgboost(project_root:str|Path, *, groups:Sequence[str]|None=None,
                selected_folds:Sequence[int]|None=None,check:bool=False,
                smoke_test:bool=False,summarize_only:bool=False,
                force:bool=False)->None:
    root=Path(project_root).expanduser().resolve()
    cfg=load_project_config(root)
    xgb_cfg=read_yaml(root/'configs'/'xgboost.yaml')
    if sum(bool(x) for x in (check,smoke_test,summarize_only))>1:
        raise ValueError('--check/--smoke-test/--summarize-only are mutually exclusive')
    if force and (check or smoke_test or summarize_only):
        raise ValueError('--force is only for formal training')
    configured=xgb_cfg['experiment']['groups']
    if list(configured)!=list(GROUPS):
        raise ValueError('XGBoost config must explicitly contain Full30 plus all six E1 2ch groups')
    if xgb_cfg['experiment']['source_key']!='X_B0_uV':
        raise ValueError('This comparison only supports already-preprocessed B0 epochs')
    chosen=list(dict.fromkeys(groups)) if groups else list(configured)
    if not chosen or not set(chosen).issubset(configured):
        raise ValueError(f'Unknown group(s): {chosen}')
    if smoke_test and groups is None:
        chosen=['Forehead2']
    all_group_map=_channel_groups(cfg,list(configured))
    paths=cfg['paths']
    labels_csv=resolve_project_path(root,paths['labels_csv'])
    folds_csv=resolve_project_path(root,paths['subject_folds_csv'])
    epochs_dir=resolve_project_path(root,paths['epochs_dir'])
    output=resolve_project_path(root,xgb_cfg['experiment']['output_dir'])
    ref_dir=resolve_project_path(root,xgb_cfg['experiment']['reference_e1_dir'])
    if smoke_test:
        output=output.parent/'_smoke_test'/'XGBoost'
    labels=read_labeled_table(labels_csv)
    expected_method=str(xgb_cfg['experiment']['expected_labeling_method'])
    if 'labeling_method' not in labels.columns:
        raise ValueError('Current labels lack labeling_method; cannot verify RT version')
    methods=sorted(labels.labeling_method.dropna().astype(str).unique())
    if methods!=[expected_method] or labels.labeling_method.isna().any():
        raise ValueError(f'RT label method mismatch: found {methods}, expected {expected_method}. '
                         'XGBoost comparison must use the same RT5-35 labels as current E1.')
    folds=pd.read_csv(folds_csv)
    all_folds=_verify_folds(labels,folds,int(cfg['baseline']['splits']['outer_folds']))
    requested=sorted(set(int(x) for x in selected_folds)) if selected_folds else list(all_folds)
    if not set(requested).issubset(all_folds):
        raise ValueError(f'Invalid fold(s) {requested}; allowed {all_folds}')
    if smoke_test and selected_folds is None:
        requested=[all_folds[0]]
    if smoke_test and len(requested)!=1:
        raise ValueError('Smoke test only supports one fold')
    feature_cfg=xgb_cfg['features']
    if int(feature_cfg['sampling_rate_hz'])!=int(cfg['baseline']['input']['sampling_rate_hz']) or \
       int(feature_cfg['expected_samples'])!=int(cfg['baseline']['input']['expected_samples']):
        raise ValueError('Feature settings must match existing B0 window and sample rate')
    if not smoke_test:
        _check_reference(labels,folds,ref_dir,list(configured))
    if check:
        # Read a handful of actual EEG epochs to check channel and data/label alignment.
        one=labels.loc[(labels.subject==labels.subject.iloc[0])&
                       (labels.session==labels.session.iloc[0])].head(3)
        batch=next(iter_session_arrays(one,epochs_dir,_canonical_channels(cfg),
                            source_key='X_B0_uV',expected_samples=int(feature_cfg['expected_samples'])))
        f=compute_features(batch[0],feature_cfg)
        print(f'[CHECK OK] subjects={labels.subject.nunique()}, sessions='
              f'{labels.groupby(["subject","session"]).ngroups}, '
              f'labeled_epochs={len(labels)}, folds={len(all_folds)}, '
              f'channels=30, features_per_channel={len(FEATURE_NAMES)}')
        print(f'  Feature tensor from 3 epochs: {f.shape}; configured groups={len(configured)}')
        return
    if smoke_test:
        # Subset a handful per subject/class while retaining existing fold roles.
        labels=labels.groupby(['subject','label'],sort=True).head(4).reset_index(drop=True)
    # --summarize-only does not read 36GB original data or recompute features.
    if summarize_only:
        # Recompute signature from metadata only; do not reopen the EEG waveforms.
        cache_signature=_feature_cache_signature(root,labels_csv,cfg,epochs_dir,feature_cfg)
        protocol=_signature(root,labels_csv,folds_csv,cfg,xgb_cfg,cache_signature,smoke_test)
        signatures={(g,f):_fold_signature(protocol,g,f,all_group_map[g])
                    for g in configured for f in all_folds}
        _aggregate(output,ref_dir,all_group_map,all_folds,signatures,
                   labels,folds,smoke_test)
        return
    cache_sig=_feature_cache_signature(root,labels_csv,cfg,epochs_dir,feature_cfg)
    protocol=_signature(root,labels_csv,folds_csv,cfg,xgb_cfg,cache_sig,smoke_test)
    signatures={(g,f):_fold_signature(protocol,g,f,all_group_map[g])
                for g in configured for f in all_folds}
    # If every requested model is complete and valid, omit feature building entirely.
    todo=[(g,f) for g in chosen for f in requested
          if force or not _is_complete(output/g/f'fold_{f}',signatures[(g,f)])]
    if todo:
        staging,meta,y=_prepare_cache(labels,cfg,feature_cfg,epochs_dir,output/'_cache',
                                      cache_sig,force=False)
        for group in chosen:
            channels=all_group_map[group]
            idx=[_canonical_channels(cfg).index(ch) for ch in channels]
            tasks=[f for f in requested if (group,f) in todo]
            if not tasks: continue
            X,cols=flatten_channel_features(staging,idx,_canonical_channels(cfg))
            print(f'[GROUP] {group}: {len(channels)} EEG channels, {len(cols)} features, '
                  f'{len(tasks)} fold(s) to train',flush=True)
            for fold in tasks:
                path=output/group/f'fold_{fold}'
                if _prepare_fold(path,signatures[(group,fold)],force):
                    _train_fold(X,meta,y,folds,fold,group,cols,xgb_cfg['model'],
                        float(xgb_cfg['report']['decision_threshold']),path,
                        signatures[(group,fold)],smoke_test)
            del X
            gc.collect()
    else:
        print('[SKIP] All requested XGBoost folds already complete',flush=True)
    _aggregate(output,ref_dir,all_group_map,all_folds,signatures,labels,folds,smoke_test)
