"""Fixed original E1 CNN on matched official P2; no model search, no relabeling."""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import json
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import read_yaml
from src.data_loader import read_labeled_table, load_arrays, EEGArrayDataset
from src.evaluation import binary_metrics,predict_probabilities,predictions_table,per_subject_metrics,save_evaluation
from src.model import build_model
from src.preprocessing import ChannelwiseStandardizer
from src.trainer import train_binary_classifier,set_random_seed

CH=('FP1','FP2')
NEEDED=('best_model.pt','run_config.json','metrics.json','predictions.csv',
        'per_subject_metrics.csv','training_history.csv','scaler.json','done.json')


def _json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def _sha(p):
    h=sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def _preflight(root):
    ep=root/'epochs/P2_official/4sec'
    labels_file=root/'data/labels/E3/P2_official/4sec/rt30_labels.csv'
    b0_file=root/'data/labels/window_length/4sec/rt30_labels.csv'
    folds_file=root/'data/splits/subject_folds.csv'
    provenance=ep/'P2_provenance.json'
    for p in (labels_file,b0_file,folds_file,provenance,root/'configs/baseline.yaml'):
        if not p.is_file():raise FileNotFoundError(p)
    if _sha(labels_file)!=_sha(b0_file):
        raise RuntimeError('P2 labels differ from E1 4-sec original; no retraining until reconciled')
    labels=read_labeled_table(labels_file)
    folds=pd.read_csv(folds_file)
    if labels.subject.nunique()!=27 or labels.groupby(['subject','session']).ngroups!=62:
        raise ValueError('P2 must use 27 subjects, 62 sessions')
    if len(labels)!=13604:
        raise ValueError(f'Expected 13604 labeled 4s trials, found {len(labels)}; check input')
    if folds.duplicated(['fold','subject']).any() or set(folds.fold.unique())!={1,2,3,4,5}:
        raise ValueError('Incorrect 5-fold subject split')
    subjects=set(labels.subject.unique())
    for fold,group in folds.groupby('fold'):
        if set(group.subject)!=subjects or set(group.role)!={'train','val','test'}:
            raise ValueError(f'Fold {fold}: missing subject/role')
    test_counts=folds[folds.role=='test'].groupby('subject').size()
    if set(test_counts.index)!=subjects or not test_counts.eq(1).all():
        raise ValueError('Each subject must appear in exactly one test fold')
    needed=sorted(set(labels.npz_file))
    if len(needed)!=62 or any(not (ep/f).is_file() for f in needed):
        raise FileNotFoundError(f'P2 NPZ missing, expected 62 sessions: {ep}')
    original=read_yaml(root/'configs/baseline.yaml')
    base=original
    model={'name':'cnn1d','conv_filters':[32,64],'kernels':[51,15],
           'dropout':0.3,'num_classes':2}
    train={'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,
           'batch_size':64,'max_epochs':60,'early_stopping_patience':8,
           'class_weighting':'train_only','decision_threshold':.5,'random_seed':42}
    for k,v in model.items():
        if base['model'].get(k)!=v:raise ValueError(f'CNN original E1 model changed: {k}')
    for k,v in train.items():
        if base['training'].get(k)!=v:raise ValueError(f'CNN original E1 training changed: {k}')
    with np.load(ep/needed[0],allow_pickle=False) as n:
        shape=n['X_P2_uV'].shape
        names=[str(v).upper() for v in n['channel_names']]
        if shape[-1]!=1000 or shape[1] not in (2,30) or float(n['sfreq_P2'])!=250:
            raise ValueError('P2 NPZ source shape/sample rate invalid')
        if not {'FP1','FP2'}.issubset(names):
            raise ValueError('P2 NPZ missing FP1/FP2')
    return ep,labels,folds,base,labels_file,folds_file,provenance


def _fingerprint(root,labels_file,folds_file,provenance,ep,base,fold,smoke):
    src_files=[root/'scripts/run_e3_p2_train.py',root/'experiments/E3_P2/train.py',
               root/'configs/baseline.yaml']
    src_files += [root/'src'/f for f in ('model.py','trainer.py','preprocessing.py',
                                        'data_loader.py','evaluation.py')]
    spec={'sources':[(_e3_training_source_sha(p),str(p.relative_to(root))) for p in src_files],
          'labels':_sha(labels_file),'folds':_sha(folds_file),
          'P2_provenance':_sha(provenance),'NPZ_meta':[
              (p.name,p.stat().st_size,p.stat().st_mtime_ns)
              for p in sorted(ep.glob('*_unlabeled.npz'))],
          'model':base['model'],'train':base['training'],
          'fold':fold,'smoke':smoke}
    return sha256(json.dumps(spec,sort_keys=True,default=str).encode()).hexdigest()


def _done(folder,hashvalue):
    if not all((folder/f).is_file() for f in NEEDED):return False
    try:return json.loads((folder/'done.json').read_text(encoding='utf-8')).get('signature')==hashvalue
    except Exception:return False


def run(root,folds_select=None,device='auto',check=False,smoke=False,force=False):
    root=Path(root).resolve()
    ep,labels,folds,base,labels_file,folds_file,provenance=_preflight(root)
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA explicitly requested but unavailable')
    fold_numbers=folds_select or ([1] if smoke else list(range(1,6)))
    for f in fold_numbers:
        if f not in range(1,6):raise ValueError('fold must be 1..5')
    print(f'P2 official: {len(labels)} labeled trials, 62 sessions, 27 subjects',flush=True)
    print('Original E1 CNN fixed; source_key=X_P2_uV, channels=FP1/FP2, 4s @250Hz',flush=True)
    if check:
        print('[CHECK PASS] Labels, folds, provenance, baseline config and P2 NPZ paths')
        return
    output=root/('results/E3/P2_official_SMOKE' if smoke else 'results/E3/P2_official')
    # First check signatures to avoid unnecessary NPZ loading.
    need=[]
    for f in fold_numbers:
        sig=_fingerprint(root,labels_file,folds_file,provenance,ep,base,f,smoke)
        dest=output/f'fold_{f}'
        if dest.exists() and _done(dest,sig) and not force:
            print(f'[SKIP] P2 fold {f} complete',flush=True)
        else:
            if dest.exists() and not force and (dest/'done.json').is_file():
                raise RuntimeError(f'Existing fold has different signature: {dest}')
            need.append((f,sig,dest))
    if need:
        use=labels
        if smoke:
            use=pd.concat([labels[labels.subject.isin(folds.loc[(folds.fold==f)&(folds.role==r),'subject'])]
                           .groupby(['subject','label'],sort=True).head(3)
                           for f,_,_ in need for r in ('train','val','test')],ignore_index=True)
            use=use.drop_duplicates(['subject','session','trial_index'])
        print('[LOAD] Loading P2 labeled FP1/FP2 epochs...',flush=True)
        X,y,meta=load_arrays(use,ep,CH,source_key='X_P2_uV',expected_samples=1000)
        for f,sig,dest in need:
            if dest.exists():shutil.rmtree(dest)
            dest.mkdir(parents=True)
            _json(dest/'run_intent.json',{'signature':sig})
            getids=lambda role: np.flatnonzero(meta.subject.isin(set(folds.loc[
                (folds.fold==f)&(folds.role==role),'subject'])).to_numpy())
            tr,va,te=(getids(role) for role in ('train','val','test'))
            for ids in (tr,va,te):
                if not len(ids) or set(y[ids])!={0,1}:
                    raise ValueError(f'Fold {f}: empty role or one-class role')
            scaler=ChannelwiseStandardizer(float(base['preprocessing']['std_epsilon']))
            xt=scaler.fit_transform(X[tr]);xv=scaler.transform(X[va]);xe=scaler.transform(X[te])
            ds=lambda xx,yy:EEGArrayDataset(xx,yy)
            cfg=dict(base['training']);seed=int(cfg['random_seed'])+f;cfg['random_seed']=seed
            if smoke:cfg.update(max_epochs=2,early_stopping_patience=2)
            set_random_seed(seed)
            dev='cuda' if (device=='cuda' or device=='auto' and torch.cuda.is_available()) else 'cpu'
            loader=lambda d,shuffle:DataLoader(d,batch_size=int(cfg['batch_size']),
                shuffle=shuffle,num_workers=0,
                generator=torch.Generator().manual_seed(seed) if shuffle else None,
                pin_memory=dev=='cuda')
            model=build_model(base['model'],n_channels=2,seed=seed)
            print(f'[TRAIN] P2 fold {f}: train={len(tr)} val={len(va)} test={len(te)} device={dev}',flush=True)
            fit=train_binary_classifier(model,loader(ds(xt,y[tr]),True),loader(ds(xv,y[va]),False),
                                        cfg,checkpoint_path=dest/'best_model.pt',device=dev)
            prob=predict_probabilities(fit['model'],loader(ds(xe,y[te]),False),device=dev)
            threshold=float(cfg.get('decision_threshold',.5))
            metrics=binary_metrics(y[te],prob,threshold)
            pred=predictions_table(meta.iloc[te].reset_index(drop=True),y[te],prob,threshold)
            save_evaluation(dest,metrics,pred,per_subject_metrics(pred))
            pd.DataFrame(fit['history']).to_csv(dest/'training_history.csv',index=False)
            _json(dest/'scaler.json',scaler.to_metadata())
            _json(dest/'run_config.json',{'signature':sig,'fold':f,'source_key':'X_P2_uV',
                 'channels':list(CH),'best_epoch':fit['best_epoch'],
                 'best_val_loss':fit['best_val_loss'],'model':base['model'],
                 'training':cfg,'device':dev,'smoke':smoke})
            _json(dest/'done.json',{'signature':sig,'balanced_accuracy':metrics['balanced_accuracy']})
            print(f'  BA={metrics["balanced_accuracy"]:.4f} F1={metrics["macro_f1"]:.4f}',flush=True)
    # Summarize completed tests without treating incomplete outer folds as final results.
    preds=[];rows=[]
    for f in range(1,6):
        dest=output/f'fold_{f}'
        if (dest/'predictions.csv').is_file() and (dest/'done.json').is_file():
            d=pd.read_csv(dest/'predictions.csv');d['fold']=f;preds.append(d)
            rows.append({'fold':f,**json.loads((dest/'metrics.json').read_text(encoding='utf-8'))})
    if rows:
        pd.DataFrame(rows).to_csv(output/'fold_summary.csv',index=False)
        big=pd.concat(preds,ignore_index=True)
        big.to_csv(output/'combined_predictions.csv',index=False,encoding='utf-8-sig')
        per_subject_metrics(big).to_csv(output/'subject_summary.csv',index=False)
        overall=binary_metrics(big.y_true,big.prob_label1)
        result={'folds_completed':len(rows),'mean_fold_BA':float(pd.DataFrame(rows).balanced_accuracy.mean()),
                'mean_fold_macro_F1':float(pd.DataFrame(rows).macro_f1.mean()),
                'pooled_test_BA':overall['balanced_accuracy'],
                'pooled_macro_F1':overall['macro_f1'],
                'n_test_predictions':len(big),
                'complete':len(rows)==5}
        _json(output/'summary.json',result)
        print('[SUMMARY]',result,flush=True)
    if len(rows)==5 and not smoke:
        print('[E3 AUTO ANALYSIS] All five P2 folds complete; comparing P0/P1/P2...', flush=True)
        summarize_e3(root)


# --- Post-training P0/P1/P2 analysis (integrated; no extra Python file) ---
"""E3 analysis integrated into existing P2 training module. No extra entrypoint."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from .prepare import SCALP
GROUPS = {'Full30': SCALP, 'Forehead2': ('FP1', 'FP2')}
IDS = ['subject', 'session', 'trial_index']

CONDITIONS = ('P0', 'P1', 'P2')
COLORS = dict(zip(CONDITIONS, ('#64748b', '#007e87', '#cf8a32')))
KEYS = {'P0':'X_B0_uV','P1':'X_P1_uV','P2':'X_P2_uV'}
METRICS = ['balanced_accuracy','macro_f1','sensitivity_label1','specificity_label0']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load_contract(root):
    label_file = root/'data/labels/window_length/4sec/rt30_labels.csv'
    fold_file = root/'data/splits/subject_folds.csv'
    raw = pd.read_csv(label_file, low_memory=False)
    if not raw.window_seconds.eq(4).all() or not raw.labeling_method.eq('session_RT_rank_0to30_vs_70to100').all():
        raise ValueError('Not the original 4s RT0-30 table')
    labels = raw.loc[raw.label.notna(), IDS+['label']].copy()
    if (len(labels)!=13604 or labels.label.value_counts().to_dict()!={0.0:6802,1.0:6802}
        or labels.subject.nunique()!=27 or labels.groupby(['subject','session']).ngroups!=62
        or labels.duplicated(IDS).any()):
        raise ValueError('Unexpected original 4s trial/subject/class counts')
    folds = pd.read_csv(fold_file)
    if set(folds.fold)!=set(range(1,6)) or folds.duplicated(['fold','subject']).any():
        raise ValueError('Invalid outer folds')
    for _, g in folds.groupby('fold'):
        if set(g.subject)!=set(labels.subject) or set(g.role)!={'train','val','test'}:
            raise ValueError('Incorrect subject/role coverage')
    test = folds.loc[folds.role.eq('test'), ['subject','fold']]
    if test.subject.duplicated().any() or set(test.subject)!=set(labels.subject):
        raise ValueError('Subjects must be tested once')
    labels = labels.merge(test,on='subject',validate='many_to_one').rename(columns={'label':'expected_label'})
    # Current prepared data, if present, must preserve the original label table.
    for cond, sub in [('P1','P1_ica'),('P2','P2_official')]:
        p = root/f'data/labels/E3/{sub}/4sec/rt30_labels.csv'
        if p.is_file() and sha(p)!=sha(label_file):
            raise ValueError(f'{cond} label CSV differs from P0')
    return labels


def metrics(df):
    truth = df.y_true.to_numpy(dtype=int); pred=df.y_pred.to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(truth,pred,labels=[0,1]).ravel()
    spec=tn/(tn+fp) if tn+fp else np.nan
    sens=tp/(tp+fn) if tp+fn else np.nan
    return dict(n_samples=len(df), n_label0=int(tn+fp), n_label1=int(tp+fn),
                balanced_accuracy=(spec+sens)/2, macro_f1=f1_score(truth,pred,average='macro',labels=[0,1],zero_division=0),
                sensitivity_label1=sens,specificity_label0=spec,tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def location(root,condition,group):
    if condition=='P0':return root/'results/diagnostics/window_length/4sec'/group
    if condition=='P1':return root/'results/E3/P1_ica'/group
    p=root/'results/E3/P2_official'
    # Support earlier Forehead2-only runner with flat output, but do not mix layouts.
    if group=='Forehead2' and any(p.glob('fold_*')):
        if (p/group).exists():raise ValueError('Ambiguous P2 output: both flat and grouped Forehead2')
        return p
    return p/group


def read_fold(directory,condition,group,fold,labels):
    d=directory/f'fold_{fold}'
    if not d.is_dir():return None,None
    required=['run_config.json','metrics.json','predictions.csv']
    if condition!='P0':required+=['done.json']
    if any(not (d/n).is_file() for n in required):return None,None
    cfg=read_json(d/'run_config.json'); saved=read_json(d/'metrics.json')
    if cfg.get('smoke') or cfg.get('smoke_test') or saved.get('smoke_test'):
        raise ValueError(f'Smoke outputs in formal results: {d}')
    if cfg.get('fold')!=fold or [str(c).upper() for c in cfg.get('channels',[])]!=list(GROUPS[group]):
        raise ValueError(f'Incorrect group/fold metadata: {d}')
    source=cfg.get('source_key',cfg.get('source'))
    if source!=KEYS[condition]:raise ValueError(f'Wrong source key at {d}: {source}')
    if condition=='P0':
        if cfg.get('window_seconds')!=4 or not cfg.get('train_only_scaling'):
            raise ValueError(f'Not the original 4s B0 run: {d}')
    else:
        if not cfg.get('signature') or read_json(d/'done.json').get('signature')!=cfg['signature']:
            raise ValueError(f'Invalid completion signature: {d}')
        expected_model={'name':'cnn1d','conv_filters':[32,64],'kernels':[51,15],'dropout':0.3,'num_classes':2}
        expected_train={'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'batch_size':64,'max_epochs':60,'early_stopping_patience':8,'class_weighting':'train_only','decision_threshold':.5,'random_seed':42+fold}
        for name, exp in [('model',expected_model),('training',expected_train)]:
            if any(cfg.get(name,{}).get(k)!=v for k,v in exp.items()):raise ValueError(f'{name} not original CNN protocol: {d}')
    p=pd.read_csv(d/'predictions.csv',low_memory=False)
    if 'fold' in p and not p.fold.eq(fold).all():raise ValueError(f'Incorrect recorded fold: {d}')
    if not p.y_true.isin([0,1]).all() or not p.y_pred.isin([0,1]).all():raise ValueError(f'Invalid labels: {d}')
    prob=p.prob_label1.to_numpy(dtype=float)
    if not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)) or not np.array_equal(p.y_pred.to_numpy(),(prob>=.5).astype(int)):
        raise ValueError(f'Invalid prediction/threshold: {d}')
    original=labels.loc[labels.fold.eq(fold),IDS+['expected_label']]
    p=p[IDS+['y_true','y_pred','prob_label1']].merge(original,on=IDS,how='outer',indicator=True,validate='one_to_one')
    if len(p)!=len(original) or not p._merge.eq('both').all() or not p.y_true.eq(p.expected_label).all():
        raise ValueError(f'Trial IDs or ground truth differ from original fold: {d}')
    p=p.drop(columns=['expected_label','_merge']);p['fold']=fold
    m=metrics(p)
    for k in ('n_samples','balanced_accuracy','macro_f1','tn','fp','fn','tp'):
        if k not in saved or not np.isclose(m[k],saved[k],rtol=1e-7,atol=1e-9):
            raise ValueError(f'Saved metrics disagree with predictions ({k}): {d}')
    row=dict(condition=condition,group=group,fold=fold,**m,
             best_epoch=cfg.get('best_epoch'),best_val_loss=cfg.get('best_val_loss'),
             train_seconds=cfg.get('train_seconds'),inference_seconds=cfg.get('inference_seconds'),
             parameter_count=cfg.get('parameter_count'),device=cfg.get('device'))
    return p,row


def save_csv(df,path):
    df.to_csv(path,index=False,encoding='utf-8-sig')


def save_fig(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)


def classification_plots(out,overall,folds,subjects):
    # Never promote incomplete folds to the main comparison.
    full=overall.loc[overall.complete].copy()
    if full.empty:return
    groups=[g for g in GROUPS if g in set(full.group)]
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),squeeze=False)
    for ax,metric,title in zip(axes[0],['BA','macro_F1'],['Balanced accuracy','Macro-F1']):
        for i,c in enumerate(CONDITIONS):
            d=full[full.condition.eq(c)].set_index('group').reindex(groups)
            vals=d[f'mean_{metric}'].to_numpy(float)*100;err=d[f'std_{metric}'].to_numpy(float)*100
            x=np.arange(len(groups))+(i-1)*.23;keep=np.isfinite(vals)
            if not keep.any():continue
            bars=ax.bar(x[keep],vals[keep],.22,yerr=err[keep],capsize=3,label=c,color=COLORS[c],alpha=.90)
            for bar,val,sd in zip(bars,vals[keep],err[keep]):ax.text(bar.get_x()+bar.get_width()/2,val+sd+1.1,f'{val:.2f}',ha='center',fontsize=8)
        ax.set_xticks(range(len(groups)),groups);ax.set_ylim(0,105);ax.set_ylabel('%');ax.set_title(title+' (5-fold mean ± SD)');ax.grid(axis='y',alpha=.2);ax.legend()
        if metric=='BA':ax.axhline(50,color='gray',ls=':',lw=1)
    save_fig(fig,out/'performance_comparison.png')
    fig,axes=plt.subplots(1,len(groups),figsize=(5.5*len(groups),4),squeeze=False,sharey=True)
    for ax,g in zip(axes[0],groups):
        for c in CONDITIONS:
            if not ((full.group==g)&(full.condition==c)).any():continue
            d=folds[(folds.group==g)&(folds.condition==c)].sort_values('fold')
            ax.plot(d.fold,100*d.balanced_accuracy,'o-',label=c,color=COLORS[c])
        ax.set_title(g);ax.set_xticks(range(1,6));ax.set_xlabel('Outer test fold');ax.set_ylabel('BA (%)');ax.axhline(50,color='gray',ls=':');ax.legend();ax.grid(alpha=.2)
    save_fig(fig,out/'fold_BA.png')
    # Subject differences are paired using exactly the same held-out trials.
    fig,axes=plt.subplots(len(groups),1,figsize=(11,3.5*len(groups)),squeeze=False)
    plotted=False
    for ax,g in zip(axes[:,0],groups):
        d=subjects[subjects.group.eq(g)].pivot(index='subject',columns='condition',values='balanced_accuracy')
        if 'P0' in d:
            for i,c in enumerate(('P1','P2')):
                if c not in d or not ((full.group==g)&(full.condition==c)).any():continue
                delta=100*(d[c]-d.P0);ax.bar(np.arange(len(d))+(i-.5)*.36,delta,.36,label=f'{c} − P0',color=COLORS[c]);plotted=True
            ax.set_xticks(range(len(d)),d.index,rotation=60,ha='right',fontsize=8)
        ax.axhline(0,color='gray',lw=1);ax.set_title(g);ax.set_ylabel('BA difference (pp)');ax.grid(axis='y',alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend()
    if plotted:save_fig(fig,out/'subject_BA_delta.png')
    else:plt.close(fig)
    # Pooled held-out confusion matrices; row normalize to compare class recall.
    fig,axes=plt.subplots(len(groups),3,figsize=(11,3.3*len(groups)),squeeze=False)
    for j,g in enumerate(groups):
        for i,c in enumerate(CONDITIONS):
            ax=axes[j,i];d=full[(full.group==g)&(full.condition==c)]
            if d.empty:ax.set_axis_off();continue
            r=d.iloc[0];cm=np.array([[r.tn,r.fp],[r.fn,r.tp]],int);norm=cm/cm.sum(axis=1,keepdims=True)
            ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
            for a in range(2):
                for b in range(2):ax.text(b,a,f'{cm[a,b]}\n{100*norm[a,b]:.1f}%',ha='center',va='center',color='white' if norm[a,b]>.6 else 'black',fontsize=10)
            ax.set_xticks([0,1],['Fast','Slow']);ax.set_yticks([0,1],['Fast','Slow']);ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(f'{g} / {c}')
    save_fig(fig,out/'confusion_matrices.png')


def ica_analysis(root,out):
    p=root/'results/E3/P1_ica';src=p/'ica_session_summary.csv'
    if not src.is_file():return ['ICA summary source missing; QA plots omitted.']
    d=pd.read_csv(src);need={'session','n_removed','n_iter'}
    if not need.issubset(d):return ['ICA summary fields missing; QA plots omitted.']
    if d.session.duplicated().any() or len(d)!=62:return ['ICA summary does not contain exactly 62 unique sessions; QA plots omitted.']
    save_csv(d,out/'ica_session_summary.csv')
    fig,axes=plt.subplots(2,1,figsize=(13,6.6),sharex=True)
    x=np.arange(len(d));axes[0].bar(x,d.n_removed,color='#007e87');axes[0].set_ylabel('Removed ICs');axes[0].set_title('P1: IC removal and convergence by recording session')
    axes[1].bar(x,d.n_iter,color='#64748b');axes[1].set_ylabel('Final fit iterations');axes[1].set_xticks(x,d.session,rotation=90,fontsize=6)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    save_fig(fig,out/'ica_sessions.png')
    msgs=[f'ICA: {len(d)} sessions; removed ICs mean={d.n_removed.mean():.2f}, min={d.n_removed.min()}, max={d.n_removed.max()}; zero-removal sessions={(d.n_removed==0).sum()}.']
    bands=[]
    for s in d.session:
        f=p/'quality'/str(s)/'band_power.csv'
        if not f.is_file():continue
        b=pd.read_csv(f);b['session']=s;bands.append(b)
    if len(bands)!=62:return msgs+['Band-power files incomplete; no aggregate band-power chart.']
    b=pd.concat(bands,ignore_index=True)
    req={'session','channel','band','B0_power_uV2','P1_power_uV2','ratio'}
    if not req.issubset(b) or len(b)!=62*2*4 or b.duplicated(['session','channel','band']).any() or set(b.channel)!={'FP1','FP2'} or set(b.band)!={'delta','theta','alpha','beta'}:
        return msgs+['Unexpected band-power data; aggregate chart omitted.']
    save_csv(b,out/'ica_band_power_sessions.csv')
    b['change_pct']=100*(b.ratio-1)
    agg=b.groupby(['channel','band'],as_index=False).agg(n_sessions=('change_pct','count'),median_change_pct=('change_pct','median'),q1=('change_pct',lambda s:s.quantile(.25)),q3=('change_pct',lambda s:s.quantile(.75)))
    save_csv(agg,out/'ica_band_power_summary.csv')
    fig,ax=plt.subplots(figsize=(8,4.5));names=['delta','theta','alpha','beta'];xx=np.arange(4)
    for i,ch in enumerate(['FP1','FP2']):
        a=agg[agg.channel.eq(ch)].set_index('band').reindex(names);v=a.median_change_pct.to_numpy(float)
        ax.bar(xx+(i-.5)*.35,v,.35,yerr=np.vstack([v-a.q1.to_numpy(float),a.q3.to_numpy(float)-v]),capsize=3,label=ch)
    ax.set_xticks(xx,names);ax.axhline(0,color='gray',lw=1);ax.set_ylabel('Power change (%)');ax.set_title('P1 vs P0: session median and IQR (all eligible epochs)');ax.legend();ax.grid(axis='y',alpha=.2)
    save_fig(fig,out/'ica_band_power_change.png')
    return msgs+['Band-power changes: median and IQR across sessions; descriptive, not evidence of neural signal preservation.']


def summarize_e3(root):
    root=Path(root).resolve(); labels=load_contract(root)
    rows=[];summaries=[];subject_rows=[];status=[]
    for g in GROUPS:
        for c in CONDITIONS:
            folder=location(root,c,g);preds=[];fs=[]
            for f in range(1,6):
                p,r=read_fold(folder,c,g,f,labels)
                if p is not None:preds.append(p);fs.append(r)
            complete=len(fs)==5; status.append({'group':g,'condition':c,'folds_completed':len(fs),'complete':complete,'source':str(folder.relative_to(root))})
            print(f'[READ] {g} / {c}: {len(fs)}/5',flush=True)
            if not fs:continue
            rows.extend(fs);m=pd.DataFrame(fs);big=pd.concat(preds,ignore_index=True)
            if big.duplicated(IDS).any() or (complete and len(big)!=len(labels)):raise ValueError('Repeated or missing held-out trial identities')
            pool=metrics(big)
            summaries.append(dict(group=g,condition=c,folds_completed=len(fs),complete=complete,n_test=len(big),n_subjects=big.subject.nunique(),
                 mean_BA=m.balanced_accuracy.mean(),std_BA=m.balanced_accuracy.std(),mean_macro_F1=m.macro_f1.mean(),std_macro_F1=m.macro_f1.std(),
                 pooled_BA=pool['balanced_accuracy'],pooled_macro_F1=pool['macro_f1'],sensitivity_label1=pool['sensitivity_label1'],specificity_label0=pool['specificity_label0'],**{k:pool[k] for k in ('tn','fp','fn','tp')}))
            for sub, ss in big.groupby('subject'):
                subject_rows.append(dict(group=g,condition=c,subject=sub,fold=int(ss.fold.iloc[0]),complete=complete,**metrics(ss)))
    if not rows:raise RuntimeError('No verified formal prediction files found.')
    overall=pd.DataFrame(summaries);folds=pd.DataFrame(rows);subjects=pd.DataFrame(subject_rows);complete=overall[overall.complete]
    delta=[];paired=[]
    for g in GROUPS:
        d=complete[complete.group.eq(g)].set_index('condition')
        for c,ref in [('P1','P0'),('P2','P0'),('P2','P1')]:
            if c not in d.index or ref not in d.index:continue
            r=d.loc[c];b=d.loc[ref]
            s=subjects[(subjects.group==g)&(subjects.condition==c)].merge(subjects[(subjects.group==g)&(subjects.condition==ref)],on=['subject','fold'],suffixes=('_a','_b'),validate='one_to_one')
            if len(s)!=27:raise ValueError('Unexpected subject paired count')
            v=100*(s.balanced_accuracy_a-s.balanced_accuracy_b);f=100*(s.macro_f1_a-s.macro_f1_b)
            delta.append(dict(group=g,comparison=f'{c}-{ref}',delta_mean_BA_pp=100*(r.mean_BA-b.mean_BA),delta_mean_macro_F1_pp=100*(r.mean_macro_F1-b.mean_macro_F1),subjects_improved=int((v>1e-9).sum()),subjects_declined=int((v< -1e-9).sum()),subjects_unchanged=int((abs(v)<=1e-9).sum())))
            for (_,ss),dv,df in zip(s.iterrows(),v,f):paired.append(dict(group=g,comparison=f'{c}-{ref}',subject=ss.subject,fold=ss.fold,delta_BA_pp=dv,delta_macro_F1_pp=df))
    out=root/'results/E3/summary';out.mkdir(parents=True,exist_ok=True)
    # Clear only known derived plots to avoid stale charts after partial analysis.
    for name in ('performance_comparison.png','fold_BA.png','subject_BA_delta.png','confusion_matrices.png','ica_sessions.png','ica_band_power_change.png','ica_session_summary.csv','ica_band_power_sessions.csv','ica_band_power_summary.csv'):
        (out/name).unlink(missing_ok=True)
    save_csv(pd.DataFrame(status),out/'status.csv');save_csv(overall,out/'comparison.csv');save_csv(folds,out/'fold_summary.csv');save_csv(subjects,out/'subject_summary.csv')
    save_csv(pd.DataFrame(delta,columns=['group','comparison','delta_mean_BA_pp','delta_mean_macro_F1_pp','subjects_improved','subjects_declined','subjects_unchanged']),out/'paired_comparison.csv')
    save_csv(pd.DataFrame(paired,columns=['group','comparison','subject','fold','delta_BA_pp','delta_macro_F1_pp']),out/'paired_subject_delta.csv')
    classification_plots(out,overall,folds,subjects); msgs=ica_analysis(root,out)
    report=['E3 preprocessing comparison (original 4-second RT0-30, 5-fold subject-wise CV)',
            'Mean BA / macro-F1 are unweighted fold means; SD is sample SD across outer folds. Pooled scores use all held-out trials.',
            'Only complete 5-fold groups enter plots and paired comparisons.',
            'P1/P2 Forehead2 use offline multichannel-cleaned data; not a fully 2-channel preprocessing pipeline.',
            'P2 differs by complete official preprocessing pipeline; differences cannot be attributed to ICA alone.',
            'Descriptive only: no statistical significance or causal attribution. Previously suspected sub-04/ses-01 vs sub-05/ses-01 duplication remains unverified here.', '', 'STATUS', pd.DataFrame(status).drop(columns='source').to_string(index=False),'','COMPLETE RESULTS']
    for _,r in complete.iterrows():report.append(f'{r.group:10s} {r.condition}: BA {100*r.mean_BA:.2f}% ± {100*r.std_BA:.2f} pp; macro-F1 {100*r.mean_macro_F1:.2f}%; pooled BA {100*r.pooled_BA:.2f}% (N={r.n_test})')
    report+=['','PAIRED COMPARISONS',pd.DataFrame(delta).to_string(index=False) if delta else 'None yet.','','ICA QA',*msgs]
    (out/'report.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report),flush=True); print('[RESULTS]',out,flush=True)



def _e3_training_source_sha(path):
    """Ignore post-training reporting changes when checking existing P2 training folds.

    Reconstruct the original pre-report training code so completed fold signatures
    from the user-supplied E3_P2/train.py continue to match. Any actual change
    to training code still changes its hash and prevents unsafe reuse.
    """
    if path.name != 'train.py' or path.parent.name != 'E3_P2':
        return _sha(path)
    text = path.read_text(encoding='utf-8')
    marker = '# --- Post-training P0/P1/P2 analysis (integrated; no extra Python file) ---'
    if marker not in text:
        return _sha(path)
    training = text.split(marker, 1)[0]
    hook = ("    if len(rows)==5 and not smoke:\n"
            "        print('[E3 AUTO ANALYSIS] All five P2 folds complete; comparing P0/P1/P2...', flush=True)\n"
            "        summarize_e3(root)\n")
    if hook not in training:
        return _sha(path)
    training = training.replace(hook, '', 1)
    training = training.replace(
        "'sources':[(_e3_training_source_sha(p),str(p.relative_to(root))) for p in src_files],",
        "'sources':[(_sha(p),str(p.relative_to(root))) for p in src_files],", 1)
    return sha256((training.rstrip('\n') + '\n').encode('utf-8')).hexdigest()
