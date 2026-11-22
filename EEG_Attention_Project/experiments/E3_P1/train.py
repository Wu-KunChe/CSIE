"""Original 1D-CNN on P1; disk-backed inputs, same fixed CV as E1/E2."""
from pathlib import Path
import gc, tempfile, shutil
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from src.data_loader import read_labeled_table,iter_session_arrays
from src.model import build_model
from src.trainer import train_binary_classifier,set_random_seed
from src.preprocessing import ChannelwiseStandardizer
from src.evaluation import binary_metrics,predict_probabilities,predictions_table,per_subject_metrics,save_evaluation
from .common import GROUPS,KEY,IDS,sha,stat,fingerprint,json_save,json_read,contract,protocol
from .prepare import verify_npz

REQUIRED=['run_config.json','best_model.pt','metrics.json','predictions.csv','per_subject_metrics.csv','scaler.json','training_history.csv','done.json']

class MemmapDataset(Dataset):
    def __init__(self,x,idx,y,scaler):
        self.x=x;self.idx=idx;self.y=y[idx];self.mean=scaler.mean_.reshape(-1,1);self.std=scaler.std_.reshape(-1,1)
    def __len__(self):return len(self.idx)
    def __getitem__(self,i):
        return torch.from_numpy(np.asarray((self.x[self.idx[i]]-self.mean)/self.std,dtype=np.float32)),torch.tensor(float(self.y[i]),dtype=torch.float32)


def indices(meta,folds,f,role):
    return np.flatnonzero(meta.subject.isin(folds.loc[(folds.fold==f)&(folds.role==role),'subject']).to_numpy())


def stage_data(ep,labels,channels,path):
    x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(labels),len(channels),1000));meta=[];ys=[];at=0
    for a,y,m in iter_session_arrays(labels,ep,channels,source_key=KEY,expected_samples=1000):
        x[at:at+len(y)]=a;at+=len(y);meta.append(m);ys.append(y)
    if at!=len(labels):raise RuntimeError('Stage row mismatch')
    x.flush();del x
    return np.concatenate(ys),pd.concat(meta,ignore_index=True)


def fit_fold(x,y,meta,folds,f,channels,base,out,sig,device,smoke):
    tr,va,te=(indices(meta,folds,f,r) for r in ('train','val','test'))
    for idx in (tr,va,te):
        if set(y[idx])!={0,1}:raise ValueError('Missing class in a fold role')
    scaler=ChannelwiseStandardizer(float(base['preprocessing']['std_epsilon']))
    scaler.fit_from_batches(x[tr[i:i+256]] for i in range(0,len(tr),256))
    cfg=dict(base['training']);seed=int(cfg['random_seed'])+f;cfg['random_seed']=seed
    if smoke:cfg.update(max_epochs=2,early_stopping_patience=2)
    set_random_seed(seed)
    dls=[DataLoader(MemmapDataset(x,idx,y,scaler),batch_size=int(cfg['batch_size']),shuffle=(i==0),num_workers=0,pin_memory=device=='cuda',generator=torch.Generator().manual_seed(seed) if i==0 else None) for i,idx in enumerate((tr,va,te))]
    model=build_model(base['model'],n_channels=len(channels),seed=seed)
    t=perf_counter();fit=train_binary_classifier(model,dls[0],dls[1],cfg,checkpoint_path=out/'best_model.pt',device=device);train_sec=perf_counter()-t
    if device=='cuda':torch.cuda.synchronize()
    t=perf_counter();p=predict_probabilities(fit['model'],dls[2],device=device)
    if device=='cuda':torch.cuda.synchronize()
    infer_sec=perf_counter()-t
    met=binary_metrics(y[te],p,float(cfg['decision_threshold']))
    pred=predictions_table(meta.iloc[te].reset_index(drop=True),y[te],p,float(cfg['decision_threshold']));pred['fold']=f
    save_evaluation(out,met,pred,per_subject_metrics(pred));pd.DataFrame(fit['history']).to_csv(out/'training_history.csv',index=False)
    json_save(out/'scaler.json',scaler.to_metadata());json_save(out/'run_config.json',{'signature':sig,'fold':f,'source_key':KEY,'channels':list(channels),'model':base['model'],'training':cfg,'best_epoch':fit['best_epoch'],'best_val_loss':fit['best_val_loss'],'train_seconds':train_sec,'inference_seconds':infer_sec,'parameter_count':sum(p.numel() for p in model.parameters()),'device':device,'smoke':smoke,'sample_counts':[len(tr),len(va),len(te)],'offline_ica':True})
    json_save(out/'done.json',{'signature':sig})
    print(f'  BA={met["balanced_accuracy"]:.4f}, F1={met["macro_f1"]:.4f}',flush=True)


def validate_predictions(pred,labels,folds,f=None):
    expect=labels.copy() if f is None else labels[labels.subject.isin(folds.loc[(folds.fold==f)&folds.role.eq('test'),'subject'])]
    a=pred[IDS+['y_true','fold']].merge(expect[IDS+['label']],on=IDS,how='outer',validate='one_to_one',indicator=True)
    test=folds[folds.role.eq('test')][['subject','fold']].rename(columns={'fold':'expected_fold'})
    a=a.merge(test,on='subject',how='left',validate='many_to_one')
    if len(a)!=len(expect) or not a._merge.eq('both').all() or not a.y_true.eq(a.label).all() or not a.fold.eq(a.expected_fold).all():raise ValueError('Predictions do not match original trial labels or folds')


def summarize_group(directory,labels,folds,signatures,smoke):
    frames=[];metrics=[]
    for f,sig in signatures.items():
        d=directory/f'fold_{f}'
        if not all((d/n).is_file() for n in REQUIRED) or json_read(d/'done.json').get('signature')!=sig:continue
        p=pd.read_csv(d/'predictions.csv');p['fold']=f
        if not smoke:validate_predictions(p,labels,folds,f)
        frames.append(p);metrics.append({'fold':f,**binary_metrics(p.y_true,p.prob_label1)})
    if not frames:return None
    big=pd.concat(frames,ignore_index=True);m=pd.DataFrame(metrics)
    m.to_csv(directory/'fold_summary.csv',index=False);big.to_csv(directory/'combined_predictions.csv',index=False,encoding='utf-8-sig');per_subject_metrics(big).to_csv(directory/'subject_summary.csv',index=False)
    pooled=binary_metrics(big.y_true,big.prob_label1)
    complete=len(m)==5 and not smoke
    if complete:validate_predictions(big,labels,folds)
    result={'folds_completed':len(m),'complete':complete,'mean_BA':float(m.balanced_accuracy.mean()),'std_BA':float(m.balanced_accuracy.std()) if len(m)>1 else None,'mean_macro_F1':float(m.macro_f1.mean()),'pooled_BA':pooled['balanced_accuracy'],'pooled_macro_F1':pooled['macro_f1'],'n_test':len(big)}
    json_save(directory/'summary.json',result);return result


def baseline_result(root,group,labels,folds):
    # Never use E1 legacy 2-sec or random search. Verify every fold's trial IDs.
    folder=root/'results/diagnostics/window_length/4sec'/group
    frames=[];mets=[]
    for f in range(1,6):
        d=folder/f'fold_{f}';p=d/'predictions.csv'
        if not p.is_file() or not (d/'run_config.json').is_file():return None
        rc=json_read(d/'run_config.json')
        if rc.get('source')!='X_B0_uV' or rc.get('window_seconds')!=4 or [str(c).upper() for c in rc.get('channels',[])]!=list(GROUPS[group]) or rc.get('fold')!=f or not rc.get('train_only_scaling'):
            raise ValueError(f'Unexpected original B0 run configuration: {d}')
        pred=pd.read_csv(p);pred['fold']=f;validate_predictions(pred,labels,folds,f)
        frames.append(pred);mets.append(binary_metrics(pred.y_true,pred.prob_label1))
    merged=pd.concat(frames,ignore_index=True);validate_predictions(merged,labels,folds)
    m=pd.DataFrame(mets); pooled=binary_metrics(merged.y_true,merged.prob_label1)
    return {'folds_completed':5,'complete':True,'mean_BA':float(m.balanced_accuracy.mean()),'std_BA':float(m.balanced_accuracy.std()),'mean_macro_F1':float(m.macro_f1.mean()),'pooled_BA':pooled['balanced_accuracy'],'pooled_macro_F1':pooled['macro_f1'],'n_test':len(merged)}


def run(root,cfg,groups,fold_numbers,device='auto',check=False,smoke=False):
    root=Path(root).resolve();allrows,lf,folds,ff=contract(root);base=protocol(root)
    ep=root/cfg['output']['epochs'];ldir=root/cfg['output']['labels'];labels_file=ldir/'rt30_labels.csv';pr=ep/'P1_provenance.json'
    if sha(labels_file)!=sha(lf):raise ValueError('P1 labels differ from original B0')
    provenance=json_read(pr)
    if provenance['recipe']['labels']!=sha(lf) or provenance['recipe']['folds']!=sha(ff) or provenance['recipe']['config']!=cfg or provenance['source_key']!=KEY:raise ValueError('P1 preparation provenance differs from current protocol')
    labels=read_labeled_table(labels_file)
    for (sub,ses),r in allrows.groupby(['subject','session']):verify_npz(ep/f'{sub}_{ses}_unlabeled.npz',r[r.included_in_npz.eq(1)].sort_values('x_row'))
    p2=root/'epochs/P2_official/4sec/P2_provenance.json';p2labels=root/'data/labels/E3/P2_official/4sec/rt30_labels.csv'
    if p2.is_file():
        if sha(p2labels)!=sha(lf):raise ValueError('Existing P2 label set is not identical to P0; resolve before comparison')
        print('[P2] original labels match; P2 not modified',flush=True)
    references={g:baseline_result(root,g,labels,folds) for g in GROUPS}
    for g,v in references.items():
        if v is None:print(f'[WARNING] Missing B0 reference prediction files for {g}; P1 may train but comparison will omit B0')
        else:print(f'[B0 verified] {g}: {v["mean_BA"]:.4f}')
    print(f'[CHECK PASS] P1 {len(labels)} labeled epochs; 62 sessions; 27 subjects; 5 folds')
    if check:return
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    dev='cuda' if device=='cuda' or device=='auto' and torch.cuda.is_available() else 'cpu'
    output=root/cfg['output']['results'];output=output.with_name(output.name+'_SMOKE') if smoke else output
    output.mkdir(parents=True,exist_ok=True)
    src=[Path(__file__),root/'configs/baseline.yaml',*[root/'src'/f'{x}.py' for x in ['model','trainer','data_loader','preprocessing','evaluation','config']]]
    context={'input':sha(pr),'files':[stat(p) for p in sorted(ep.glob('*_unlabeled.npz'))],'code':[sha(p) for p in src],'folds':sha(ff),'smoke':smoke,'device':dev,'torch':torch.__version__}
    selected=fold_numbers or ([1] if smoke else list(range(1,6)))
    if smoke and selected!=[1]:raise ValueError('Smoke test uses fold 1 only')
    summary=[]
    for g,channels in GROUPS.items():
        directory=output/g;signatures={f:fingerprint([context,g,f]) for f in range(1,6)}
        if g in groups:
            need=[]
            for f in selected:
                d=directory/f'fold_{f}';sig=signatures[f]
                if d.exists():
                    if (d/'done.json').is_file() and json_read(d/'done.json').get('signature')==sig and all((d/n).is_file() for n in REQUIRED): print(f'[SKIP] {g} fold {f}');continue
                    if not (d/'run_intent.json').is_file() or json_read(d/'run_intent.json').get('signature')!=sig:raise RuntimeError(f'Existing incompatible fold {d}; move aside before re-running')
                    shutil.rmtree(d)
                need.append(f)
            if need:
                use=labels if not smoke else labels.groupby(['subject','label'],sort=True).head(4)
                print(f'[STAGE] {g}, data={len(use)}x{len(channels)}x1000',flush=True)
                with tempfile.TemporaryDirectory(prefix='P1_',dir=output) as tmp:
                    path=Path(tmp)/'data.npy';y,meta=stage_data(ep,use,channels,path);x=np.load(path,mmap_mode='r')
                    try:
                        for f in need:
                            d=directory/f'fold_{f}';d.mkdir(parents=True);json_save(d/'run_intent.json',{'signature':signatures[f]})
                            print(f'[TRAIN] {g} fold {f} on {dev}',flush=True)
                            fit_fold(x,y,meta,folds,f,channels,base,d,signatures[f],dev,smoke)
                    finally:del x;gc.collect()
        result=summarize_group(directory,labels,folds,signatures,smoke)
        if not smoke and references[g]:summary.append({'group':g,'condition':'P0',**references[g]})
        if result:summary.append({'group':g,'condition':'P1',**result})
    pd.DataFrame(summary).to_csv(output/'comparison.csv',index=False)
    print('[RESULTS]',output,flush=True)
