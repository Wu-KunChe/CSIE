"""P1: verify original timeline and P0 numerics, full-session ICA, matched epochs."""
from pathlib import Path
from time import perf_counter
import importlib.metadata as im
import hashlib
import shutil
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfilt,resample_poly,welch
from .common import SCALP,KEY,contract,sha,stat,fingerprint,json_save,json_read
from .ica import fit_session,apply_to_original,choose_components,CLASSES,remove_affine,RETRY_MAX_ITER

FS=500
N=2000
WARM=2000
SOS=butter(4,[1.,30.],btype='bandpass',fs=FS,output='sos')


def deps():
    # Check exact runtime imports, not just package names.
    import mne, picard, torch
    from mne_icalabel.iclabel import iclabel_label_components
    return {k:im.version(k) for k in ['mne','python-picard','mne-icalabel','torch','numpy','scipy','pandas','threadpoolctl']}


def input_files(raw_root,sub,ses):
    prefix=raw_root/sub/ses/'eeg'/f'{sub}_{ses}_task-driving'
    eeg=Path(str(prefix)+'_eeg.set');ev=Path(str(prefix)+'_events.tsv')
    for p in (eeg,ev):
        if not p.is_file():raise FileNotFoundError(p)
    return eeg,ev


def open_source(eeg):
    import mne
    raw=mne.io.read_raw_eeglab(str(eeg),preload=False,verbose='ERROR')
    try:
        if abs(float(raw.info['sfreq'])-FS)>1e-3:raise ValueError('Raw EEG must be 500 Hz')
        names={str(n).strip().upper():n for n in raw.ch_names}
        if len(names)!=len(raw.ch_names) or not set(SCALP).issubset(names):raise ValueError('Missing/duplicate canonical scalp channels')
        return raw,[names[n] for n in SCALP]
    except Exception:raw.close();raise


def check_events(path,rows,n_times):
    ev=pd.read_csv(path,sep='\t')
    t=pd.to_numeric(ev.onset,errors='coerce').to_numpy(float)
    c=pd.to_numeric(ev.value,errors='coerce').to_numpy(float)
    valid=np.isfinite(t)&np.isfinite(c)&(t>=0)
    # Match original parse_trials: int(float(code)); stable sorted by time.
    t=t[valid];c=c[valid].astype(int);order=np.argsort(t,kind='stable');t=t[order];c=c[order]
    dev=np.isin(c,[251,252]);ix=np.rint(t*FS).astype(np.int64)
    if not np.array_equal(ix[dev],np.rint(rows.onset_s.to_numpy(float)*FS).astype(int)) or not np.array_equal(c[dev],rows.deviation_code.to_numpy(int)):
        raise ValueError('BIDS deviation codes/onsets differ from 4sec label table')
    if 'sample' in ev:
        sample=pd.to_numeric(ev['sample'],errors='coerce');onset=pd.to_numeric(ev.onset,errors='coerce')
        if ((sample/FS-onset).abs().dropna()>1/FS+1e-6).any():raise ValueError('BIDS sample/onset disagreement')
    kept=rows[rows.included_in_npz.eq(1)].sort_values('x_row')
    windows=[]
    for row in kept.itertuples(index=False):
        end=round(float(row.onset_s)*FS);start=end-N
        lo=np.searchsorted(ix,start,side='left');hi=np.searchsorted(ix,end,side='left')
        if start<0 or end>n_times or lo!=hi:raise ValueError(f'Trial {row.trial_index}: invalid existing window')
        begin=max(0 if lo==0 else int(ix[lo-1])+1,start-WARM)
        if begin>start:raise ValueError('Invalid warmup boundary')
        if hasattr(row,'filter_warmup_s') and not np.isclose((start-begin)/FS,float(row.filter_warmup_s),atol=1e-7):raise ValueError('Warmup differs from original B0')
        windows.append((begin,end))
    return kept,windows


def epoch(seg):
    # The original B0 extraction uses uV input, causal SOS, restricted warmup,
    # then resample the final 2000 points (not the extended segment).
    seg=np.asarray(seg,dtype=np.float64)*1e6
    return resample_poly(sosfilt(SOS,seg,axis=-1)[:,-N:],1,2,axis=-1,padtype='line').astype(np.float32)


def audit_baseline(b0,kept):
    with np.load(b0,allow_pickle=False) as z:
        names=[str(n).strip().upper() for n in z['channel_names']]
        if len(names)!=30 or set(names)!=set(SCALP):raise ValueError('B0 canonical channel names invalid')
        if float(z['sfreq_B0'])!=250 or not np.array_equal(z['trial_index'],kept.trial_index.to_numpy(int)):raise ValueError('B0 NPZ IDs or frequency differ')
        if not np.allclose(z['onset_s'],kept.onset_s.to_numpy(float),rtol=0,atol=1e-8):raise ValueError('B0 onset mismatch')
        arr=z['X_B0_uV'][:,[names.index(n) for n in SCALP],:]
        if arr.shape!=(len(kept),30,1000) or not np.isfinite(arr).all():raise ValueError('B0 shape/nonfinite')
        return arr


def figures_and_quality(b0,p1,quality):
    # Means across the same eligible epochs; no RT labels used for QA.
    f,b=welch(b0[:,:2,:],fs=250,nperseg=500,axis=-1)
    _,a=welch(p1[:,:2,:],fs=250,nperseg=500,axis=-1)
    b=b.mean(axis=0);a=a.mean(axis=0)
    rows=[]
    for name,lo,hi in [('delta',1,4),('theta',4,8),('alpha',8,13),('beta',13,30)]:
        mask=(f>=lo)&(f<hi)
        for ch,n in enumerate(SCALP[:2]):
            before=float(b[ch,mask].sum()*(f[1]-f[0]));after=float(a[ch,mask].sum()*(f[1]-f[0]))
            rows.append(dict(channel=n,band=name,B0_power_uV2=before,P1_power_uV2=after,ratio=after/before if before>0 else np.nan))
    pd.DataFrame(rows).to_csv(quality/'band_power.csv',index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,6));t=np.arange(1000)/250
        for i,n in enumerate(SCALP[:2]):
            axes[i,0].plot(t,b0[0,i],label='P0',lw=.8);axes[i,0].plot(t,p1[0,i],label='P1',lw=.8);axes[i,0].set(xlabel='Time (s)',ylabel=f'{n} (uV)');axes[i,0].legend()
            axes[i,1].semilogy(f,np.maximum(b[i],1e-20),label='P0');axes[i,1].semilogy(f,np.maximum(a[i],1e-20),label='P1');axes[i,1].set(xlim=(1,30),xlabel='Hz',ylabel='PSD (uV²/Hz)');axes[i,1].legend()
        fig.tight_layout();fig.savefig(quality/'forehead_waveform_psd.png',dpi=120);plt.close(fig)
    except Exception as ex:(quality/'qa_plot_warning.txt').write_text(str(ex),encoding='utf-8')


def verify_npz(path,kept):
    with np.load(path,allow_pickle=False) as z:
        if z[KEY].shape!=(len(kept),30,1000) or not np.isfinite(z[KEY]).all() or float(z['sfreq_P1'])!=250 or list(z['channel_names'])!=list(SCALP):raise ValueError(f'Invalid P1 NPZ {path}')
        if not np.array_equal(z['trial_index'],kept.trial_index.to_numpy(int)) or not np.allclose(z['onset_s'],kept.onset_s.to_numpy(float),rtol=0,atol=1e-8):raise ValueError(f'P1 IDs/onsets differ {path}')


def read_saved_fit(quality, info, cfg):
    """Recover the fitting-only artifacts of a verified earlier checkpoint.

    Prior epochs/QA were generated from a potentially mutated raw buffer and
    must not be reused. ICA itself was fitted on the intended CAR/FIR data.
    """
    from mne.preprocessing import read_ica
    rank=int(info['rank']); excluded=list(info['removed'])
    limit=int(info.get('max_iter_used',cfg['max_iter']))
    if not (2<=rank<=29 and 0<int(info['n_iter'])<limit and limit in (int(cfg['max_iter']),RETRY_MAX_ITER)):
        raise ValueError('Saved fit convergence metadata invalid')
    if len(excluded)!=int(info['n_removed']) or len(excluded)>=rank:
        raise ValueError('Saved fit exclusion metadata invalid')
    table=pd.read_csv(quality/'components.csv')
    if not np.array_equal(table.component.to_numpy(), np.arange(rank)):
        raise ValueError('Saved IC table component indices differ')
    if choose_components(table[CLASSES].to_numpy(),cfg)!=excluded:
        raise ValueError('Saved ICLabel probabilities do not match fixed rule')
    with np.load(quality/'artifact_operator.npz',allow_pickle=False) as z:
        op=z['operator'];bias=z['bias_volts']
        if list(z['channels'])!=list(SCALP) or z['excluded'].tolist()!=excluded:
            raise ValueError('Saved artifact operator channels/exclusions differ')
    if op.shape!=(30,30) or bias.shape!=(30,1) or not np.isfinite(op).all() or not np.isfinite(bias).all():
        raise ValueError('Saved artifact operator shape/nonfinite')
    if not excluded and (np.any(op!=0) or np.any(bias!=0)):
        raise ValueError('Empty exclusions must have zero operator/bias')
    # Verify that the serialized ICA and artifact operator agree (allow FIF roundoff).
    ica=read_ica(quality/'solution-ica.fif',verbose='ERROR')
    if int(ica.n_components_)!=rank or list(ica.ch_names)!=list(SCALP) or list(ica.exclude)!=excluded or ica.info is None:
        raise ValueError('Saved ICA solution metadata differs')
    op_check,bias_check=remove_affine(ica,ica.info,excluded)
    if not np.allclose(op,op_check,rtol=2e-5,atol=2e-6) or not np.allclose(bias,bias_check,rtol=2e-5,atol=1e-10):
        raise ValueError('Saved ICA solution and artifact operator disagree')
    return op,bias


def data_digest(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast('B')).hexdigest()


def extract_session(eeg,ev,rows,b0_path,out,quality,cfg,signature,recovery_signatures=None):
    sub,ses=rows.iloc[0][['subject','session']];key=f'{sub}_{ses}'
    done=quality/'done.json';path=out/f'{key}_unlabeled.npz';manifest=out/f'{key}_manifest.csv'
    kept=rows[rows.included_in_npz.eq(1)].sort_values('x_row')
    saved_fit=None
    if done.is_file():
        info=json_read(done)
        old=info.get('signature')
        if old==signature:
            verify_npz(path,kept)
            if not manifest.is_file() or not (quality/'components.csv').is_file():raise RuntimeError(f'{key}: incomplete saved session')
            print(f'[SKIP] {key}',flush=True);return info
        generation=(recovery_signatures or {}).get(old)
        if generation is None:
            raise RuntimeError(f'{key}: completion signature changed; source/config not verified; no unsafe reuse')
        if generation=='original':
            if not (0<int(info.get('n_iter',0))<int(cfg['max_iter'])):
                raise RuntimeError(f'{key}: invalid original convergence record')
            info.update(max_iter_used=int(cfg['max_iter']),retry_used=False,resumed_original_release=True,
                        fit_attempts=[{'max_iter':int(cfg['max_iter']),'n_iter':int(info['n_iter']),'converged':True}])
        op,bias=read_saved_fit(quality,info,cfg)
        saved_fit=(op,bias,dict(info,previous_signature=old,recovered_fit_generation=generation))
        print(f'[REBUILD] {key}: reuse verified ICA fit; regenerate epochs from untouched raw',flush=True)
    quality.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    raw,names=open_source(eeg)
    try:
        kept,windows=check_events(ev,rows,raw.n_times)
        b0=audit_baseline(b0_path,kept)
        print(f'  Loading {raw.n_times:,} samples × 30; validating P0 numerical reconstruction',flush=True)
        x=raw.get_data(picks=names)
        if not np.isfinite(x).all():raise ValueError('Nonfinite continuous raw EEG')
    finally:raw.close()
    # Require numerical identity within float tolerance on EVERY P0 epoch before ICA.
    maxerr=0.
    for j,(start,end) in enumerate(windows):
        ref=epoch(x[:,start:end]);err=float(np.max(np.abs(ref-b0[j])));maxerr=max(maxerr,err)
        if not np.allclose(ref,b0[j],rtol=2e-5,atol=1e-3):raise ValueError(f'{key} trial {kept.iloc[j].trial_index}: P0 reconstruction mismatch {err:.6g} uV; check raw version/filter')
    if saved_fit is None:
        print('  Fitting Picard-Extended Infomax, then ICLabel...',flush=True)
        raw_digest=data_digest(x)
        op,bias,stats=fit_session(x,FS,cfg,quality)
        if data_digest(x)!=raw_digest:
            raise RuntimeError('ICA fitting unexpectedly mutated original raw buffer')
    else:
        op,bias,stats=saved_fit
    print(f'  rank={stats["rank"]}, iterations={stats["n_iter"]}, removing {stats["removed"]}',flush=True)
    t=perf_counter();arr=np.empty_like(b0)
    for j,(start,end) in enumerate(windows):
        seg=x[:,start:end];clean=apply_to_original(seg,op,bias)
        arr[j]=epoch(clean)
    if not np.isfinite(arr).all():raise ValueError('Nonfinite reconstructed P1 epochs')
    if stats['n_removed']==0:
        # Do not waive the no-op test. Verify the freshly reconstructed data,
        # then use the exact existing P0 float32 values for bitwise identity.
        if np.any(op!=0) or np.any(bias!=0) or not np.allclose(arr,b0,rtol=2e-5,atol=1e-3):
            raise RuntimeError('No-exclusion P1 should reproduce B0')
        arr=b0.copy()
    stats.update(epoch_rebuilt_from_saved_ica=saved_fit is not None,
                 original_buffer_protected=True,reference_reconstruction='untouched_raw_minus_artifact',
                 no_exclusion_exact_B0=stats['n_removed']==0)
    stats.update(session=key,signature=signature,n_epochs=len(kept),n_labeled=int(kept.label.notna().sum()),raw_B0_max_error_uV=maxerr,epoch_reconstruction_seconds=perf_counter()-t,fp_B0_rms_uV=float(np.sqrt(np.mean(b0[:,:2,:].astype(float)**2))),fp_P1_rms_uV=float(np.sqrt(np.mean(arr[:,:2,:].astype(float)**2))))
    del x
    figures_and_quality(b0,arr,quality)
    tmp=path.with_name('.'+path.stem+'_building.npz')
    np.savez(tmp,**{KEY:arr},trial_index=kept.trial_index.to_numpy(np.int32),onset_s=kept.onset_s.to_numpy(float),rt_s=kept.rt_s.to_numpy(float),channel_names=np.array(SCALP),forehead_names=np.array(SCALP[:2]),sfreq_P1=np.array(250.),source=np.array('full30_ICA_ICLabel'))
    verify_npz(tmp,kept);tmp.replace(path)
    rows.to_csv(manifest,index=False,encoding='utf-8-sig')
    json_save(done,stats)
    (quality/'failure.json').unlink(missing_ok=True)
    return stats


# Exact SHA256 hashes of the unmodified initial distribution. Only allow
# legacy checkpoints when the original experiment recipe still matches.
LEGACY_CODE={
    'prepare.py':'f0660c0b96793ab22167cbf23e3fd26c038392687dd7d39f430359f62be159f2',
    'ica.py':'c690a75acc04f08b04d6017824f34ab89d084b20145c57b2eb838fb7864490fa',
}

# Exact bytes from the convergence-resume release (no tests folder needed).
RETRY_RELEASE_CODE={
    'prepare.py':'0e24eb5405d7562533e0c1276e3d9818c182335479473caf707bae9371b0cdc8',
    'ica.py':'00bfdbb099539c5f100936784b56d14d06ef72f88b03eb531fd76f5b78b1dd69',
}

def recipe_signatures(recipe):
    legacy={**recipe,'code':{**recipe['code'],**LEGACY_CODE}}
    legacy_sig=fingerprint(legacy)
    policy={'initial_max_iter':recipe['config']['ica']['max_iter'],
        'retry_max_iter':RETRY_MAX_ITER,'restart_same_seed':True,'only_on_nonconvergence':True,
        'unchanged_tolerance':True,'legacy_original_recipe_signature':legacy_sig}
    retry_recipe={**recipe,'code':{**recipe['code'],**RETRY_RELEASE_CODE},'ica_retry_policy':policy}
    retry_sig=fingerprint(retry_recipe)
    current={**recipe,'ica_retry_policy':policy,'reference_reconstruction_version':'isolated_raw_buffer_v2',
             'recovery_fit_recipe_signatures':{'original':legacy_sig,'convergence_retry':retry_sig}}
    return current,fingerprint(current),{legacy_sig:'original',retry_sig:'convergence_retry'}

def run(root,raw_root,cfg,mode='check'):
    from .common import protocol
    protocol(root);versions=deps();labels,lf,folds,ff=contract(root)
    out=root/cfg['output']['epochs'];ldir=root/cfg['output']['labels'];reports=root/cfg['output']['inspection'];result=root/cfg['output']['results']
    reports.mkdir(parents=True,exist_ok=True)
    sources=[];checks=[]
    for (sub,ses),rows in labels.groupby(['subject','session'],sort=True):
        key=f'{sub}_{ses}';b0=root/'epochs/4sec'/f'{key}_unlabeled.npz'
        try:
            eeg,ev=input_files(raw_root,sub,ses);raw,names=open_source(eeg)
            try:kept,_=check_events(ev,rows,raw.n_times);n_times=raw.n_times
            finally:raw.close()
            # Verify each NPZ contract; complete numerical audit happens at extraction.
            audit_baseline(b0,kept)
            # Include external EEGLAB FDT if present. No provenance inference from ordering.
            sidecars=sorted(eeg.parent.glob('*.fdt'))
            sources.append((key,eeg,ev,rows,b0,[stat(p) for p in (eeg,ev,b0,*sidecars)]))
            checks.append({'session':key,'status':'PASS','samples':n_times,'eligible':len(kept),'error':''})
        except Exception as ex:checks.append({'session':key,'status':'FAIL','error':str(ex)})
        print(f'[CHECK {len(checks)}/62] {key}: {checks[-1]["status"]} {checks[-1]["error"]}',flush=True)
    pd.DataFrame(checks).to_csv(reports/'session_check.csv',index=False)
    report={'status':'PASS' if len(sources)==62 else 'FAIL','sessions_pass':len(sources),'versions':versions,'labels_sha256':sha(lf),'folds_sha256':sha(ff)};json_save(reports/'preflight.json',report)
    if len(sources)!=62:raise RuntimeError('P1 preflight failed; see session_check.csv. No new official epochs published.')
    if mode=='check':return
    code={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py')) if p.name!='train.py'}
    recipe={'config':cfg,'versions':versions,'code':code,'labels':sha(lf),'folds':sha(ff),'source_files':[s[-1] for s in sources]}
    recipe,sig,old_signatures=recipe_signatures(recipe)
    if mode=='pilot':
        stage=root/'epochs/_P1_ica_pilot/4sec';quality=result.parent/'P1_ica_PILOT'/'quality';items=sources[:1]
    else:
        stage=root/'epochs/_P1_ica_staging/4sec';quality=result/'quality';items=sources
        if out.exists():
            provenance=out/'P1_provenance.json'
            if provenance.is_file() and json_read(provenance).get('signature')==sig and (ldir/'rt30_labels.csv').is_file() and sha(ldir/'rt30_labels.csv')==sha(lf):
                for key,_,_,rows,_,_ in sources:verify_npz(out/f'{key}_unlabeled.npz',rows[rows.included_in_npz.eq(1)].sort_values('x_row'))
                print('[ALREADY COMPLETE] P1 matches current sources',flush=True);return
            raise FileExistsError(f'Published P1 exists but differs: {out}; no overwrite')
        if ldir.exists():raise FileExistsError(f'P1 labels target exists: {ldir}')
    summaries=[]
    for i,(key,eeg,ev,rows,b0,_) in enumerate(items,1):
        print(f'[ICA {i}/{len(items)}] {key}',flush=True)
        try:summaries.append(extract_session(eeg,ev,rows,b0,stage,quality/key,cfg['ica'],fingerprint([sig,key]),recovery_signatures={fingerprint([old,key]):generation for old,generation in old_signatures.items()}))
        except Exception as ex:
            json_save(quality/key/'failure.json',{'error':f'{type(ex).__name__}: {ex}'});raise
    pd.DataFrame(summaries).to_csv(quality.parent/'ica_session_summary.csv',index=False)
    if mode=='pilot':print('[PILOT PASS] isolated first session; no official P1 data published');return
    for key,_,_,rows,_,_ in sources:verify_npz(stage/f'{key}_unlabeled.npz',rows[rows.included_in_npz.eq(1)].sort_values('x_row'))
    provenance={'name':'E3_P1_full30_ICA','signature':sig,'recipe':recipe,'saved_channels':list(SCALP),'source_key':KEY,'eligible_epochs':22777,'labeled_epochs':13604,'session_count':62,'offline_test_session_unlabeled_fit':True}
    json_save(stage/'P1_provenance.json',provenance)
    labelstage=ldir.parent/'_4sec_staging';labelstage.mkdir(parents=True,exist_ok=True)
    shutil.copy2(lf,labelstage/'rt30_labels.csv')
    if (lf.parent/'rt30_summary.csv').is_file():shutil.copy2(lf.parent/'rt30_summary.csv',labelstage/'rt30_summary.csv')
    if (root/'epochs/4sec/all_trials_UNLABELED.csv').is_file():shutil.copy2(root/'epochs/4sec/all_trials_UNLABELED.csv',stage/'all_trials_UNLABELED.csv')
    json_save(labelstage/'P1_provenance.json',provenance)
    (stage/'extraction_report.txt').write_text('P1: 62 session-specific Full30 ICA fits; 22777 aligned epochs; 13604 unchanged RT0–30 labels; offline.\n',encoding='utf-8')
    out.parent.mkdir(parents=True,exist_ok=True);stage.replace(out)
    try:labelstage.replace(ldir)
    except Exception:out.replace(stage);raise
    print(f'[SUCCESS] {out}; labels {ldir}',flush=True)
