"""Offline, per-session full scalp ICA. No labels used in fitting or selection."""
from time import perf_counter
import warnings
import numpy as np
import pandas as pd
from .common import SCALP, json_save

RETRY_MAX_ITER=2000

CLASSES=['brain','muscle','eye','heart','line_noise','channel_noise','other']

def make_fit_raw(x, fs, cfg):
    import mne
    # Explicit canonical aliases, deterministic template coordinates for all sessions.
    aliases={'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
    m=mne.channels.make_standard_montage(cfg['montage'])
    lookup={n.upper():v for n,v in m.get_positions()['ch_pos'].items()}
    pos={n:lookup[aliases.get(n,n)] for n in SCALP}
    fid=m.get_positions()
    montage=mne.channels.make_dig_montage(ch_pos=pos,nasion=fid['nasion'],lpa=fid['lpa'],rpa=fid['rpa'],coord_frame=fid['coord_frame'])
    # RawArray(copy='auto') can alias float64 input; CAR/FIR mutate in place.
    # Explicitly own the data before any fitting-only preprocessing.
    raw=mne.io.RawArray(x, mne.create_info(list(SCALP),fs,'eeg'), copy='both', verbose='ERROR')
    raw.set_montage(montage,on_missing='raise',verbose='ERROR')
    raw.set_eeg_reference('average',projection=False,verbose='ERROR')
    raw.filter(*cfg['fit_band'],method='fir',phase='zero',verbose='ERROR')
    raw.resample(cfg['fit_sfreq'],verbose='ERROR')
    return raw

def remove_affine(ica, info, exclude):
    """Probe MNE's reconstruction on the CAR subspace; return artifact operator in volts.
    This preserves the original reference's common mode rather than rereferencing P1.
    All PCA residuals are retained by MNE. No excls => exact identity behavior.
    """
    import mne
    n=len(info['ch_names'])
    if not exclude:return np.zeros((n,n)),np.zeros((n,1))
    step=1e-6
    basis=np.column_stack([np.zeros(n),np.eye(n)*step])
    basis-=basis.mean(axis=0,keepdims=True)
    raw=mne.io.RawArray(basis.copy(),info.copy(),verbose='ERROR')
    ica.apply(raw,exclude=list(exclude),n_pca_components=None,verbose='ERROR')
    delta=basis-raw.get_data()
    bias=delta[:,:1]; op=(delta[:,1:]-bias)/step
    return op,bias

def apply_to_original(x, op, bias):
    # op already includes the CAR projection. Never subtract the average itself.
    return x-(op@x+bias)

def choose_components(prob,cfg):
    prob=np.asarray(prob,dtype=float)
    if prob.ndim!=2 or prob.shape[1]!=7 or not np.isfinite(prob).all() or (prob<0).any() or (prob>1.000001).any() or not np.allclose(prob.sum(axis=1),1,atol=.003):
        raise ValueError('ICLabel must return all 7 valid probabilities per component')
    return np.flatnonzero((prob[:,1]>=cfg['muscle_threshold'])|(prob[:,2]>=cfg['eye_threshold'])).tolist()

def fit_with_retry(fit, rank, cfg, quality_dir, factory):
    """Restart from the same seed only when convergence fails; record each attempt."""
    from threadpoolctl import threadpool_limits
    quality_dir.mkdir(parents=True,exist_ok=True)
    limits=[int(cfg['max_iter'])]
    if limits[0]<RETRY_MAX_ITER:limits.append(RETRY_MAX_ITER)
    attempts=[]
    for limit in limits:
        ica=factory(n_components=rank,method='picard',fit_params={'ortho':False,'extended':True,'tol':cfg['tolerance']},max_iter=limit,random_state=cfg['random_state'],verbose='ERROR')
        started=perf_counter()
        with threadpool_limits(limits=4), warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            ica.fit(fit,picks=list(SCALP),reject=None,reject_by_annotation=False,verbose='ERROR')
        messages=[str(w.message) for w in ws]
        n_iter=int(ica.n_iter_)
        converged=n_iter<limit and not any('converg' in m.lower() and ('not' in m.lower() or 'fail' in m.lower()) for m in messages)
        attempts.append({'max_iter':limit,'n_iter':n_iter,'converged':converged,'seconds':perf_counter()-started,'warnings':messages})
        json_save(quality_dir/'fit_attempts.json',attempts)
        (quality_dir/'fit_warnings.txt').write_text('\n'.join(f"Attempt {i+1} (max_iter={a['max_iter']}): "+' | '.join(a['warnings']) for i,a in enumerate(attempts)),encoding='utf-8')
        if converged:return ica,attempts
        if limit!=limits[-1]:
            print(f'  [RETRY] ICA not converged at {n_iter} iterations; restart with max_iter={limits[-1]}, same seed/tolerance',flush=True)
        del ica
    raise RuntimeError(f'ICA did not converge after limits {limits} (last iterations={n_iter}); no session published')

def fit_session(x,fs,cfg,quality_dir):
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel.iclabel import iclabel_label_components
    from threadpoolctl import threadpool_limits
    t=perf_counter()
    fit=make_fit_raw(x,fs,cfg)
    z=fit.get_data()
    eig=np.linalg.eigvalsh(np.cov(z)); rank=int(np.sum(eig>eig[-1]*cfg['rank_relative_eigenvalue_floor']))
    if rank<2 or rank>29:raise ValueError(f'Unexpected CAR data rank {rank}; eigenvalues={eig}')
    if not np.isfinite(eig).all() or not eig[-1]>0:raise ValueError('Invalid covariance')
    del z
    ica,attempts=fit_with_retry(fit,rank,cfg,quality_dir,ICA)
    n_iter=int(ica.n_iter_)
    fitted=perf_counter()-t; t=perf_counter()
    # Low-level API returns all seven probabilities, not just winning-class score.
    prob=iclabel_label_components(fit,ica,inplace=True,backend=cfg['backend'])
    excluded=choose_components(prob,cfg)
    if len(excluded)>=rank:raise RuntimeError('All ICA components flagged; review session before publication')
    ica.exclude=excluded
    table=pd.DataFrame(prob,columns=CLASSES)
    table.insert(0,'component',np.arange(rank));table['predicted_class']=[CLASSES[i] for i in np.argmax(prob,axis=1)]
    table['removed']=table.component.isin(excluded)
    table.to_csv(quality_dir/'components.csv',index=False)
    ica.save(quality_dir/'solution-ica.fif',overwrite=True,verbose='ERROR')
    op,bias=remove_affine(ica,fit.info,excluded)
    # Cross-check affine reconstruction against direct MNE.apply on a real CAR segment.
    n=min(2500,fit.n_times)
    car=fit.get_data(start=0,stop=n)
    direct=mne.io.RawArray(car.copy(),fit.info.copy(),verbose='ERROR')
    ica.apply(direct,exclude=excluded,n_pca_components=None,verbose='ERROR')
    expected=direct.get_data() if excluded else car
    err=float(np.max(np.abs(apply_to_original(car,op,bias)-expected)))
    if not np.isfinite(err) or err>1e-10:raise RuntimeError(f'ICA affine vs direct MNE.apply mismatch: {err} V')
    np.savez(quality_dir/'artifact_operator.npz',operator=op,bias_volts=bias,channels=np.array(SCALP),excluded=np.array(excluded,dtype=int))
    # Component maps for removed ICs; noncritical visualization failure is logged.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if excluded:
            figs=ica.plot_components(picks=excluded,show=False)
            if not isinstance(figs,list):figs=[figs]
            for i,fig in enumerate(figs):fig.savefig(quality_dir/f'removed_topography_{i+1}.png',dpi=120);plt.close(fig)
    except Exception as ex:(quality_dir/'plot_warning.txt').write_text(str(ex),encoding='utf-8')
    return op,bias,{'rank':rank,'n_iter':n_iter,'max_iter_used':attempts[-1]['max_iter'],'retry_used':len(attempts)>1,'fit_attempts':attempts,'removed':excluded,'n_removed':len(excluded),'eye_count':int((prob[:,2]>=cfg['eye_threshold']).sum()),'muscle_count':int((prob[:,1]>=cfg['muscle_threshold']).sum()),'fit_seconds':fitted,'label_and_audit_seconds':perf_counter()-t,'projection_error_volts':err,'rank_eigenvalues':eig.tolist(),'fit_samples':fit.n_times,'montage':cfg['montage']}
