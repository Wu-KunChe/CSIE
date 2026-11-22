"""Strict, non-destructive preparation of Cao et al. official preprocessed EEG for E3 P2.

P2 is a distinct Figshare product, NOT the NEMAR BIDS Raw tree.
A P2 session is accepted only when its original session identity, sample clock,
30 scalp channels, onset sequence and existing 4sec labels match the B0 source.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, resample_poly

SCALP = ('FP1','FP2','F7','F3','FZ','F4','F8','FT7','FC3','FCZ','FC4',
         'FT8','T3','C3','CZ','C4','T4','TP7','CP3','CPZ','CP4','TP8',
         'T5','P3','PZ','P4','T6','O1','OZ','O2')
FS, OUT_FS, N_SEC, WARMUP_SEC = 500, 250, 4, 4
SOURCE_KEY = 'X_P2_uV'
LABEL_METHOD = 'session_RT_rank_0to30_vs_70to100'


def _to_json(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, Path): return str(obj)
    raise TypeError(repr(obj))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_to_json), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def stat_fingerprint(path):
    st = path.stat()
    return [str(path.resolve()), st.st_size, st.st_mtime_ns]


@dataclass(frozen=True)
class Session:
    subject: str
    session: str
    original: str
    bids_set: Path
    bids_events: Path
    preproc_file: Path | None

    @property
    def key(self): return f'{self.subject}_{self.session}'


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


def discover(raw_root, preproc_root, labels):
    if not raw_root.is_dir(): raise FileNotFoundError(f'Raw NEMAR folder missing: {raw_root}')
    if raw_root.resolve()==preproc_root.resolve():
        raise ValueError('Preprocessed root must be different from NEMAR Raw root')
    # Restrict exact mapping; never infer session identity by file ordering.
    available={}
    for p in (preproc_root.rglob('*') if preproc_root.is_dir() else []):
        if p.is_file() and (p.name.lower().endswith('.set') or p.name.lower().endswith('.set.zip')):
            available.setdefault(p.name.lower(),[]).append(p)
    sessions=[]
    for sub in sorted(labels.subject.unique()):
        table_path=raw_root/sub/f'{sub}_sessions.tsv'
        if not table_path.is_file(): raise FileNotFoundError(table_path)
        tab=pd.read_csv(table_path, sep='\t', dtype=str).fillna('')
        if not {'session_id','original_filename'}.issubset(tab):
            raise ValueError(f'{table_path}: missing original_filename/session_id')
        for ses in sorted(labels.loc[labels.subject.eq(sub),'session'].unique()):
            row=tab.loc[tab.session_id.eq(ses)]
            if len(row)!=1: raise ValueError(f'{sub}/{ses}: session table identity ambiguous')
            original=str(row.iloc[0].original_filename).strip()
            if original.lower().endswith('.set'): original=original[:-4]
            if not re.fullmatch(r's\d\d_[A-Za-z0-9_]+',original):
                raise ValueError(f'{sub}/{ses}: unsupported original filename {original!r}')
            eeg=raw_root/sub/ses/'eeg'/f'{sub}_{ses}_task-driving_eeg.set'
            ev=raw_root/sub/ses/'eeg'/f'{sub}_{ses}_task-driving_events.tsv'
            if not eeg.is_file() or not ev.is_file(): raise FileNotFoundError(f'{eeg} or {ev}')
            matches=available.get((original+'.set').lower(),[])+available.get((original+'.set.zip').lower(),[])
            if len(matches)>1: raise ValueError(f'{sub}/{ses}: duplicate official files {matches}')
            sessions.append(Session(sub,ses,original,eeg,ev,matches[0] if matches else None))
    if len(sessions)!=62 or len(set(x.key for x in sessions))!=62:
        raise ValueError(f'Expected exactly 62 sessions, got {len(sessions)}')
    return sessions


@contextlib.contextmanager
def official_set_file(session):
    source=session.preproc_file
    if source is None: raise FileNotFoundError(f'{session.key}: missing {session.original}.set(.zip)')
    if source.suffix.lower()=='.set':
        yield source
        return
    with tempfile.TemporaryDirectory(prefix='e3_p2_official_') as tmp:
        root=Path(tmp)
        with zipfile.ZipFile(source) as z:
            members=[i for i in z.infolist() if not i.is_dir()]
            if not members or sum(i.file_size for i in members)>6*1024**3:
                raise ValueError(f'{source}: empty/oversized archive')
            for info in members:
                target=(root/info.filename.replace('\\','/')).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError(f'Unsafe ZIP path: {info.filename}')
                if info.flag_bits & 0x1: raise ValueError(f'Encrypted ZIP entry: {source}')
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as inp, target.open('wb') as out:
                    shutil.copyfileobj(inp,out,1024*1024)
        files=[p for p in root.rglob('*')
               if p.is_file() and p.name.lower()==(session.original+'.set').lower()]
        if len(files)!=1: raise ValueError(f'{source}: expected {session.original}.set, found {files}')
        yield files[0]


def event_code(s):
    val=str(s).strip().lower()
    words={'deviation_onset_left':251,'deviation_onset_right':252,
           'response_onset':253,'response_offset':254}
    if val in words: return words[val]
    # MNE often uses string annotations '251', 'S 251', 'S251', '251.0'.
    m=re.fullmatch(r'(?:s\s*)?0*(25[1-5])(?:\.0+)?',val)
    return int(m.group(1)) if m else None


def check_session(session, labels):
    import mne
    result={'session':session.key,'original':session.original,
            'official_file':str(session.preproc_file) if session.preproc_file else '',
            'status':'FAIL','error':''}
    if session.preproc_file is None:
        result['error']='Missing official preprocessed .set/.set.zip'
        return result
    b0 = mne.io.read_raw_eeglab(str(session.bids_set), preload=False, verbose='ERROR')
    try:
        with official_set_file(session) as official_path:
            p2 = mne.io.read_raw_eeglab(str(official_path), preload=False, verbose='ERROR')
            try:
                fs1=float(b0.info['sfreq']);fs2=float(p2.info['sfreq'])
                result.update(raw_fs=fs1,official_fs=fs2,
                              raw_samples=b0.n_times,official_samples=p2.n_times,
                              raw_channels=len(b0.ch_names),official_channels=len(p2.ch_names))
                if abs(fs1-FS)>0.001 or abs(fs2-FS)>0.001:
                    raise ValueError(f'Sampling rates must BOTH be 500 Hz (raw {fs1}, P2 {fs2})')
                if b0.n_times!=p2.n_times:
                    raise ValueError(f'Signal length changed: {b0.n_times} vs {p2.n_times}. '
                                     'Cannot reuse original onset seconds without validated time mapping')
                name_map={str(c).upper().strip():c for c in p2.ch_names}
                if len(name_map)!=len(p2.ch_names): raise ValueError('Duplicate case-insensitive channels')
                missing=set(SCALP)-set(name_map)
                if missing: raise ValueError(f'Missing 30 scalp channels: {sorted(missing)}')
                orig_map={str(c).upper().strip():c for c in b0.ch_names}
                if set(SCALP)-set(orig_map): raise ValueError('NEMAR BIDS raw missing canonical channels')
                if not {'FP1','FP2'}.issubset(name_map): raise ValueError('No FP1/FP2 in official dataset')
                dev=sorted([(int(np.rint(t*fs2)), code)
                            for t,d in zip(p2.annotations.onset,p2.annotations.description)
                            if (code:=event_code(d)) in (251,252)])
                sublabels=labels.loc[labels.subject.eq(session.subject)&labels.session.eq(session.session)]
                expected=np.rint(sublabels.onset_s.to_numpy(dtype=float)*FS).astype(np.int64)
                codes=sublabels.deviation_code.to_numpy(dtype=int)
                result['expected_deviations']=len(expected)
                result['official_deviations']=len(dev)
                if len(dev)!=len(expected):
                    raise ValueError(f'Deviation marker count differs: B0 {len(expected)}, P2 {len(dev)}')
                actual=np.array([v[0] for v in dev],dtype=np.int64)
                p2codes=np.array([v[1] for v in dev],dtype=int)
                maxdelta=int(np.abs(actual-expected).max(initial=0))
                result['max_onset_error_samples']=maxdelta
                if not np.array_equal(codes,p2codes) or maxdelta>2:
                    raise ValueError('Deviation codes or onset indices do not match '
                                     f'(max difference {maxdelta} samples); P2 time alignment unverified')
                # Official boundary annotations indicate temporal discontinuities.
                boundaries=np.array([int(np.rint(t*FS)) for t,d in zip(p2.annotations.onset,p2.annotations.description)
                                      if 'boundary' in str(d).lower()],dtype=np.int64)
                usable=sublabels.loc[sublabels.included_in_npz.eq(1)]
                starts=np.rint((usable.onset_s.to_numpy(dtype=float)-N_SEC)*FS).astype(np.int64)
                ends=np.rint(usable.onset_s.to_numpy(dtype=float)*FS).astype(np.int64)
                if (starts<0).any() or (ends>b0.n_times).any():
                    raise ValueError('B0 eligible epochs no longer fit P2 signal length')
                for a,b in zip(starts,ends):
                    if np.any((boundaries>=a)&(boundaries<b)):
                        raise ValueError('Official boundary annotation lies within B0 eligible 4sec window')
                result['eligible_epochs']=int(len(usable))
                result['training_epochs']=int(usable.label.notna().sum())
                # Compare representative samples for units and accidental Raw substitution.
                n=min(FS*2,b0.n_times)
                orig=b0.get_data(picks=[orig_map['FP1'],orig_map['FP2']],start=0,stop=n)
                proc=p2.get_data(picks=[name_map['FP1'],name_map['FP2']],start=0,stop=n)
                if not (np.isfinite(orig).all() and np.isfinite(proc).all()):
                    raise ValueError('Non-finite values in representative samples')
                if np.allclose(orig,proc,rtol=1e-7,atol=1e-14):
                    raise ValueError('Official signal appears identical to Raw; check that P2 files are genuine')
                norm_raw=float(np.sqrt(np.mean(orig**2))); norm_p2=float(np.sqrt(np.mean(proc**2)))
                result['sample_raw_rms_uV']=norm_raw*1e6
                result['sample_p2_rms_uV']=norm_p2*1e6
                if norm_raw>1e-14 and not (0.01 <= norm_p2/norm_raw <= 100.0):
                    raise ValueError('Suspicious signal scale difference; verify EEG units')
                result['status']='PASS'
            finally: p2.close()
    finally: b0.close()
    return result


def preflight(project,raw_root,preproc_root):
    labels,label_file,folds,fold_file=validate_baseline(project)
    sessions=discover(raw_root,preproc_root,labels)
    reports=[]
    # Check missing set files before expensive EEG/ZIP reads.
    missing=[s for s in sessions if s.preproc_file is None]
    if missing:
        for s in sessions:
            reports.append({'session':s.key,'original':s.original,
                            'official_file':str(s.preproc_file) if s.preproc_file else '',
                            'status':'NOT_CHECKED' if s.preproc_file else 'MISSING',
                            'error':'Missing official preprocessed .set(.zip)' if not s.preproc_file else ''})
    else:
        for i,s in enumerate(sessions,1):
            print(f'[CHECK {i}/{len(sessions)}] {s.key} ⇔ {s.original}',flush=True)
            try: r=check_session(s,labels)
            except Exception as e:
                r={'session':s.key,'original':s.original,
                   'official_file':str(s.preproc_file),'status':'FAIL',
                   'error':f'{type(e).__name__}: {e}'}
            reports.append(r)
            print(' ',r['status'],r.get('error',''),flush=True)
    reportdir=project/'results'/'E3'/'P2_preflight'
    reportdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(reports).to_csv(reportdir/'session_check.csv',index=False,encoding='utf-8-sig')
    report={'status':'PASS' if all(x['status']=='PASS' for x in reports) else 'FAIL',
            'raw_root':str(raw_root),'preproc_root':str(preproc_root),
            'expected_sessions':62,'checked_sessions':len(reports),
            'pass':sum(x['status']=='PASS' for x in reports),
            'missing':sum(x['status']=='MISSING' for x in reports),
            'baseline_events':len(labels),
            'baseline_valid_epochs':int(labels.included_in_npz.sum()),
            'baseline_labeled':int(labels.label.notna().sum()),
            'source_label_sha256':sha256(label_file),
            'source_folds_sha256':sha256(fold_file)}
    save_json(reportdir/'preflight.json',report)
    (reportdir/'README.txt').write_text(
        'P2 preflight: '+report['status']+'\n'
        'Official P2 must be separately downloaded from https://doi.org/10.6084/m9.figshare.7666055.v3\n'
        'No ICA, no relabeling, no training happens during this step.\n'
        'See session_check.csv for missing files / signal or event differences.\n',encoding='utf-8')
    return report,sessions,labels,label_file,fold_file


def _input_signature(s,label_sha,script_sha,forehead_only=False):
    return hashlib.sha256(json.dumps([s.key,stat_fingerprint(s.bids_set),
                   stat_fingerprint(s.preproc_file),label_sha,script_sha,forehead_only],sort_keys=True).encode()).hexdigest()


def export_session(s,labels,outdir,label_sha,script_sha,forehead_only=False):
    import mne
    base=outdir/f'{s.key}_unlabeled.npz'
    manifest=outdir/f'{s.key}_manifest.csv'
    done=outdir/f'{s.key}_p2_done.json'
    signature=_input_signature(s,label_sha,script_sha,forehead_only)
    target_names=SCALP[:2] if forehead_only else SCALP
    target_c=len(target_names)
    sub=labels.loc[labels.subject.eq(s.subject)&labels.session.eq(s.session)].copy()
    kept=sub.loc[sub.included_in_npz.eq(1)].sort_values('x_row')
    if done.is_file() and base.is_file() and manifest.is_file():
        try:
            info=json.loads(done.read_text(encoding='utf-8'))
            if info.get('signature')==signature and info.get('epochs')==len(kept):
                with np.load(base,allow_pickle=False) as n:
                    ids=n['trial_index']
                    if (np.array_equal(ids,kept.trial_index.to_numpy(dtype=np.int32))
                            and n[SOURCE_KEY].shape==(len(kept),target_c,1000)):
                        print(f'[RESUME] {s.key}: saved session verified',flush=True)
                        return
        except Exception: pass
    with official_set_file(s) as path:
        raw=mne.io.read_raw_eeglab(str(path),preload=False,verbose='ERROR')
        try:
            fmap={str(c).strip().upper():c for c in raw.ch_names}
            names=[fmap[c] for c in target_names]
            sos=butter(4,[1.,30.],btype='bandpass',fs=FS,output='sos')
            events=pd.read_csv(s.bids_events,sep='\t')
            if not {'onset','value'}.issubset(events.columns):
                raise ValueError(f'{s.bids_events}: events.tsv missing onset/value')
            etimes=pd.to_numeric(events.onset,errors='coerce').to_numpy(dtype=float)
            etimes=etimes[np.isfinite(etimes)]
            event_ix=np.sort(np.rint(etimes*FS).astype(np.int64))
            xs=[]
            for row in kept.itertuples(index=False):
                end=int(round(float(row.onset_s)*FS))
                start=end-N_SEC*FS
                pos=np.searchsorted(event_ix,start,side='left')-1
                boundary=0 if pos<0 else int(event_ix[pos])+1
                begin=max(boundary,start-WARMUP_SEC*FS)
                # Same EVENT-RESTRICTED causal 4th-order 1–30 Hz filter and
                # 500→250Hz resampling as the original B0 extractor.
                seg=raw.get_data(picks=names,start=begin,stop=end)*1e6
                if seg.shape!=(target_c,end-begin) or not np.isfinite(seg).all():
                    raise ValueError(f'{s.key} trial {row.trial_index}: nonfinite or unexpected P2 segment')
                out=resample_poly(sosfilt(sos,seg,axis=-1)[:,-N_SEC*FS:],
                                  1,2,axis=-1,padtype='line')
                if out.shape!=(target_c,1000) or not np.isfinite(out).all():
                    raise ValueError(f'{s.key} trial {row.trial_index}: filtering failed')
                xs.append(out.astype(np.float32))
            arr=np.stack(xs,axis=0)
            tmp=outdir/f'.{s.key}_building.npz'
            try:
                np.savez(tmp,**{SOURCE_KEY:arr},
                         trial_index=kept.trial_index.to_numpy(dtype=np.int32),
                         onset_s=kept.onset_s.to_numpy(dtype=np.float64),
                         rt_s=kept.rt_s.to_numpy(dtype=np.float64),
                         channel_names=np.array(target_names,dtype='U16'),
                         forehead_names=np.array(['FP1','FP2'],dtype='U16'),
                         sfreq_P2=np.array(OUT_FS,dtype=np.float64),
                         source=np.array('official_figshare_7666055'))
                with np.load(tmp,allow_pickle=False) as chk:
                    if chk[SOURCE_KEY].shape!=(len(kept),target_c,1000):
                        raise AssertionError('Saved NPZ has wrong shape')
                    if not np.array_equal(chk['trial_index'],kept.trial_index.to_numpy(dtype=int)):
                        raise AssertionError('Saved trial order incorrect')
                tmp.replace(base)
                sub.to_csv(manifest,index=False,encoding='utf-8-sig')
                save_json(done,{'signature':signature,'epochs':len(kept),
                                'source_file':str(s.preproc_file),
                                'array_key':SOURCE_KEY,'channels':list(target_names),
                                'sampling_rate_hz':OUT_FS,'window_seconds':N_SEC})
            finally:
                tmp.unlink(missing_ok=True)
        finally: raw.close()
    print(f'[EXTRACT] {s.key}: {len(kept)} epochs ×{target_c}×1000',flush=True)


def publish(staged_epochs,staged_labels,epochs_dest,labels_dest):
    if epochs_dest.exists() or labels_dest.exists():
        raise FileExistsError('P2 output exists; refusing to overwrite:\n'
                              f'{epochs_dest}\n{labels_dest}')
    epochs_dest.parent.mkdir(parents=True,exist_ok=True)
    labels_dest.parent.mkdir(parents=True,exist_ok=True)
    staged_epochs.replace(epochs_dest)
    try: staged_labels.replace(labels_dest)
    except Exception:
        epochs_dest.replace(staged_epochs)
        raise


def run(project,raw_root,preproc_root,check_only=True,forehead_only=False):
    project=Path(project).resolve();raw_root=Path(raw_root).resolve();preproc_root=Path(preproc_root).resolve()
    report,sessions,labels,label_file,fold_file=preflight(project,raw_root,preproc_root)
    print('\n[PREFLIGHT]',json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if report['status']!='PASS':
        raise RuntimeError('P2 verification FAILED. See results/E3/P2_preflight/session_check.csv. '
                           'No P2 epochs or labels were published.')
    if check_only:
        print('[PASS] Metadata, 30-channel layout, timeline, original labels and events aligned. '
              'Run with --run to generate P2 epochs.',flush=True)
        return
    epochs_dest=project/'epochs'/'P2_official'/'4sec'
    labels_dest=project/'data'/'labels'/'E3'/'P2_official'/'4sec'
    if epochs_dest.exists() or labels_dest.exists():
        raise FileExistsError('Final P2 target already exists; nothing overwritten. '
                              f'{epochs_dest} | {labels_dest}')
    stage=project/'epochs'/'_P2_official_staging'/'4sec'
    labels_stage=project/'data'/'labels'/'E3'/'_P2_official_staging'/'4sec'
    stage.mkdir(parents=True,exist_ok=True);labels_stage.mkdir(parents=True,exist_ok=True)
    labelsha=sha256(label_file);scriptsha=sha256(Path(__file__))
    for i,s in enumerate(sessions,1):
        print(f'[P2 {i}/{len(sessions)}] {s.key}',flush=True)
        export_session(s,labels,stage,labelsha,scriptsha,forehead_only=forehead_only)
    for s in sessions:
        done=stage/f'{s.key}_p2_done.json'
        if not done.is_file(): raise RuntimeError(f'Missing completion flag: {s.key}')
        with np.load(stage/f'{s.key}_unlabeled.npz',allow_pickle=False) as n:
            expected=labels.loc[labels.subject.eq(s.subject)&labels.session.eq(s.session)&
                                labels.included_in_npz.eq(1)].sort_values('x_row')
            if not np.array_equal(n['trial_index'],expected.trial_index.to_numpy()):
                raise ValueError(f'{s.key}: final trial audit failed')
            if n[SOURCE_KEY].shape!=(len(expected),2 if forehead_only else 30,1000):
                raise ValueError(f'{s.key}: final array audit failed')
    shutil.copy2(label_file,labels_stage/'rt30_labels.csv')
    base_summary=project/'data'/'labels'/'window_length'/'4sec'/'rt30_summary.csv'
    if base_summary.is_file(): shutil.copy2(base_summary,labels_stage/'rt30_summary.csv')
    unlabeled=project/'epochs'/'4sec'/'all_trials_UNLABELED.csv'
    if unlabeled.is_file(): shutil.copy2(unlabeled,stage/'all_trials_UNLABELED.csv')
    source_report=project/'epochs'/'4sec'/'extraction_report.txt'
    if source_report.is_file(): shutil.copy2(source_report,stage/'B0_extraction_report_reference.txt')
    provenance={'name':'E3_P2_official','source_dataset':'figshare_preprocessed_7666055_v3',
                'raw_dataset':str(raw_root),'preproc_root':str(preproc_root),
                'baseline_labels':str(label_file),'baseline_label_sha256':labelsha,
                'folds':str(fold_file),'folds_sha256':sha256(fold_file),
                'session_count':len(sessions),'events':len(labels),
                'eligible_epochs':int(labels.included_in_npz.sum()),
                'labeled_epochs':int(labels.label.notna().sum()),
                'verified_full_channels':list(SCALP),'saved_channels':list(SCALP[:2] if forehead_only else SCALP),
                'source_key':SOURCE_KEY,
                'preparation':'official continuous EEG + event-restricted 1-30Hz causal Butterworth + resample 500->250; same B0 label, no rerank',
                'official_preprocessing_already_done':'1-50 Hz FIR and artifact removal in source release',
                'notes':'Not raw, not ICA; checked exact onset/code alignment before extraction. Default Full30 NPZ supports FP1/FP2; --forehead-only saves compact 2ch.'}
    save_json(stage/'P2_provenance.json',provenance)
    save_json(labels_stage/'P2_provenance.json',provenance)
    (stage/'extraction_report.txt').write_text(
        f'P2 official, {len(sessions)} sessions; {int(labels.included_in_npz.sum())} 4sec windows; '
        f'{int(labels.label.notna().sum())} RT30 training labels.\n'
        'Labels copied byte-for-byte from baseline 4sec table. Original train/val/test folds reused.\n'
        'All extracted arrays are official P2 EEG after matching B0-style 1-30Hz/250Hz processing.\n',encoding='utf-8')
    publish(stage,labels_stage,epochs_dest,labels_dest)
    print(f'[SUCCESS] P2 EEG: {epochs_dest}\n[P2 labels] {labels_dest}\n'
          f'[Training source_key] {SOURCE_KEY}\n[Reports] results/E3/P2_preflight/',flush=True)
