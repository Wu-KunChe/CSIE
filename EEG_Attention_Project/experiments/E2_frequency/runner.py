"""E2: fixed E1 CNN and fixed 4-second RT0-30 labels; only EEG passband varies.

Uses existing src/{model,trainer,data_loader,preprocessing,evaluation}.py.
Disk-backed temporary arrays keep Full30 memory use bounded. One band/group
is staged at a time, and a completed fold can be safely skipped on resume.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Sequence
import gc
import json
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import read_yaml, load_project_config, resolve_project_path
from src.data_loader import read_labeled_table, iter_session_arrays, select_fold_rows
from src.model import build_model
from src.preprocessing import ChannelwiseStandardizer
from src.trainer import train_binary_classifier, set_random_seed
from src.evaluation import binary_metrics, predict_probabilities, predictions_table, per_subject_metrics, save_evaluation
from .filtering import filter_epochs, make_sos

BANDS = ("theta", "alpha", "beta")
GROUPS = ("Full30", "Forehead2")


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _signature(root: Path, e2: dict, base: dict, labels_path: Path,
               folds_path: Path, epochs_dir: Path, band: str,
               group: str, channels: list[str], fold: int, smoke: bool) -> str:
    sources = [root / "configs" / "e2_frequency.yaml",
               root / "configs" / "baseline.yaml",
               root / "configs" / "channels.yaml",
               root / "scripts" / "run_e2_frequency.py",
               root / "experiments" / "E2_frequency" / "runner.py",
               root / "experiments" / "E2_frequency" / "filtering.py"]
    sources += [root / "src" / f for f in ("model.py", "trainer.py", "preprocessing.py",
                                           "data_loader.py", "evaluation.py", "config.py")]
    payload = {
        "files": {str(p.relative_to(root)): _sha(p) for p in sources},
        "labels_sha": _sha(labels_path), "folds_sha": _sha(folds_path),
        "npz_stat": [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                     for p in sorted(epochs_dir.glob("*_unlabeled.npz"))],
        "e2": e2, "baseline": base["baseline"],
        "band": band, "group": group, "channels": channels, "fold": fold,
        "smoke": smoke,
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _channel_map(base: dict) -> dict[str, list[str]]:
    canon = [str(n).upper() for n in base["channels"]["canonical_30"]]
    e1 = base["channels"]["E1"]
    output = {}
    for group in GROUPS:
        if group not in e1:
            raise ValueError(f"channels.yaml E1 lacks {group}")
        spec = e1[group]
        names = canon if spec == "all" else [str(x).upper() for x in spec]
        if not names or len(names) != len(set(names)) or not set(names).issubset(set(canon)):
            raise ValueError(f"Invalid E1 {group}: {names}")
        output[group] = names
    if len(output["Full30"]) != 30 or len(output["Forehead2"]) != 2:
        raise ValueError("Expected original E1 Full30 and Forehead2 (2 channels)")
    return output


def _preflight(root: Path, e2: dict, base: dict) -> tuple:
    study = e2["study"]
    if (int(study["window_seconds"]) != 4 or int(study["sampling_rate_hz"]) != 250
            or study["source_key"] != "X_B0_uV"):
        raise ValueError("E2 must use 4-second B0, 250 Hz, X_B0_uV")
    if tuple(study["bands"]) != BANDS or tuple(study["groups"]) != GROUPS:
        raise ValueError(f"E2 expected bands={BANDS}, groups={GROUPS}")
    if int(e2["filtering"]["butterworth_order"]) != 4:
        raise ValueError("Expected fixed 4th-order Butterworth filters")
    if e2["filtering"]["type"] != "butterworth_sos_zero_phase_reflect_padded":
        raise ValueError("Unknown frequency filtering method")
    # Fail early if the user's shared baseline YAML was replaced by tuned CNN values.
    # This E2 protocol deliberately uses the unchanged original E1 architecture.
    original_model = {"name": "cnn1d", "conv_filters": [32, 64],
                      "kernels": [51, 15], "dropout": 0.30, "num_classes": 2}
    for field, expected_value in original_model.items():
        if base["baseline"]["model"].get(field) != expected_value:
            raise ValueError(f"Original E1 model changed at {field}; "
                             "E2 must not use Random Search tuned settings")
    original_training = {"optimizer": "AdamW", "learning_rate": 0.001,
                         "weight_decay": 0.0001, "batch_size": 64,
                         "max_epochs": 60, "early_stopping_patience": 8,
                         "class_weighting": "train_only", "decision_threshold": 0.5,
                         "random_seed": 42}
    for field, expected_value in original_training.items():
        if base["baseline"]["training"].get(field) != expected_value:
            raise ValueError(f"Original E1 training changed at {field}; "
                             "E2 must not use Random Search tuned settings")
    expected = 4 * int(study["sampling_rate_hz"])
    paths = e2["paths"]
    epochs_dir = resolve_project_path(root, paths["epochs_dir"])
    labels_path = resolve_project_path(root, paths["labels_csv"])
    folds_path = resolve_project_path(root, paths["subject_folds_csv"])
    outroot = resolve_project_path(root, paths["results_root"])
    reference = resolve_project_path(root, paths["b0_reference_dir"])
    if not epochs_dir.is_dir():
        raise FileNotFoundError(f"Missing 4-second epochs folder: {epochs_dir}")
    for path in (labels_path, folds_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
    labels = read_labeled_table(labels_path)
    if ("window_seconds" not in labels or not labels.window_seconds.eq(4).all()
            or "labeling_method" not in labels or not labels.labeling_method.eq(
                "session_RT_rank_0to30_vs_70to100").all()):
        raise ValueError("E2 requires original per-session RT0-30 4sec labels")
    if (labels.subject.nunique() != 27 or
            labels.groupby(["subject", "session"]).ngroups != 62 or
            len(labels) != 13604):
        raise ValueError("Expected unchanged 4-second dataset: 13604 labels, 27 subjects, 62 sessions")
    folds = pd.read_csv(folds_path)
    must = {"fold", "subject", "role"}
    if not must.issubset(folds.columns) or folds.duplicated(["fold", "subject"]).any():
        raise ValueError("Invalid subject_folds.csv")
    fold_ids = sorted(folds.fold.unique().astype(int).tolist())
    if len(fold_ids) != 5 or not folds.role.isin(["train", "val", "test"]).all():
        raise ValueError("Expected original subject-wise 5-fold train/val/test roles")
    subjects = set(labels.subject.unique())
    for f in fold_ids:
        subfold = folds[folds.fold == f]
        if set(subfold.subject) != subjects or set(subfold.role) != {"train", "val", "test"}:
            raise ValueError(f"Fold {f} lacks complete subject assignments")
        for role in ("train", "val", "test"):
            role_df = select_fold_rows(labels, folds, f, role)
            if set(role_df.label.unique()) != {0, 1}:
                raise ValueError(f"Fold {f}/{role} lacks both labels")
    test_ct = folds[folds.role == "test"].groupby("subject").size()
    if set(test_ct.index) != subjects or not test_ct.eq(1).all():
        raise ValueError("Each subject must be TEST in exactly one fold")
    needed = labels.npz_file.unique().tolist()
    missing = [n for n in needed if not (epochs_dir / str(n)).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} 4sec NPZ files: {missing[:3]}")
    with np.load(epochs_dir / str(needed[0]), allow_pickle=False) as z:
        if study["source_key"] not in z or "channel_names" not in z:
            raise ValueError("First epoch NPZ missing expected arrays")
        x = z[study["source_key"]]
        if x.ndim != 3 or x.shape[1:] != (30, expected):
            raise ValueError(f"Expected [N,30,1000], got {x.shape}")
        if [str(v).upper() for v in z["channel_names"].tolist()] != _channel_map(base)["Full30"]:
            raise ValueError("First NPZ channel_names disagree with canonical_30")
    for band in BANDS:
        lo, hi = study["bands"][band]
        make_sos(float(lo), float(hi), float(study["sampling_rate_hz"]), 4)
    return labels, folds, epochs_dir, labels_path, folds_path, outroot, reference


@dataclass
class Stage:
    path: Path
    labels: np.ndarray
    metadata: pd.DataFrame
    shape: tuple[int, int, int]
    filtering_seconds: float


def _make_stage(labeled: pd.DataFrame, epochs_dir: Path, channels: list[str],
                array_path: Path, band: tuple[float, float], cfg: dict) -> Stage:
    samples = 1000
    fs = float(cfg["study"]["sampling_rate_hz"])
    order = int(cfg["filtering"]["butterworth_order"])
    pad = int(cfg["filtering"]["reflect_pad_samples"])
    sos = make_sos(*band, fs_hz=fs, order=order)
    arr = np.lib.format.open_memmap(array_path, mode="w+", dtype="float32",
                                   shape=(len(labeled), len(channels), samples))
    metadata, y_all = [], []
    at = 0
    filter_seconds = 0.0
    for X, y, meta in iter_session_arrays(labeled, epochs_dir, channels,
                                           source_key="X_B0_uV", expected_samples=samples):
        # Bounded work chunks: only 32 epochs per filtering call, even for Full30.
        for start in range(0, len(y), 32):
            stop = min(start + 32, len(y))
            start_time = perf_counter()
            arr[at + start: at + stop] = filter_epochs(X[start:stop], sos, pad)
            filter_seconds += perf_counter() - start_time
        at += len(y)
        metadata.append(meta)
        y_all.append(y)
    arr.flush()
    del arr
    if at != len(labeled):
        raise RuntimeError("Stage row-count mismatch")
    md = pd.concat(metadata, ignore_index=True)
    ys = np.concatenate(y_all).astype(np.int64)
    if not np.array_equal(md.label.to_numpy(dtype=np.int64), ys):
        raise RuntimeError("Stage metadata/y mismatch")
    return Stage(array_path, ys, md, (len(ys), len(channels), samples), filter_seconds)


class MemmapDataset(Dataset):
    def __init__(self, arr: np.ndarray, idx: np.ndarray, y: np.ndarray,
                 scaler: ChannelwiseStandardizer):
        self.arr, self.idx = arr, np.asarray(idx, dtype=np.int64)
        self.y = y[self.idx].astype(np.int64)
        self.mean = scaler.mean_.reshape(-1, 1)
        self.std = scaler.std_.reshape(-1, 1)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, n: int):
        raw = self.arr[int(self.idx[n])]
        v = np.asarray((raw - self.mean) / self.std, dtype=np.float32)
        return torch.from_numpy(v), torch.tensor(float(self.y[n]), dtype=torch.float32)


class LoggingLoader:
    def __init__(self, loader: DataLoader, name: str):
        self.loader = loader
        self.dataset = loader.dataset
        self.name = name
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        if self.epoch == 1 or self.epoch % 5 == 0:
            print(f"      Epoch {self.epoch}: {self.name}", flush=True)
        return iter(self.loader)


def _indices(md: pd.DataFrame, folds: pd.DataFrame, fold: int, role: str) -> np.ndarray:
    people = set(folds.loc[(folds.fold == fold) & (folds.role == role), "subject"])
    return np.flatnonzero(md.subject.isin(people).to_numpy())


def _smoke_rows(labels: pd.DataFrame, folds: pd.DataFrame, fold: int) -> pd.DataFrame:
    chunks = []
    for role in ("train", "val", "test"):
        group = select_fold_rows(labels, folds, fold, role)
        chunks.append(group.groupby(["subject", "label"], sort=True).head(4))
    return pd.concat(chunks, ignore_index=True).drop_duplicates(
        ["subject", "session", "trial_index"])


def _completed(out: Path, signature: str) -> bool:
    required = ("best_model.pt", "metrics.json", "predictions.csv", "per_subject_metrics.csv",
                "run_config.json", "training_history.csv", "scaler.json")
    if not all((out / name).is_file() for name in required):
        return False
    try:
        return _read_json(out / "run_config.json").get("signature") == signature
    except (OSError, ValueError, KeyError):
        return False


def _prepare(out: Path, signature: str, force: bool) -> bool:
    if out.exists():
        if not force and _completed(out, signature):
            print(f"  [SKIP] {out} already complete", flush=True)
            return True
        if not force and any(out.iterdir()):
            try:
                matching = _read_json(out / "run_intent.json").get("signature") == signature
            except (OSError, ValueError, KeyError):
                matching = False
            if not matching:
                raise RuntimeError(f"Existing result differs from current experiment: {out}. "
                                   "Do not merge settings; use --force only after reviewing outputs")
            print(f"  [RESUME] Repeating unfinished fold: {out}", flush=True)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _save_json(out / "run_intent.json", {"signature": signature})
    return False


def _train_fold(stage: Stage, folds: pd.DataFrame, fold: int,
                band: str, group: str, channels: list[str], base: dict,
                out: Path, signature: str, device: str, smoke: bool) -> None:
    mmap = np.load(stage.path, mmap_mode="r", allow_pickle=False)
    tr, va, te = (_indices(stage.metadata, folds, fold, name)
                  for name in ("train", "val", "test"))
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(f"Empty fold {fold}")
    print(f"    Fold {fold}: train={len(tr):,}, val={len(va):,}, test={len(te):,}", flush=True)
    for role, idx in (("train", tr), ("val", va), ("test", te)):
        if set(stage.labels[idx].tolist()) != {0, 1}:
            raise ValueError(f"Fold {fold}/{role} missing class")
    scaler = ChannelwiseStandardizer(float(base["preprocessing"]["std_epsilon"]))
    scaler.fit_from_batches(mmap[tr[start:start + 256]] for start in range(0, len(tr), 256))
    d_train, d_val, d_test = (MemmapDataset(mmap, idx, stage.labels, scaler)
                              for idx in (tr, va, te))
    opts = dict(base["training"])
    if smoke:
        opts["max_epochs"] = opts["early_stopping_patience"] = 2
    seed = int(opts["random_seed"]) + fold
    opts["random_seed"] = seed
    set_random_seed(seed)
    on_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    target = "cuda" if on_cuda else "cpu"
    n_batch = int(opts["batch_size"])
    loader_kwargs = {"num_workers": 0, "pin_memory": on_cuda}
    train_dl = DataLoader(d_train, batch_size=n_batch, shuffle=True,
                          generator=torch.Generator().manual_seed(seed), **loader_kwargs)
    val_dl = DataLoader(d_val, batch_size=n_batch, shuffle=False, **loader_kwargs)
    test_dl = DataLoader(d_test, batch_size=n_batch, shuffle=False, **loader_kwargs)
    model = build_model(base["model"], n_channels=len(channels), seed=seed)
    count_params = sum(int(p.numel()) for p in model.parameters())
    if on_cuda:
        torch.cuda.synchronize()
    train_start = perf_counter()
    result = train_binary_classifier(
        model, LoggingLoader(train_dl, "train"), LoggingLoader(val_dl, "val"),
        opts, checkpoint_path=out / "best_model.pt", device=target)
    if on_cuda:
        torch.cuda.synchronize()
    train_seconds = perf_counter() - train_start
    if on_cuda:
        torch.cuda.synchronize()
    infer_start = perf_counter()
    prob = predict_probabilities(result["model"], test_dl, device=target)
    if on_cuda:
        torch.cuda.synchronize()
    infer_seconds = perf_counter() - infer_start
    threshold = float(opts.get("decision_threshold", 0.5))
    truth = stage.labels[te]
    metric = binary_metrics(truth, prob, threshold)
    metric.update({"band": band, "group": group, "fold": int(fold),
                   "window_seconds": 4, "n_train": len(tr), "n_val": len(va),
                   "best_epoch": int(result["best_epoch"]),
                   "best_val_loss": float(result["best_val_loss"]),
                   "train_positive_weight": float(result["train_positive_weight"]),
                   "train_seconds": float(train_seconds),
                   "inference_seconds": float(infer_seconds),
                   "inference_ms_per_epoch": float(1000 * infer_seconds / len(te)),
                   "n_model_parameters": count_params, "smoke_test": bool(smoke)})
    pred = predictions_table(stage.metadata.iloc[te], truth, prob, threshold)
    pd.DataFrame(result["history"]).to_csv(out / "training_history.csv", index=False)
    _save_json(out / "scaler.json", scaler.to_metadata())
    save_evaluation(out, metric, pred, per_subject_metrics(pred))
    _save_json(out / "run_config.json", {
        "signature": signature, "band": band, "group": group, "fold": int(fold),
        "channels": channels, "source_key": "X_B0_uV", "sampling_rate_hz": 250,
        "window_seconds": 4, "train_only_scaling": True,
        "model": base["model"], "training": opts,
        "best_epoch": int(result["best_epoch"]), "device": target,
        "filter": "4th-order Butterworth sosfiltfilt, reflect pad 250 per edge"
    })
    print(f"    [DONE] {band}/{group}/fold_{fold} BA={metric['balanced_accuracy']:.4f} "
          f"F1={metric['macro_f1']:.4f} epoch={metric['best_epoch']}", flush=True)
    del result, d_train, d_val, d_test, train_dl, val_dl, test_dl, mmap
    gc.collect()
    if on_cuda:
        torch.cuda.empty_cache()


def _read_fold(out: Path, sig: str, band: str, group: str, fold: int,
               labels: pd.DataFrame, folds: pd.DataFrame, smoke: bool):
    if not _completed(out, sig):
        return None
    cfg = _read_json(out / "run_config.json")
    metric = _read_json(out / "metrics.json")
    if (cfg["band"] != band or cfg["group"] != group or cfg["fold"] != fold or
            metric["band"] != band or metric["group"] != group or metric["fold"] != fold or
            bool(metric["smoke_test"]) != smoke):
        raise ValueError(f"Fold result identity mismatch: {out}")
    pred = pd.read_csv(out / "predictions.csv", encoding="utf-8-sig")
    keys = ["npz_file", "x_row"]
    expected = select_fold_rows(labels, folds, fold, "test")
    if pred.empty or pred.duplicated(keys).any():
        raise ValueError(f"Bad prediction rows: {out}")
    check = pred[keys + ["subject", "session", "trial_index", "y_true"]].merge(
        expected[keys + ["subject", "session", "trial_index", "label"]],
        on=keys, how="left", indicator=True, validate="one_to_one",
        suffixes=("", "_expected"))
    if ((not smoke and len(check) != len(expected)) or
            not check._merge.eq("both").all() or
            not check.subject.eq(check.subject_expected).all() or
            not check.session.eq(check.session_expected).all() or
            not check.trial_index.eq(check.trial_index_expected).all() or
            not check.y_true.eq(check.label).all()):
        raise ValueError(f"Predictions mismatched with subject-wise test labels: {out}")
    computed = binary_metrics(pred.y_true, pred.prob_label1, float(metric["threshold"]))
    for n in ("n_samples", "tn", "fp", "fn", "tp"):
        if computed[n] != metric[n]:
            raise ValueError(f"Stored confusion matrix mismatch: {out}, field={n}")
    for n in ("balanced_accuracy", "macro_f1"):
        if not np.isclose(computed[n], metric[n], atol=1.e-10):
            raise ValueError(f"Stored metric mismatch: {out}, field={n}")
    pred.insert(0, "fold", fold)
    subj = per_subject_metrics(pred)
    subj.insert(0, "fold", fold)
    return metric, pred, subj


def _reference_rows(reference: Path, labels: pd.DataFrame,
                    folds: pd.DataFrame, smoke: bool) -> list[dict]:
    if smoke:
        return []
    rows = []
    for group in GROUPS:
        src = reference / group
        summa, preds = src / "summary.csv", src / "combined_predictions.csv"
        if not (summa.is_file() and preds.is_file()):
            print(f"[REFERENCE] Missing B0 4sec E1 files for {group}: {src}", flush=True)
            continue
        table = pd.read_csv(summa, encoding="utf-8-sig")
        ref_pred = pd.read_csv(preds, encoding="utf-8-sig")
        if len(table) != 1 or len(ref_pred) != len(labels):
            raise ValueError(f"B0 reference is incomplete: {src}")
        expected = labels[["npz_file", "x_row", "subject", "session", "trial_index", "label"]]
        check = ref_pred[["npz_file", "x_row", "subject", "session", "trial_index", "y_true"]].merge(
            expected, on=["npz_file", "x_row"], validate="one_to_one",
            how="inner", suffixes=("", "_expected"))
        if (len(check) != len(labels) or not check.subject.eq(check.subject_expected).all()
                or not check.session.eq(check.session_expected).all()
                or not check.trial_index.eq(check.trial_index_expected).all()
                or not check.y_true.eq(check.label).all()):
            raise ValueError(f"B0 reference test trial IDs/labels differ: {src}")
        if "fold" in ref_pred:
            subject_to_fold = folds.loc[folds.role.eq("test"), ["subject", "fold"]]
            check_folds = ref_pred[["subject", "fold"]].merge(
                subject_to_fold, on="subject", how="left", validate="many_to_one",
                suffixes=("", "_expected"))
            if not check_folds.fold.eq(check_folds.fold_expected).all():
                raise ValueError(f"B0 reference used different subject-wise test folds: {src}")
        # Do not alter E1 files; use their own documented five-fold mean and pooled BA.
        r = table.iloc[0]
        if (not bool(r["all_folds_complete"]) or int(r["folds_completed"]) != 5
                or int(r["total_test_predictions"]) != len(labels)):
            raise ValueError(f"B0 reference not five-fold complete: {src}")
        rows.append({"band": "B0_1to30", "group": group,
                     "folds_completed": 5, "folds_expected": 5, "all_folds_complete": True,
                     "mean_BA": float(r["mean_BA"]), "std_BA": float(r["std_BA"]),
                     "mean_macro_F1": float(r["mean_macro_F1"]),
                     "pooled_BA": float(r["pooled_BA"]),
                     "pooled_macro_F1": float(r["pooled_macro_F1"]),
                     "total_test_predictions": int(r["total_test_predictions"]),
                     "source": "E1_original_4sec_read_only"})
    return rows


def _summarize(root: Path, e2: dict, base: dict, labels: pd.DataFrame,
               folds: pd.DataFrame, epochs: Path, labels_path: Path,
               folds_path: Path, output: Path, reference: Path, smoke: bool) -> None:
    summary_rows, fold_rows, subject_rows = [], [], []
    for band in BANDS:
        for group, channels in _channel_map(base).items():
            directory = output / band / group
            pred_frames, metric_rows = [], []
            for fold in sorted(folds.fold.unique().astype(int).tolist()):
                sig = _signature(root, e2, base, labels_path, folds_path,
                                 epochs, band, group, channels, fold, smoke)
                valid = _read_fold(directory / f"fold_{fold}", sig, band, group,
                                   fold, labels, folds, smoke)
                if valid is None:
                    continue
                metric, pred, subj = valid
                pred_frames.append(pred)
                metric_rows.append(metric)
                subj.insert(0, "band", band)
                subj.insert(1, "group", group)
                subject_rows.append(subj)
                fold_rows.append(metric)
            if not metric_rows:
                continue
            met = pd.DataFrame(metric_rows).sort_values("fold")
            merged = pd.concat(pred_frames, ignore_index=True)
            if merged.duplicated(["npz_file", "x_row"]).any():
                raise RuntimeError(f"Duplicate test epoch across folds: {band}/{group}")
            score = binary_metrics(merged.y_true, merged.prob_label1,
                                   float(met.threshold.iloc[0]))
            complete = len(met) == 5 and len(merged) == len(labels) and not smoke
            if len(met) == 5 and not smoke and not complete:
                raise RuntimeError(f"Complete folds with incomplete coverage: {directory}")
            row = {"band": band, "group": group,
                   "folds_completed": len(met), "folds_expected": 5,
                   "all_folds_complete": complete,
                   "mean_BA": float(met.balanced_accuracy.mean()),
                   "std_BA": float(met.balanced_accuracy.std(ddof=1)) if len(met) > 1 else np.nan,
                   "mean_macro_F1": float(met.macro_f1.mean()),
                   "pooled_BA": float(score["balanced_accuracy"]),
                   "pooled_macro_F1": float(score["macro_f1"]),
                   "total_test_predictions": len(merged),
                   "source": "E2_retrained"}
            summary_rows.append(row)
            directory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(directory / "summary.csv", index=False, encoding="utf-8-sig")
            met.to_csv(directory / "fold_summary.csv", index=False, encoding="utf-8-sig")
            merged.to_csv(directory / "combined_predictions.csv", index=False, encoding="utf-8-sig")
            _save_json(directory / "combined_metrics.json", {
                "fold_mean_BA": row["mean_BA"], "fold_std_BA": None if pd.isna(row["std_BA"]) else row["std_BA"],
                "pooled_out_of_fold": score, "full_five_fold_complete": complete})
            print(f"[REPORT] {band}/{group} {len(met)}/5, BA={row['mean_BA']:.4f}")
    if not summary_rows:
        print("[REPORT] No E2 results available yet")
        return
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(output / "fold_summary.csv", index=False, encoding="utf-8-sig")
    if subject_rows:
        pd.concat(subject_rows, ignore_index=True).to_csv(
            output / "subject_summary.csv", index=False, encoding="utf-8-sig")
    ref = _reference_rows(reference, labels, folds, smoke)
    comp = pd.concat([pd.DataFrame(ref), summary], ignore_index=True)
    b0_scores = {r["group"]: r["mean_BA"] for r in ref}
    comp["delta_BA_vs_B0"] = comp.apply(
        lambda row: row["mean_BA"] - b0_scores[row["group"]]
        if row["group"] in b0_scores and row["band"] != "B0_1to30" else
        (0.0 if row["band"] == "B0_1to30" else np.nan), axis=1)
    comp.to_csv(output / "e2_comparison.csv", index=False, encoding="utf-8-sig")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        fig, ax = plt.subplots(figsize=(9, 5))
        widths = .19
        x = np.arange(4)
        bands = ["B0_1to30", *BANDS]
        for k, group in enumerate(GROUPS):
            vals = []
            for b in bands:
                subset = comp.loc[comp.band.eq(b) & comp.group.eq(group)]
                vals.append(float(subset.mean_BA.iloc[0]) if len(subset) else np.nan)
            ax.bar(x + (k - .5) * widths, vals, widths, label=group)
        ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x, ["B0 (1-30)", "Theta (4-8)", "Alpha (8-13)", "Beta (13-30)"])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean held-out fold Balanced Accuracy")
        ax.set_title("E2: input EEG frequency bands (original 1D-CNN)" +
                     (" [SMOKE]" if smoke else ""))
        ax.legend()
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(output / "comparison.png", dpi=180)
        plt.close(fig)
    except ImportError:
        print("[PLOT] matplotlib not installed; reports saved without PNG")
    report = ["E2 FREQUENCY BAND INPUT STUDY", "",
              "Fixed: 4sec, 250Hz, original CNN (NOT Random Search), RT0-30 per session, 5 subject folds.",
              "Independent input: preprocessed B0 (1-30Hz), then SOS-filtered theta/alpha/beta.",
              "B0 reference: existing window_length/4sec (not retrained/modified).",
              "Filter: 4th-order Butterworth SOS zero-phase, reflect-pad 250 samples inside epoch.",
              "WARNING: filtering after epoch extraction has unavoidable boundary approximation;",
              "re-filtering continuous EEG before epoching would be preferable for physiological conclusions.",
              "Performance = fold mean BA; pooled BA is separately reported.",
              "Timing is machine-dependent and does not include real-time acquisition.",
              "", comp.to_string(index=False, float_format=lambda a: f"{a:.4f}")]
    (output / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_e2(project_root: str | Path, *, bands: Sequence[str] | None = None,
           groups: Sequence[str] | None = None, selected_folds: Sequence[int] | None = None,
           device: str = "auto", check: bool = False, smoke: bool = False,
           summarize_only: bool = False, force: bool = False) -> None:
    root = Path(project_root).expanduser().resolve()
    if sum(map(bool, (check, smoke, summarize_only))) > 1:
        raise ValueError("--check, --smoke-test, --summarize-only are exclusive")
    if force and (check or summarize_only):
        raise ValueError("Cannot --force with --check or --summarize-only")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unknown device")
    if device == "cuda" and not torch.cuda.is_available() and not (check or summarize_only):
        raise RuntimeError("CUDA unavailable")
    e2 = read_yaml(root / "configs" / "e2_frequency.yaml")
    base = load_project_config(root)
    labels, folds, epochs, labels_path, folds_path, outroot, reference = _preflight(root, e2, base)
    chosen_bands = list(BANDS if bands is None else dict.fromkeys(bands))
    chosen_groups = list(GROUPS if groups is None else dict.fromkeys(groups))
    valid_folds = sorted(folds.fold.unique().astype(int).tolist())
    chosen_folds = valid_folds if selected_folds is None else sorted(set(selected_folds))
    if (not chosen_bands or set(chosen_bands) - set(BANDS) or not chosen_groups or
            set(chosen_groups) - set(GROUPS) or not chosen_folds or
            set(chosen_folds) - set(valid_folds)):
        raise ValueError("Unknown/empty band, group or fold selection")
    if smoke:
        if bands is None:
            chosen_bands = ["theta"]
        if groups is None:
            chosen_groups = ["Forehead2"]
        if selected_folds is None:
            chosen_folds = [1]
        outroot = outroot / "_smoke_test"
    if check:
        print(f"[CHECK OK] 4sec NPZ (N,30,1000), 13604 labels/27 subjects/62 sessions, "
              f"five subject folds, E1 Full30/Forehead2, bands={list(BANDS)}")
        print(f"          B0 reference directory: {reference}")
        return
    if summarize_only:
        _summarize(root, e2, base, labels, folds, epochs, labels_path,
                   folds_path, outroot, reference, smoke=False)
        return
    print("[PROTOCOL] E2 fixed original 1D-CNN, 4sec, RT0-30 per session, 250 Hz")
    print(f"[PLAN] bands={chosen_bands}, groups={chosen_groups}, folds={chosen_folds}; "
          f"{len(chosen_bands)*len(chosen_groups)*len(chosen_folds)} CNN trainings maximum")
    scratch = outroot / ".scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    cmap = _channel_map(base)
    for band in chosen_bands:
        band_hz = tuple(float(x) for x in e2["study"]["bands"][band])
        for group in chosen_groups:
            channels = cmap[group]
            needed = []
            for fold in chosen_folds:
                dest = outroot / band / group / f"fold_{fold}"
                sig = _signature(root, e2, base, labels_path, folds_path,
                                 epochs, band, group, channels, fold, smoke)
                if _prepare(dest, sig, force):
                    continue
                needed.append((fold, dest, sig))
            if not needed:
                continue
            subset = labels if not smoke else pd.concat(
                [_smoke_rows(labels, folds, fold) for fold, _, _ in needed],
                ignore_index=True).drop_duplicates(["subject", "session", "trial_index"])
            print(f"[STAGE] {band}/{group} {band_hz}Hz | "
                  f"{len(subset):,} trials × {len(channels)} ch × 1000 samples", flush=True)
            with tempfile.TemporaryDirectory(prefix=f"e2_{band}_{group}_", dir=scratch) as tmp:
                stage_start = perf_counter()
                stage = _make_stage(subset, epochs, channels, Path(tmp) / "band.npy", band_hz, e2)
                stage_seconds = perf_counter() - stage_start
                _save_json(outroot / band / group / "filter_timing.json", {
                    "band_hz": list(band_hz), "group": group,
                    "n_staged_epochs": len(subset), "n_channels": len(channels),
                    "filter_computation_seconds": stage.filtering_seconds,
                    "total_staging_seconds_including_io": stage_seconds,
                    "machine_dependent": True,
                    "offline_reflection_padding": True,
                    "note": "Time for preparing current staged EEG; may cover only smoke subset. "
                            "Not directly comparable to B0 original extraction time."
                })
                print(f"[FILTER] computation {stage.filtering_seconds:.2f}s; "
                      f"staging (with IO) {stage_seconds:.2f}s", flush=True)
                for fold, dest, sig in needed:
                    _train_fold(stage, folds, fold, band, group, channels,
                                base["baseline"], dest, sig, device, smoke)
            print(f"[CLEAN] {band}/{group} temporary EEG removed", flush=True)
    _summarize(root, e2, base, labels, folds, epochs, labels_path,
               folds_path, outroot, reference, smoke)
    print("[DONE] E2 experiment; new results stored only in", outroot)
