"""Window-length study: rerun the existing E1 1D-CNN under 2/3/4/5s.

No EEG extraction, re-filtering, RT relabeling, or random epoch splitting occurs
here. The runner uses NEW per-window RT0-30 labels, fixed subject-wise folds and
preprocessed .npz files, and delegates reusable work to src/ modules.

Outputs (per group/fold): best_model.pt, metrics.json, predictions.csv,
per_subject_metrics.csv, training_history.csv, scaler.json, run_config.json.

Full30 is memory-conscious: a group is assembled one session at a time into
an on-disk temporary .npy array, and training reads it on demand. Temporary
arrays are removed when the group finishes; no extra EEG archive is retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import load_project_config, resolve_project_path
from src.data_loader import iter_session_arrays, read_labeled_table, select_fold_rows
from src.evaluation import (binary_metrics, per_subject_metrics,
                            predict_probabilities, predictions_table, save_evaluation)
from src.model import build_model
from src.preprocessing import ChannelwiseStandardizer
from src.trainer import train_binary_classifier, set_random_seed

BASELINE_GROUPS = ("Full30",)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def _json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for buf in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def _protocol_hash(root: Path, cfg: dict, labels_csv: Path, folds_csv: Path,
                   smoke_test: bool) -> str:
    """Do not reuse completed folds if protocol, inputs, or code change."""
    payload = {
        "baseline": cfg["baseline"],
        "channels": cfg["channels"],
        "label_sha256": _file_sha256(labels_csv),
        "splits_sha256": _file_sha256(folds_csv),
        "source_sha256": {
            str(p.relative_to(root)): _file_sha256(p)
            for p in sorted([* (root / 'src').glob('*.py'),
                             root / 'experiments' / 'window_length' / 'runner.py',
                             root / 'scripts' / 'run_window_length.py',
                             root / 'configs' / 'window_length.yaml'])
        },
        "smoke_test": smoke_test,
        "paths": {"epochs_dir": str(cfg["paths"]["epochs_dir"]),
                  "labels_csv": str(cfg["paths"]["labels_csv"])},
        "npz_stat": [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                     for p in sorted(resolve_project_path(root, cfg["paths"]["epochs_dir"]).glob("*_unlabeled.npz"))],
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _group_signature(protocol_hash: str, group: str, channels: list[str],
                     fold: int) -> str:
    return sha256(json.dumps({"protocol": protocol_hash, "group": group,
                              "channels": channels, "fold": fold},
                             sort_keys=True).encode()).hexdigest()


def _e1_groups(cfg: dict) -> dict[str, list[str]]:
    groups = cfg["channels"]["E1"]
    canonical = [str(x).upper() for x in cfg["channels"]["canonical_30"]]
    result: dict[str, list[str]] = {}
    for name, entry in groups.items():
        ch = canonical if entry == "all" else [str(x).upper() for x in entry]
        if not ch or len(ch) != len(set(ch)) or not set(ch).issubset(set(canonical)):
            raise ValueError(f"Invalid channel group {name}: {ch}")
        result[str(name)] = ch
    if not set(BASELINE_GROUPS).issubset(result):
        raise ValueError("channels.yaml E1 must contain Full30 and Forehead2")
    return result


def _validate_folds(labels: pd.DataFrame, folds: pd.DataFrame,
                    expected_fold_count: int) -> list[int]:
    required = {"fold", "subject", "role"}
    if not required.issubset(folds):
        raise ValueError(f"Folds CSV missing columns: {sorted(required-set(folds))}")
    if folds.duplicated(["fold", "subject"]).any():
        raise ValueError("Fold assignments have duplicate fold/subject pairs")
    if not folds.role.isin({"train", "val", "test"}).all():
        raise ValueError("Invalid role in subject_folds.csv")
    fold_numbers = sorted(int(x) for x in folds.fold.unique())
    if len(fold_numbers) != expected_fold_count:
        raise ValueError(f"Expected {expected_fold_count} folds, found {fold_numbers}")
    subject_set = set(labels.subject.unique())
    for number in fold_numbers:
        ff = folds[folds.fold == number]
        if set(ff.subject) != subject_set:
            raise ValueError(f"Fold {number}: subject assignments incomplete/mismatched")
        if set(ff.role) != {"train", "val", "test"}:
            raise ValueError(f"Fold {number}: missing train/val/test role")
        for role in ("train", "val", "test"):
            role_labels = labels[labels.subject.isin(ff.loc[ff.role == role, "subject"])]
            if len(role_labels) == 0 or set(role_labels.label.unique()) != {0, 1}:
                raise ValueError(f"Fold {number}, {role}: missing one or both classes")
    test_counts = folds[folds.role == "test"].groupby("subject").size()
    if not test_counts.eq(1).all() or set(test_counts.index) != subject_set:
        raise ValueError("Each subject must be TEST in exactly one outer fold")
    return fold_numbers


def _preflight(cfg: dict, groups: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path, Path]:
    root: Path = cfg["project_root"]
    paths = cfg["paths"]
    labels_path = resolve_project_path(root, paths["labels_csv"])
    folds_path = resolve_project_path(root, paths["subject_folds_csv"])
    epochs_dir = resolve_project_path(root, paths["epochs_dir"])
    results_dir = resolve_project_path(root, paths["results_dir"])
    for p in (labels_path, folds_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p} (extract the data.zip first)")
    if not epochs_dir.is_dir():
        raise FileNotFoundError(f"Missing B0 Epoch directory: {epochs_dir}; check configs/paths.yaml")
    labels = read_labeled_table(labels_path)
    folds = pd.read_csv(folds_path)
    _validate_folds(labels, folds, int(cfg["baseline"]["splits"]["outer_folds"]))
    required_npz = sorted(labels.npz_file.unique())
    missing = [name for name in required_npz if not (epochs_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} .npz archives; first: {missing[:3]}")
    expected_samples = int(cfg["baseline"]["input"]["expected_samples"])
    expected_duration = int(cfg["window_seconds"])
    hz = int(cfg["baseline"]["input"]["sampling_rate_hz"])
    if expected_samples != expected_duration * hz:
        raise ValueError(f"Window length/sample mismatch: {expected_duration}s at {hz}Hz != {expected_samples}")
    if not required_npz:
        raise RuntimeError("No labeled NPZ archives found")
    if labels["subject"].nunique() != 27 or labels.groupby(["subject","session"]).ngroups != 62:
        raise ValueError("This NM000275 study expects 27 subjects and 62 sessions")
    if 'labeling_method' not in labels.columns or not labels.labeling_method.eq(
            'session_RT_rank_0to30_vs_70to100').all():
        raise ValueError("Wrong label version: Window Length must use new RT0-30 labels")
    if 'window_seconds' not in labels.columns or not labels.window_seconds.eq(expected_duration).all():
        raise ValueError("Wrong labels CSV window_seconds, refusing to train")
    with np.load(epochs_dir / required_npz[0], allow_pickle=False) as data:
        if "channel_names" not in data or "X_B0_uV" not in data:
            raise ValueError("Input NPZ does not contain expected B0 arrays")
        names = [str(x).upper() for x in data["channel_names"].tolist()]
        requested_channels = {ch for name in groups for ch in _e1_groups(cfg)[name]}
        if not requested_channels.issubset(set(names)):
            raise ValueError("The first NPZ does not contain all requested channels")
        sample_x = data["X_B0_uV"]
        if sample_x.ndim != 3 or sample_x.shape[2] != expected_samples:
            raise ValueError(f"Unexpected X_B0_uV shape: {sample_x.shape}")
    return labels, folds, epochs_dir, results_dir, labels_path, folds_path


@dataclass
class GroupArray:
    path: Path
    metadata: pd.DataFrame
    labels: np.ndarray
    shape: tuple[int, int, int]


def _build_group_array(labeled: pd.DataFrame, epochs_dir: Path,
                       channels: list[str], path: Path,
                       expected_samples: int) -> GroupArray:
    """Single pass through sessions; never concatenate full 30-channel arrays."""
    n = len(labeled)
    arr = np.lib.format.open_memmap(path, mode="w+", dtype="float32",
                                    shape=(n, len(channels), expected_samples))
    metadata, y_parts = [], []
    pos = 0
    for X, y, meta in iter_session_arrays(labeled, epochs_dir, channels,
                                           source_key="X_B0_uV",
                                           expected_samples=expected_samples):
        next_pos = pos + len(y)
        arr[pos:next_pos] = X
        metadata.append(meta)
        y_parts.append(y)
        pos = next_pos
    arr.flush()
    del arr
    if pos != n:
        raise RuntimeError(f"Staging mismatch: expected {n}, got {pos}")
    md = pd.concat(metadata, ignore_index=True)
    ys = np.concatenate(y_parts).astype(np.int64)
    if len(md) != n or not np.array_equal(md.label.to_numpy(), ys):
        raise RuntimeError("Metadata and temporary EEG labels do not align")
    return GroupArray(path, md, ys, (n, len(channels), expected_samples))


class MemmapEEGDataset(Dataset):
    """Lazy read + train-fitted standardization; no duplicated 1GB arrays."""
    def __init__(self, array: np.ndarray, indices: np.ndarray, y: np.ndarray,
                 scaler: ChannelwiseStandardizer):
        self.array = array
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = np.asarray(y[self.indices], dtype=np.int64)
        if scaler.mean_ is None or scaler.std_ is None:
            raise RuntimeError("Train-fitted scaler required")
        self.mean = scaler.mean_.reshape(-1, 1)
        self.std = scaler.std_.reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        raw = self.array[int(self.indices[i])]
        x = (raw - self.mean) / self.std
        # This creates a writable float32 copy, safe for torch.from_numpy.
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.tensor(
            float(self.y[i]), dtype=torch.float32)


class LoggingLoader:
    """Expose dataset.y as expected by src.trainer, print brief live progress."""
    def __init__(self, loader: DataLoader, name: str, every: int = 5):
        self.loader, self.dataset, self.name, self.every = (
            loader, loader.dataset, name, every)
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        if self.epoch == 1 or self.epoch % self.every == 0:
            print(f"    Epoch {self.epoch}: {self.name}...", flush=True)
        return iter(self.loader)


def _role_indices(meta: pd.DataFrame, folds: pd.DataFrame, fold: int,
                  role: str) -> np.ndarray:
    role_subs = set(folds.loc[(folds.fold == fold) & (folds.role == role), "subject"])
    return np.flatnonzero(meta.subject.isin(role_subs).to_numpy())


def _smoke_subset(labels: pd.DataFrame, folds: pd.DataFrame,
                  fold: int, per_subject_label: int = 8) -> pd.DataFrame:
    all_parts = []
    for role in ("train", "val", "test"):
        part = select_fold_rows(labels, folds, fold, role)
        # Deliberately tiny, deterministic and class-diverse; NEVER for research.
        part = part.groupby(["subject", "label"], sort=True).head(per_subject_label)
        all_parts.append(part)
    result = pd.concat(all_parts, ignore_index=True)
    if not len(result):
        raise ValueError("Empty smoke-test subset")
    return result


def _fold_complete(output: Path, signature: str) -> bool:
    needed = ("run_config.json", "metrics.json", "predictions.csv",
              "per_subject_metrics.csv", "best_model.pt", "training_history.csv")
    if not all((output / name).is_file() for name in needed):
        return False
    try:
        return _json_read(output / "run_config.json")["signature"] == signature
    except (OSError, ValueError, KeyError):
        return False


def _prepare_output(output: Path, signature: str, force: bool) -> bool:
    """Skip completed folds; retry interrupted same-config folds safely."""
    if output.exists():
        if not force and _fold_complete(output, signature):
            print(f"  [SKIP] {output} (same settings)", flush=True)
            return True
        if not force and any(output.iterdir()):
            intent = output / "run_intent.json"
            try:
                same_run = _json_read(intent).get("signature") == signature
            except (OSError, ValueError, KeyError):
                same_run = False
            if not same_run:
                raise RuntimeError(
                    f"Existing incomplete/outdated result: {output}. "
                    "Use --force to overwrite this experiment's selected outputs.")
            print(f"  [RESUME] Retrying unfinished fold: {output}", flush=True)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "run_intent.json", {"signature": signature})
    return False


def _save_fold(output: Path, signature: str, group: str,
               channels: list[str], fold: int, train_result: dict,
               scaler: ChannelwiseStandardizer,
               metrics: dict, preds: pd.DataFrame) -> None:
    pd.DataFrame(train_result["history"]).to_csv(
        output / "training_history.csv", index=False)
    _write_json(output / "scaler.json", scaler.to_metadata())
    _write_json(output / "run_config.json", {
        "signature": signature, "group": group, "fold": fold,
        "channels": channels, "source": "X_B0_uV", "labels": "session RT 0-30% vs 70-100% after per-window event rejection",
        "train_only_scaling": True,
        "best_epoch": int(train_result["best_epoch"]),
        "best_val_loss": float(train_result["best_val_loss"]),
        "device": train_result["device"],
        "window_seconds": int(metrics["window_seconds"]),
    })
    save_evaluation(output, metrics, preds, per_subject_metrics(preds))


def _run_fold(array: GroupArray, fold: int, group: str, channels: list[str],
              cfg: dict, folds: pd.DataFrame, output: Path, signature: str,
              device: str, smoke: bool) -> None:
    mmap = np.load(array.path, mmap_mode="r", allow_pickle=False)
    train_ids = _role_indices(array.metadata, folds, fold, "train")
    val_ids = _role_indices(array.metadata, folds, fold, "val")
    test_ids = _role_indices(array.metadata, folds, fold, "test")
    if min(len(train_ids), len(val_ids), len(test_ids)) == 0:
        raise ValueError(f"Fold {fold}: empty train/val/test role")
    if (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids)
            or set(val_ids) & set(test_ids)):
        raise RuntimeError("Epoch train/val/test overlap")
    for role, indices in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        if set(np.unique(array.labels[indices])) != {0, 1}:
            raise ValueError(f"Fold {fold}/{role}: only one label available")
    print(f"  Fold {fold}: train={len(train_ids)}, val={len(val_ids)}, "
          f"test={len(test_ids)}", flush=True)

    base = cfg["baseline"]
    scaler = ChannelwiseStandardizer(float(base["preprocessing"]["std_epsilon"]))
    # Source uses only training subjects to compute channel mean/std.
    scaler.fit_from_batches(mmap[train_ids[start:start+256]]
                            for start in range(0, len(train_ids), 256))
    train_ds = MemmapEEGDataset(mmap, train_ids, array.labels, scaler)
    val_ds = MemmapEEGDataset(mmap, val_ids, array.labels, scaler)
    test_ds = MemmapEEGDataset(mmap, test_ids, array.labels, scaler)
    train_cfg = dict(base["training"])
    if smoke:
        train_cfg["max_epochs"] = 2
        train_cfg["early_stopping_patience"] = 2
    seed = int(train_cfg["random_seed"]) + fold
    train_cfg["random_seed"] = seed
    set_random_seed(seed)
    is_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    torch_device = "cuda" if is_cuda else "cpu"
    batch = int(train_cfg["batch_size"])
    if batch < 1:
        raise ValueError("Batch size must be >= 1")
    kwargs = {"num_workers": 0, "pin_memory": is_cuda}
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, **kwargs)
    model = build_model(base["model"], n_channels=len(channels), seed=seed)
    train_result = train_binary_classifier(
        model, LoggingLoader(train_loader, "train"),
        LoggingLoader(val_loader, "validation"), train_cfg,
        checkpoint_path=output / "best_model.pt", device=torch_device)
    probs = predict_probabilities(train_result["model"], test_loader,
                                  device=torch_device)
    y_true = array.labels[test_ids]
    threshold = float(train_cfg.get("decision_threshold", 0.5))
    metrics = binary_metrics(y_true, probs, threshold)
    metrics.update({"window_seconds": int(cfg["window_seconds"]),
                    "group": group, "fold": fold,
                    "n_train": len(train_ids), "n_val": len(val_ids),
                    "best_epoch": int(train_result["best_epoch"]),
                    "best_val_loss": float(train_result["best_val_loss"]),
                    "train_positive_weight": train_result["train_positive_weight"],
                    "smoke_test": bool(smoke)})
    preds = predictions_table(array.metadata.iloc[test_ids], y_true, probs,
                              threshold)
    # Save signature before metrics, so complete runs are recognized on resume.
    _save_fold(output, signature, group, channels, fold,
               train_result, scaler, metrics, preds)
    print(f"  [DONE] {group}/fold_{fold}: BA={metrics['balanced_accuracy']:.4f}, "
          f"MacroF1={metrics['macro_f1']:.4f}, "
          f"best_epoch={metrics['best_epoch']}", flush=True)
    del train_result, train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds, mmap
    gc.collect()  # Release Windows memmap handles before temp directory cleanup.
    if is_cuda:
        torch.cuda.empty_cache()


def _required_fold_files(output: Path) -> bool:
    """Presence test only. --summarize-only can accept earlier runner versions."""
    files = ("run_config.json", "metrics.json", "predictions.csv",
             "per_subject_metrics.csv", "best_model.pt", "training_history.csv")
    return all((output / f).is_file() for f in files)


def _read_verified_fold(output: Path, group: str, fold: int,
                        channels: list[str], labels: pd.DataFrame,
                        folds: pd.DataFrame, signature: str,
                        allow_legacy: bool, smoke_test: bool = False) -> tuple[dict, pd.DataFrame, pd.DataFrame] | None:
    """Read one finished held-out fold; verify exact trial coverage and RT30 labels.

    Old source-code fingerprints are allowed ONLY during explicit report-only mode.
    We cannot independently recover old optimizer settings from legacy run_config.
    """
    if not _required_fold_files(output):
        return None
    if not allow_legacy and not _fold_complete(output, signature):
        return None
    conf = _json_read(output / "run_config.json")
    if (str(conf.get("group")) != group or int(conf.get("fold", -1)) != fold
            or [str(c).upper() for c in conf.get("channels", [])] != channels
            or conf.get("source") != "X_B0_uV"):
        raise ValueError(f"Stored fold configuration mismatch: {output}")
    metric = _json_read(output / "metrics.json")
    if metric.get("group") != group or int(metric.get("fold", -1)) != fold:
        raise ValueError(f"Stored metrics group/fold mismatch: {output}")
    if bool(metric.get("smoke_test", False)) != smoke_test:
        raise ValueError(f"Smoke/formal result mismatch: {output}")
    pred = pd.read_csv(output / "predictions.csv", encoding="utf-8-sig")
    must = {"subject", "session", "trial_index", "npz_file", "x_row",
            "y_true", "y_pred", "prob_label1"}
    if not must.issubset(pred.columns) or pred.empty:
        raise ValueError(f"Missing columns or empty predictions: {output}")
    keys = ["npz_file", "x_row"]
    if pred.duplicated(keys).any():
        raise ValueError(f"Duplicate test epochs: {output}")
    expected = select_fold_rows(labels, folds, fold, "test")
    expected = expected[keys + ["subject", "session", "trial_index", "label"]]
    checked = pred[keys + ["subject", "session", "trial_index", "y_true"]].merge(
        expected, on=keys, how="left", suffixes=("", "_expected"),
        indicator=True, validate="one_to_one")
    if ((not smoke_test and len(checked) != len(expected))
            or not checked._merge.eq("both").all()
            or not checked.subject.eq(checked.subject_expected).all()
            or not checked.session.eq(checked.session_expected).all()
            or not checked.trial_index.eq(checked.trial_index_expected).all()
            or not checked.y_true.eq(checked.label).all()):
        raise ValueError(f"Predictions do not match current labels/test subjects: {output}")
    threshold = float(metric.get("threshold", 0.5))
    if not np.array_equal(pred.y_pred.to_numpy(dtype=int),
                          (pred.prob_label1.to_numpy() >= threshold).astype(int)):
        raise ValueError(f"Predictions use an unexpected threshold: {output}")
    recomputed = binary_metrics(pred.y_true, pred.prob_label1, threshold)
    for field in ("n_samples", "tn", "fp", "fn", "tp"):
        if recomputed[field] != metric[field]:
            raise ValueError(f"Stored {field} inconsistent with predictions: {output}")
    for field in ("balanced_accuracy", "macro_f1"):
        if not np.isclose(recomputed[field], float(metric[field]), atol=1e-10):
            raise ValueError(f"Stored {field} inconsistent with predictions: {output}")
    subj = per_subject_metrics(pred, threshold=threshold)
    pred.insert(0, "fold", fold)
    return metric, pred, subj


def _write_confusion(path: Path, metrics: dict) -> None:
    confusion = pd.DataFrame({
        "actual_class": ["Label0_fast", "Label1_slow"],
        "predicted_Label0_fast": [metrics["tn"], metrics["fn"]],
        "predicted_Label1_slow": [metrics["fp"], metrics["tp"]],
    })
    confusion.to_csv(path, index=False, encoding="utf-8-sig")


def _plot_comparison(base: Path, summary: pd.DataFrame, experiment: str,
                     ordered_groups: list[str]) -> None:
    """Plot five-fold mean BA ± SD with labels showing the exact bar values."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        print("[PLOT] matplotlib is not installed; CSV/JSON reports were saved")
        return
    present = [g for g in ordered_groups if g in set(summary.group)]
    frame = summary.set_index("group").loc[present]
    fig, ax = plt.subplots(figsize=(max(8, len(present) * 1.55), 5.7))
    x = np.arange(len(present))
    means = frame.mean_BA.to_numpy(dtype=float)
    stdev = pd.to_numeric(frame.std_BA, errors="coerce").fillna(0).to_numpy(dtype=float)
    bars = ax.bar(x, means, yerr=stdev, capsize=4, alpha=0.85,
                  width=0.66, error_kw={"elinewidth": 1.3})
    for i, bar in enumerate(bars):
        y = float(means[i] + stdev[i])
        ax.annotate(f"{means[i]*100:.2f}%", (bar.get_x()+bar.get_width()/2, y),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.75,
               label="Chance level (50%)")
    ax.set_xticks(x, present, rotation=20, ha="right")
    ax.set_ylabel("Balanced Accuracy (mean across held-out folds)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, max(1, float(np.max(means + stdev)) + 0.085))
    ax.set_xlim(-0.85, max(len(present) - 0.15, 0.85))
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)
    incomplete = not frame.all_folds_complete.all()
    ax.set_title(("Window-length 1D-CNN — regional channel comparison") +
                 (" (INCOMPLETE)" if incomplete else ""))
    ax.legend(loc="upper right")
    fig.text(0.5, 0.015,
             "Bars: five-fold mean BA; error bars: fold SD; labels: exact mean BA.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(base / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _aggregate_results(base: Path, cfg: dict, experiment: str,
                       all_folds: list[int], protocol_hash: str,
                       labels: pd.DataFrame, folds: pd.DataFrame,
                       allow_legacy: bool = False, smoke_test: bool = False) -> None:
    """Produce per-group COMBINED five-fold predictions, pooled metrics,
    mean±SD over folds, and a labeled comparison chart. No model fitting.
    """
    all_groups = list(_e1_groups(cfg)) if experiment == "E1" else list(BASELINE_GROUPS)
    rows: list[dict] = []
    subject_rows: list[pd.DataFrame] = []
    combined_summaries: list[dict] = []
    base.mkdir(parents=True, exist_ok=True)
    for group in all_groups:
        channels = _e1_groups(cfg)[group]
        group_dir = base / group
        fold_rows: list[dict] = []
        pred_frames: list[pd.DataFrame] = []
        for fold in all_folds:
            output = group_dir / f"fold_{fold}"
            signature = _group_signature(protocol_hash, group, channels, fold)
            verified = _read_verified_fold(output, group, fold, channels,
                                            labels, folds, signature, allow_legacy, smoke_test)
            if verified is None:
                continue
            metric, pred, subj = verified
            fold_rows.append(metric)
            pred_frames.append(pred)
            subj.insert(0, "group", group)
            subj.insert(1, "fold", fold)
            subject_rows.append(subj)
        if not fold_rows:
            print(f"[REPORT] {group}: no completed folds (group omitted)")
            continue
        detailed_group = pd.DataFrame(fold_rows).sort_values("fold")
        rows.extend(fold_rows)
        combined = pd.concat(pred_frames, ignore_index=True)
        if combined.duplicated(["npz_file", "x_row"]).any():
            raise ValueError(f"Same test epoch appears in multiple folds: {group}")
        if not combined.prob_label1.between(0, 1).all():
            raise ValueError(f"Invalid probabilities in {group}")
        thresholds = detailed_group.threshold.unique()
        if len(thresholds) != 1:
            raise ValueError(f"Cannot pool folds with different thresholds: {group}")
        pooled = binary_metrics(combined.y_true, combined.prob_label1,
                                float(thresholds[0]))
        num_folds = int(detailed_group.fold.nunique())
        complete = (num_folds == len(all_folds) and len(combined) == len(labels))
        if num_folds == len(all_folds) and not complete:
            raise ValueError(f"All folds finished but test set coverage incomplete: {group}")
        ba_sd = float(detailed_group.balanced_accuracy.std(ddof=1)) if num_folds > 1 else None
        f1_sd = float(detailed_group.macro_f1.std(ddof=1)) if num_folds > 1 else None
        s = {
            "group": group, "channels": ",".join(channels),
            "folds_completed": num_folds, "folds_expected": len(all_folds),
            "all_folds_complete": complete,
            "mean_BA": float(detailed_group.balanced_accuracy.mean()),
            "std_BA": ba_sd,
            "mean_macro_F1": float(detailed_group.macro_f1.mean()),
            "std_macro_F1": f1_sd,
            "pooled_BA": pooled["balanced_accuracy"],
            "pooled_macro_F1": pooled["macro_f1"],
            "pooled_specificity_label0": pooled["specificity_label0"],
            "pooled_sensitivity_label1": pooled["sensitivity_label1"],
            "tn": pooled["tn"], "fp": pooled["fp"],
            "fn": pooled["fn"], "tp": pooled["tp"],
            "total_test_predictions": len(combined),
            "n_test_subjects": combined.subject.nunique(),
        }
        combined_summaries.append(s)
        group_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([s]).to_csv(group_dir / "summary.csv", index=False,
                                  encoding="utf-8-sig")
        detailed_group.to_csv(group_dir / "fold_summary.csv", index=False,
                              encoding="utf-8-sig")
        combined.to_csv(group_dir / "combined_predictions.csv", index=False,
                        encoding="utf-8-sig")
        _write_confusion(group_dir / "confusion_matrix.csv", pooled)
        _write_json(group_dir / "combined_metrics.json", {
            "group": group,
            "interpretation": {
                "mean_BA": "Unweighted mean of the fold-specific balanced accuracies",
                "pooled_BA": "Balanced accuracy recomputed using all out-of-fold test predictions",
                "label0": "RT relatively fast within session",
                "label1": "RT relatively slow within session",
            },
            "fold_summary": {"count": num_folds, "expected": len(all_folds),
                             "complete": complete,
                             "mean_BA": s["mean_BA"], "std_BA": s["std_BA"],
                             "mean_macro_F1": s["mean_macro_F1"],
                             "std_macro_F1": s["std_macro_F1"]},
            "pooled_out_of_fold_metrics": pooled,
            "n_test_subjects": s["n_test_subjects"],
        })
        sd_label = f"{ba_sd:.4f}" if ba_sd is not None else "N/A (one fold)"
        (group_dir / "report.txt").write_text(
            f"Group: {group}\nChannels: {s['channels']}\n"
            f"Status: {'COMPLETE' if complete else 'PARTIAL'} "
            f"({num_folds}/{len(all_folds)} folds)\n"
            f"Test predictions: {len(combined):,}; "
            f"test subjects: {s['n_test_subjects']}\n"
            f"Fold mean BA: {s['mean_BA']:.4f} +/- {sd_label}\n"
            f"Fold mean Macro-F1: {s['mean_macro_F1']:.4f}\n"
            f"Pooled out-of-fold BA: {s['pooled_BA']:.4f}\n"
            f"Pooled out-of-fold Macro-F1: {s['pooled_macro_F1']:.4f}\n"
            f"Pooled confusion matrix: TN={s['tn']}, FP={s['fp']}, "
            f"FN={s['fn']}, TP={s['tp']}\n"
            "Mean across folds differs conceptually from pooled out-of-fold metrics.\n",
            encoding="utf-8")
        print(f"[REPORT] {group}: {num_folds}/{len(all_folds)} folds, "
              f"mean BA={s['mean_BA']:.4f}, pooled BA={s['pooled_BA']:.4f}")
    if not combined_summaries:
        raise RuntimeError(f"No complete fold results found under {base}")
    summary = pd.DataFrame(combined_summaries)
    summary.to_csv(base / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(base / "fold_summary.csv", index=False,
                              encoding="utf-8-sig")
    _plot_comparison(base, summary, experiment, all_groups)
    print("\n=== Window-length five-fold summary ===")
    print(summary[["group", "folds_completed", "mean_BA", "std_BA",
                   "pooled_BA", "pooled_macro_F1", "total_test_predictions"]]
          .to_string(index=False, float_format=lambda f: f"{f:.4f}"))
    if subject_rows:
        subject_frame = pd.concat(subject_rows, ignore_index=True)
        subject_frame.to_csv(base / "subject_summary.csv", index=False,
                             encoding="utf-8-sig")
        old_comparison = base / "per_subject_comparison.csv"
        if experiment == "E1":
            ref = (subject_frame.loc[subject_frame.group.eq("Full30"),
                                     ["subject", "balanced_accuracy"]]
                   .rename(columns={"balanced_accuracy": "Full30_BA"}))
            comparison = (subject_frame.loc[~subject_frame.group.eq("Full30")]
                          .merge(ref, on="subject", how="inner"))
            comparison["delta_BA_vs_Full30"] = (
                comparison.balanced_accuracy - comparison.Full30_BA)
            comparison.to_csv(old_comparison, index=False, encoding="utf-8-sig")
        elif old_comparison.is_file():
            old_comparison.unlink()  # B0 has no second channel group to compare.
    (base / "report.txt").write_text(
        f"Experiment: {experiment}\n"
        f"Summary rows: {len(summary)}\n"
        "Group reports: <group>/summary.csv, <group>/combined_metrics.json, "
        "<group>/combined_predictions.csv, <group>/confusion_matrix.csv\n"
        "comparison.png shows five-fold mean BA (numeric labels) +/- SD.\n"
        "pooled_BA is independently recomputed from all held-out test predictions.\n",
        encoding="utf-8")



# ---------------------------------------------------------------------------
# Dedicated window-length orchestration (the original E1 runner is unchanged).
# ---------------------------------------------------------------------------
from copy import deepcopy
from src.config import read_yaml

WINDOWS = (2, 3, 4, 5)
SEVEN_GROUPS = ("Full30", "Forehead2", "Frontal2", "Central2", "Temporal2",
                "Parietal2", "Occipital2")


def _window_config(root: Path, base_config: dict, window_config: dict,
                   seconds: int) -> dict:
    cfg = deepcopy(base_config)
    paths = window_config["paths"]
    hz = int(window_config["preprocessing"]["output_sampling_rate_hz"])
    if hz != 250:
        raise ValueError("Current B0 epochs are expected at 250 Hz")
    cfg["paths"]["epochs_dir"] = str(Path(paths["epochs_root"]) / f"{seconds}sec")
    cfg["paths"]["labels_csv"] = str(Path(paths["labels_root"]) /
                                      f"{seconds}sec" / "rt30_labels.csv")
    cfg["paths"]["subject_folds_csv"] = paths["subject_folds_csv"]
    cfg["paths"]["results_dir"] = paths["results_root"]
    cfg["baseline"]["input"].update({
        "source_key": "X_B0_uV", "sampling_rate_hz": hz,
        "window_seconds": seconds, "expected_samples": seconds * hz,
    })
    cfg["window_seconds"] = seconds
    return cfg


def _aggregate_across_windows(root: Path, out_root: Path, windows: list[int],
                              *, smoke: bool) -> None:
    """Display only held-out test performance. Incomplete runs are marked."""
    records, fold_records, subject_records = [], [], []
    for seconds in windows:
        wbase = out_root / f"{seconds}sec"
        summary_path = wbase / "summary.csv"
        if not summary_path.is_file():
            continue
        summary = pd.read_csv(summary_path)
        if summary.empty:
            continue
        if "window_seconds" in summary.columns:
            if not summary.window_seconds.eq(seconds).all():
                raise ValueError(f"Wrong stored window_seconds in {summary_path}")
        else:
            summary.insert(0, "window_seconds", seconds)
        records.append(summary)
        for filename, collect in (("fold_summary.csv", fold_records),
                                  ("subject_summary.csv", subject_records)):
            path = wbase / filename
            if path.is_file():
                data = pd.read_csv(path)
                if not data.empty:
                    if "window_seconds" in data.columns:
                        if not data.window_seconds.eq(seconds).all():
                            raise ValueError(f"Wrong stored window_seconds in {path}")
                    else:
                        data.insert(0, "window_seconds", seconds)
                    collect.append(data)
    if not records:
        print("[SUMMARY] No finished fold results to summarize", flush=True)
        return
    merged = pd.concat(records, ignore_index=True)
    out_root.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")
    if fold_records:
        pd.concat(fold_records, ignore_index=True).to_csv(
            out_root / "fold_summary.csv", index=False, encoding="utf-8-sig")
    if subject_records:
        pd.concat(subject_records, ignore_index=True).to_csv(
            out_root / "subject_summary.csv", index=False, encoding="utf-8-sig")
    comparison = merged.pivot(index="group", columns="window_seconds", values="mean_BA")
    comparison = comparison.reindex([g for g in SEVEN_GROUPS if g in comparison.index])
    comparison.to_csv(out_root / "window_length_comparison.csv", encoding="utf-8-sig")

    report = ["WINDOW LENGTH 1D-CNN COMPARISON", "", "Model: existing src/model.py CNN1D",
              "Labels: per-session RT0-30% vs RT70-100% AFTER previous-event rejection",
              "CV: same subject-wise 5-fold split for each window; no test subjects in training",
              "Scores: held-out test BA and Macro F1, never training accuracy",
              "Caution: different window lengths have different retained trials and re-ranked labels.",
              "Mean_BA = unweighted average of test-fold BAs; pooled_BA = all OOF test predictions.",
              "", merged[["window_seconds", "group", "folds_completed", "folds_expected",
                           "mean_BA", "std_BA", "pooled_BA", "mean_macro_F1"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}")]
    (out_root / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(comparison.index))
        n = len(windows)
        width = 0.8 / n
        plotted = False
        for i, seconds in enumerate(windows):
            ss = merged.loc[merged.window_seconds.eq(seconds)].set_index("group")
            means = np.array([ss.loc[g, "mean_BA"] if g in ss.index else np.nan
                              for g in comparison.index], dtype=float)
            errs = np.array([ss.loc[g, "std_BA"] if g in ss.index and
                             pd.notna(ss.loc[g, "std_BA"]) else 0.0
                             for g in comparison.index], dtype=float)
            ax.bar(x + (i-(n-1)/2)*width, means, width, yerr=errs,
                   label=f"{seconds} sec", capsize=2)
            plotted = True
        if plotted:
            ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xticks(x, comparison.index, rotation=20, ha="right")
            ax.set_ylim(0, 1.07)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.set_ylabel("Mean balanced accuracy on held-out subjects")
            ax.set_title("Input window length — 1D-CNN / seven channel groups" +
                         (" [SMOKE TEST]" if smoke else ""))
            ax.legend(ncol=min(n, 4))
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_root / "comparison.png", dpi=180)
        plt.close(fig)
    except ImportError:
        print("[PLOT] matplotlib is not installed; CSV reports still saved")
    print(f"\n[SUMMARY] Saved cross-window reports in {out_root}", flush=True)


def run_window_length(project_root: str | Path, *,
                      windows: Sequence[int] | None = None,
                      groups: Sequence[str] | None = None,
                      selected_folds: Sequence[int] | None = None,
                      device: str = "auto", force: bool = False,
                      check: bool = False, smoke_test: bool = False,
                      summarize_only: bool = False) -> None:
    """One command for all 2/3/4/5-second, seven-region, five-fold CNN runs."""
    root = Path(project_root).expanduser().resolve()
    if not (root / "src" / "model.py").is_file():
        raise FileNotFoundError(f"Expected project root containing src/model.py: {root}")
    if sum(bool(x) for x in (check, smoke_test, summarize_only)) > 1:
        raise ValueError("Choose only one of --check/--smoke-test/--summarize-only")
    if summarize_only and force:
        raise ValueError("--force cannot be combined with --summarize-only")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid device")
    if device == "cuda" and not torch.cuda.is_available() and not (check or summarize_only):
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")

    win_yaml = read_yaml(root / "configs" / "window_length.yaml")
    base_cfg = load_project_config(root)
    declared = tuple(int(x) for x in win_yaml["study"]["window_seconds"])
    if declared != WINDOWS:
        raise ValueError(f"window_length.yaml must specify {WINDOWS}, got {declared}")
    selected_windows = list(WINDOWS if not windows else sorted(set(int(x) for x in windows)))
    if set(selected_windows) - set(WINDOWS):
        raise ValueError(f"Invalid windows: {selected_windows}")
    group_map = _e1_groups(base_cfg)
    if set(group_map) != set(SEVEN_GROUPS):
        raise ValueError(f"Expected exactly seven E1 groups: {SEVEN_GROUPS}; got {list(group_map)}")
    chosen_groups = list(SEVEN_GROUPS if not groups else dict.fromkeys(groups))
    if set(chosen_groups) - set(SEVEN_GROUPS):
        raise ValueError(f"Unknown group(s): {set(chosen_groups) - set(SEVEN_GROUPS)}")

    if smoke_test:
        if groups is None:
            chosen_groups = ["Forehead2"]
        if selected_folds is None:
            selected_folds = [1]
    print(f"Project root: {root}\nWindows: {selected_windows}; groups: {chosen_groups}")
    print("[PROTOCOL] Four newly split 2-5s datasets, RT0-30 after event rejection, "
          "fixed subject-wise folds, train-only normalization, original CNN hyperparameters")
    outroot = resolve_project_path(root, win_yaml["paths"]["results_root"])
    if smoke_test:
        outroot = outroot / "_smoke_test"
    scratch = outroot / ".scratch"
    completed_windows = []
    for seconds in selected_windows:
        print("\n" + "=" * 72 + f"\n[WINDOW {seconds} SEC]" + "\n" + "=" * 72,
              flush=True)
        cfg = _window_config(root, base_cfg, win_yaml, seconds)
        labels, folds, epochs_dir, _, labels_path, folds_path = _preflight(cfg, chosen_groups)
        all_folds = sorted(int(x) for x in folds.fold.unique())
        use_folds = (all_folds if selected_folds is None else
                     sorted(set(int(x) for x in selected_folds)))
        if set(use_folds) - set(all_folds):
            raise ValueError(f"Unknown fold(s) {use_folds}; available {all_folds}")
        print(f"Data: {epochs_dir}; Labels: {labels_path}")
        print(f"Labeled {len(labels):,} epochs; test folds {use_folds}")
        print(f"Input shape (C,T) = (selected channels,{seconds*250})")
        if "rt_quality_flag" in labels:
            flagged = int(labels.rt_quality_flag.fillna("ok").ne("ok").sum())
            print(f"[NOTE] {flagged} labeled RT-quality flags remain; no new RT exclusion")
        if check:
            print("[CHECK OK] input file(s), labels, E1 groups, folds and first NPZ shape")
            completed_windows.append(seconds)
            continue

        phash = _protocol_hash(root, cfg, labels_path, folds_path, smoke_test)
        output = outroot / f"{seconds}sec"
        if summarize_only:
            print("[REPORT ONLY] Recompute summaries from existing verified folds")
            _aggregate_results(output, cfg, "E1", all_folds, phash,
                               labels, folds, allow_legacy=False,
                               smoke_test=False)
            completed_windows.append(seconds)
            continue

        output.mkdir(parents=True, exist_ok=True)
        scratch.mkdir(parents=True, exist_ok=True)
        for group in chosen_groups:
            channels = group_map[group]
            pending = []
            for fold in use_folds:
                dest = output / group / f"fold_{fold}"
                signature = _group_signature(phash, group, channels, fold)
                if _fold_complete(dest, signature) and not force:
                    print(f"[SKIP] {seconds}sec/{group}/fold_{fold}: already finished")
                    continue
                if _prepare_output(dest, signature, force):
                    continue
                pending.append((fold, dest, signature))
            if not pending:
                continue
            if smoke_test:
                needed = pd.concat([_smoke_subset(labels, folds, fold)
                                    for fold, _, _ in pending], ignore_index=True)
                needed = needed.drop_duplicates(["subject", "session", "trial_index"])
            else:
                needed = labels
            print(f"\n[STAGE] {seconds}sec/{group}: {len(needed):,} epochs x "
                  f"{len(channels)} ch x {seconds*250} samples", flush=True)
            with tempfile.TemporaryDirectory(prefix=f"eeg_window_{seconds}s_", dir=scratch) as tmp:
                stage = _build_group_array(needed, epochs_dir, channels,
                                           Path(tmp) / "group.npy", seconds*250)
                for fold, dest, signature in pending:
                    _run_fold(stage, fold, group, channels, cfg, folds,
                              dest, signature, device, smoke_test)
            print(f"[CLEAN] Finished {seconds}sec/{group}; temporary EEG memmap removed",
                  flush=True)
        _aggregate_results(output, cfg, "E1", all_folds, phash,
                           labels, folds, allow_legacy=False,
                           smoke_test=smoke_test)
        completed_windows.append(seconds)

    if not check:
        _aggregate_across_windows(root, outroot, list(WINDOWS),
                                  smoke=smoke_test)
        print("\n[DONE] All requested window-length CNN runs have finished.")
    else:
        print("\n[CHECK OK] All requested windows passed basic preflight.")
