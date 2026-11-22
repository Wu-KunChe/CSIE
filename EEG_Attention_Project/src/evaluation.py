"""Shared binary evaluation, subject-level results and prediction exports."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import torch


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Evaluate 0=fast RT proxy, 1=slow RT proxy; fixed threshold."""
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if truth.shape != prob.shape or not len(truth):
        raise ValueError('Need nonempty equal-length ground truth and probabilities')
    if not np.isin(truth, [0,1]).all() or not np.isfinite(prob).all():
        raise ValueError('Invalid label or predicted probability')
    if np.any((prob < 0) | (prob > 1)):
        raise ValueError('Probabilities must be within [0,1]')
    pred = (prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0,1]).ravel().tolist()
    n0, n1 = int(np.count_nonzero(truth == 0)), int(np.count_nonzero(truth == 1))
    # Undefined per-class scores are reported as None rather than pretending
    # a one-class subject has a meaningful balanced accuracy.
    specificity = float(tn/n0) if n0 else None
    sensitivity = float(tp/n1) if n1 else None
    ba = float(balanced_accuracy_score(truth,pred)) if n0 and n1 else None
    return {
        'n_samples':len(truth), 'n_label0':n0, 'n_label1':n1,
        'balanced_accuracy':ba,
        'macro_f1':float(f1_score(truth,pred,average='macro',labels=[0,1],zero_division=0)),
        'sensitivity_label1':sensitivity, 'specificity_label0':specificity,
        'tn':tn,'fp':fp,'fn':fn,'tp':tp,
        'threshold':float(threshold),
    }


@torch.no_grad()
def predict_probabilities(model, data_loader, device: str | None = None) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    scores = []
    for X, _ in data_loader:
        logits = model(X.to(device=device,dtype=torch.float32)).reshape(-1)
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty((0,), dtype=np.float32)


def predictions_table(metadata: pd.DataFrame, y_true, y_prob,
                      threshold: float = 0.5) -> pd.DataFrame:
    if not (len(metadata)==len(y_true)==len(y_prob)):
        raise ValueError('Prediction metadata and arrays lengths differ')
    result = metadata.copy().reset_index(drop=True)
    result['y_true'] = np.asarray(y_true,dtype=np.int64)
    result['prob_label1'] = np.asarray(y_prob,dtype=np.float64)
    result['y_pred'] = (result.prob_label1 >= threshold).astype(int)
    return result


def per_subject_metrics(predictions: pd.DataFrame, threshold:float=0.5) -> pd.DataFrame:
    required = {'subject','y_true','prob_label1'}
    if not required.issubset(predictions):
        raise ValueError(f'Missing prediction columns: {required-set(predictions)}')
    rows = []
    for subject, grp in predictions.groupby('subject', sort=True):
        rows.append({'subject':subject, **binary_metrics(grp.y_true,grp.prob_label1,threshold)})
    return pd.DataFrame(rows)


def save_evaluation(output_dir: str | Path, metrics: dict,
                    predictions: pd.DataFrame, subject_metrics: pd.DataFrame) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True,exist_ok=True)
    (path/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    predictions.to_csv(path/'predictions.csv',index=False,encoding='utf-8-sig')
    subject_metrics.to_csv(path/'per_subject_metrics.csv',index=False,encoding='utf-8-sig')
