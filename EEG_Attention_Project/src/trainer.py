"""Shared CNN fitting loop (E1/E2/E3 runners will call this module)."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _one_pass(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)
    running_loss, count = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32).reshape(-1)
            logits = model(X).reshape(-1)
            if logits.shape != y.shape:
                raise ValueError('Model logits and label dimensions disagree')
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            running_loss += float(loss.item()) * len(y)
            count += len(y)
    if not count:
        raise ValueError('Empty train/validation loader')
    return running_loss / count


def train_binary_classifier(model: nn.Module, train_loader, val_loader,
                            config: dict, checkpoint_path: str | Path | None = None,
                            device: str | None = None) -> dict:
    """Fit on train subjects, early-stop on VAL subjects. Never touch TEST rows.

    Train loader dataset must expose .y for balanced loss weights. Val subject
    labels determine early stopping only, not fit scaler, weight stats or SGD.
    """
    seed = int(config.get('random_seed', 42))
    set_random_seed(seed)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    lr = float(config.get('learning_rate', .001))
    if config.get('optimizer', 'AdamW').lower() != 'adamw':
        raise ValueError('Only AdamW implemented at this stage')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=float(config.get('weight_decay', .0001)))
    train_y = np.asarray(train_loader.dataset.y, dtype=np.int64)
    n_pos = int(np.count_nonzero(train_y == 1))
    n_neg = int(np.count_nonzero(train_y == 0))
    if not n_pos or not n_neg:
        raise ValueError('Training fold needs both binary labels')
    use_weight = config.get('class_weighting', 'train_only') == 'train_only'
    pos_weight = torch.tensor([n_neg/n_pos if use_weight else 1.],
                              dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state = deepcopy(model.state_dict())
    best_loss, best_epoch, failures = float('inf'), 0, 0
    history = []
    patience = int(config.get('early_stopping_patience', 8))
    if patience < 1:
        raise ValueError('Early-stopping patience must be >=1')
    for epoch in range(1, int(config.get('max_epochs',60)) + 1):
        train_loss = _one_pass(model, train_loader, criterion, optimizer, device)
        val_loss = _one_pass(model, val_loader, criterion, None, device)
        history.append({'epoch':epoch, 'train_loss':train_loss, 'val_loss':val_loss})
        if val_loss < best_loss - 1e-7:
            best_epoch, best_loss, failures = epoch, val_loss, 0
            best_state = deepcopy(model.state_dict())
        else:
            failures += 1
            if failures >= patience:
                break
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict':best_state, 'best_epoch':best_epoch,
                    'best_val_loss':best_loss, 'n_train':len(train_y),
                    'n_channels':next(model.parameters()).shape[1]}, path)
    return {'model':model, 'history':history, 'best_epoch':best_epoch,
            'best_val_loss':best_loss, 'train_positive_weight':float(pos_weight.item()),
            'device':str(device)}
