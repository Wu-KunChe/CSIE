"""XGBoost model factory. No PyTorch optimizer or EEGNet architecture used."""
from __future__ import annotations

from xgboost import XGBClassifier


def build_xgboost(config: dict, fold: int, smoke_test: bool = False) -> XGBClassifier:
    kwargs = dict(config)
    seed = int(kwargs.pop('random_seed', 42)) + int(fold)
    if smoke_test:
        kwargs['n_estimators'] = min(int(kwargs['n_estimators']), 20)
        kwargs['early_stopping_rounds'] = min(int(kwargs['early_stopping_rounds']), 5)
    return XGBClassifier(objective='binary:logistic', random_state=seed,
                         **kwargs)
