from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class WalkForwardResult:
    oof_predictions: pd.Series
    oof_probabilities: pd.Series
    feature_importances: pd.DataFrame
    fold_metrics: List[Dict[str, float]]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred))
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:  # noqa: BLE001
        out["roc_auc"] = float("nan")
    return out


def walk_forward_train(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 6,
    model_params: Dict | None = None,
    random_state: int = 42,
) -> Tuple[XGBClassifier, WalkForwardResult]:
    """
    TimeSeriesSplit walk-forward OOF training and evaluation.
    Returns last fitted model and OOF predictions/probabilities.
    """
    if model_params is None:
        model_params = {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "objective": "binary:logistic",
            "n_jobs": -1,
            "tree_method": "hist",
        }

    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_pred = pd.Series(index=X.index, dtype=float)
    oof_prob = pd.Series(index=X.index, dtype=float)
    importances: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, float]] = []
    fitted_model: XGBClassifier | None = None

    for fold_idx, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBClassifier(**model_params)
        model.fit(X_tr, y_tr)

        prob_val = model.predict_proba(X_val)[:, 1]
        pred_val = (prob_val >= 0.5).astype(int)

        oof_prob.iloc[valid_idx] = prob_val
        oof_pred.iloc[valid_idx] = pred_val

        # XGBoost feature importance (gain)
        booster = model.get_booster()
        gain_dict = booster.get_score(importance_type="gain")
        importance_values = [gain_dict.get(f"f{i}", 0.0) for i in range(len(X.columns))]
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": importance_values,
            "fold": fold_idx,
        })
        importances.append(fi)

        fold_metrics.append(_metrics(y_val.values, pred_val, prob_val))
        fitted_model = model

    assert fitted_model is not None

    fi_df = pd.concat(importances, axis=0).reset_index(drop=True)
    result = WalkForwardResult(
        oof_predictions=oof_pred,
        oof_probabilities=oof_prob,
        feature_importances=fi_df,
        fold_metrics=fold_metrics,
    )
    return fitted_model, result


def save_model(model: XGBClassifier, model_path: str) -> None:
    dump(model, model_path)

