from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    accuracy: float
    sharpe: float
    cumulative_return: float
    hit_rate: float


def evaluate_predictions(
    price_df: pd.DataFrame,
    oof_predictions: pd.Series,
    oof_probabilities: pd.Series,
    trading_threshold: float = 0.5,
    horizon: int = 1,
) -> Dict[str, float]:
    close = price_df["AdjClose" if "AdjClose" in price_df.columns else "Close"].astype(float)
    future_return = close.shift(-horizon) / close - 1.0

    aligned = pd.DataFrame({
        "pred": oof_predictions,
        "prob": oof_probabilities,
        "ret": future_return,
    }).dropna()

    # Decision rule: long if prob >= threshold else flat
    positions = (aligned["prob"] >= trading_threshold).astype(int)
    strategy_ret = positions * aligned["ret"]

    # Metrics
    accuracy = float((aligned["pred"] == (aligned["ret"] > 0).astype(int)).mean())
    sharpe = float(np.sqrt(252) * (strategy_ret.mean() / (strategy_ret.std() + 1e-12)))
    cumulative_return = float((1.0 + strategy_ret).prod() - 1.0)
    hit_rate = float((strategy_ret > 0).mean())

    return {
        "accuracy": accuracy,
        "sharpe": sharpe,
        "cumulative_return": cumulative_return,
        "hit_rate": hit_rate,
    }

