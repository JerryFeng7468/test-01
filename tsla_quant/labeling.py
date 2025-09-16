from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd


def make_labels(price_df: pd.DataFrame, horizon: int = 1, method: Literal["direction", "classification", "returns"] = "direction", threshold: float = 0.0) -> Tuple[pd.Series, pd.Series]:
    """
    Create prediction targets and future returns.

    - direction: binary label 1 if next horizon return > 0, else 0
    - classification: -1, 0, +1 with threshold around zero
    - returns: continuous next-horizon return as y, plus binary direction for evaluation
    Returns (y, y_direction) where y is main label as per method, y_direction is binary direction label.
    """
    df = price_df.copy()
    close = df["AdjClose" if "AdjClose" in df.columns else "Close"].astype(float)
    future_return = close.shift(-horizon) / close - 1.0
    binary_direction = (future_return > 0.0).astype(int)

    if method == "direction":
        y = binary_direction
    elif method == "classification":
        y = np.select(
            [future_return > threshold, future_return < -threshold],
            [1, -1],
            default=0,
        )
    elif method == "returns":
        y = future_return
    else:
        raise ValueError(f"Unknown labeling method: {method}")

    y = pd.Series(y, index=price_df.index, name="y")
    binary_direction = pd.Series(binary_direction, index=price_df.index, name="y_dir")
    return y, binary_direction

