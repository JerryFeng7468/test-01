from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import CCIIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def build_features(price_df: pd.DataFrame, feature_windows: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Create a robust set of technical features from OHLCV dataframe.
    Returns a dataframe aligned with input index containing feature columns only.
    """
    if feature_windows is None:
        feature_windows = [5, 10, 20, 50]

    df = price_df.copy()
    if "AdjClose" in df.columns:
        close = df["AdjClose"].astype(float)
    else:
        close = df["Close"].astype(float)

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    feats: dict[str, pd.Series] = {}

    # Simple and exponential moving averages and momentum
    for window in feature_windows:
        feats[f"ret_{window}"] = _safe_pct_change(close, periods=window)
        feats[f"sma_{window}"] = close.rolling(window).mean()
        feats[f"ema_{window}"] = EMAIndicator(close=close, window=window).ema_indicator()
        feats[f"vol_{window}"] = _safe_pct_change(close, periods=1).rolling(window).std()
        feats[f"volume_z_{window}"] = ((volume - volume.rolling(window).mean()) / (volume.rolling(window).std() + 1e-9))

    # MACD
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    feats["macd"] = macd.macd()
    feats["macd_signal"] = macd.macd_signal()
    feats["macd_hist"] = macd.macd_diff()

    # RSI, Stoch
    rsi = RSIIndicator(close=close, window=14)
    feats["rsi14"] = rsi.rsi()
    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    feats["stoch_k"] = stoch.stoch()
    feats["stoch_d"] = stoch.stoch_signal()

    # Bollinger
    bb = BollingerBands(close=close, window=20, window_dev=2.0)
    feats["bb_high"] = bb.bollinger_hband()
    feats["bb_low"] = bb.bollinger_lband()
    feats["bb_pct"] = bb.bollinger_pband()

    # ATR, CCI
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    feats["atr14"] = atr.average_true_range()
    cci = CCIIndicator(high=high, low=low, close=close, window=20)
    feats["cci20"] = cci.cci()

    # Price and volume microstructure
    feats["open_close_pct"] = (df["Open"].astype(float) / close - 1.0)
    feats["high_low_range"] = (high / low - 1.0)
    feats["close_high_pct"] = (close / high - 1.0)
    feats["close_low_pct"] = (close / low - 1.0)
    feats["ret_1"] = _safe_pct_change(close, periods=1)

    feature_df = pd.DataFrame(feats, index=df.index)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    return feature_df

