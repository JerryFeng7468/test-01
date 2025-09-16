import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def _get_cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _make_cache_path(start: str, end: Optional[str], interval: str) -> Path:
    safe_end = end if end is not None else datetime.utcnow().strftime("%Y-%m-%d")
    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"TSLA_{interval}_{start}_{safe_end}.parquet"
    return cache_dir / filename


def fetch_tsla_data(start: str = "2015-01-01", end: Optional[str] = None, interval: str = "1d", force_refresh: bool = False) -> pd.DataFrame:
    """
    Download TSLA OHLCV data from Yahoo Finance with simple disk caching.
    Returns a DataFrame indexed by Datetime with columns: Open, High, Low, Close, Adj Close, Volume
    """
    cache_path = _make_cache_path(start, end, interval)

    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        return df

    retries = 3
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            ticker = yf.Ticker("TSLA")
            df = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
            if df is None or df.empty:
                raise RuntimeError("Empty dataframe returned from yfinance")
            df = df.rename(columns={"Adj Close": "AdjClose"})
            df = df.sort_index()
            df.to_parquet(cache_path)
            return df
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Failed to download TSLA data after {retries} attempts: {last_err}")

