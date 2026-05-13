"""Yahoo Finance Lit Market Data Feed.

Fetches OHLCV data from Yahoo Finance for lit market comparison.
Used alongside FINRA ATS data to estimate dark pool activity.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)


class YFinanceFeed:
    """Yahoo Finance market data feed for lit market reference."""

    def __init__(
        self,
        tickers: Optional[list[str]] = None,
        cache_dir: str = "data/raw/yfinance",
        max_cache_age_hours: int = 4,
    ):
        self.tickers = tickers or [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
            "META", "TSLA", "SPY", "QQQ", "IWM",
        ]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_age = timedelta(hours=max_cache_age_hours)
        self._data: dict[str, pd.DataFrame] = {}

    def fetch_intraday(
        self,
        tickers: Optional[list[str]] = None,
        interval: str = "1m",
        period: str = "7d",
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch intraday OHLCV for one or more tickers."""
        tickers = tickers or self.tickers
        results = {}

        for ticker in tickers:
            cache_path = self.cache_dir / f"{ticker}_{interval}_{period}.parquet"

            if use_cache and self._cache_valid(cache_path):
                logger.debug(f"Using cached data for {ticker}")
                df = pd.read_parquet(cache_path)
            else:
                df = self._download_ticker(ticker, interval, period)
                if not df.empty:
                    df.to_parquet(cache_path, index=False)
                    logger.info(f"Cached {ticker} ({len(df)} rows)")

            if not df.empty:
                df = self._enrich(df)
                self._data[ticker] = df
                results[ticker] = df

        logger.info(f"Fetched {len(results)}/{len(tickers)} tickers")
        return results

    def compute_lit_baseline(
        self, df: pd.DataFrame, window: int = 20
    ) -> pd.DataFrame:
        """Compute lit market baseline metrics for anomaly comparison."""
        df = df.copy()
        if "volume" not in df.columns:
            return df

        df["volume_ma"] = df["volume"].rolling(window, min_periods=1).mean()
        df["volume_std"] = df["volume"].rolling(window, min_periods=1).std()
        df["volume_zscore"] = (
            (df["volume"] - df["volume_ma"]) / df["volume_std"].replace(0, np.nan)
        ).fillna(0)

        if all(c in df.columns for c in ["high", "low", "close"]):
            df["spread_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan) * 100

        if "close" in df.columns:
            df["returns"] = df["close"].pct_change().fillna(0)
            df["volatility"] = df["returns"].rolling(window, min_periods=1).std()

        if all(c in df.columns for c in ["open", "high", "low", "close", "volume"]):
            df["vwap_est"] = (
                (df["open"] + df["high"] + df["low"] + df["close"]) / 4
                * df["volume"]
            ).cumsum() / df["volume"].cumsum().replace(0, np.nan)
            df["vwap_deviation_pct"] = (
                (df["close"] - df["vwap_est"]) / df["vwap_est"].replace(0, np.nan) * 100
            ).fillna(0)

        return df

    def detect_volume_anomalies(
        self, df: pd.DataFrame, zscore_threshold: float = 3.0,
    ) -> pd.DataFrame:
        if "volume_zscore" not in df.columns:
            df = self.compute_lit_baseline(df)
        anomalies = df[df["volume_zscore"].abs() > zscore_threshold].copy()
        anomalies["anomaly_type"] = np.where(
            anomalies["volume_zscore"] > 0, "volume_spike", "volume_drop"
        )
        return anomalies

    def estimate_dark_volume(
        self, lit_df: pd.DataFrame, total_volume_estimate: Optional[float] = None,
    ) -> dict:
        lit_volume = float(lit_df["volume"].sum()) if "volume" in lit_df.columns else 0.0
        if total_volume_estimate and total_volume_estimate > 0:
            estimated_dark = max(0, total_volume_estimate - lit_volume)
            dark_pct = estimated_dark / total_volume_estimate * 100
        else:
            total_est = lit_volume / 0.65 if lit_volume > 0 else 0
            estimated_dark = total_est - lit_volume
            dark_pct = 35.0
        return {
            "lit_volume": lit_volume,
            "estimated_dark_volume": estimated_dark,
            "total_volume_estimate": lit_volume + estimated_dark,
            "dark_share_pct": dark_pct,
            "n_bars": len(lit_df),
        }

    def fetch_history(
        self, ticker: str, start: str, end: Optional[str] = None, interval: str = "1h",
    ) -> pd.DataFrame:
        end = end or datetime.now().strftime("%Y-%m-%d")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=end, interval=interval)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            time_col = next((c for c in df.columns if c in ("datetime", "date", "timestamp")), None)
            if time_col and time_col != "timestamp":
                df = df.rename(columns={time_col: "timestamp"})
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype("int64") // 1_000_000
            df["ticker"] = ticker
            return self._enrich(df)
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def _download_ticker(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            time_col = next((c for c in df.columns if c in ("datetime", "date", "timestamp")), None)
            if time_col and time_col != "timestamp":
                df = df.rename(columns={time_col: "timestamp"})
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype("int64") // 1_000_000
            df["ticker"] = ticker
            return df
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            return pd.DataFrame()

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.compute_lit_baseline(df)
        if "timestamp" in df.columns:
            df["hour"] = (df["timestamp"] // 3_600_000) % 24
        return df

    def _cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return (datetime.now() - mtime) < self.max_cache_age

    @property
    def available_tickers(self) -> list[str]:
        return list(self._data.keys())

    def get_data(self, ticker: str) -> Optional[pd.DataFrame]:
        return self._data.get(ticker)
