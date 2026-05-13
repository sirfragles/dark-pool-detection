"""FINRA ATS Data Scraper.

Downloads weekly Alternative Trading System (ATS) transparency reports
from FINRA. These reports contain dark pool trading volume by venue
and symbol — the primary public source of US dark pool data.

Data format (weekly CSV):
- Week ending date
- ATS name/MPID
- Security symbol
- Total shares traded
- Number of trades
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)


FINRA_ATS_URL = (
    "https://www.finra.org/finra-data/"
    "browse-catalog/alternative-trading-system"
)


class FinraATSScraper:
    """Scrape FINRA ATS weekly transparency reports."""

    def __init__(self, data_dir: str = "data/raw/finra"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._reports: pd.DataFrame | None = None

    def fetch_weekly_reports(
        self, year: Optional[int] = None, week: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch FINRA ATS weekly data.

        Args:
            year: Report year (defaults to current).
            week: ISO week number (defaults to latest available).

        Returns DataFrame with columns: week, venue, symbol, share_volume, trade_count
        """
        logger.info(f"Fetching FINRA ATS data (year={year}, week={week})")

        try:
            # FINRA provides downloadable CSV files per week
            # Example URL pattern (simplified — real URL varies):
            # https://www.finra.org/finra-data/.../ats-weekly-YYYY-WW.csv

            resp = requests.get(FINRA_ATS_URL, timeout=30)
            resp.raise_for_status()

            # Parse download page for CSV links
            soup = BeautifulSoup(resp.text, "lxml")
            csv_links = [
                a["href"] for a in soup.find_all("a", href=True)
                if a["href"].endswith(".csv") and "ats" in a["href"].lower()
            ]

            if not csv_links:
                logger.warning("No CSV links found on FINRA ATS page")
                return pd.DataFrame(
                    columns=["week", "venue", "symbol", "share_volume", "trade_count"]
                )

            # Download the first/latest CSV
            csv_url = csv_links[0]
            if not csv_url.startswith("http"):
                csv_url = "https://www.finra.org" + csv_url

            logger.info(f"Downloading: {csv_url}")
            df = pd.read_csv(csv_url)

            # Normalize column names (FINRA changes them occasionally)
            df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

            # Map to standard column names
            col_map = {}
            for col in df.columns:
                if "week" in col or "date" in col:
                    col_map[col] = "week"
                elif "ats" in col or "venue" in col or "mpid" in col:
                    col_map[col] = "venue"
                elif "symbol" in col or "ticker" in col or "security" in col:
                    col_map[col] = "symbol"
                elif "share" in col or "volume" in col:
                    col_map[col] = "share_volume"
                elif "trade" in col or "count" in col:
                    col_map[col] = "trade_count"

            df = df.rename(columns=col_map)

            # Keep only standard columns
            keep_cols = [c for c in ["week", "venue", "symbol", "share_volume", "trade_count"] if c in df.columns]
            df = df[keep_cols]

            self._reports = df
            logger.info(f"Downloaded {len(df)} FINRA ATS records")

            # Cache locally
            cache_path = self.data_dir / f"finra_ats_{year or 'latest'}_w{week or 'latest'}.parquet"
            df.to_parquet(cache_path)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch FINRA ATS data: {e}")
            return pd.DataFrame(
                columns=["week", "venue", "symbol", "share_volume", "trade_count"]
            )

    def load_cached(self, year: int, week: int) -> pd.DataFrame:
        """Load previously cached FINRA data."""
        path = self.data_dir / f"finra_ats_{year}_w{week}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    def aggregate_by_symbol(self) -> pd.DataFrame:
        """Aggregate ATS data by symbol."""
        if self._reports is None or self._reports.empty:
            return pd.DataFrame()

        vol_col = "share_volume" if "share_volume" in self._reports.columns else None
        if vol_col is None:
            return pd.DataFrame()

        return (
            self._reports.groupby("symbol")[vol_col]
            .sum()
            .reset_index()
            .sort_values(vol_col, ascending=False)
        )

    def aggregate_by_venue(self) -> pd.DataFrame:
        """Aggregate ATS data by venue (which dark pools are most active)."""
        if self._reports is None or self._reports.empty:
            return pd.DataFrame()

        vol_col = "share_volume" if "share_volume" in self._reports.columns else None
        if vol_col is None:
            return pd.DataFrame()

        return (
            self._reports.groupby("venue")[vol_col]
            .sum()
            .reset_index()
            .sort_values(vol_col, ascending=False)
        )

    @property
    def summary(self) -> dict:
        if self._reports is None or self._reports.empty:
            return {"n_records": 0}

        vol_col = "share_volume" if "share_volume" in self._reports.columns else None
        return {
            "n_records": len(self._reports),
            "n_venues": self._reports["venue"].nunique() if "venue" in self._reports.columns else 0,
            "n_symbols": self._reports["symbol"].nunique() if "symbol" in self._reports.columns else 0,
            "total_volume": int(self._reports[vol_col].sum()) if vol_col else 0,
        }
