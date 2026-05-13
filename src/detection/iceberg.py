"""Iceberg Order Detection — CME Iceberg Detection Method.

Implements the approach from Zotikov (2019): arXiv:1909.09495
"CME Iceberg Order Detection and Prediction"

Detects:
1. Native icebergs — managed by exchange (order book discrepancy)
2. Synthetic icebergs — managed by trader (rapid order resubmission)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class IcebergOrder:
    """Detected iceberg order."""

    id: str
    timestamp: int | float
    price: float
    displayed_size: int
    estimated_total: int
    hidden_size: int
    side: str
    iceberg_type: str
    confidence: float
    fills: list[dict] = field(default_factory=list)
    active: bool = True


class IcebergDetector:
    """Detect iceberg orders from order book snapshots or tick data."""

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        synthetic_window_ms: int = 50,
        sizing_window_ticks: int = 100,
    ):
        self.min_confidence = min_detection_confidence
        self.synthetic_window_ms = synthetic_window_ms
        self.sizing_window_ticks = sizing_window_ticks
        self._active_icebergs: dict[str, IcebergOrder] = {}
        self._order_book: dict[float, dict] = {}
        self._trade_history: list[dict] = []
        self._iceberg_id_counter = 0

    def update_order_book(
        self, timestamp: float,
        bids: list[tuple[float, int]],
        asks: list[tuple[float, int]],
    ) -> None:
        for price, size in bids:
            self._order_book[price] = {"bid_size": size, "ask_size": 0}
        for price, size in asks:
            if price in self._order_book:
                self._order_book[price]["ask_size"] = size
            else:
                self._order_book[price] = {"bid_size": 0, "ask_size": size}

    def process_trade(
        self, timestamp: float, price: float, size: int, side: str,
    ) -> list[IcebergOrder]:
        self._trade_history.append({"time": timestamp, "price": price, "size": size, "side": side})
        if len(self._trade_history) > 10000:
            self._trade_history = self._trade_history[-5000:]

        detected = []
        book_level = self._order_book.get(price, {})
        opposite_side = "ask_size" if side == "buy" else "bid_size"
        resting_volume = book_level.get(opposite_side, 0)

        if resting_volume > 0 and resting_volume >= size:
            post_trade_volume = resting_volume
            if post_trade_volume >= size * 0.8:
                iceberg = self._detect_native_iceberg(timestamp, price, size, side, resting_volume)
                if iceberg and iceberg.confidence >= self.min_confidence:
                    detected.append(iceberg)
                    self._active_icebergs[iceberg.id] = iceberg

        syn_icebergs = self._detect_synthetic_icebergs(timestamp, price, size, side)
        detected.extend(i for i in syn_icebergs if i.confidence >= self.min_confidence)

        for ib in detected:
            ib.fills.append({"time": timestamp, "size": size, "price": price})
        return detected

    def _detect_native_iceberg(
        self, timestamp: float, price: float, trade_size: int,
        side: str, resting_volume: int,
    ) -> Optional[IcebergOrder]:
        if resting_volume >= trade_size:
            discrepancy = resting_volume / max(trade_size, 1)
            confidence = min(0.95, discrepancy / 5.0)
        else:
            confidence = 0.0
        if confidence < 0.3:
            return None
        self._iceberg_id_counter += 1
        estimated_total = self._estimate_iceberg_size(timestamp, price, trade_size, side)
        return IcebergOrder(
            id=f"native_{self._iceberg_id_counter}", timestamp=timestamp,
            price=price, displayed_size=trade_size, estimated_total=estimated_total,
            hidden_size=estimated_total - trade_size, side=side,
            iceberg_type="native", confidence=confidence,
        )

    def _detect_synthetic_icebergs(
        self, timestamp: float, price: float, trade_size: int, side: str,
    ) -> list[IcebergOrder]:
        detected = []
        window_start = timestamp - self.synthetic_window_ms
        recent_trades = [
            t for t in self._trade_history[-200:]
            if window_start <= t["time"] <= timestamp
            and t["price"] == price and t["size"] == trade_size
        ]
        if len(recent_trades) >= 2:
            self._iceberg_id_counter += 1
            estimated_total = trade_size * len(recent_trades) * 3
            confidence = min(0.95, 0.5 + (len(recent_trades) - 1) * 0.15)
            detected.append(IcebergOrder(
                id=f"synthetic_{self._iceberg_id_counter}", timestamp=timestamp,
                price=price, displayed_size=trade_size, estimated_total=estimated_total,
                hidden_size=estimated_total - trade_size, side=side,
                iceberg_type="synthetic", confidence=confidence,
            ))
        return detected

    def _estimate_iceberg_size(
        self, timestamp: float, price: float, trade_size: int, side: str,
    ) -> int:
        recent_related = [
            t for t in self._trade_history[-self.sizing_window_ticks:]
            if abs(t["price"] - price) / max(price, 1) < 0.001 and t["side"] == side
        ]
        if len(recent_related) >= 3:
            total_filled = sum(t["size"] for t in recent_related)
            return max(trade_size * 5, total_filled * 2)
        return trade_size * 10

    @property
    def active_icebergs(self) -> pd.DataFrame:
        if not self._active_icebergs:
            return pd.DataFrame()
        records = []
        for ib in self._active_icebergs.values():
            if ib.active:
                records.append({
                    "id": ib.id, "timestamp": ib.timestamp, "price": ib.price,
                    "displayed": ib.displayed_size, "estimated_total": ib.estimated_total,
                    "hidden": ib.hidden_size, "side": ib.side, "type": ib.iceberg_type,
                    "confidence": ib.confidence, "fills": len(ib.fills),
                })
        return pd.DataFrame(records).sort_values("confidence", ascending=False)

    def summary(self) -> dict:
        active = self.active_icebergs
        if active.empty:
            return {"n_active": 0, "n_native": 0, "n_synthetic": 0,
                    "total_estimated_hidden": 0, "avg_confidence": 0.0}
        return {
            "n_active": len(active),
            "n_native": int((active["type"] == "native").sum()),
            "n_synthetic": int((active["type"] == "synthetic").sum()),
            "total_estimated_hidden": int(active["hidden"].sum()),
            "avg_confidence": float(active["confidence"].mean()),
            "estimated_volume_share": float(
                active["estimated_total"].sum()
                / max(sum(t["size"] for t in self._trade_history[-100:]), 1)
            ),
        }

    def reset(self) -> None:
        self._active_icebergs.clear()
        self._order_book.clear()
        self._trade_history.clear()
        self._iceberg_id_counter = 0
