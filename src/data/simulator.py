"""Order Book Simulator — Synthetic market microstructure data.

Generates realistic limit order book snapshots with embedded iceberg
orders, dark pool trades, and informed trader activity for testing
detection algorithms.
"""

from typing import Optional

import numpy as np
import pandas as pd


class OrderBookSimulator:
    """Generate realistic synthetic market data for backtesting.

    Features:
    - Realistic price paths (geometric Brownian motion)
    - Level 2 order book with depth
    - Embedded iceberg orders (native + synthetic)
    - Dark pool trade simulation
    - Informed trader agent
    - Noise traders + market makers
    """

    def __init__(
        self,
        n_tickers: int = 5,
        n_ticks: int = 23400,  # ~1 tick/s for 6.5h
        seed: int = 42,
        initial_prices: Optional[dict[str, float]] = None,
    ):
        self.n_tickers = n_tickers
        self.n_ticks = n_ticks
        self.rng = np.random.default_rng(seed)

        # Tickers
        self.tickers = [f"STOCK_{chr(65 + i)}" for i in range(n_tickers)]

        # Initial prices
        if initial_prices:
            self.prices = initial_prices
        else:
            self.prices = {
                t: 100.0 + i * 50.0 for i, t in enumerate(self.tickers)
            }

    def generate(self) -> pd.DataFrame:
        """Generate complete simulation dataset.

        Returns DataFrame with columns:
            timestamp, ticker, price, bid, ask, bid_size, ask_size,
            volume, trade_side, is_dark, is_iceberg, vpin, trader_type
        """
        records = []

        # Price processes (GBM)
        mu = 0.0  # no drift
        sigma = 0.001  # ~1.5% daily vol
        dt = 1.0 / self.n_ticks
        current_prices = dict(self.prices)

        # Order book levels (per ticker)
        spread_base = {t: 0.001 * p for t, p in self.prices.items()}  # 0.1% spread

        # Iceberg state
        icebergs: dict[str, dict] = {}  # ticker -> {price, side, remaining, peak}

        # Informed trader state
        informed_knowledge: dict[str, float] = {}  # ticker -> target price

        tick = 0
        while tick < self.n_ticks:
            timestamp = tick * 1000  # ms

            for ticker in self.tickers:
                p = current_prices[ticker]
                spread = spread_base[ticker]

                # Random walk for price
                p += p * self.rng.normal(mu * dt, sigma * np.sqrt(dt))
                p = max(p, 1.0)
                current_prices[ticker] = p

                # Bid/Ask
                bid = p - spread / 2
                ask = p + spread / 2

                # Normal depth
                base_size = int(self.rng.lognormal(mean=5.0, sigma=1.0))  # ~150 shares
                bid_size = base_size
                ask_size = base_size

                # =========================================
                # Simulate: Informed Trader
                # =========================================
                trader_type = "noise"
                is_dark = False
                is_iceberg = False
                trade_volume = 0
                trade_side = ""

                if tick % 50 == 0:
                    # Every ~50 ticks, maybe spawn an informed trader
                    if ticker not in informed_knowledge:
                        informed_knowledge[ticker] = p * self.rng.uniform(
                            0.97, 1.03
                        )

                # Informed trader acts
                if ticker in informed_knowledge:
                    target = informed_knowledge[ticker]
                    if abs(p - target) / p > 0.005:  # 0.5% mispricing
                        if self.rng.random() < 0.3:  # 30% chance to trade
                            trade_side = "buy" if p < target else "sell"
                            trade_volume = int(
                                self.rng.lognormal(mean=4.5, sigma=0.8)
                            )
                            trader_type = "informed"
                            # Informed traders use dark pools ~40% of time
                            if self.rng.random() < 0.4:
                                is_dark = True
                            # Clear knowledge after acting
                            if self.rng.random() < 0.1:
                                del informed_knowledge[ticker]

                # =========================================
                # Simulate: Noise Trader
                # =========================================
                if trade_volume == 0 and self.rng.random() < 0.15:
                    trade_side = "buy" if self.rng.random() < 0.5 else "sell"
                    trade_volume = int(self.rng.lognormal(mean=3.0, sigma=1.0))
                    trader_type = "retail" if trade_volume < 200 else "institution"

                # =========================================
                # Simulate: Iceberg Order
                # =========================================
                if (
                    ticker in icebergs
                    and icebergs[ticker]["side"] == trade_side
                    and abs(icebergs[ticker]["price"] - (bid if trade_side == "buy" else ask)) / p < 0.002
                ):
                    is_iceberg = True
                    remaining = icebergs[ticker]["remaining"]
                    peak = icebergs[ticker]["peak"]
                    actual_fill = min(trade_volume, peak)
                    remaining -= actual_fill
                    icebergs[ticker]["remaining"] = remaining

                    if remaining <= 0:
                        del icebergs[ticker]
                    else:
                        # Synthetic iceberg: replenish peak
                        icebergs[ticker]["peak"] = min(peak, remaining)

                # Spawn new iceberg
                if ticker not in icebergs and self.rng.random() < 0.005:
                    side = "buy" if self.rng.random() < 0.5 else "sell"
                    iceberg_price = bid if side == "buy" else ask
                    total_size = int(self.rng.lognormal(mean=8.0, sigma=1.5))  # ~3000
                    peak_size = min(int(total_size * 0.3), 500)
                    icebergs[ticker] = {
                        "price": iceberg_price,
                        "side": side,
                        "remaining": total_size,
                        "peak": peak_size,
                    }

                # =========================================
                # Simulate: Dark pool trade
                # =========================================
                if not is_dark and trade_volume > 0 and self.rng.random() < 0.2:
                    is_dark = True
                    trader_type = "institution" if trade_volume > 500 else trader_type

                # =========================================
                # Record
                # =========================================
                records.append(
                    {
                        "timestamp": timestamp,
                        "ticker": ticker,
                        "price": p,
                        "bid": bid,
                        "ask": ask,
                        "bid_size": bid_size,
                        "ask_size": ask_size,
                        "volume": trade_volume,
                        "trade_side": trade_side,
                        "is_dark": is_dark,
                        "is_iceberg": is_iceberg,
                        "informed_active": ticker in informed_knowledge,
                        "trader_type": trader_type,
                        "iceberg_active": ticker in icebergs,
                    }
                )

            tick += 1

        return pd.DataFrame(records)

    def generate_summary(self, df: pd.DataFrame) -> dict:
        """Generate statistics about the simulated data."""
        trades = df[df["volume"] > 0]
        return {
            "total_ticks": len(df),
            "total_trades": len(trades),
            "dark_trades": int(trades["is_dark"].sum()),
            "iceberg_trades": int(trades["is_iceberg"].sum()),
            "informed_trades": int(
                (trades["trader_type"] == "informed").sum()
            ),
            "dark_pct": float(trades["is_dark"].mean()) * 100,
            "iceberg_pct": float(trades["is_iceberg"].mean()) * 100,
            "informed_pct": float(
                (trades["trader_type"] == "informed").mean()
            ) * 100,
            "n_icebergs_placed": len(
                df[df["iceberg_active"]].drop_duplicates(
                    subset=["ticker", "iceberg_active"]
                )
            ),
            "trader_types": trades["trader_type"].value_counts().to_dict(),
        }
