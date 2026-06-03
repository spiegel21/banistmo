#!/usr/bin/env python3
"""
generate_sample_data.py — one-shot generator for sample P&L dashboard data.

Run from anywhere inside the project:
    python projects/02-pnl-dashboard/generate_sample_data.py

Or from the project root:
    cd projects/02-pnl-dashboard && python generate_sample_data.py

Writes (or overwrites) CSV files under data/ so the Streamlit dashboard
can be explored before Bloomberg static data is imported.

Portfolio design
----------------
  HY  25714PFB9  (bond static data comes from Bloomberg)
  HY  195325ER2  (bond static data comes from Bloomberg)
  IG  25714PEF1  (bond static data comes from Bloomberg)

bonds_static.csv is intentionally written with only the CUSIP column.
Run the dashboard Bloomberg workflow (Prepare → Excel → Import prices & static)
to populate coupon, maturity, day-count, etc.

Timeline
--------
  Inception mark (initial_positions.csv) : 2025-04-30
  Trades                                 : 2025-05-05 → 2026-05-15
  Price history                          : 2025-04-30 → 2026-05-29

The random walk uses a fixed seed (42) so every run produces identical files.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PRICES_DIR = DATA_DIR / "prices"

INCEPTION_DATE = date(2025, 4, 30)   # end-of-month mark that seeds positions
END_DATE       = date(2026, 5, 29)   # last business day of price history

RNG = np.random.default_rng(42)      # reproducible


# ── bond definitions (internal only — NOT written to bonds_static.csv) ────────
# Bloomberg will supply the real coupon/maturity/day-count via Static sheet.
# These placeholder values are only used to compute sample accrued interest
# and net cash in the generated trades.csv.

class _Bond(NamedTuple):
    cusip: str
    currency: str
    country: str
    coupon_rate: float
    coupon_frequency: int
    day_count_convention: str
    maturity_date: date
    first_coupon_date: date


BONDS: list[_Bond] = [
    _Bond("25714PFB9", "USD", "US", 0.055, 2, "30/360", date(2028, 4, 15), date(2025, 4, 15)),
    _Bond("195325ER2", "USD", "US", 0.060, 2, "30/360", date(2031, 7, 30), date(2025, 7, 30)),
    _Bond("25714PEF1", "USD", "US", 0.050, 2, "30/360", date(2030, 1, 15), date(2025, 1, 15)),
]

_BOND_BY_CUSIP: dict[str, _Bond] = {b.cusip: b for b in BONDS}


# ── accruals helpers (mirrors accruals.py — standalone, no src/ import) ───────

def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return (date(year + 1, 1, 1) - date(year, 12, 1)).days
    return (date(year, month + 1, 1) - date(year, month, 1)).days


def _coupon_dates(bond: _Bond) -> list[date]:
    months = 12 // bond.coupon_frequency
    dates: list[date] = []
    d = bond.first_coupon_date
    while d <= bond.maturity_date:
        dates.append(d)
        m = d.month + months
        y = d.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        d = d.replace(year=y, month=m, day=min(d.day, _days_in_month(y, m)))
    return dates


def _prev_period_start(bond: _Bond) -> date:
    months = 12 // bond.coupon_frequency
    m = bond.first_coupon_date.month - months
    y = bond.first_coupon_date.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    day = min(bond.first_coupon_date.day, _days_in_month(y, m))
    return bond.first_coupon_date.replace(year=y, month=m, day=day)


def _last_coupon_date(bond: _Bond, as_of: date) -> date:
    past = [d for d in _coupon_dates(bond) if d <= as_of]
    return max(past) if past else _prev_period_start(bond)


def _days_30_360(start: date, end: date) -> int:
    d1 = min(start.day, 30)
    d2 = min(end.day, 30) if d1 == 30 else end.day
    return (end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)


def _accrued_pct(bond: _Bond, as_of: date) -> float:
    lcd = _last_coupon_date(bond, as_of)
    if bond.day_count_convention == "30/360":
        days = _days_30_360(lcd, as_of)
    else:
        days = (as_of - lcd).days
    basis = 365.0 if bond.day_count_convention == "Act/365" else 360.0
    return bond.coupon_rate * days / basis


# ── calendar ──────────────────────────────────────────────────────────────────

def business_days(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _t2_settle(trade_date: date) -> date:
    d = trade_date + timedelta(days=2)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ── price simulation ──────────────────────────────────────────────────────────

_PRICE_PARAMS: dict[str, tuple[float, float]] = {
    "25714PFB9": (97.50, 0.18),
    "195325ER2": (95.25, 0.16),
    "25714PEF1": (98.75, 0.07),
}


def generate_price_history(days: list[date]) -> pd.DataFrame:
    """Random walk for each CUSIP; clipped to [82, 108] to stay plausible."""
    n = len(days)
    rows: list[dict] = []
    for cusip, (start, vol) in _PRICE_PARAMS.items():
        shocks = RNG.normal(0.0, vol, n)
        prices = [start]
        for s in shocks[1:]:
            prices.append(max(82.0, min(108.0, prices[-1] + s)))
        for d, px in zip(days, prices):
            rows.append({"date": d.isoformat(), "cusip": cusip, "px_last": round(px, 4)})
    return pd.DataFrame(rows)


def _px(prices_df: pd.DataFrame, cusip: str, as_of: date) -> float:
    sub = prices_df[(prices_df["cusip"] == cusip) & (prices_df["date"] <= as_of.isoformat())]
    if sub.empty:
        return _PRICE_PARAMS[cusip][0]
    return float(sub.iloc[-1]["px_last"])


# ── bonds_static ──────────────────────────────────────────────────────────────
# Only the CUSIP column is written. All other fields (name, coupon, maturity,
# day-count, etc.) are left blank and must be populated via Bloomberg.

def build_bonds_static() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "cusip": b.cusip,
            "name": "", "currency": "", "country": "",
            "coupon_rate": "", "coupon_frequency": "",
            "day_count_convention": "", "maturity_date": "",
            "first_coupon_date": "", "bbg_ticker": "",
        }
        for b in BONDS
    ])


# ── initial positions ─────────────────────────────────────────────────────────

_INCEPTION_SPECS = [
    ("HY", "25714PFB9", 5_000_000),
    ("HY", "195325ER2", 3_000_000),
    ("IG", "25714PEF1", 10_000_000),
]


def build_initial_positions(prices_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, cusip, nominal in _INCEPTION_SPECS:
        px = _px(prices_df, cusip, INCEPTION_DATE)
        book_value = round(-(nominal * px / 100), 2)
        rows.append({
            "portfolio": portfolio,
            "cusip": cusip,
            "nominal": nominal,
            "price": round(px, 4),
            "book_value": book_value,
            "inception_date": INCEPTION_DATE.isoformat(),
        })
    return pd.DataFrame(rows)


# ── trades ────────────────────────────────────────────────────────────────────

_TRADE_SPECS: list[tuple] = [
    # ── HY: 25714PFB9 ─────────────────────────────────────────────────────────
    ("HY", "25714PFB9", "ALICE", "buy",  2_000_000, date(2025,  5,  5)),
    ("HY", "25714PFB9", "ALICE", "buy",  1_000_000, date(2025,  7, 15)),
    ("HY", "25714PFB9", "BOB",   "sell", 4_000_000, date(2025,  9, 10)),
    ("HY", "25714PFB9", "ALICE", "buy",  2_500_000, date(2025, 11, 20)),
    ("HY", "25714PFB9", "BOB",   "sell", 1_500_000, date(2026,  2, 10)),
    ("HY", "25714PFB9", "ALICE", "buy",    500_000, date(2026,  4, 22)),

    # ── HY: 195325ER2 ─────────────────────────────────────────────────────────
    ("HY", "195325ER2", "BOB",   "buy",  1_500_000, date(2025,  5, 20)),
    ("HY", "195325ER2", "BOB",   "sell", 2_000_000, date(2025,  8,  5)),
    ("HY", "195325ER2", "ALICE", "buy",  2_000_000, date(2025, 10, 15)),
    ("HY", "195325ER2", "BOB",   "sell",   500_000, date(2026,  3,  5)),
    ("HY", "195325ER2", "ALICE", "buy",    500_000, date(2026,  5, 12)),

    # ── IG: 25714PEF1 ─────────────────────────────────────────────────────────
    ("IG", "25714PEF1", "CAROL", "buy",  5_000_000, date(2025,  5, 12)),
    ("IG", "25714PEF1", "CAROL", "buy",  2_000_000, date(2025,  8, 18)),
    ("IG", "25714PEF1", "CAROL", "sell", 6_000_000, date(2025, 11,  5)),
    ("IG", "25714PEF1", "CAROL", "buy",  3_000_000, date(2026,  1, 20)),
]

_YTM = {
    "25714PFB9": 5.75,
    "195325ER2": 6.20,
    "25714PEF1": 5.05,
}


def build_trades(prices_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, cusip, trader, side, nominal, trade_date in _TRADE_SPECS:
        bond = _BOND_BY_CUSIP[cusip]
        px = _px(prices_df, cusip, trade_date)

        acc_dollar = round(nominal * _accrued_pct(bond, trade_date), 2)
        principal  = round(nominal * px / 100, 2)

        if side == "buy":
            net = round(-(principal + acc_dollar), 2)
        else:
            net = round(principal + acc_dollar, 2)

        settle = _t2_settle(trade_date)

        rows.append({
            "Timestamp":    f"{trade_date.isoformat()}T09:00:00",
            "cusip":        cusip,
            "side":         side,
            "nominal":      nominal,
            "principal":    principal,
            "net":          net,
            "accrued":      acc_dollar,
            "price":        round(px, 4),
            "yield_closed": _YTM.get(cusip, ""),
            "trade_date":   trade_date.strftime("%m/%d/%y"),
            "settle_date":  settle.strftime("%m/%d/%y"),
            "trader":       trader,
            "portfolio":    portfolio,
        })
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    days = business_days(INCEPTION_DATE, END_DATE)
    prices_df = generate_price_history(days)
    prices_df = prices_df.sort_values(["cusip", "date"]).reset_index(drop=True)

    # bonds_static.csv — CUSIPs only; static data populated via Bloomberg
    bonds_df = build_bonds_static()
    bonds_df.to_csv(DATA_DIR / "bonds_static.csv", index=False)
    print(f"  bonds_static.csv      {len(bonds_df)} CUSIPs (static fields blank — use Bloomberg to populate)")

    # initial_positions.csv
    init_df = build_initial_positions(prices_df)
    init_df.to_csv(DATA_DIR / "initial_positions.csv", index=False)
    print(f"  initial_positions.csv {len(init_df)} positions as of {INCEPTION_DATE}")

    # price_history.csv  (gitignored — market data)
    prices_df.to_csv(DATA_DIR / "price_history.csv", index=False)
    n_days = len(days)
    print(f"  price_history.csv     {len(prices_df)} rows  "
          f"({n_days} business days × {len(BONDS)} CUSIPs)")

    # prices/manual_prices.csv
    latest = prices_df["date"].max()
    manual = (
        prices_df[prices_df["date"] == latest][["cusip", "px_last"]]
        .assign(date=latest)
        .reset_index(drop=True)
    )
    manual.to_csv(PRICES_DIR / "manual_prices.csv", index=False)
    print(f"  manual_prices.csv     current snapshot as of {latest}")

    # trades.csv  (gitignored — real trade data)
    trades_df = build_trades(prices_df)
    trades_df.to_csv(DATA_DIR / "trades.csv", index=False)
    print(f"  trades.csv            {len(trades_df)} trades  "
          f"({trades_df['portfolio'].nunique()} portfolios, "
          f"{trades_df['cusip'].nunique()} CUSIPs)")

    print()
    print(f"All files written to {DATA_DIR.resolve()}")
    print()
    print("Next steps:")
    print("  1. Use the dashboard '① Prepare & open template' to write tickers to Excel")
    print("  2. Bloomberg populates Static sheet → Save & Close Excel")
    print("  3. Click '③ Import prices & static' to fill bonds_static.csv from Bloomberg")
    print("  4. streamlit run src/dashboard.py")


if __name__ == "__main__":
    main()

