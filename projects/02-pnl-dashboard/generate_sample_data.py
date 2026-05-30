#!/usr/bin/env python3
"""
generate_sample_data.py — one-shot generator for sample P&L dashboard data.

Run from anywhere inside the project:
    python projects/02-pnl-dashboard/generate_sample_data.py

Or from the project root:
    cd projects/02-pnl-dashboard && python generate_sample_data.py

Writes (or overwrites) all CSV files under data/ so the Streamlit dashboard
can be explored on a machine with no real trades and no Bloomberg access.

Portfolio design
----------------
  HY  CHARTER COMM OPS  5.50% 2027  CUSIP 23312NAC1
  HY  FORD MOTOR CREDIT 6.10% 2032  CUSIP 34540XAB1
  IG  APPLE INC         5.00% 2030  CUSIP 037833100
  IG  US TREASURY       4.50% 2033  CUSIP 912828Z78

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


# ── bond definitions (mirrors BondStatic; no src/ import needed) ──────────────

class _Bond(NamedTuple):
    cusip: str
    name: str
    currency: str
    country: str
    coupon_rate: float
    coupon_frequency: int
    day_count_convention: str
    maturity_date: date
    first_coupon_date: date


BONDS: list[_Bond] = [
    _Bond("23312NAC1", "CHARTER COMM OPS 5.5% 2027",  "USD", "US",
          0.055, 2, "30/360",  date(2027, 10, 30), date(2024, 10, 30)),
    _Bond("34540XAB1", "FORD MOTOR CREDIT 6.1% 2032", "USD", "US",
          0.061, 2, "30/360",  date(2032, 8,  15), date(2024, 8,  15)),
    _Bond("037833100", "APPLE INC 5.0% 2030",          "USD", "US",
          0.050, 2, "30/360",  date(2030, 6,  15), date(2024, 6,  15)),
    _Bond("912828Z78", "US TREASURY 4.5% 2033",        "USD", "US",
          0.045, 2, "Act/360", date(2033, 2,  15), date(2024, 2,  15)),
]

_BOND_BY_CUSIP: dict[str, _Bond] = {b.cusip: b for b in BONDS}


# ── accruals helpers (mirrors accruals.py — standalone, no src/ import) ───────

def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return (date(year + 1, 1, 1) - date(year, 12, 1)).days
    return (date(year, month + 1, 1) - date(year, month, 1)).days


def _coupon_dates(bond: _Bond) -> list[date]:
    """All coupon dates from first_coupon_date to maturity (inclusive)."""
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
    """Accrued interest per 100 par, matching the day-count convention."""
    lcd = _last_coupon_date(bond, as_of)
    if bond.day_count_convention == "30/360":
        days = _days_30_360(lcd, as_of)
    else:
        days = (as_of - lcd).days          # actual days for Act/360 and Act/365
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

# (start_price, daily_vol_price_points)
_PRICE_PARAMS: dict[str, tuple[float, float]] = {
    "23312NAC1": (97.50, 0.18),   # HY Charter: moderate vol
    "34540XAB1": (95.25, 0.16),   # HY Ford: moderate vol
    "037833100": (98.75, 0.07),   # IG Apple: low vol
    "912828Z78": (96.00, 0.10),   # UST: low vol, longer duration
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
    """Latest price on or before as_of."""
    sub = prices_df[(prices_df["cusip"] == cusip) & (prices_df["date"] <= as_of.isoformat())]
    if sub.empty:
        return _PRICE_PARAMS[cusip][0]
    return float(sub.iloc[-1]["px_last"])


# ── bonds_static ─────────────────────────────────────────────────────────────

def build_bonds_static() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "cusip": b.cusip, "name": b.name, "currency": b.currency,
            "country": b.country, "coupon_rate": b.coupon_rate,
            "coupon_frequency": b.coupon_frequency,
            "day_count_convention": b.day_count_convention,
            "maturity_date": b.maturity_date.isoformat(),
            "first_coupon_date": b.first_coupon_date.isoformat(),
        }
        for b in BONDS
    ])


# ── initial positions ─────────────────────────────────────────────────────────

# (portfolio, cusip, nominal) as of INCEPTION_DATE
_INCEPTION_SPECS = [
    ("HY", "23312NAC1", 5_000_000),
    ("HY", "34540XAB1", 3_000_000),
    ("IG", "037833100", 10_000_000),
    ("IG", "912828Z78",  8_000_000),
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

# (portfolio, cusip, trader, side, nominal, trade_date)
_TRADE_SPECS: list[tuple] = [
    # ── HY: Charter 5.5% 2027 ─────────────────────────────────────────────────
    # Inception: 5M long as of Apr-30. Buy more on May 5, then reduce position.
    ("HY", "23312NAC1", "ALICE", "buy",  2_000_000, date(2025,  5,  5)),
    ("HY", "23312NAC1", "ALICE", "buy",  1_000_000, date(2025,  7, 15)),
    ("HY", "23312NAC1", "BOB",   "sell", 4_000_000, date(2025,  9, 10)),  # partial close → realized
    ("HY", "23312NAC1", "ALICE", "buy",  2_500_000, date(2025, 11, 20)),
    ("HY", "23312NAC1", "BOB",   "sell", 1_500_000, date(2026,  2, 10)),  # another partial close
    ("HY", "23312NAC1", "ALICE", "buy",    500_000, date(2026,  4, 22)),

    # ── HY: Ford 6.1% 2032 ────────────────────────────────────────────────────
    # Inception: 3M long. Add, trim, rebuild.
    ("HY", "34540XAB1", "BOB",   "buy",  1_500_000, date(2025,  5, 20)),
    ("HY", "34540XAB1", "BOB",   "sell", 2_000_000, date(2025,  8,  5)),  # partial close → realized
    ("HY", "34540XAB1", "ALICE", "buy",  2_000_000, date(2025, 10, 15)),
    ("HY", "34540XAB1", "BOB",   "sell",   500_000, date(2026,  3,  5)),
    ("HY", "34540XAB1", "ALICE", "buy",    500_000, date(2026,  5, 12)),

    # ── IG: Apple 5% 2030 ─────────────────────────────────────────────────────
    # Inception: 10M long. Add then reduce.
    ("IG", "037833100", "CAROL", "buy",  5_000_000, date(2025,  5, 12)),
    ("IG", "037833100", "CAROL", "buy",  2_000_000, date(2025,  8, 18)),
    ("IG", "037833100", "CAROL", "sell", 6_000_000, date(2025, 11,  5)),  # large close → realized
    ("IG", "037833100", "CAROL", "buy",  3_000_000, date(2026,  1, 20)),

    # ── IG: US Treasury 4.5% 2033 ─────────────────────────────────────────────
    # Inception: 8M long. Trim on rate rally, rebuild when rates back up.
    ("IG", "912828Z78", "CAROL", "buy",  4_000_000, date(2025,  5, 28)),
    ("IG", "912828Z78", "CAROL", "sell", 3_000_000, date(2025,  9, 22)),  # rate rally → sell
    ("IG", "912828Z78", "CAROL", "buy",  2_500_000, date(2025, 12, 15)),
    ("IG", "912828Z78", "CAROL", "sell", 1_000_000, date(2026,  4,  1)),
]

# Approximate yield-to-maturity by portfolio / cusip (informational only)
_YTM = {
    "23312NAC1": 5.75,
    "34540XAB1": 6.20,
    "037833100": 5.05,
    "912828Z78": 4.52,
}


def build_trades(prices_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, cusip, trader, side, nominal, trade_date in _TRADE_SPECS:
        bond = _BOND_BY_CUSIP[cusip]
        px = _px(prices_df, cusip, trade_date)

        # accrued interest in dollar terms (paid by buyer, received by seller)
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
            "nominal":      nominal,             # always positive in CSV
            "principal":    principal,
            "net":          net,
            "accrued":      acc_dollar,
            "price":        round(px, 4),
            "yield_closed": _YTM.get(cusip, ""),
            "trade_date":   trade_date.strftime("%m/%d/%y"),   # mm/dd/yy
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
    # sort for deterministic lookup
    prices_df = prices_df.sort_values(["cusip", "date"]).reset_index(drop=True)

    # bonds_static.csv  (static reference — safe to commit)
    bonds_df = build_bonds_static()
    bonds_df.to_csv(DATA_DIR / "bonds_static.csv", index=False)
    print(f"  bonds_static.csv      {len(bonds_df)} bonds")

    # initial_positions.csv  (inception mark — safe to commit)
    init_df = build_initial_positions(prices_df)
    init_df.to_csv(DATA_DIR / "initial_positions.csv", index=False)
    print(f"  initial_positions.csv {len(init_df)} positions as of {INCEPTION_DATE}")

    # price_history.csv  (gitignored — market data)
    prices_df.to_csv(DATA_DIR / "price_history.csv", index=False)
    n_days = len(days)
    print(f"  price_history.csv     {len(prices_df)} rows  "
          f"({n_days} business days × {len(BONDS)} CUSIPs)")

    # prices/manual_prices.csv  (current snapshot = last date in history)
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
    print("  1. cd projects/02-pnl-dashboard")
    print("  2. pip install -r requirements.txt")
    print("  3. streamlit run src/dashboard.py")
    print()
    print("Portfolio summary:")
    for portfolio, group in trades_df.groupby("portfolio"):
        buys  = group[group["side"] == "buy"]["nominal"].sum()
        sells = group[group["side"] == "sell"]["nominal"].sum()
        print(f"  {portfolio}: {len(group)} trades, "
              f"{buys:,.0f} bought, {sells:,.0f} sold")


if __name__ == "__main__":
    main()
