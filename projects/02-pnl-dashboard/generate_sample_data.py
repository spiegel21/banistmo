#!/usr/bin/env python3
"""
generate_sample_data.py — one-shot generator for sample P&L dashboard data.

Run from anywhere inside the project:
    python projects/02-pnl-dashboard/generate_sample_data.py

Or from the project root:
    cd projects/02-pnl-dashboard && python generate_sample_data.py

Writes (or overwrites) CSV files under data/:
  bonds_static.csv       — CUSIPs only; static fields populated via Bloomberg
  initial_positions.csv  — inception holdings as of 2026-01-01
  trades.csv             — 1 sample trade confirmation

Price history is NOT generated here. Use the dashboard Bloomberg workflow:
  1. streamlit run src/dashboard.py
  2. Sidebar → Today's Prices: ① Prepare & open template → Bloomberg → ③ Import prices & static
  3. Sidebar → Price History:  ① Prepare & open history template → Bloomberg → ③ Import history
     (P&L recomputes automatically after import)

Portfolio design
----------------
  HY  25714PFB9  (bond static data comes from Bloomberg)
  HY  195325ER2  (bond static data comes from Bloomberg)
  IG  25714PEF1  (bond static data comes from Bloomberg)
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PRICES_DIR = DATA_DIR / "prices"

INCEPTION_DATE = date(2026, 1, 1)   # inception mark that seeds positions


# ── bond definitions (internal only — NOT written to bonds_static.csv) ────────
# Bloomberg supplies the real coupon/maturity/day-count via the Static sheet.
# These placeholder values are used only to compute accrued interest and net
# cash in the generated trade confirmations.

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

# Representative clean prices (% of par) used for inception position book values.
# Real market prices come from Bloomberg via the import flow.
_SPOT_PRICES: dict[str, float] = {
    "25714PFB9": 107.53,
    "195325ER2": 105.62,
    "25714PEF1":  96.36,
}

# Enterprise reference/classification data per bond. This is *static* reference
# data (not market data), so it belongs in the committed bonds_static.csv —
# leaving it blank makes every held bond fail to load (maturity_date missing)
# and floods the Debug view with "no bond static row" warnings. Credit ratings
# are intentionally left blank: they are authoritative agency data the user must
# supply (Bloomberg RTG_* or manual entry).
class _Static(NamedTuple):
    name: str
    instrument_type: str
    issuer: str
    sector: str
    seniority: str
    market: str          # Local / Global


_STATIC_BY_CUSIP: dict[str, _Static] = {
    "25714PFB9": _Static("Sample HY Corp 5.5% 2028", "Corporate",
                         "Sample HY Issuer", "Financials", "Sr Unsecured", "Local"),
    "195325ER2": _Static("Sample HY Corp 6.0% 2031", "Corporate",
                         "Sample Energy Issuer", "Energy", "Sr Unsecured", "Local"),
    "25714PEF1": _Static("Sample IG Corp 5.0% 2030", "Corporate",
                         "Sample IG Issuer", "Financials", "Sr Unsecured", "Local"),
}


# ── accruals helpers (mirrors accruals.py — standalone, no src/ import) ───────

def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return (date(year + 1, 1, 1) - date(year, 12, 1)).days
    return (date(year, month + 1, 1) - date(year, month, 1)).days


def _coupon_dates(bond: _Bond) -> list[date]:
    if not bond.coupon_frequency:   # non-coupon-bearing: no schedule
        return []
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
    if not bond.coupon_frequency:   # no period to step back to
        return bond.first_coupon_date
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

def _t2_settle(trade_date: date) -> date:
    d = trade_date + timedelta(days=2)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ── bonds_static ──────────────────────────────────────────────────────────────
# Full static reference data (indenture terms + enterprise classification) is
# written so every held bond loads cleanly and the Debug view is meaningful.
# Only the credit ratings are left blank — those are authoritative agency data
# the user supplies via Bloomberg (RTG_SP/RTG_MOODY/RTG_FITCH) or manual entry.

def build_bonds_static() -> pd.DataFrame:
    rows = []
    for b in BONDS:
        s = _STATIC_BY_CUSIP[b.cusip]
        rows.append({
            "cusip": b.cusip,
            "name": s.name,
            "currency": b.currency,
            "country": b.country,
            "coupon_rate": b.coupon_rate,
            "coupon_frequency": b.coupon_frequency,
            "day_count_convention": b.day_count_convention,
            "maturity_date": b.maturity_date.isoformat(),
            "first_coupon_date": b.first_coupon_date.isoformat(),
            "bbg_ticker": f"{b.cusip} Corp",
            "instrument_type": s.instrument_type,
            "issuer": s.issuer,
            "country_of_risk": b.country,
            "sector": s.sector,
            "seniority": s.seniority,
            "market": s.market,
            "rating_sp": "",        # ← authoritative agency data; user must supply
            "rating_moody": "",
            "rating_fitch": "",
        })
    return pd.DataFrame(rows)


# ── initial positions ─────────────────────────────────────────────────────────

_INCEPTION_SPECS = [
    ("HY", "25714PFB9", 1_000_000),
    ("HY", "195325ER2", 1_000_000),
    ("IG", "25714PEF1", 1_000_000),
]


def build_initial_positions() -> pd.DataFrame:
    rows = []
    for portfolio, cusip, nominal in _INCEPTION_SPECS:
        px = _SPOT_PRICES[cusip]
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
# Tuple format: (portfolio, cusip, trader, side, nominal, trade_date[, price])
# Optional 7th element overrides _SPOT_PRICES for the trade confirmation price.

_TRADE_SPECS: list[tuple] = [
    ("HY", "25714PFB9", "ALICE", "buy", 1_000_000, date(2026, 5, 5), 107.53),
    ("HY", "195325ER2", "ALICE", "buy", 1_000_000, date(2026, 5, 5), 105.62),
    ("IG", "25714PEF1", "ALICE", "buy", 1_000_000, date(2026, 5, 5),  96.36),
]

_YTM = {
    "25714PFB9": 5.75,
    "195325ER2": 6.20,
    "25714PEF1": 5.05,
}


def build_trades() -> pd.DataFrame:
    rows = []
    for spec in _TRADE_SPECS:
        portfolio, cusip, trader, side, nominal, trade_date = spec[:6]
        px = spec[6] if len(spec) > 6 else _SPOT_PRICES[cusip]
        bond = _BOND_BY_CUSIP[cusip]

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


# ── manual prices (offline / no-Bloomberg fallback) ───────────────────────────
# Bloomberg is the source of truth for prices in production, but for an offline
# demo the import step falls back to these manual files (see CLAUDE.md). Writing
# them here means the dashboard shows live P&L out of the box instead of "—",
# and the Debug view stops flagging "no current price" for every held bond.

def _recent_business_days(end: date, n: int) -> list[date]:
    days: list[date] = []
    d = end
    while len(days) < n:
        if d.weekday() < 5:        # Mon–Fri
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


def build_manual_prices(as_of: date) -> pd.DataFrame:
    """Current clean price per CUSIP (cusip, px_last, date)."""
    rows = [{"cusip": c, "px_last": round(px, 4), "date": as_of.isoformat()}
            for c, px in _SPOT_PRICES.items()]
    return pd.DataFrame(rows, columns=["cusip", "px_last", "date"])


def build_manual_price_history(as_of: date, days: int = 10) -> pd.DataFrame:
    """A short daily price path per CUSIP (date, cusip, px_last).

    Values drift gently into each bond's spot price, so they are all distinct —
    the Debug view's stale-price heuristic is not tripped.
    """
    bdays = _recent_business_days(as_of, days)
    rows = []
    for cusip, spot in _SPOT_PRICES.items():
        for i, d in enumerate(bdays):
            drift = (len(bdays) - 1 - i) * 0.05      # ~5bp/day up toward spot
            rows.append({"date": d.isoformat(), "cusip": cusip,
                         "px_last": round(spot - drift, 4)})
    return pd.DataFrame(rows, columns=["date", "cusip", "px_last"])


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    # bonds_static.csv — CUSIPs only; static data populated via Bloomberg
    bonds_df = build_bonds_static()
    bonds_df.to_csv(DATA_DIR / "bonds_static.csv", index=False)
    print(f"  bonds_static.csv      {len(bonds_df)} CUSIPs (static fields blank — use Bloomberg to populate)")

    # initial_positions.csv
    init_df = build_initial_positions()
    init_df.to_csv(DATA_DIR / "initial_positions.csv", index=False)
    print(f"  initial_positions.csv {len(init_df)} positions as of {INCEPTION_DATE}")

    # trades.csv  (gitignored — real trade data)
    trades_df = build_trades()
    trades_df.to_csv(DATA_DIR / "trades.csv", index=False)
    print(f"  trades.csv            {len(trades_df)} trade(s)  "
          f"({trades_df['portfolio'].nunique()} portfolio(s), "
          f"{trades_df['cusip'].nunique()} CUSIP(s))")

    # manual prices — offline fallback so P&L shows without Bloomberg
    as_of = date.today()
    mp_df = build_manual_prices(as_of)
    mp_df.to_csv(PRICES_DIR / "manual_prices.csv", index=False)
    print(f"  prices/manual_prices.csv          {len(mp_df)} current price(s) as of {as_of}")

    mph_df = build_manual_price_history(as_of)
    mph_df.to_csv(PRICES_DIR / "manual_price_history.csv", index=False)
    print(f"  prices/manual_price_history.csv   {len(mph_df)} history row(s)")

    print()
    print(f"All files written to {DATA_DIR.resolve()}")
    print()
    print("The dashboard now works offline (manual prices). For live Bloomberg prices:")
    print("  1. streamlit run src/dashboard.py")
    print("  2. Sidebar → Today's Prices:")
    print("       ① Prepare & open template → Bloomberg populates → ③ Import prices & static")
    print("  3. Sidebar → Price History:")
    print("       ① Prepare & open history template → Bloomberg populates → ③ Import history")
    print("       (P&L recomputes automatically after import)")
    print()
    print("Optional (not flagged in Debug): credit ratings and the Local/Global")
    print("market flag. Add ratings (Bloomberg RTG_* or manual) for richer risk reporting.")


if __name__ == "__main__":
    main()
