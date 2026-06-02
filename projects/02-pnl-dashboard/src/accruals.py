"""
Accrued interest calculations for fixed-income bonds.

Supports day-count conventions: Act/360, Act/365, 30/360.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

import config
from config import get_logger
from models import Position, BondStatic

log = get_logger(__name__)


def load_bonds_static(path: Path | None = None) -> dict[str, BondStatic]:
    """Load bond reference data, validating each row via BondStatic.__post_init__."""
    path = Path(path) if path is not None else config.BONDS_STATIC_PATH
    if not path.exists() or path.stat().st_size == 0:
        return {}

    df = pd.read_csv(path, dtype={"cusip": str}, parse_dates=["maturity_date", "first_coupon_date"])
    result: dict[str, BondStatic] = {}
    for _, row in df.iterrows():
        cusip = str(row.get("cusip", ""))
        missing = [f for f in ("coupon_frequency", "coupon_rate", "maturity_date", "first_coupon_date")
                   if pd.isna(row.get(f))]
        if missing:
            log.warning("bonds_static row for %s missing required field(s) %s — row skipped",
                        cusip, missing)
            continue
        try:
            bond = BondStatic(
                cusip=cusip,
                name=row.get("name", ""),
                currency=row.get("currency", ""),
                country=row.get("country", "") if pd.notna(row.get("country", "")) else "",
                coupon_rate=float(row["coupon_rate"]),
                coupon_frequency=int(row["coupon_frequency"]),
                day_count_convention=str(row["day_count_convention"]),
                maturity_date=row["maturity_date"].date(),
                first_coupon_date=row["first_coupon_date"].date(),
            )
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("Skipping invalid bonds_static row for %s: %s", cusip, exc)
            continue
        result[bond.cusip] = bond
    return result


def _coupon_dates(bond: BondStatic) -> list[date]:
    """Generate all coupon dates from first_coupon_date up to and including maturity."""
    months_per_period = 12 // bond.coupon_frequency
    dates: list[date] = []
    d = bond.first_coupon_date
    while d <= bond.maturity_date:
        dates.append(d)
        month = d.month + months_per_period
        year = d.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        # clamp day to end of month (handles 31st → shorter months)
        day = min(d.day, _days_in_month(year, month))
        d = d.replace(year=year, month=month, day=day)
    return dates


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    return (nxt - date(year, month, 1)).days


def _prev_period_start(bond: BondStatic) -> date:
    """The coupon date one period before first_coupon_date (proxy for issue date)."""
    months = 12 // bond.coupon_frequency
    m = bond.first_coupon_date.month - months
    y = bond.first_coupon_date.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    day = min(bond.first_coupon_date.day, _days_in_month(y, m))
    return bond.first_coupon_date.replace(year=y, month=m, day=day)


def last_coupon_date(bond: BondStatic, as_of: date) -> date:
    """Most recent coupon date on or before as_of."""
    past = [d for d in _coupon_dates(bond) if d <= as_of]
    if past:
        return max(past)
    return _prev_period_start(bond)


def _days_30_360(start: date, end: date) -> int:
    """30/360 (bond basis) day count between two dates."""
    d1 = min(start.day, 30)
    d2 = min(end.day, 30) if d1 == 30 else end.day
    return (end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)


def days_accrued(bond: BondStatic, as_of: date) -> int:
    """Days since last coupon using the bond's day-count convention."""
    lcd = last_coupon_date(bond, as_of)
    if bond.day_count_convention == "30/360":
        return _days_30_360(lcd, as_of)
    return (as_of - lcd).days  # actual days for Act/360 and Act/365


def accrued_interest(nominal: float, bond: BondStatic, as_of: date) -> float:
    """
    Accrued interest in currency units, per the bond's day-count convention.

    Act/360, Act/365: annual_coupon * days / basis
    30/360:           annual_coupon * days_30_360 / 360
    """
    days = days_accrued(bond, as_of)
    annual_coupon = nominal * bond.coupon_rate
    if bond.day_count_convention == "Act/360":
        return annual_coupon * days / 360
    if bond.day_count_convention == "Act/365":
        return annual_coupon * days / 365
    # 30/360
    return annual_coupon * days / 360


def total_portfolio_accruals(
    positions: dict[str, Position],
    bonds_static: dict[str, BondStatic],
    as_of: date | None = None,
) -> pd.DataFrame:
    """
    Accrued interest (carry) for each position as of a date.

    Equals the accrued portion of each position's current market value:
        accrued = net_nominal * accrued_interest(100, bond, as_of) / 100
    Short positions carry a negative accrual (liability).
    """
    if as_of is None:
        as_of = date.today()

    rows = []
    for cusip, pos in positions.items():
        if pos.net_nominal == 0:
            continue
        bond = bonds_static.get(cusip)
        if bond is None:
            rows.append({"cusip": cusip, "net_nominal": pos.net_nominal,
                         "accrued": None, "note": "missing bond static"})
            continue
        accrued_pct = accrued_interest(100, bond, as_of)  # per 100 par
        accrued = pos.net_nominal * accrued_pct / 100      # signed with position
        rows.append({
            "cusip": cusip,
            "net_nominal": pos.net_nominal,
            "accrued": round(accrued, 2),
            "note": "",
        })
    return pd.DataFrame(rows, columns=["cusip", "net_nominal", "accrued", "note"])
