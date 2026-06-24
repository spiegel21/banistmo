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
from data_io import parse_bond_static_row

log = get_logger(__name__)


def load_bonds_static(path: Path | None = None) -> dict[str, BondStatic]:
    """Load bond reference data, validating each row via BondStatic.__post_init__."""
    path = Path(path) if path is not None else config.BONDS_STATIC_PATH
    if not path.exists() or path.stat().st_size == 0:
        return {}

    df = pd.read_csv(path, dtype={"cusip": str})
    result: dict[str, BondStatic] = {}
    for _, row in df.iterrows():
        cusip = str(row.get("cusip", ""))
        # All field fallbacks (blank/None/#N/A → sane defaults) live in
        # parse_bond_static_row so the loader and the editor validator stay in
        # sync. maturity_date is the only field with no fallback; a missing value
        # raises and the row is skipped.
        try:
            bond = parse_bond_static_row(row)
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("Skipping invalid bonds_static row for %s: %s", cusip, exc)
            continue
        result[bond.cusip] = bond
    return result


def _coupon_dates(bond: BondStatic) -> list[date]:
    """Generate all coupon dates from first_coupon_date up to and including maturity."""
    # A frequency of 0 (or any missing/invalid value) means non-coupon-bearing:
    # no coupon schedule to walk.
    if not bond.coupon_frequency:
        return []
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
    """The coupon date one period before first_coupon_date (proxy for issue date).

    Non-coupon-bearing bonds (``coupon_frequency`` 0 / missing) have no period to
    step back to, so the first_coupon_date is returned unchanged. This keeps the
    function safe to call for any bond (no ZeroDivisionError).
    """
    if not bond.coupon_frequency:
        return bond.first_coupon_date
    months = 12 // bond.coupon_frequency
    m = bond.first_coupon_date.month - months
    y = bond.first_coupon_date.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    day = min(bond.first_coupon_date.day, _days_in_month(y, m))
    return bond.first_coupon_date.replace(year=y, month=m, day=day)


def last_coupon_date(bond: BondStatic, as_of: date) -> date:
    """Most recent coupon date on or before as_of.

    For non-coupon-bearing bonds (``coupon_frequency`` 0 / missing) there is no
    coupon schedule, so ``_prev_period_start`` returns first_coupon_date as a
    safe, no-accrual reference date rather than raising.
    """
    past = [d for d in _coupon_dates(bond) if d <= as_of]
    if past:
        return max(past)
    return _prev_period_start(bond)


def next_coupon_date(bond: BondStatic, as_of: date) -> date | None:
    """Earliest coupon date strictly after as_of, or None if the bond has matured
    or is non-coupon-bearing."""
    if not bond.coupon_frequency:
        return None
    future = [d for d in _coupon_dates(bond) if d > as_of]
    return min(future) if future else None


def coupon_amount_per_period(nominal: float, bond: BondStatic) -> float:
    """Cash coupon paid each period for a given nominal (annual coupon / frequency)."""
    if not bond.coupon_frequency or not bond.coupon_rate:
        return 0.0
    return nominal * bond.coupon_rate / bond.coupon_frequency


def upcoming_coupons(
    positions: dict[str, Position],
    bonds_static: dict[str, BondStatic],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Forecast coupon cash flows for held positions between start and end (inclusive).

    One row per (coupon_date, cusip): the position-scaled coupon amount. Short
    positions produce negative (paid-away) coupons. Sorted by date.
    """
    cols = ["coupon_date", "cusip", "net_nominal", "coupon_rate", "coupon_amount"]
    rows = []
    for cusip, pos in positions.items():
        if pos.net_nominal == 0:
            continue
        bond = bonds_static.get(cusip)
        if bond is None or bond.coupon_frequency == 0:
            continue
        per = coupon_amount_per_period(pos.net_nominal, bond)
        for d in _coupon_dates(bond):
            if start <= d <= end:
                rows.append({
                    "coupon_date": d.isoformat(),
                    "cusip": cusip,
                    "net_nominal": pos.net_nominal,
                    "coupon_rate": bond.coupon_rate,
                    "coupon_amount": round(per, 2),
                })
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["coupon_date", "cusip"]).reset_index(drop=True) if not df.empty else df


def _days_30_360(start: date, end: date) -> int:
    """30/360 (bond basis) day count between two dates."""
    d1 = min(start.day, 30)
    d2 = min(end.day, 30) if d1 == 30 else end.day
    return (end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)


def days_accrued(bond: BondStatic, as_of: date) -> int:
    """Days since last coupon using the bond's day-count convention.

    A matured bond (as_of on/after maturity) has redeemed at par — the final
    coupon was paid at maturity, so there is no further carry. Without this
    clamp ``last_coupon_date`` returns the last coupon and the day count would
    grow without bound for every day past maturity, inflating accrued interest
    and dirty-price MTM for a position that was never closed out.
    """
    if not bond.coupon_frequency:
        return 0
    if bond.is_matured(as_of):
        return 0
    lcd = last_coupon_date(bond, as_of)
    if bond.day_count_convention == "30/360":
        return _days_30_360(lcd, as_of)
    return (as_of - lcd).days  # actual days for Act/360, Act/365, Act/Act


def accrued_interest(nominal: float, bond: BondStatic, as_of: date) -> float:
    """
    Accrued interest in currency units, per the bond's day-count convention.

    Act/360, Act/365: annual_coupon * days / basis
    Act/Act:          annual_coupon * days / 365 (365-day approximation)
    30/360:           annual_coupon * days_30_360 / 360
    """
    if not bond.coupon_frequency or not bond.coupon_rate:
        return 0.0
    days = days_accrued(bond, as_of)
    annual_coupon = nominal * bond.coupon_rate
    if bond.day_count_convention == "Act/360":
        return annual_coupon * days / 360
    if bond.day_count_convention in ("Act/365", "Act/Act"):
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
