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
from models import Position, BondStatic, normalise_day_count

log = get_logger(__name__)


def load_bonds_static(path: Path | None = None) -> dict[str, BondStatic]:
    """Load bond reference data, validating each row via BondStatic.__post_init__."""
    path = Path(path) if path is not None else config.BONDS_STATIC_PATH
    if not path.exists() or path.stat().st_size == 0:
        return {}

    df = pd.read_csv(path, dtype={"cusip": str}, parse_dates=["maturity_date"])
    # first_coupon_date is intentionally NOT in parse_dates: Bloomberg returns "#N/A"
    # for zero-coupon / bullet bonds, and we need the raw string so the fallback
    # logic below can detect it.  If the column exists, parse it ourselves.
    if "first_coupon_date" in df.columns:
        df["first_coupon_date"] = pd.to_datetime(df["first_coupon_date"], errors="coerce")
    result: dict[str, BondStatic] = {}
    for _, row in df.iterrows():
        cusip = str(row.get("cusip", ""))
        # Only block on fields that have NO fallback path. maturity_date is the
        # one field we cannot synthesise. Everything else (coupon_rate #N/A → 0,
        # coupon_frequency #N/A → 0, first_coupon_date #N/A → maturity_date) is
        # handled below and must NOT cause a skip here.
        if pd.isna(row.get("maturity_date")):
            log.warning("bonds_static row for %s missing required field 'maturity_date' — row skipped",
                        cusip)
            continue
        try:
            # coupon_frequency: Bloomberg can return "#N/A" for non-coupon bonds → 0
            raw_freq = row.get("coupon_frequency")
            try:
                coupon_freq = int(round(float(raw_freq)))
            except (ValueError, TypeError):
                coupon_freq = 0

            # coupon_rate: blank / None / "#N/A" → 0.0 (treat as zero-coupon) so the
            # bond still loads rather than vanishing from the dashboard.
            raw_rate = row.get("coupon_rate")
            try:
                coupon_rate = float(raw_rate)
            except (ValueError, TypeError):
                coupon_rate = 0.0
            if pd.isna(coupon_rate):
                coupon_rate = 0.0
            if coupon_rate == 0.0 and coupon_freq != 0:
                # Missing/zero coupon with a stated frequency: keep it consistent
                # as a non-coupon-bearing bond so no phantom accrual is computed.
                coupon_freq = 0

            maturity = row["maturity_date"].date()

            # first_coupon_date: Bloomberg returns "#N/A" for zero-coupon / bullet
            # → fall back to maturity_date so accrual calculations are consistent.
            raw_fcd = row.get("first_coupon_date")
            if pd.isna(raw_fcd) if not isinstance(raw_fcd, str) else str(raw_fcd).strip() in ("#N/A", "N/A", ""):
                first_coupon = maturity
            else:
                first_coupon = pd.to_datetime(raw_fcd).date() if not hasattr(raw_fcd, "date") else raw_fcd.date()

            # day_count_convention: blank / None → default to 30/360 so the bond
            # still loads (irrelevant for zero-coupon bonds; a sane default otherwise).
            raw_dcc = row.get("day_count_convention")
            dcc = normalise_day_count(str(raw_dcc).strip()) if pd.notna(raw_dcc) and str(raw_dcc).strip() else ""
            if not dcc:
                log.warning("bonds_static row for %s missing day_count_convention — defaulting to 30/360", cusip)
                dcc = "30/360"

            bond = BondStatic(
                cusip=cusip,
                name="" if pd.isna(row.get("name")) else str(row.get("name", "")).strip(),
                currency="" if pd.isna(row.get("currency")) else str(row.get("currency", "")).strip(),
                country="" if pd.isna(row.get("country")) else str(row.get("country", "")).strip(),
                coupon_rate=coupon_rate,
                coupon_frequency=coupon_freq,
                day_count_convention=dcc,
                maturity_date=maturity,
                first_coupon_date=first_coupon,
                bbg_ticker="" if pd.isna(row.get("bbg_ticker")) else str(row.get("bbg_ticker", "")),
                instrument_type="" if pd.isna(row.get("instrument_type")) else str(row.get("instrument_type", "")),
            )
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("Skipping invalid bonds_static row for %s: %s", cusip, exc)
            continue
        result[bond.cusip] = bond
    return result


def _coupon_dates(bond: BondStatic) -> list[date]:
    """Generate all coupon dates from first_coupon_date up to and including maturity."""
    if bond.coupon_frequency == 0:
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
    if bond.coupon_frequency == 0:
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
    if bond.coupon_frequency == 0:
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
