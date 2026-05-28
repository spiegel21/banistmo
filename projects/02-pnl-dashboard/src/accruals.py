"""
Accrued interest calculations for fixed-income bonds.

Supports day-count conventions: Act/360, Act/365, 30/360.
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from models import Position, BondStatic

BONDS_STATIC_PATH = Path(__file__).parent.parent / "data" / "bonds_static.csv"


def load_bonds_static(path: Path = BONDS_STATIC_PATH) -> dict[str, BondStatic]:
    if not Path(path).exists():
        return {}

    df = pd.read_csv(path, parse_dates=["maturity_date", "first_coupon_date"])
    result = {}
    for _, row in df.iterrows():
        b = BondStatic(
            cusip=row["cusip"],
            name=row["name"],
            currency=row["currency"],
            country=row.get("country", ""),
            coupon_rate=float(row["coupon_rate"]),
            coupon_frequency=int(row["coupon_frequency"]),
            day_count_convention=row["day_count_convention"],
            maturity_date=row["maturity_date"].date(),
            first_coupon_date=row["first_coupon_date"].date(),
        )
        result[b.cusip] = b
    return result


def _coupon_dates(bond: BondStatic) -> list[date]:
    """Generate all coupon dates from first_coupon_date up to and including maturity."""
    months_per_period = 12 // bond.coupon_frequency
    dates = []
    d = bond.first_coupon_date
    while d <= bond.maturity_date:
        dates.append(d)
        month = d.month + months_per_period
        year = d.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        d = d.replace(year=year, month=month)
    return dates


def last_coupon_date(bond: BondStatic, as_of: date) -> date:
    """Most recent coupon date on or before as_of."""
    all_dates = _coupon_dates(bond)
    past = [d for d in all_dates if d <= as_of]
    if past:
        return max(past)
    # before first coupon — accrue from issue (approximate as first_coupon_date minus period)
    months = 12 // bond.coupon_frequency
    m = bond.first_coupon_date.month - months
    y = bond.first_coupon_date.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    return bond.first_coupon_date.replace(year=y, month=m)


def _days_30_360(start: date, end: date) -> int:
    """30/360 day count."""
    d1 = min(start.day, 30)
    d2 = min(end.day, 30) if d1 == 30 else end.day
    return (end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)


def days_accrued(bond: BondStatic, as_of: date) -> int:
    """Days since last coupon using the bond's day-count convention."""
    lcd = last_coupon_date(bond, as_of)
    if bond.day_count_convention == "30/360":
        return _days_30_360(lcd, as_of)
    return (as_of - lcd).days  # actual days for Act/360 and Act/365


def _period_basis(bond: BondStatic, as_of: date) -> float:
    """Day-count denominator (full coupon period length)."""
    if bond.day_count_convention == "Act/360":
        return 360.0
    if bond.day_count_convention == "Act/365":
        return 365.0
    # 30/360: full period is always 360 / frequency days
    return 360.0 / bond.coupon_frequency


def accrued_interest(nominal: float, bond: BondStatic, as_of: date) -> float:
    """
    Accrued interest in currency units.

    Formula: nominal × coupon_rate × days_accrued / day_count_basis / coupon_frequency
    For Act/360 and Act/365 the denominator is the annual basis.
    For 30/360 we use 360/frequency as the full-period length.
    """
    days = days_accrued(bond, as_of)
    basis = _period_basis(bond, as_of)
    annual_coupon = nominal * bond.coupon_rate
    if bond.day_count_convention in ("Act/360", "Act/365"):
        return annual_coupon * days / basis
    # 30/360: accrual within the coupon period
    period_length = 360 / bond.coupon_frequency
    return annual_coupon / bond.coupon_frequency * days / period_length


def total_portfolio_accruals(
    positions: dict[str, Position],
    bonds_static: dict[str, BondStatic],
    as_of: date | None = None,
) -> pd.DataFrame:
    """Return DataFrame with accrued interest for each position as of today."""
    if as_of is None:
        as_of = date.today()

    rows = []
    for cusip, pos in positions.items():
        if pos.net_nominal == 0:
            continue
        bond = bonds_static.get(cusip)
        if bond is None:
            rows.append({"cusip": cusip, "net_nominal": pos.net_nominal, "accrued": None,
                         "note": "missing bond static"})
            continue
        ai = accrued_interest(abs(pos.net_nominal), bond, as_of)
        if pos.net_nominal < 0:
            ai = -ai  # short position accrues as a liability
        rows.append({
            "cusip": cusip,
            "net_nominal": pos.net_nominal,
            "accrued": round(ai, 2),
            "note": "",
        })
    return pd.DataFrame(rows)
