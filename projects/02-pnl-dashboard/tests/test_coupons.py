"""Tests for coupon-schedule transparency helpers."""
from datetime import date

import pytest

from accruals import (
    next_coupon_date, coupon_amount_per_period, upcoming_coupons,
)
from models import BondStatic, Position


def _bond(**ov) -> BondStatic:
    base = dict(
        cusip="X", name="X 6% 2030", currency="USD", country="US",
        coupon_rate=0.06, coupon_frequency=2, day_count_convention="30/360",
        maturity_date=date(2030, 6, 15), first_coupon_date=date(2024, 6, 15),
    )
    base.update(ov)
    return BondStatic(**base)


def test_next_coupon_date():
    b = _bond()
    # semi-annual on Jun/Dec 15 → next after 2026-01-01 is 2026-06-15
    assert next_coupon_date(b, date(2026, 1, 1)) == date(2026, 6, 15)
    assert next_coupon_date(b, date(2026, 6, 15)) == date(2026, 12, 15)


def test_next_coupon_none_for_zero_coupon():
    assert next_coupon_date(_bond(coupon_frequency=0, coupon_rate=0.0), date(2026, 1, 1)) is None


def test_coupon_amount_per_period():
    # 1,000,000 nominal, 6% semi-annual → 30,000 per period
    assert coupon_amount_per_period(1_000_000, _bond()) == pytest.approx(30_000)
    assert coupon_amount_per_period(1_000_000, _bond(coupon_frequency=0)) == 0.0


def test_upcoming_coupons_window_and_sign():
    positions = {
        "X": Position("X", net_nominal=1_000_000, wavg_price=100, book_value=-1e6, last_settle=date(2025, 1, 1)),
        "S": Position("S", net_nominal=-500_000, wavg_price=100, book_value=5e5, last_settle=date(2025, 1, 1)),
    }
    bonds = {"X": _bond(cusip="X"), "S": _bond(cusip="S")}
    cal = upcoming_coupons(positions, bonds, date(2026, 1, 1), date(2026, 12, 31))
    # two coupon dates (Jun, Dec) × two bonds = 4 rows
    assert len(cal) == 4
    # short position pays away (negative coupon)
    s_rows = cal[cal["cusip"] == "S"]
    assert (s_rows["coupon_amount"] < 0).all()
    # sorted by date
    assert list(cal["coupon_date"]) == sorted(cal["coupon_date"])


def test_upcoming_coupons_empty_when_no_positions():
    cal = upcoming_coupons({}, {}, date(2026, 1, 1), date(2026, 12, 31))
    assert cal.empty
