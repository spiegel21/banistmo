"""Tests for the analytical risk engine (YTM, duration, DV01, convexity, VaR)."""
from datetime import date

import pandas as pd
import pytest

import analytics as an
from models import BondStatic, Position


def _bond(**ov) -> BondStatic:
    base = dict(
        cusip="X", name="X 5% 2030", currency="USD", country="US",
        coupon_rate=0.05, coupon_frequency=2, day_count_convention="30/360",
        maturity_date=date(2030, 6, 15), first_coupon_date=date(2020, 6, 15),
    )
    base.update(ov)
    return BondStatic(**base)


def test_remaining_cashflows_include_redemption():
    b = _bond()
    cfs = an.remaining_cashflows(b, date(2026, 1, 1))
    # last flow is at maturity and includes the 100 redemption + final coupon
    assert cfs[-1][0] == date(2030, 6, 15)
    assert cfs[-1][1] == pytest.approx(100 + 2.5, abs=1e-6)
    # interim coupons are 2.5 per 100 (5% semi)
    assert cfs[0][1] == pytest.approx(2.5, abs=1e-6)


def test_price_at_coupon_yield_is_par_on_coupon_date():
    # On a coupon date, discounting at the coupon rate gives ~par (clean ≈ 100).
    b = _bond()
    on_cpn = date(2026, 6, 15)
    dirty = an.price_from_yield(b, on_cpn, 0.05)
    # on a coupon date accrued ≈ 0, so dirty ≈ clean ≈ par
    assert dirty == pytest.approx(100.0, abs=0.3)


def test_ytm_recovers_yield_roundtrip():
    b = _bond()
    as_of = date(2026, 3, 10)
    # price the bond at a known yield, then solve YTM back from the clean price
    y_true = 0.062
    dirty = an.price_from_yield(b, as_of, y_true)
    accrued = an.accrued_interest(100, b, as_of)
    clean = dirty - accrued
    y_solved = an.yield_to_maturity(b, as_of, clean)
    assert y_solved == pytest.approx(y_true, abs=1e-5)


def test_risk_measures_sane_signs_and_magnitudes():
    b = _bond()
    rm = an.risk_measures(b, date(2026, 3, 10), clean_price=98.0)
    assert rm["ytm"] > 0
    # ~4.3y bond: modified duration should be a few years and positive
    assert 2.0 < rm["mod_duration"] < 5.0
    assert rm["mac_duration"] > rm["mod_duration"]   # Mac = Mod*(1+y/f)
    assert rm["dv01_per_100"] > 0
    assert rm["convexity"] > 0


def test_zero_coupon_has_no_interim_flows():
    z = _bond(coupon_rate=0.0, coupon_frequency=0)
    cfs = an.remaining_cashflows(z, date(2026, 1, 1))
    assert cfs == [(date(2030, 6, 15), 100.0)]
    rm = an.risk_measures(z, date(2026, 1, 1), clean_price=80.0)
    assert rm["ytm"] > 0


def test_portfolio_risk_aggregates():
    positions = {
        "A": Position("A", 1_000_000, 100, -1e6, date(2025, 1, 1)),
        "B": Position("B", 2_000_000, 100, -2e6, date(2025, 1, 1)),
    }
    bonds = {"A": _bond(cusip="A"),
             "B": _bond(cusip="B", coupon_rate=0.03, maturity_date=date(2028, 6, 15))}
    prices = {"A": 99.0, "B": 97.0}
    df, summary = an.portfolio_risk(positions, bonds, prices, as_of=date(2026, 3, 10))
    assert len(df) == 2
    assert summary["n_priced"] == 2
    # dollar DV01 sums additively and is positive for long positions
    assert summary["dv01_dollar"] > 0
    assert summary["mod_duration"] > 0


def test_portfolio_risk_handles_missing_price():
    positions = {"A": Position("A", 1_000_000, 100, -1e6, date(2025, 1, 1))}
    bonds = {"A": _bond(cusip="A")}
    df, summary = an.portfolio_risk(positions, bonds, {}, as_of=date(2026, 3, 10))
    assert df.iloc[0]["note"] == "no price"
    assert summary["n_priced"] == 0


def test_risk_by_group():
    positions = {
        "A": Position("A", 1_000_000, 100, -1e6, date(2025, 1, 1)),
        "B": Position("B", 2_000_000, 100, -2e6, date(2025, 1, 1)),
    }
    bonds = {"A": _bond(cusip="A"),
             "B": _bond(cusip="B", maturity_date=date(2028, 6, 15))}
    prices = {"A": 99.0, "B": 97.0}
    risk_df, _ = an.portfolio_risk(positions, bonds, prices, as_of=date(2026, 3, 10))
    grouped = an.risk_by_group(risk_df, {"A": "CO", "B": "US"})
    assert set(grouped["group"]) == {"CO", "US"}
    assert grouped["n_bonds"].sum() == 2
    # group DV01 sums to portfolio DV01
    assert grouped["dv01_dollar"].sum() == pytest.approx(risk_df["dv01_dollar"].sum(), abs=1.0)


def test_risk_by_group_unmapped_is_unknown():
    positions = {"A": Position("A", 1_000_000, 100, -1e6, date(2025, 1, 1))}
    bonds = {"A": _bond(cusip="A")}
    risk_df, _ = an.portfolio_risk(positions, bonds, {"A": 99.0}, as_of=date(2026, 3, 10))
    grouped = an.risk_by_group(risk_df, {})  # no mapping
    assert list(grouped["group"]) == ["Unknown"]


def test_var_historical():
    # symmetric-ish series with a couple of bad days
    s = pd.Series([100, -50, 20, -200, 10, -30, 40, -500, 5, -10])
    v = an.var_historical(s, confidence=0.9)
    assert v["var"] > 0
    assert v["es"] >= v["var"]          # ES is at least as large as VaR
    assert v["worst_day"] == -500
    assert v["n_obs"] == 10


def test_var_insufficient_data():
    v = an.var_historical(pd.Series([1.0]))
    assert v["var"] == 0.0 and v["n_obs"] == 1
