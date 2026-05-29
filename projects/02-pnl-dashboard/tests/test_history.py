from datetime import date

import pytest

from history import business_days, compute_daily_pnl, load_pnl_history


def test_business_days_skips_weekends():
    # 2025-03-01 is a Saturday; 03-03 Monday
    days = business_days(date(2025, 2, 28), date(2025, 3, 4))
    assert date(2025, 3, 1) not in days  # Sat
    assert date(2025, 3, 2) not in days  # Sun
    assert date(2025, 2, 28) in days     # Fri
    assert date(2025, 3, 3) in days      # Mon


def test_realized_appears_on_trade_day_not_after():
    # Sell on 03-03: realized shows that day; March 4 (no trades) should be zero.
    df = compute_daily_pnl(date(2025, 3, 3), date(2025, 3, 4), portfolio="HY")
    d303 = df[df["date"] == "2025-03-03"]
    d304 = df[df["date"] == "2025-03-04"]
    assert d303["realized_gain"].sum() > 0
    assert d304["realized_gain"].sum() == pytest.approx(0.0)


def test_daily_total_equals_components():
    df = compute_daily_pnl(date(2025, 3, 3), date(2025, 3, 4), portfolio="HY")
    recon = (df["price_pnl"].fillna(0) + df["accrued"].fillna(0) + df["realized_gain"]).round(2)
    assert (recon == df["total_pnl"]).all()


def test_price_pnl_is_daily_clean_price_change():
    # On Mar 3: SOD nominal = 1.8M (before the sell), prev close = 100.3, today close = 100.5
    # price_pnl = 1_800_000 * (100.5 - 100.3) / 100 = 3_600
    df = compute_daily_pnl(date(2025, 3, 3), date(2025, 3, 3), portfolio="HY")
    d = df[df["date"] == "2025-03-03"]
    assert d["price_pnl"].sum() == pytest.approx(1_800_000 * (100.5 - 100.3) / 100, abs=1)


def test_inception_price_used_as_prev_on_first_day():
    # Jan 3 (Friday): prev business day = Jan 2, no price in history.
    # Fallback: initial_positions.csv price = 97.0.
    # SOD position (as of Jan 2) = inception 300k only (Jan 3 buy not yet settled).
    # price_pnl = 300_000 * (97.5 - 97.0) / 100 = 1_500
    df = compute_daily_pnl(date(2025, 1, 3), date(2025, 1, 3), portfolio="HY")
    d = df[df["date"] == "2025-01-03"]
    assert not d.empty
    assert d["price_pnl"].sum() == pytest.approx(300_000 * (97.5 - 97.0) / 100, abs=1)


def test_load_pnl_history_scope_isolation():
    compute_daily_pnl(date(2025, 3, 3), date(2025, 3, 4), portfolio="HY")
    compute_daily_pnl(date(2025, 3, 3), date(2025, 3, 4), portfolio=None)  # "all" scope
    hy = load_pnl_history(portfolio="HY")
    every = load_pnl_history(portfolio=None)  # ALL_PORTFOLIOS scope only
    assert (hy["portfolio"] == "HY").all()
    assert (every["portfolio"] == "all").all()
