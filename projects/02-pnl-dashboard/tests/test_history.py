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


def test_daily_pnl_persists_realized_after_close():
    # On 03-04, the HY 600k sell (03-03) is realized and the position is still open.
    df = compute_daily_pnl(date(2025, 3, 3), date(2025, 3, 4), portfolio="HY")
    d304 = df[df["date"] == "2025-03-04"]
    assert not d304.empty
    assert d304["realized_gain"].sum() > 0


def test_daily_total_equals_components():
    df = compute_daily_pnl(date(2025, 3, 3), date(2025, 3, 4), portfolio="HY")
    recon = (df["price_pnl"].fillna(0) + df["accrued"].fillna(0) + df["realized_gain"]).round(2)
    assert (recon == df["total_pnl"]).all()


def test_load_pnl_history_scope_isolation():
    compute_daily_pnl(date(2025, 3, 3), date(2025, 3, 4), portfolio="HY")
    compute_daily_pnl(date(2025, 3, 3), date(2025, 3, 4), portfolio=None)  # "all" scope
    hy = load_pnl_history(portfolio="HY")
    every = load_pnl_history(portfolio=None)  # ALL_PORTFOLIOS scope only
    assert (hy["portfolio"] == "HY").all()
    assert (every["portfolio"] == "all").all()
