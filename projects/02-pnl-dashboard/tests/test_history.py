from datetime import date

import pandas as pd
import pytest

from history import (
    business_days, compute_daily_pnl, load_pnl_history,
    daily_snapshot, position_timeseries, accrual_breakdown,
)
from position_manager import get_positions_as_of
from accruals import load_bonds_static, accrued_interest


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


def test_daily_snapshot_exposes_clean_and_dirty():
    snap = daily_snapshot(date(2025, 3, 3), portfolio="HY")
    row = snap[snap["cusip"] == "037833100"].iloc[0]
    assert row["clean_px"] == 100.5
    # dirty = clean + accrued_pct
    assert row["dirty_px"] == pytest.approx(row["clean_px"] + row["accrued_pct"], abs=1e-6)


def test_daily_snapshot_flags_missing_price():
    # No price in history for 2025-03-05.
    snap = daily_snapshot(date(2025, 3, 5), portfolio="HY")
    row = snap[snap["cusip"] == "037833100"].iloc[0]
    assert row["note"] == "no price"
    assert pd.isna(row["clean_px"])


def test_accrual_breakdown_matches_accrued_interest():
    day = date(2025, 3, 3)
    bonds = load_bonds_static()
    pos = get_positions_as_of(day, "HY")
    acc = accrual_breakdown(pos, bonds, day)
    row = acc[acc["cusip"] == "037833100"].iloc[0]
    bond = bonds["037833100"]
    assert row["accrued_per_100"] == pytest.approx(round(accrued_interest(100, bond, day), 6))


def test_position_timeseries_one_row_per_day_per_cusip():
    ts = position_timeseries(date(2025, 3, 3), date(2025, 3, 4), portfolio="HY")
    # 2 business days × 1 held CUSIP in HY
    assert len(ts) == 2
    assert set(ts["date"]) == {"2025-03-03", "2025-03-04"}
    assert (ts["cusip"] == "037833100").all()


def test_price_history_falls_back_to_manual_offline(data_dir):
    """When Bloomberg's price_history.csv is absent, load_price_history and the
    daily P&L must fall back to manual_price_history.csv (offline workflow).

    Mirrors the current-price side, where get_prices/load_latest_prices already
    fall back to manual_prices.csv."""
    import config
    from bloomberg import load_price_history, last_priced_date

    # Simulate a no-Bloomberg environment: remove the accumulated history file
    # and seed the manual fallback the dashboard's ③ Import never wrote to.
    if config.PRICE_HISTORY_PATH.exists():
        config.PRICE_HISTORY_PATH.unlink()
    pd.DataFrame([
        dict(date="2025-02-28", cusip="037833100", px_last=100.3),
        dict(date="2025-03-03", cusip="037833100", px_last=100.5),
    ]).to_csv(config.MANUAL_HISTORY_PATH, index=False)

    ph = load_price_history()
    assert len(ph) == 2
    assert set(ph["cusip"]) == {"037833100"}
    assert last_priced_date() == date(2025, 3, 3)

    # Daily P&L now computes a real price move instead of blanks.
    df = compute_daily_pnl(date(2025, 3, 3), date(2025, 3, 3), portfolio="HY")
    d = df[df["date"] == "2025-03-03"]
    assert d["price_pnl"].notna().any()
    assert d["price_pnl"].sum() == pytest.approx(1_800_000 * (100.5 - 100.3) / 100, abs=1)
