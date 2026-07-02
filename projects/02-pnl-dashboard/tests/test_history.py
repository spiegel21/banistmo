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


def test_coupon_collected_into_realized_not_accrued():
    # UST 912828XY9 (first_coupon 2024-09-20, semi-annual, Act/360) pays on
    # 2025-03-20. IG bought 2M on 2025-02-10. Running the daily P&L from just
    # after the buy through the coupon, interest accrues into `accrued` and is
    # collected into `realized_gain` (as coupon income) on the coupon date —
    # accrued sawtooths back down rather than carrying the coupon.
    df = compute_daily_pnl(date(2025, 2, 11), date(2025, 3, 21), portfolio="IG")
    df = df.sort_values("date")
    coupon_row = df[df["date"] == "2025-03-20"]
    # Coupon income lands in realized on the coupon date (it was ~0 before).
    assert coupon_row["realized_gain"].sum() > 0
    assert df[df["date"] < "2025-03-20"]["realized_gain"].sum() == pytest.approx(0.0, abs=1)
    # Accrued draws down on the coupon date (the receivable is collected).
    assert coupon_row["accrued"].sum() < 0
    # The interest earned window-start → coupon (2M * 3% * ~38/360 days).
    assert coupon_row["realized_gain"].sum() == pytest.approx(2_000_000 * 0.03 * 38 / 360, abs=200)


def test_accrued_never_negative_across_a_coupon():
    # Cumulative accrued for a long must never dip below zero: it builds up as a
    # receivable and resets to ~0 when the coupon is collected — the core fix.
    df = compute_daily_pnl(date(2025, 2, 11), date(2025, 3, 21), portfolio="IG")
    df = df.sort_values("date")
    assert df["accrued"].fillna(0).cumsum().min() >= -0.01


def test_price_pnl_is_daily_clean_price_change():
    # Mar 4 has no trades, so price_pnl (Valuation) is the raw daily clean mark on
    # the SOD position. SOD nominal = 1.2M (after the Mar 3 sell), prev close 100.5,
    # today close 100.6 → price_pnl = 1_200_000 * (100.6 - 100.5) / 100 = 1_200.
    df = compute_daily_pnl(date(2025, 3, 4), date(2025, 3, 4), portfolio="HY")
    d = df[df["date"] == "2025-03-04"]
    assert d["price_pnl"].sum() == pytest.approx(1_200_000 * (100.6 - 100.5) / 100, abs=1)


def test_trade_day_price_pnl_excludes_realized_no_double_count():
    # Mar 3 sells 600k @100.0. The day's clean price P&L =
    #   SOD mark   = 1.8M × (100.5 − 100.3)/100          = +3_600
    #   trade adj  = (100.5×(−600k) − (−600k×100))/100    = −3_000   (sold below close)
    #   price_total                                        = +600
    # The realised slice (600k × (1.00 − wavg_clean)) is carved OUT of Valuation, so
    # price_pnl = price_total − realized and total is counted exactly once.
    df = compute_daily_pnl(date(2025, 3, 3), date(2025, 3, 3), portfolio="HY")
    d = df[df["date"] == "2025-03-03"].iloc[0]
    price_total = 600.0
    assert (d["price_pnl"] + d["realized_gain"]) == pytest.approx(price_total, abs=1)
    # total reconciles with the reported components (no double count)
    assert d["total_pnl"] == pytest.approx(d["price_pnl"] + d["accrued"] + d["realized_gain"], abs=0.01)


def test_inception_price_used_as_prev_on_first_day():
    # Jan 3 (Friday): prev business day = Jan 2, no price in history → prev-price
    # fallback to initial_positions.csv price = 97.0. SOD = inception 300k; the Jan 3
    # buy of 1M @98.5 is marked execution→close on the trade date:
    #   SOD mark  = 300k × (97.5 − 97.0)/100                = +1_500
    #   trade adj = (97.5×1_000_000 − 1_000_000×98.5)/100   = −10_000  (bought above close)
    #   price_pnl = −8_500
    df = compute_daily_pnl(date(2025, 1, 3), date(2025, 1, 3), portfolio="HY")
    d = df[df["date"] == "2025-01-03"]
    assert not d.empty
    assert d["price_pnl"].sum() == pytest.approx(
        300_000 * (97.5 - 97.0) / 100 + (97.5 * 1_000_000 - 1_000_000 * 98.5) / 100, abs=1
    )


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

    # Daily P&L now computes a real price move instead of blanks. Mar 4 (no trades)
    # isolates the raw clean mark on the 1.2M SOD position (100.5 → 100.6).
    pd.DataFrame([
        dict(date="2025-02-28", cusip="037833100", px_last=100.3),
        dict(date="2025-03-03", cusip="037833100", px_last=100.5),
        dict(date="2025-03-04", cusip="037833100", px_last=100.6),
    ]).to_csv(config.MANUAL_HISTORY_PATH, index=False)
    df = compute_daily_pnl(date(2025, 3, 4), date(2025, 3, 4), portfolio="HY")
    d = df[df["date"] == "2025-03-04"]
    assert d["price_pnl"].notna().any()
    assert d["price_pnl"].sum() == pytest.approx(1_200_000 * (100.6 - 100.5) / 100, abs=1)
