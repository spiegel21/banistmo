from datetime import date

import pytest

from position_manager import load_trades, load_all_trades, compute_positions


def test_side_signs_nominal():
    t = load_trades()
    sells = t[t["side"] == "sell"]
    buys = t[t["side"] == "buy"]
    assert (sells["nominal"] < 0).all()
    assert (buys["nominal"] > 0).all()


def test_side_signs_net():
    t = load_trades()
    # net is signed on load: buys negative (cash out), sells positive (cash in)
    assert (t[t["side"] == "buy"]["net"] < 0).all()
    assert (t[t["side"] == "sell"]["net"] > 0).all()


def test_net_signed_from_absolute_csv(tmp_path):
    """A trades.csv storing net as an absolute (positive) value is signed on load."""
    import pandas as pd
    from position_manager import load_trades

    p = tmp_path / "trades.csv"
    pd.DataFrame([
        dict(Timestamp="2025-01-05T09:00", cusip="037833100", side="buy",
             nominal=1_000_000, principal=985000, net=986000, accrued=1000,
             price=98.5, yield_closed=5.1, trade_date="2025-01-03",
             settle_date="2025-01-07", trader="ALICE", portfolio="HY"),
        dict(Timestamp="2025-03-05T09:00", cusip="037833100", side="sell",
             nominal=600_000, principal=600000, net=601000, accrued=1000,
             price=100.0, yield_closed=4.5, trade_date="2025-03-03",
             settle_date="2025-03-05", trader="BOB", portfolio="HY"),
    ]).to_csv(p, index=False)

    t = load_trades(p)
    buy = t[t["side"] == "buy"].iloc[0]
    sell = t[t["side"] == "sell"].iloc[0]
    assert buy["net"] == pytest.approx(-986000)   # absolute in CSV → negative on load
    assert sell["net"] == pytest.approx(601000)


def test_net_position_all():
    pos = compute_positions(load_all_trades())
    # 300k initial + 1M + 0.5M - 0.6M = 1.2M
    assert pos["037833100"].net_nominal == pytest.approx(1_200_000)
    assert pos["912828XY9"].net_nominal == pytest.approx(2_000_000)


def test_as_of_filters_future_trades():
    allt = load_all_trades()
    pos = compute_positions(allt, as_of=date(2025, 1, 15), portfolio="HY")
    # only initial (300k) + first buy (1M) settled by 2025-01-15
    assert pos["037833100"].net_nominal == pytest.approx(1_300_000)


def test_portfolio_filter_isolates_books():
    allt = load_all_trades()
    hy = compute_positions(allt, portfolio="HY")
    ig = compute_positions(allt, portfolio="IG")
    assert "912828XY9" not in hy
    assert "037833100" not in ig


def test_wavg_price_is_buy_weighted():
    pos = compute_positions(load_all_trades(), portfolio="HY")["037833100"]
    # buys: 300k@97, 1M@98.5, 0.5M@99.5  → weighted by nominal
    expected = (300_000 * 97 + 1_000_000 * 98.5 + 500_000 * 99.5) / 1_800_000
    assert pos.wavg_price == pytest.approx(expected)


def test_duplicate_reappend_dropped_but_distinct_clips_kept(tmp_path):
    import pandas as pd
    p = tmp_path / "trades.csv"
    base = dict(Timestamp="2025-01-05T09:00", cusip="X", side="buy", nominal=100, principal=98,
                net=-98, accrued=0, price=98.0, yield_closed=5.0,
                trade_date="2025-01-03", settle_date="2025-01-07", trader="A", portfolio="HY")
    # row1 and row2 identical re-append; row3 distinct (different Timestamp)
    rows = [base, dict(base), dict(base, Timestamp="2025-01-05T10:00")]
    pd.DataFrame(rows).to_csv(p, index=False)
    t = load_trades(p)
    assert len(t) == 2  # one duplicate dropped, distinct clip retained
