from datetime import date

import pandas as pd
import pytest

from position_manager import load_all_trades, compute_positions
from accruals import load_bonds_static, total_portfolio_accruals
from trading_gains import realized_pnl, total_realized_pnl
from mtm import mark_to_market


def test_realized_wavg_long():
    detail = realized_pnl(load_all_trades())
    g = detail[detail["cusip"] == "037833100"]["realized_gain"].sum()
    # Inception 300k@0.97 + buy 1M@0.986 + buy 500k@0.996 → wavg = 1775000/1800000
    # Sell 600k at unit 601000/600000; gain = 600k × (sell_unit − wavg)
    wavg = (300_000 * 0.97 + 1_000_000 * 0.986 + 500_000 * 0.996) / 1_800_000
    expected = 600_000 * (601_000 / 600_000 - wavg)
    assert g == pytest.approx(expected, abs=1)


def test_realized_handles_short_open_and_cover():
    # sell 100 @ 1.01 (open short), buy 100 @ 1.00 (cover) → +1.0 gain
    trades = pd.DataFrame([
        dict(cusip="S", portfolio="P", nominal=-100, net=101.0, price=101.0,
             trade_date=pd.Timestamp("2025-01-01")),
        dict(cusip="S", portfolio="P", nominal=100, net=-100.0, price=100.0,
             trade_date=pd.Timestamp("2025-01-02")),
    ])
    detail = realized_pnl(trades)
    assert detail["realized_gain"].sum() == pytest.approx(1.0, abs=1e-6)


def test_oversell_does_not_invent_profit():
    # buy 100 @ 1.00, sell 150 @ 1.00 → realized 0 on matched 100; 50 opens short
    trades = pd.DataFrame([
        dict(cusip="O", portfolio="P", nominal=100, net=-100.0, price=100.0,
             trade_date=pd.Timestamp("2025-01-01")),
        dict(cusip="O", portfolio="P", nominal=-150, net=150.0, price=100.0,
             trade_date=pd.Timestamp("2025-01-02")),
    ])
    detail = realized_pnl(trades)
    # only the matched 100 realizes (at flat price → 0); the extra 50 is a new short lot
    assert detail["realized_gain"].sum() == pytest.approx(0.0, abs=1e-6)
    assert detail["closed_nominal"].sum() == pytest.approx(100.0)


def test_total_pnl_no_double_count():
    """The headline invariant: realized + price + accrued must NOT double-count accrued."""
    allt = load_all_trades()
    pos = compute_positions(allt, portfolio="HY")
    bonds = load_bonds_static()
    prices = {"037833100": 100.5}
    as_of = date(2025, 3, 15)

    m = mark_to_market(pos, prices, bonds, as_of)
    accruals = total_portfolio_accruals(pos, bonds, as_of)

    # accrued_pnl in mtm must equal the standalone accrual calc (same definition)
    assert m["accrued_pnl"].sum() == pytest.approx(accruals["accrued"].sum())
    # price + accrued must reconstruct full dirty-price mtm_gain
    assert (m["price_pnl"] + m["accrued_pnl"]).sum() == pytest.approx(m["mtm_gain"].sum())


def test_mtm_missing_price_is_flagged():
    pos = compute_positions(load_all_trades(), portfolio="IG")
    m = mark_to_market(pos, {}, load_bonds_static(), date(2025, 3, 15))
    assert (m["note"] == "no price").all()
    assert m["price_pnl"].isna().all()
