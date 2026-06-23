"""Tests for per-bond movement lineage."""
import pandas as pd
import pytest

from movements import position_movements, MOVEMENT_COLUMNS
from position_manager import load_all_trades
from trading_gains import realized_pnl


def test_empty_returns_columns():
    out = position_movements(pd.DataFrame())
    assert list(out.columns) == MOVEMENT_COLUMNS
    assert out.empty


def test_running_position_and_wavg():
    trades = pd.DataFrame([
        dict(cusip="X", portfolio="P", trader="A", nominal=100, net=-100.0,
             price=100.0, trade_date=pd.Timestamp("2025-01-01")),
        dict(cusip="X", portfolio="P", trader="A", nominal=100, net=-102.0,
             price=102.0, trade_date=pd.Timestamp("2025-01-02")),
    ])
    mv = position_movements(trades, cusip="X")
    assert mv.iloc[0]["running_nominal"] == 100
    assert mv.iloc[1]["running_nominal"] == 200
    # WAVG of (100@1.00, 100@1.02) = 1.01
    assert mv.iloc[1]["running_wavg_cost"] == pytest.approx(1.01, abs=1e-6)


def test_realized_matches_trading_gains_total():
    """Per-trade realized in movements must sum to the engine's realized total."""
    trades = load_all_trades()
    mv = position_movements(trades)
    mv_total = mv["realized_gain"].sum()
    eng_total = realized_pnl(trades)["realized_gain"].sum()
    assert mv_total == pytest.approx(eng_total, abs=1.0)


def test_close_books_realized():
    trades = pd.DataFrame([
        dict(cusip="S", portfolio="P", trader="A", nominal=-100, net=101.0,
             price=101.0, trade_date=pd.Timestamp("2025-01-01")),  # short open
        dict(cusip="S", portfolio="P", trader="A", nominal=100, net=-100.0,
             price=100.0, trade_date=pd.Timestamp("2025-01-02")),  # cover
    ])
    mv = position_movements(trades, cusip="S")
    assert mv.iloc[1]["running_nominal"] == 0
    assert mv.iloc[1]["realized_gain"] == pytest.approx(1.0, abs=1e-6)
    assert mv.iloc[1]["cumulative_realized"] == pytest.approx(1.0, abs=1e-6)


def test_cumulative_cash_tracks_net():
    trades = pd.DataFrame([
        dict(cusip="X", portfolio="P", trader="A", nominal=100, net=-100.0,
             price=100.0, trade_date=pd.Timestamp("2025-01-01")),
        dict(cusip="X", portfolio="P", trader="A", nominal=-50, net=51.0,
             price=102.0, trade_date=pd.Timestamp("2025-01-03")),
    ])
    mv = position_movements(trades, cusip="X")
    assert mv.iloc[-1]["cumulative_cash"] == pytest.approx(-49.0, abs=1e-6)
