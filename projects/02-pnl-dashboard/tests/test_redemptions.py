"""Tests for automatic par-redemption of matured bonds.

A matured bond is auto-redeemed at par: the position closes, the pull-to-par
P&L is realized, and the face value moves off the book as cash. These exercise
position_manager.synthesize_redemptions and its interaction with the existing
WAVG position / realized-gain machinery.
"""
from datetime import date

import pandas as pd
import pytest

from models import BondStatic
from mtm import mark_to_market
from position_manager import (
    synthesize_redemptions, compute_positions, REDEMPTION_TRADER,
)
from trading_gains import realized_pnl


def _bond(cusip, maturity, **ov) -> BondStatic:
    base = dict(
        cusip=cusip, name=cusip, currency="USD", country="US",
        coupon_rate=0.05, coupon_frequency=2, day_count_convention="30/360",
        maturity_date=maturity, first_coupon_date=date(2020, 1, 15),
    )
    base.update(ov)
    return BondStatic(**base)


def _trade(cusip, side, nominal, price, trade_date, portfolio="HY") -> dict:
    """A signed-nominal trade row shaped like load_all_trades output."""
    signed = nominal if side == "buy" else -nominal
    principal = price * nominal / 100
    return dict(
        Timestamp="", cusip=cusip, side=side, nominal=signed,
        principal=principal, net=(-principal if side == "buy" else principal),
        accrued=0.0, price=price, yield_closed=float("nan"),
        trade_date=pd.Timestamp(trade_date), settle_date=pd.Timestamp(trade_date),
        trader="alice", portfolio=portfolio,
    )


AS_OF = date(2026, 1, 1)   # well after the matured bonds below


def test_matured_long_redeems_at_par():
    trades = pd.DataFrame([_trade("A", "buy", 1_000_000, 107.53, "2020-06-15")])
    bonds = {"A": _bond("A", date(2024, 6, 15))}

    reds = synthesize_redemptions(trades, bonds, as_of=AS_OF)
    assert len(reds) == 1
    r = reds.iloc[0]
    assert r["trader"] == REDEMPTION_TRADER
    assert r["side"] == "sell"
    assert r["price"] == 100.0
    assert r["nominal"] == -1_000_000           # flattens the long
    assert r["net"] == pytest.approx(1_000_000)  # cash in = face value
    assert r["trade_date"] == pd.Timestamp("2024-06-15")


def test_redemption_closes_position_and_realizes_pull_to_par():
    trades = pd.DataFrame([_trade("A", "buy", 1_000_000, 107.53, "2020-06-15")])
    bonds = {"A": _bond("A", date(2024, 6, 15))}
    augmented = pd.concat([trades, synthesize_redemptions(trades, bonds, as_of=AS_OF)],
                          ignore_index=True)

    # Position is flat after maturity — it's now cash, not a holding, so it
    # drops out of mark-to-market / exposure (which skip zero positions).
    pos = compute_positions(augmented, as_of=AS_OF)
    assert pos["A"].net_nominal == 0.0
    mtm = mark_to_market(pos, {"A": 100.0}, bonds, as_of=AS_OF)
    assert "A" not in set(mtm["cusip"])

    # Premium bond pulled to par → realized loss of (100 - 107.53)% of face.
    detail = realized_pnl(augmented)
    assert detail["realized_gain"].sum() == pytest.approx(-75_300.0, abs=1.0)


def test_discount_bond_realizes_gain():
    trades = pd.DataFrame([_trade("B", "buy", 1_000_000, 95.0, "2020-06-15")])
    bonds = {"B": _bond("B", date(2024, 6, 15))}
    augmented = pd.concat([trades, synthesize_redemptions(trades, bonds, as_of=AS_OF)],
                          ignore_index=True)
    detail = realized_pnl(augmented)
    assert detail["realized_gain"].sum() == pytest.approx(50_000.0, abs=1.0)


def test_short_position_covered_at_par():
    trades = pd.DataFrame([_trade("A", "sell", 1_000_000, 98.0, "2020-06-15")])
    bonds = {"A": _bond("A", date(2024, 6, 15))}
    reds = synthesize_redemptions(trades, bonds, as_of=AS_OF)
    r = reds.iloc[0]
    assert r["side"] == "buy"                     # cover the short
    assert r["nominal"] == 1_000_000
    assert r["net"] == pytest.approx(-1_000_000)  # cash out to repay par
    augmented = pd.concat([trades, reds], ignore_index=True)
    assert compute_positions(augmented, as_of=AS_OF)["A"].net_nominal == 0.0


def test_future_maturity_not_redeemed():
    trades = pd.DataFrame([_trade("C", "buy", 1_000_000, 100.0, "2020-06-15")])
    bonds = {"C": _bond("C", date(2030, 6, 15))}
    assert synthesize_redemptions(trades, bonds, as_of=AS_OF).empty
    # Position stays open.
    assert "C" in compute_positions(trades, as_of=AS_OF)


def test_redeemed_on_maturity_date_boundary():
    trades = pd.DataFrame([_trade("A", "buy", 1_000_000, 100.0, "2020-06-15")])
    bonds = {"A": _bond("A", date(2026, 1, 1))}
    # as_of exactly on the maturity date → redeemed that day.
    reds = synthesize_redemptions(trades, bonds, as_of=date(2026, 1, 1))
    assert len(reds) == 1
    # The day before maturity → still open.
    assert synthesize_redemptions(trades, bonds, as_of=date(2025, 12, 31)).empty


def test_per_portfolio_independent_redemption():
    trades = pd.DataFrame([
        _trade("A", "buy", 1_000_000, 107.53, "2020-06-15", portfolio="HY"),
        _trade("A", "buy", 2_000_000, 95.00, "2020-06-15", portfolio="IG"),
    ])
    bonds = {"A": _bond("A", date(2024, 6, 15))}
    reds = synthesize_redemptions(trades, bonds, as_of=AS_OF)
    assert set(reds["portfolio"]) == {"HY", "IG"}
    assert sorted(reds["nominal"].tolist()) == [-2_000_000, -1_000_000]


def test_already_flat_position_not_redeemed():
    # Bought then fully sold before maturity → nothing to redeem.
    trades = pd.DataFrame([
        _trade("A", "buy", 1_000_000, 100.0, "2020-06-15"),
        _trade("A", "sell", 1_000_000, 101.0, "2021-06-15"),
    ])
    bonds = {"A": _bond("A", date(2024, 6, 15))}
    assert synthesize_redemptions(trades, bonds, as_of=AS_OF).empty


def test_empty_stream_returns_empty():
    empty = pd.DataFrame(columns=["cusip", "nominal", "trade_date", "portfolio", "net"])
    assert synthesize_redemptions(empty, {}, as_of=AS_OF).empty
