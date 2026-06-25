"""Tests for deterministic CUSIP → portfolio assignment."""
import sys
from pathlib import Path

import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

import config  # noqa: E402
import data_io  # noqa: E402
import bond_portfolio as bp  # noqa: E402
from models import BondStatic  # noqa: E402
from position_manager import load_all_trades, compute_positions  # noqa: E402


def _bond(cusip, issuer="", name=""):
    return BondStatic(
        cusip=cusip, name=name or cusip, currency="USD", country="US",
        coupon_rate=0.05, coupon_frequency=2, day_count_convention="30/360",
        maturity_date=pd.Timestamp("2030-01-15").date(),
        first_coupon_date=pd.Timestamp("2025-01-15").date(),
        issuer=issuer,
    )


def _initial(rows):
    return pd.DataFrame(rows, columns=["portfolio", "cusip"])


# ── core resolution rules ─────────────────────────────────────────────────────

def test_initial_position_wins():
    initial = _initial([("HY", "AAA111111")])
    asg = bp.resolve_portfolios(initial, {}, ["AAA111111"])
    assert asg.portfolio["AAA111111"] == "HY"
    assert asg.source["AAA111111"] == bp.SOURCE_INITIAL
    assert asg.assigned["AAA111111"] == "HY"


def test_issuer_inference_same_issuer_one_book():
    # Two CUSIPs, same issuer name; only one is seeded in initial positions.
    bonds = {"AAA111111": _bond("AAA111111", issuer="Acme"),
             "BBB222222": _bond("BBB222222", issuer="Acme")}
    initial = _initial([("HY", "AAA111111")])
    asg = bp.resolve_portfolios(initial, bonds, ["AAA111111", "BBB222222"])
    # The un-seeded bond inherits the issuer's single book.
    assert asg.portfolio["BBB222222"] == "HY"
    assert asg.source["BBB222222"] == bp.SOURCE_ISSUER


def test_issuer_inference_via_cusip6_prefix_when_no_issuer_name():
    # No issuer field → 6-digit CUSIP prefix is the issuer proxy.
    initial = _initial([("IG", "037833100")])
    asg = bp.resolve_portfolios(initial, {}, ["037833100", "037833AB1"])
    assert asg.portfolio["037833AB1"] == "IG"
    assert asg.source["037833AB1"] == bp.SOURCE_ISSUER


def test_ambiguous_issuer_left_unassigned():
    # Same issuer name seeded into two different books → cannot auto-assign.
    bonds = {"AAA111111": _bond("AAA111111", issuer="Acme"),
             "AAA222222": _bond("AAA222222", issuer="Acme"),
             "AAA333333": _bond("AAA333333", issuer="Acme")}
    initial = _initial([("HY", "AAA111111"), ("IG", "AAA222222")])
    asg = bp.resolve_portfolios(initial, bonds, list(bonds))
    # The third Acme bond is ambiguous → unassigned, default fallback, no pin.
    assert "AAA333333" in asg.unassigned
    assert asg.source["AAA333333"] == bp.SOURCE_UNASSIGNED
    assert asg.portfolio["AAA333333"] == config.DEFAULT_PORTFOLIO
    assert "AAA333333" not in asg.assigned
    assert "Acme" in asg.ambiguous_issuers
    assert sorted(asg.ambiguous_issuers["Acme"]) == ["HY", "IG"]


def test_distinct_issuers_sharing_cusip6_are_not_ambiguous():
    # Same 6-digit prefix but different issuer NAMES in different books → the
    # issuer-name field keeps them apart, so neither is ambiguous.
    bonds = {"25714PFB9": _bond("25714PFB9", issuer="HY Issuer"),
             "25714PEF1": _bond("25714PEF1", issuer="IG Issuer")}
    initial = _initial([("HY", "25714PFB9"), ("IG", "25714PEF1")])
    asg = bp.resolve_portfolios(initial, bonds, list(bonds))
    assert asg.ambiguous_issuers == {}
    assert asg.portfolio["25714PFB9"] == "HY"
    assert asg.portfolio["25714PEF1"] == "IG"


def test_manual_pin_overrides_everything():
    bonds = {"AAA111111": _bond("AAA111111", issuer="Acme")}
    initial = _initial([("HY", "AAA111111")])
    asg = bp.resolve_portfolios(initial, bonds, ["AAA111111"], manual_map={"AAA111111": "SPECIAL"})
    assert asg.portfolio["AAA111111"] == "SPECIAL"
    assert asg.source["AAA111111"] == bp.SOURCE_MANUAL


def test_unknown_unrelated_bond_is_unassigned():
    initial = _initial([("HY", "AAA111111")])
    asg = bp.resolve_portfolios(initial, {}, ["ZZZ999999"])
    assert asg.source["ZZZ999999"] == bp.SOURCE_UNASSIGNED
    assert "ZZZ999999" not in asg.assigned  # never clobbers an unrelated bond


def test_empty_inputs_safe():
    asg = bp.resolve_portfolios(pd.DataFrame(), {}, [])
    assert asg.portfolio == {}
    assert asg.unassigned == []


# ── apply_to_trades ───────────────────────────────────────────────────────────

def test_apply_to_trades_overrides_only_assigned():
    trades = pd.DataFrame([
        {"cusip": "AAA111111", "portfolio": "WRONG"},
        {"cusip": "ZZZ999999", "portfolio": "KEEP"},
    ])
    asg = bp.PortfolioAssignment(
        portfolio={}, source={}, assigned={"AAA111111": "HY"},
        issuer_label={}, ambiguous_issuers={},
    )
    out = bp.apply_to_trades(trades, asg)
    assert out.loc[0, "portfolio"] == "HY"     # resolved bond overridden
    assert out.loc[1, "portfolio"] == "KEEP"   # unresolved bond untouched


# ── persistence round-trip ────────────────────────────────────────────────────

def test_manual_map_round_trip(data_dir):
    df = pd.DataFrame([
        {"cusip": "AAA111111", "portfolio": "SPECIAL"},
        {"cusip": "BBB222222", "portfolio": ""},        # blank → dropped
        {"cusip": "", "portfolio": "X"},                # blank cusip → dropped
    ])
    data_io.save_bond_portfolio_map(df)
    loaded = bp.load_manual_map()
    assert loaded == {"AAA111111": "SPECIAL"}


# ── end-to-end through load_all_trades (uses conftest fixtures) ────────────────

def test_load_all_trades_applies_assignment():
    # conftest seeds 037833100→HY in initial positions; 912828XY9 is IG by trade.
    allt = load_all_trades()
    by_cusip = allt.groupby("cusip")["portfolio"].unique()
    assert set(by_cusip["037833100"]) == {"HY"}
    # The unrelated treasury keeps its own book (not auto-moved).
    assert set(by_cusip["912828XY9"]) == {"IG"}


def test_assignment_can_be_disabled():
    raw = load_all_trades(assign_portfolios=False)
    assert not raw.empty


def test_manual_pin_moves_trades_end_to_end():
    # Pin the treasury (booked under IG) to HY and confirm load_all_trades moves
    # every one of its trades. Clean up the pin file so other tests are isolated.
    data_io.save_bond_portfolio_map(
        pd.DataFrame([{"cusip": "912828XY9", "portfolio": "HY"}])
    )
    try:
        allt = load_all_trades()
        moved = allt[allt["cusip"] == "912828XY9"]["portfolio"].unique()
        assert set(moved) == {"HY"}
    finally:
        config.BOND_PORTFOLIO_MAP_PATH.unlink(missing_ok=True)
