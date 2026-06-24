"""Tests for the data-quality / reconciliation engine."""
from datetime import date

import pandas as pd

import reconciliation as rec
from models import BondStatic


def _bond(cusip="037833100", **ov) -> BondStatic:
    base = dict(
        cusip=cusip, name="APPLE 5% 2030", currency="USD", country="US",
        coupon_rate=0.05, coupon_frequency=2, day_count_convention="30/360",
        maturity_date=date(2030, 6, 15), first_coupon_date=date(2024, 6, 15),
        instrument_type="Corporate", country_of_risk="US", sector="Tech",
        market="Local", rating_sp="AA",
    )
    base.update(ov)
    return BondStatic(**base)


def test_trade_blank_cusip_and_bad_side():
    raw = pd.DataFrame([dict(
        cusip="", side="hold", nominal=100, price=99, principal=99, net=-99,
        accrued=0, trade_date="2025-01-01", settle_date="2025-01-03")])
    f = rec.check_trades(raw, {})
    assert any(x["field"] == "cusip" and x["severity"] == "error" for x in f)
    assert any(x["field"] == "side" for x in f)


def test_trade_unknown_cusip_warns():
    raw = pd.DataFrame([dict(
        cusip="999999999", side="buy", nominal=100, price=100, principal=100,
        net=-100, accrued=0, trade_date="2025-01-01", settle_date="2025-01-03")])
    f = rec.check_trades(raw, {})  # empty bonds_static → unknown
    assert any("no bond static" in x["issue"] for x in f)


def test_trade_economic_mismatch_principal():
    # principal should be 100*1000/100 = 1000, but stored 1234 → flag
    raw = pd.DataFrame([dict(
        cusip="037833100", side="buy", nominal=1000, price=100, principal=1234,
        net=-1000, accrued=0, trade_date="2025-01-01", settle_date="2025-01-03")])
    f = rec.check_trades(raw, {"037833100": _bond()})
    assert any(x["field"] == "principal" for x in f)


def test_trade_settle_before_trade():
    raw = pd.DataFrame([dict(
        cusip="037833100", side="buy", nominal=1000, price=100, principal=1000,
        net=-1000, accrued=0, trade_date="2025-01-05", settle_date="2025-01-01")])
    f = rec.check_trades(raw, {"037833100": _bond()})
    assert any(x["field"] == "settle_date" for x in f)


def test_trade_clean_row_no_findings():
    raw = pd.DataFrame([dict(
        cusip="037833100", side="buy", nominal=1000, price=100, principal=1000,
        net=-1000, accrued=0, trade_date="2025-01-01", settle_date="2025-01-03")])
    f = rec.check_trades(raw, {"037833100": _bond()})
    assert f == []


def test_bond_missing_name_and_classification():
    b = _bond(name="", instrument_type="", country_of_risk="", country="",
              sector="", market="", currency="", rating_sp="")
    f = rec.check_bonds({b.cusip: b}, held_cusips={b.cusip})
    fields = {x["field"] for x in f}
    assert "name" in fields
    assert "instrument_type" in fields
    assert "country_of_risk" in fields
    assert "sector" in fields
    assert "rating" in fields
    # held bond with missing classification escalates to needs_input
    assert any(x["field"] == "country_of_risk" and x["severity"] == "needs_input" for x in f)


def test_bond_fully_populated_minimal_findings():
    b = _bond(sector="Technology")
    f = rec.check_bonds({b.cusip: b}, held_cusips={b.cusip})
    # AA rating + all classification present → no findings
    assert f == []


def test_price_missing_weird_and_nonpositive():
    held = {"A", "B", "C"}
    prices = {"B": 5.0, "C": -3.0}  # A missing, B too low, C negative
    f = rec.check_prices(held, prices, pd.DataFrame(), {})
    by_key = {(x["key"], x["severity"]) for x in f}
    assert ("A", "warning") in by_key       # missing
    assert any(x["key"] == "B" and "plausible" in x["issue"] for x in f)
    assert any(x["key"] == "C" and x["severity"] == "error" for x in f)


def test_price_stale_detection():
    held = {"A"}
    ph = pd.DataFrame([
        dict(date=f"2025-01-0{i}", cusip="A", px_last=99.0) for i in range(1, 7)
    ])
    f = rec.check_prices(held, {"A": 99.0}, ph, {})
    assert any("stale" in x["issue"] for x in f)


def test_matured_held_bond_flagged():
    b = _bond(maturity_date=date(2024, 6, 15))   # already matured vs as_of below
    f = rec.check_matured_bonds({b.cusip}, {b.cusip: b}, as_of=date(2026, 1, 1))
    assert len(f) == 1
    assert f[0]["field"] == "maturity_date"
    assert f[0]["severity"] == "warning"
    assert "matured" in f[0]["issue"]


def test_matured_on_maturity_date_flagged():
    # On the maturity date the bond has redeemed → still flag the open position.
    b = _bond(maturity_date=date(2026, 1, 1))
    f = rec.check_matured_bonds({b.cusip}, {b.cusip: b}, as_of=date(2026, 1, 1))
    assert len(f) == 1


def test_live_bond_not_flagged():
    b = _bond(maturity_date=date(2030, 6, 15))
    f = rec.check_matured_bonds({b.cusip}, {b.cusip: b}, as_of=date(2026, 1, 1))
    assert f == []


def test_matured_but_not_held_not_flagged():
    # Only held positions matter — a reference-only matured bond is not noise.
    b = _bond(maturity_date=date(2024, 6, 15))
    f = rec.check_matured_bonds(set(), {b.cusip: b}, as_of=date(2026, 1, 1))
    assert f == []


def test_run_all_checks_includes_matured(tmp_path):
    held_b = _bond(maturity_date=date(2024, 6, 15))
    df, summary = rec.run_all_checks(
        raw_trades=pd.DataFrame(), bonds_static={held_b.cusip: held_b},
        held_cusips={held_b.cusip}, current_prices={held_b.cusip: 100.0},
        price_history=pd.DataFrame(), as_of=date(2026, 1, 1))
    assert any(x == "maturity_date" for x in df["field"])


def test_run_all_checks_summary_shape():
    raw = pd.DataFrame([dict(
        cusip="037833100", side="buy", nominal=1000, price=100, principal=1000,
        net=-1000, accrued=0, trade_date="2025-01-01", settle_date="2025-01-03")])
    df, summary = rec.run_all_checks(
        raw_trades=raw, bonds_static={"037833100": _bond()},
        held_cusips=set(), current_prices={}, price_history=pd.DataFrame())
    assert set(rec.FINDING_COLUMNS) == set(df.columns)
    assert summary["total"] == len(df)
    for k in ("error", "warning", "needs_input"):
        assert k in summary
