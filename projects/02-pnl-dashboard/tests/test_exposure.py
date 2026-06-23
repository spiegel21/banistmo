"""Tests for exposure, concentration, and maturity-ladder analytics."""
from datetime import date

import pandas as pd
import pytest

import exposure as exp
from models import BondStatic, Position


def _pos(cusip, nominal=1_000_000):
    return Position(cusip=cusip, net_nominal=nominal, wavg_price=100.0,
                    book_value=-nominal, last_settle=date(2025, 1, 1))


def _bond(cusip, **ov):
    base = dict(
        cusip=cusip, name=f"BND {cusip}", currency="USD", country="US",
        coupon_rate=0.05, coupon_frequency=2, day_count_convention="30/360",
        maturity_date=date(2030, 6, 15), first_coupon_date=date(2024, 6, 15),
        instrument_type="Corporate", country_of_risk="US", sector="Tech",
        market="Local",
    )
    base.update(ov)
    return BondStatic(**base)


def _setup():
    positions = {"A": _pos("A"), "B": _pos("B"), "C": _pos("C")}
    bonds = {
        "A": _bond("A", country_of_risk="CO", currency="USD", market="Global", sector="Financials"),
        "B": _bond("B", country_of_risk="CO", currency="COP", market="Local", sector="Financials"),
        "C": _bond("C", country_of_risk="US", instrument_type="Sovereign", sector="Government"),
    }
    prices = {"A": 100.0, "B": 100.0, "C": 100.0}
    return positions, bonds, prices


def test_exposure_base_has_dimensions():
    base = exp.exposure_base(*_setup(), as_of=date(2026, 1, 1))
    assert not base.empty
    for col in ("country_of_risk", "market", "sector", "instrument_type", "mtm_value"):
        assert col in base.columns
    assert len(base) == 3


def test_aggregate_by_country_of_risk():
    base = exp.exposure_base(*_setup(), as_of=date(2026, 1, 1))
    agg = exp.aggregate_exposure(base, "country_of_risk")
    co = agg[agg["country_of_risk"] == "CO"]
    assert co.iloc[0]["n_bonds"] == 2
    # percentages sum to ~100
    assert agg["pct_mtm"].sum() == pytest.approx(100.0, abs=0.1)


def test_aggregate_local_global_split():
    base = exp.exposure_base(*_setup(), as_of=date(2026, 1, 1))
    agg = exp.aggregate_exposure(base, "market")
    markets = set(agg["market"])
    assert {"Local", "Global"} <= markets


def test_concentration():
    base = exp.exposure_base(*_setup(), as_of=date(2026, 1, 1))
    c = exp.concentration(base, "country_of_risk", top_n=1)
    assert c["largest"] in ("CO", "US")
    assert 0 <= c["largest_pct"] <= 100


def test_maturity_ladder_buckets():
    positions, bonds, prices = _setup()
    bonds["C"].maturity_date = date(2027, 1, 1)   # ~1y from as_of
    base = exp.exposure_base(positions, bonds, prices, as_of=date(2026, 1, 1))
    ladder = exp.maturity_ladder(base, as_of=date(2026, 1, 1))
    assert not ladder.empty
    assert ladder["n_bonds"].sum() == 3
    # A and B mature 2030 (~4.5y) → 3-5y band present
    assert "3-5y" in set(ladder["bucket"])


def test_maturity_ladder_unknown_bucket():
    # A bond with no maturity date must land in the Unknown bucket, not vanish.
    base = pd.DataFrame([
        dict(cusip="A", name="A", net_nominal=1_000_000, mtm_value=1_000_000,
             currency="USD", maturity_date=None),
        dict(cusip="B", name="B", net_nominal=1_000_000, mtm_value=1_000_000,
             currency="USD", maturity_date=date(2030, 6, 15)),
    ])
    ladder = exp.maturity_ladder(base, as_of=date(2026, 1, 1))
    assert ladder["n_bonds"].sum() == 2
    assert exp.UNKNOWN in set(ladder["bucket"])
