"""Tests for the enterprise classification derivation helpers."""
from datetime import date

import classification as clf
from models import BondStatic, normalise_instrument_type


def _bond(**overrides) -> BondStatic:
    base = dict(
        cusip="TEST00000", name="Test 5% 2030", currency="USD", country="US",
        coupon_rate=0.05, coupon_frequency=2, day_count_convention="30/360",
        maturity_date=date(2030, 6, 15), first_coupon_date=date(2024, 6, 15),
    )
    base.update(overrides)
    return BondStatic(**base)


def test_normalise_instrument_type_buckets():
    assert normalise_instrument_type("Govt") == "Sovereign"
    assert normalise_instrument_type("CORP") == "Corporate"
    assert normalise_instrument_type("agency") == "Agency"
    assert normalise_instrument_type("Supranational") == "Supranational"
    assert normalise_instrument_type("") == ""
    assert normalise_instrument_type("weird thing") == "Other"


def test_instrument_type_explicit_wins():
    assert clf.instrument_type(_bond(instrument_type="Sovereign")) == "Sovereign"


def test_instrument_type_from_ticker_suffix():
    b = _bond(instrument_type="", bbg_ticker="912828Z78 Govt")
    assert clf.instrument_type(b) == "Sovereign"
    b2 = _bond(instrument_type="", bbg_ticker="037833DT4 Corp")
    assert clf.instrument_type(b2) == "Corporate"


def test_instrument_type_unknown():
    assert clf.instrument_type(_bond(instrument_type="", bbg_ticker="")) == clf.UNKNOWN


def test_country_of_risk_fallback_to_country():
    assert clf.country_of_risk(_bond(country_of_risk="CO")) == "CO"
    assert clf.country_of_risk(_bond(country_of_risk="", country="US")) == "US"
    assert clf.country_of_risk(_bond(country_of_risk="", country="")) == clf.UNKNOWN


def test_market_scope_heuristic_local_vs_global():
    # Colombian issuer in COP → Local; same issuer in USD → Global.
    assert clf.market_scope(_bond(currency="COP", country_of_risk="CO")) == "Local"
    assert clf.market_scope(_bond(currency="USD", country_of_risk="CO")) == "Global"
    # US issuer in USD → Local.
    assert clf.market_scope(_bond(currency="USD", country="US")) == "Local"


def test_market_scope_explicit_override():
    assert clf.market_scope(_bond(market="Global", currency="USD", country="US")) == "Global"


def test_market_scope_unknown_when_unmappable():
    assert clf.market_scope(_bond(currency="XYZ", country_of_risk="ZZ")) == clf.UNKNOWN


def test_is_sovereign():
    assert clf.is_sovereign(_bond(instrument_type="Sovereign"))
    assert clf.is_sovereign(_bond(instrument_type="Agency"))
    assert not clf.is_sovereign(_bond(instrument_type="Corporate"))


def test_classify_returns_all_dimensions():
    out = clf.classify(_bond(instrument_type="Corporate", currency="COP",
                             country_of_risk="CO", sector="Financials"))
    assert out["instrument_type"] == "Corporate"
    assert out["market"] == "Local"
    assert out["country_of_risk"] == "CO"
    assert out["sector"] == "Financials"
    assert out["seniority"] == clf.UNKNOWN  # not provided
