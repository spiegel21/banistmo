"""Tests for CUSIP/ISIN normalisation (security_id)."""
import pandas as pd
import pytest

import security_id as sid


# Real reference securities (CUSIP, ISIN) — Apple & IBM senior notes.
APPLE = ("037833100", "US0378331005")
IBM = ("459200101", "US4592001014")


# ── check digits ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cusip", [APPLE[0], IBM[0]])
def test_cusip_check_digit_valid(cusip):
    assert sid.cusip_check_digit(cusip[:8]) == cusip[8]
    assert sid.is_valid_cusip(cusip)


def test_cusip_check_digit_rejects_wrong_digit():
    assert not sid.is_valid_cusip("037833101")  # last digit should be 0


@pytest.mark.parametrize("cusip,isin", [APPLE, IBM])
def test_isin_check_digit_valid(cusip, isin):
    assert sid.isin_check_digit(isin[:11]) == isin[11]
    assert sid.is_valid_isin(isin)


def test_isin_check_digit_rejects_wrong_digit():
    assert not sid.is_valid_isin("US0378331004")  # check digit should be 5


# ── structural ISIN → CUSIP ───────────────────────────────────────────────────

@pytest.mark.parametrize("cusip,isin", [APPLE, IBM])
def test_isin_to_cusip_us(cusip, isin):
    assert sid.isin_to_cusip(isin) == cusip


def test_isin_to_cusip_rejects_non_us_prefix():
    # A well-formed non-US ISIN carries no embedded CUSIP — must not be guessed.
    assert sid.isin_to_cusip("XS1234567890") is None


def test_isin_to_cusip_rejects_invalid_check_digit():
    assert sid.isin_to_cusip("US0378331004") is None


# ── canonical_id ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cusip,isin", [APPLE, IBM])
def test_canonical_id_collapses_us_isin(cusip, isin):
    assert sid.canonical_id(isin) == cusip
    # already-canonical CUSIP is returned unchanged (upper-cased)
    assert sid.canonical_id(cusip) == cusip


def test_canonical_id_is_idempotent():
    once = sid.canonical_id(APPLE[1])
    assert sid.canonical_id(once) == once


def test_canonical_id_lowercase_and_whitespace():
    assert sid.canonical_id("  us0378331005  ") == APPLE[0]


def test_canonical_id_blank_inputs():
    assert sid.canonical_id("") == ""
    assert sid.canonical_id(None) == ""
    assert sid.canonical_id("nan") == "nan"  # cleaned to empty -> stripped original


def test_canonical_id_unknown_isin_without_map_is_left_alone():
    # Non-US ISIN, no crosswalk → cannot be safely merged, stays as itself.
    assert sid.canonical_id("XS1234567890") == "XS1234567890"


def test_canonical_id_uses_explicit_crosswalk():
    amap = {"XS1234567890": "AB1234567"}
    assert sid.canonical_id("XS1234567890", amap) == "AB1234567"


def test_canonical_id_crosswalk_overrides_structural():
    # An explicit map entry wins even over the structural rule.
    amap = {APPLE[1]: "ZZZZZZZZZ"}
    assert sid.canonical_id(APPLE[1], amap) == "ZZZZZZZZZ"


def test_canonicalize_series_preserves_nulls():
    s = pd.Series([APPLE[1], IBM[0], None])
    out = sid.canonicalize_series(s)
    assert out.iloc[0] == APPLE[0]
    assert out.iloc[1] == IBM[0]
    assert pd.isna(out.iloc[2])


# ── alias-map loader ──────────────────────────────────────────────────────────

def test_load_alias_map_from_bonds_static(tmp_path):
    bs = tmp_path / "bonds_static.csv"
    bs.write_text("cusip,name,isin\nAB1234567,Foo,XS1234567890\n")
    amap = sid.load_alias_map(bonds_static_path=bs, id_map_path=tmp_path / "none.csv")
    assert amap == {"XS1234567890": "AB1234567"}


def test_load_alias_map_from_id_map(tmp_path):
    idmap = tmp_path / "id_map.csv"
    idmap.write_text("alias,cusip\nXS9999999999,AB1234567\n")
    amap = sid.load_alias_map(bonds_static_path=tmp_path / "none.csv", id_map_path=idmap)
    assert amap == {"XS9999999999": "AB1234567"}


def test_load_alias_map_canonicalises_target(tmp_path):
    # If the crosswalk target is itself a US ISIN, store its CUSIP.
    idmap = tmp_path / "id_map.csv"
    idmap.write_text(f"alias,cusip\nXS9999999999,{APPLE[1]}\n")
    amap = sid.load_alias_map(bonds_static_path=tmp_path / "none.csv", id_map_path=idmap)
    assert amap == {"XS9999999999": APPLE[0]}


def test_load_alias_map_merges_rows_sharing_isin(tmp_path):
    # Two bonds_static rows under different CUSIPs but the SAME ISIN are the same
    # bond fetched twice — they must collapse onto one canonical id automatically.
    bs = tmp_path / "bonds_static.csv"
    bs.write_text(
        "cusip,name,isin\n"
        f"{APPLE[0]},Apple 5% 2030,XS1234567890\n"
        "ZZ9999999,Apple 5% 2030,XS1234567890\n"
    )
    amap = sid.load_alias_map(bonds_static_path=bs, id_map_path=tmp_path / "none.csv")
    # Both the ISIN and the non-canonical CUSIP resolve to the valid CUSIP.
    assert sid.canonical_id("ZZ9999999", amap) == APPLE[0]
    assert sid.canonical_id("XS1234567890", amap) == APPLE[0]
    assert sid.canonical_id(APPLE[0], amap) == APPLE[0]


def test_load_alias_map_singleton_isin_unchanged(tmp_path):
    # A single row's ISIN still maps to its CUSIP (no regression).
    bs = tmp_path / "bonds_static.csv"
    bs.write_text("cusip,name,isin\nAB1234567,Foo,XS1234567890\n")
    amap = sid.load_alias_map(bonds_static_path=bs, id_map_path=tmp_path / "none.csv")
    assert amap == {"XS1234567890": "AB1234567"}
