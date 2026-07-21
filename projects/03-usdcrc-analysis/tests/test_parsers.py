"""Validate the parsers' clean-CSV output schema and basic sanity.

Cheap guards against a silently-broken parse (renamed column, empty file, unsorted or
duplicated dates) that would poison every downstream number.
"""
from __future__ import annotations

import re

from conftest import DATA, read_csv_rows

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def test_monex_clean_schema_and_sanity():
    rows = read_csv_rows(DATA / "monex_clean.csv")
    assert len(rows) > 2000
    expected = {"date", "open", "close", "low", "high", "vwap", "volume_usd", "n_trades"}
    assert expected <= set(rows[0].keys())
    dates = [r["date"] for r in rows]
    assert all(_DATE.fullmatch(d) for d in dates), "malformed date in monex_clean.csv"
    assert dates == sorted(dates), "monex_clean.csv dates not sorted"
    assert len(dates) == len(set(dates)), "duplicate dates in monex_clean.csv"
    # vwap is never negative; on actual trading days it is strictly positive
    # (holidays are carried with vwap 0 and is_trading_day False)
    trading_days = 0
    for r in rows:
        v = _to_float(r["vwap"])
        if v is None:
            continue
        assert v >= 0, f"negative vwap on {r['date']}"
        if str(r.get("is_trading_day", "")).strip().lower() in ("true", "1"):
            trading_days += 1
            assert v > 0, f"non-positive vwap on trading day {r['date']}"
    assert trading_days > 2000, "too few trading days parsed"


def test_bccr_reserves_schema():
    rows = read_csv_rows(DATA / "bccr_reserves_clean.csv")
    assert len(rows) > 50
    assert {"date", "reserves_usd_mn"} <= set(rows[0].keys())
    for r in rows:
        assert _DATE.fullmatch(r["date"])
        res = _to_float(r["reserves_usd_mn"])
        assert res is None or res > 0


def test_bccr_intervention_schema():
    rows = read_csv_rows(DATA / "bccr_intervention_clean.csv")
    assert len(rows) > 2000
    cols = set(rows[0].keys())
    assert {"date", "off_net_usd", "off_gross_usd", "is_intervention_day"} <= cols
    assert all(_DATE.fullmatch(r["date"]) for r in rows)
