"""Every JSON output declares the one canonical basis, and they agree on the sample.

If a module is re-run on a different basis, or the sample window shifts, these fail.
"""
from __future__ import annotations

from conftest import (COST_CRC_PER_SIDE, NOTIONAL_USD, SESSIONS_PER_YEAR, _load_json)

N_DAYS = 2663  # the analysis frame (2014-12-08 -> 2026-06-19)


def test_ranking_meta_is_canonical(ranking):
    m = ranking["_meta"]
    assert m["sessions_per_year_actual"] == SESSIONS_PER_YEAR
    assert m["sessions_per_year_legacy"] == 252
    assert m["cost_crc_per_side"] == COST_CRC_PER_SIDE
    assert m["notional_usd"] == NOTIONAL_USD
    assert m["n_days"] == N_DAYS
    assert m["date_min"] == "2014-12-08" and m["date_max"] == "2026-06-19"
    assert m["split_date"] == "2021-11-18"


def test_exit_lab_and_intervention_declare_231(exit_lab, intervention):
    assert exit_lab["_meta"]["sessions_per_year"] == SESSIONS_PER_YEAR
    assert exit_lab["_meta"]["notional_usd"] == NOTIONAL_USD
    assert intervention["_meta"]["sessions_per_year"] == SESSIONS_PER_YEAR
    assert intervention["_meta"]["notional_usd"] == NOTIONAL_USD
    assert intervention["_meta"]["cost_crc_per_side"] == COST_CRC_PER_SIDE


def test_legacy_modules_now_on_231(quincena, dynamics):
    # These were on 252 before the consolidation; they must now carry the shared basis.
    assert quincena["_meta"]["sessions_per_year"] == SESSIONS_PER_YEAR
    assert dynamics["_meta"]["sessions_per_year"] == SESSIONS_PER_YEAR
    # years == n_days / 231 (11.5), not n_days / 252 (10.6)
    assert quincena["_meta"]["years"] == round(N_DAYS / SESSIONS_PER_YEAR, 1)
    assert dynamics["_meta"]["years"] == round(N_DAYS / SESSIONS_PER_YEAR, 1)


def test_all_outputs_share_the_sample():
    for name in ["ranking.json", "exit_lab.json", "intervention_results.json",
                 "quincena_results.json", "dynamics_results.json",
                 "backtest_results.json", "backtest_vwap_results.json", "vm_results.json"]:
        meta = _load_json(name)["_meta"]
        assert meta["n_days"] == N_DAYS, f"{name} has n_days={meta['n_days']}"
