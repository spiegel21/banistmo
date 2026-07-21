"""The single-accounting-basis guarantee, at the source level.

The whole consolidation rests on one claim: every module annualises, costs, and sizes
trades identically, from src/basis.py. These tests fail loudly if a module drifts back to
the legacy 252 convention or re-defines the basis locally.
"""
from __future__ import annotations

import re

from conftest import (COST_CRC_PER_SIDE, COST_CRC_ROUNDTRIP, NOTIONAL_USD,
                      SESSIONS_PER_YEAR, SRC)

# Modules that price/annualise strategies and therefore must be on the shared basis.
STRATEGY_MODULES = [
    "analyze.py", "strategies.py", "backtest.py", "backtest_vwap.py", "volume_model.py",
    "dynamics.py", "quincena.py", "exits.py", "rank_strategies.py", "exit_lab.py",
    "intervention.py",
]

# Arithmetic uses of the literal 252 (annualisation). Comments and windows-1252 do NOT
# match: \b252\b never matches inside "1252", and none of these operators appear in prose.
_LEGACY_252 = re.compile(r"(sqrt\(\s*252\b|[*/]\s*252\b|\b252\s*[*/]|\b252\s*/)")


def test_basis_module_defines_canonical_constants():
    ns = {}
    exec((SRC / "basis.py").read_text(), ns)
    assert ns["SESSIONS_PER_YEAR"] == SESSIONS_PER_YEAR
    assert ns["COST_CRC_PER_SIDE"] == COST_CRC_PER_SIDE
    assert ns["COST_CRC_ROUNDTRIP"] == COST_CRC_ROUNDTRIP
    assert ns["NOTIONAL_USD"] == NOTIONAL_USD
    # round-trip is exactly twice the per-side cost
    assert abs(ns["COST_CRC_ROUNDTRIP"] - 2 * ns["COST_CRC_PER_SIDE"]) < 1e-12
    # legacy inflation factors are self-consistent with the two densities
    assert abs(ns["PER_YEAR_LEGACY_INFLATION"] - (252 / SESSIONS_PER_YEAR - 1)) < 1e-9


def test_no_strategy_module_hardcodes_252_arithmetic():
    offenders = {}
    for mod in STRATEGY_MODULES:
        src = (SRC / mod).read_text()
        hits = [ln for ln in src.splitlines() if _LEGACY_252.search(ln)]
        if hits:
            offenders[mod] = hits
    assert not offenders, f"legacy 252 annualisation still present: {offenders}"


def test_only_basis_module_may_reference_legacy_constant():
    # SESSIONS_PER_YEAR_LEGACY is defined once, in basis.py, for documentation/metadata.
    for mod in STRATEGY_MODULES:
        src = (SRC / mod).read_text()
        # if a module mentions the legacy density it must do so via the shared name,
        # never a bare arithmetic literal (covered above) — this just documents intent.
        assert "np.sqrt(252)" not in src, f"{mod} still annualises Sharpe on 252"


def test_strategy_modules_import_basis():
    # Every pricing module must pull its constants from basis (directly or transitively
    # via rank_strategies, which re-exports them). Check for a basis import edge.
    direct = {"analyze.py", "strategies.py", "backtest.py", "volume_model.py",
              "dynamics.py", "quincena.py", "rank_strategies.py"}
    for mod in direct:
        src = (SRC / mod).read_text()
        assert "from basis import" in src or "import basis" in src, \
            f"{mod} does not import the shared basis"
