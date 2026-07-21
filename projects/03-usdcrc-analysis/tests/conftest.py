"""Shared fixtures/paths for the USD/CRC consolidation test suite.

Deliberately stdlib-only (json, csv, re, pathlib) + pytest, so the whole suite runs in
any environment without pandas/numpy/matplotlib. These tests validate the *consolidated
report*: that every module shares one accounting basis, that the JSON outputs are
internally consistent, and that the prose (FINDINGS.md, report.html) matches the JSON
source of truth to the digit.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "out"
DATA = ROOT / "data"

# Canonical basis (must match src/basis.py and every JSON _meta).
SESSIONS_PER_YEAR = 231.0
COST_CRC_PER_SIDE = 0.325
COST_CRC_ROUNDTRIP = 0.65
NOTIONAL_USD = 1_000_000


def _load_json(name):
    return json.loads((OUT / name).read_text())


@pytest.fixture(scope="session")
def ranking():
    return _load_json("ranking.json")


@pytest.fixture(scope="session")
def ranking_by_label(ranking):
    return {r["label"]: r for r in ranking["ranking"]}


@pytest.fixture(scope="session")
def exit_lab():
    return _load_json("exit_lab.json")


@pytest.fixture(scope="session")
def intervention():
    return _load_json("intervention_results.json")


@pytest.fixture(scope="session")
def quincena():
    return _load_json("quincena_results.json")


@pytest.fixture(scope="session")
def dynamics():
    return _load_json("dynamics_results.json")


@pytest.fixture(scope="session")
def findings_text():
    return (ROOT / "FINDINGS.md").read_text()


@pytest.fixture(scope="session")
def report_html():
    return (OUT / "report.html").read_text()


def read_csv_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
