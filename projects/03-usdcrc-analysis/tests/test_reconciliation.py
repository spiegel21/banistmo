"""Prose must match the JSON source of truth to the digit.

The report's whole failure mode over its history was hand-typed numbers drifting from the
runs that produced them. These tests re-derive every headline figure from the JSON and
assert it appears, correctly, in FINDINGS.md and out/report.html — so a stale number can
never ship silently again.
"""
from __future__ import annotations

import re


def _norm(s):
    """Normalise doc text for numeric matching: unify minus signs, drop $ and NBSP."""
    return (s.replace("−", "-").replace("–", "-")
             .replace(" ", " ").replace("$", ""))


def _num(cell):
    """Parse a table cell like '$100,673' or '-$15,986' or '3.70' to a number."""
    c = _norm(cell).replace(",", "").strip()
    m = re.search(r"-?\d+\.?\d*", c)
    return float(m.group()) if m else None


# --------------------------------------------------------------------------- #
# A. Curated headline facts — the numbers the recommendation is sold on.
# --------------------------------------------------------------------------- #
def test_headline_facts_present_in_findings(findings_text, ranking_by_label, exit_lab, intervention):
    doc = _norm(findings_text)
    top = ranking_by_label["Calendar + trail 30 + floor 40 (exit_lab)"]
    rec = exit_lab["variants"][exit_lab["recommended"]]
    facts = [
        f"{top['is']['sharpe']:.2f}",                    # 3.70 IS Sharpe of the top rule
        f"{top['oos']['sharpe']:.2f}",                   # 3.92 OOS
        f"{top['is']['per_year_usd']:,}",                # 100,673 IS $/yr
        f"{abs(top['is']['maxdd_usd']):,}",              # 15,986 IS max DD
        f"{exit_lab['baseline']['is']['sharpe']:.2f}",   # 3.14 baseline IS
        f"{exit_lab['baseline']['oos']['sharpe']:.2f}",  # 2.79 baseline OOS
        f"{abs(rec['oos']['maxdd_usd']):,}",             # 26,192 overlay OOS max DD
        f"{intervention['strategy']['+ reserve-regime long-trim']['oos']['sharpe']:.2f}",  # 4.09
    ]
    missing = [f for f in facts if f not in doc]
    assert not missing, f"headline facts missing from FINDINGS.md: {missing}"


def test_reserve_significance_stated(findings_text, intervention):
    rvb = intervention["reserve_vs_blind"]
    doc = _norm(findings_text)
    assert f"{rvb['paired_t']:.2f}" in doc          # t = 3.31
    assert "0.001" in doc                            # p = 0.001
    assert f"{rvb['oos_total_delta_usd']//1000}" in doc.replace(",", "")  # ~$56k delta


def test_official_flow_mechanism_matches_json(findings_text, intervention):
    flow = intervention["flow"]
    doc = _norm(findings_text)
    # median official share ~52% and the contemporaneous correlation ~-0.25
    assert str(round(flow["official_share_median"] * 100)) in doc          # 52
    assert f"{flow['corr_contemp']:.2f}".lstrip("-") in doc                 # 0.25
    assert f"{flow['next_move_by_official_quintile_bps'][-1]:.1f}" in doc   # -13.6 bottom quintile


# --------------------------------------------------------------------------- #
# B. The unified-ranking table in FINDINGS.md, reconciled row-by-row (by rank).
# --------------------------------------------------------------------------- #
def _findings_ranking_rows(findings_text):
    section = findings_text.split("## 2.")[1].split("## 3.")[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and re.fullmatch(r"\d+", cells[0]):   # a numbered data row
            rows.append(cells)
    return rows


def test_findings_ranking_table_reconciles_with_json(findings_text, ranking):
    doc_rows = _findings_ranking_rows(findings_text)
    json_rows = sorted(ranking["ranking"], key=lambda r: r["rank_is"])
    assert len(doc_rows) == len(json_rows) == 16, f"expected 16 rows, got {len(doc_rows)}"
    for cells, r in zip(doc_rows, json_rows):
        assert int(cells[0]) == r["rank_is"]
        assert _num(cells[3]) == round(r["is"]["sharpe"], 2), f"IS Sharpe row {cells[0]}"
        assert _num(cells[4]) == round(r["oos"]["sharpe"], 2), f"OOS Sharpe row {cells[0]}"
        assert _num(cells[5]) == r["is"]["per_year_usd"], f"IS $/yr row {cells[0]}"
        assert _num(cells[6]) == r["is"]["maxdd_usd"], f"IS max DD row {cells[0]}"


# --------------------------------------------------------------------------- #
# C. The rendered HTML report carries the same source-of-truth numbers.
# --------------------------------------------------------------------------- #
def test_report_html_headline_numbers(report_html, ranking_by_label, quincena):
    # The report renders raw JSON floats (e.g. 3.7), so match the value as rendered in a
    # table cell (>3.7<) rather than a re-formatted string.
    top = ranking_by_label["Calendar + trail 30 + floor 40 (exit_lab)"]
    for val in (top["is"]["sharpe"], top["oos"]["sharpe"],
                quincena["Calendar + slow-vol (recommended)"]["sharpe"]):
        assert f">{val}<" in report_html, f"{val} not rendered in report.html"


def test_report_html_is_single_basis(report_html):
    # the old dual-basis warning must be gone; the single-basis statement must be present
    assert "Two accounting bases" not in report_html
    assert "One accounting basis" in report_html
