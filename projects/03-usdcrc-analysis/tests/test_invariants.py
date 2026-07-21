"""Structural invariants that must hold for the accounting to be trustworthy.

These are the checks that catch a silently-broken P&L engine even when every number
"looks plausible": the long/short mirror, the ranking ordering, window arithmetic, and
the held-out non-decay of the recommended overlay.
"""
from __future__ import annotations


def test_ranking_is_ordered_by_in_sample_sharpe(ranking):
    rows = ranking["ranking"]
    is_sharpes = [r["is"]["sharpe"] for r in rows]
    assert is_sharpes == sorted(is_sharpes, reverse=True), "ranking not sorted by IS Sharpe"
    assert [r["rank_is"] for r in rows] == list(range(1, len(rows) + 1))


def test_always_long_and_short_are_mirrors(ranking_by_label):
    """Two opposite always-on positions must mirror to within slippage.

    They are not bit-exact (cost is charged on the single entry either way and the daily
    return sign flips), but their in- and out-of-sample Sharpes must sum to ~0.
    """
    lo = ranking_by_label["Always long USD"]
    sh = ranking_by_label["Always short USD"]
    assert abs(lo["is"]["sharpe"] + sh["is"]["sharpe"]) < 0.05
    assert abs(lo["oos"]["sharpe"] + sh["oos"]["sharpe"]) < 0.05


def test_windows_partition_the_sample(ranking_by_label):
    # For any always-in-market rule, IS n_days + OOS n_days == full n_days.
    for label in ["Calendar + slow-vol", "Always long USD"]:
        r = ranking_by_label[label]
        assert r["is"]["n_days"] + r["oos"]["n_days"] == r["full"]["n_days"]


def test_cost_reduces_activity_pnl(ranking_by_label):
    # A high-turnover flow rule must show a worse (more negative or smaller) IS result than
    # the low-turnover calendar rule at similar gross edge — sanity on cost accounting.
    flow = ranking_by_label["VWAP-skew rule"]
    cal = ranking_by_label["Calendar + slow-vol"]
    assert flow["is"]["roundtrips_yr"] > 5 * cal["is"]["roundtrips_yr"]
    assert flow["is"]["maxdd_usd"] < cal["is"]["maxdd_usd"]  # deeper drawdown (more negative)


def test_recommended_overlay_does_not_decay_out_of_sample(exit_lab):
    base = exit_lab["baseline"]
    rec = exit_lab["variants"][exit_lab["recommended"]]
    # held-out confirmation: OOS Sharpe >= IS Sharpe for the recommended overlay
    assert rec["oos"]["sharpe"] >= rec["is"]["sharpe"]
    # and the overlay improves risk-adjusted return over the no-exit baseline in both windows
    assert rec["is"]["sharpe"] > base["is"]["sharpe"]
    assert rec["oos"]["sharpe"] > base["oos"]["sharpe"]
    # it is a RISK improvement: it cuts drawdown (less negative) out of sample
    assert rec["oos"]["maxdd_usd"] > base["oos"]["maxdd_usd"]


def test_take_profit_hurts_out_of_sample(exit_lab):
    # The stated negative result: tightening the take-profit degrades OOS monotonically.
    sweep = {r["target_bps"]: r for r in exit_lab["take_profit_sweep"]}
    none = sweep[None]["oos_sharpe"]
    assert sweep[160]["oos_sharpe"] < none
    assert sweep[80]["oos_sharpe"] < sweep[160]["oos_sharpe"]


def test_reserve_trim_beats_blind_control(intervention):
    rvb = intervention["reserve_vs_blind"]
    assert rvb["oos_total_delta_usd"] > 0          # reserve-selective earns more OOS
    assert rvb["paired_p"] <= 0.05                  # and the difference is significant
    res = intervention["strategy"]["+ reserve-regime long-trim"]
    base = intervention["strategy"]["Calendar + trail30/floor40"]
    assert res["oos"]["sharpe"] > base["oos"]["sharpe"]      # higher OOS Sharpe
    assert res["full"]["maxdd_usd"] > base["full"]["maxdd_usd"]  # shallower drawdown
