"""Rank EVERY strategy in this project on ONE common basis, selected in-sample.

Why this module exists
----------------------
The strategies in this repo were developed in separate modules that each grew
their own accounting convention, so their headline numbers are NOT comparable:

  * strategies.py   reports GROSS Sharpe (no slippage at all) and, for the
                    opening-gap rule, an INTRADAY (open->close) return stream
                    rather than the next-session stream everything else uses.
  * backtest.py     prices close-to-close, in bps.
  * backtest_vwap.py prices VWAP-to-VWAP, in bps.
  * quincena.py     prices VWAP-to-VWAP, in USD at 1M notional.
  * dynamics.py     prices VWAP-to-VWAP, in USD, with its own frame().

Ranking those side by side is meaningless. This module re-implements every
position rule against a SINGLE frame and a SINGLE accounting basis:

    return basis   VWAP-to-VWAP next session   (the fill a desk can work)
    notional       USD 1,000,000 per unit position
    cost           0.325 CRC per side (0.65 CRC round trip), charged on |dpos|
    annualisation  derived FROM THE DATA (see ANN_DAYS below), not hardcoded 252
    selection      first 60% of history in-sample; last 40% reported untouched

Annualisation
-------------
Every other module hardcodes 252 trading days/year. MONEX actually trades ~231
sessions/year (2,663 sessions over 11.53 calendar years), so the 252 convention
overstates per-year P&L by ~9% and Sharpe by ~4.5%. This module measures the
real session density and annualises with it. Both numbers are reported so the
legacy figures can still be reconciled.

Ranking is by IN-SAMPLE net Sharpe. The out-of-sample column is carried along
for honesty but never feeds the ordering.

Writes out/ranking.json and out/ranking.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import OUT
from basis import NOTIONAL_USD as NOTIONAL
from basis import COST_CRC_PER_SIDE as COST_SIDE_CRC
from basis import ann_days, SESSIONS_PER_YEAR_LEGACY
from exits import apply_exit
from quincena import (CAL_PRE, OOS_FRAC, SHORT_END, SHORT_START, frame,
                      pos_base, pos_calendar, pos_calendar_slowvol,
                      pos_refined, pos_refined_slowvol)
NAVY, GREEN, RED, GREY, PURPLE, ORANGE = (
    "#1f3b73", "#2e8b57", "#c0392b", "#7f8c8d", "#7d3c98", "#e08a2b")


# --------------------------------------------------------------------------- #
# accounting
# --------------------------------------------------------------------------- #
def pnl(pos, df):
    """Daily net P&L in USD for a position series, VWAP-to-VWAP, net of slippage.

    `pos` is in units of NOTIONAL (so -1 = short USD 1M, 0.5 = half-size long).
    Cost is charged on every change in position, including the initial entry,
    priced at that session's VWAP. The final row has no next-session return and
    is dropped, so the series is strictly realisable.
    """
    pos = pd.Series(np.asarray(pos, float)).reset_index(drop=True)
    turn = pos.diff().abs()
    turn.iloc[0] = abs(pos.iloc[0])
    gross = pos * NOTIONAL * df.r_next.reset_index(drop=True)
    cost = turn * NOTIONAL * COST_SIDE_CRC / df.vwap.reset_index(drop=True)
    return (gross - cost).dropna(), turn


def stats(pos, df, apy):
    """Summary of a rule on a window. `apy` = sessions/year used to annualise."""
    net, turn = pnl(pos, df)
    if len(net) < 5 or net.std() == 0:
        return None
    cum = net.cumsum()
    yrs = len(net) / apy
    bps = net / (NOTIONAL / 1e4)          # USD -> bps of notional
    exposed = np.abs(np.asarray(pos, float)) > 0
    return {
        "sharpe": round(float(bps.mean() / bps.std() * np.sqrt(apy)), 2),
        "per_year_usd": round(float(net.sum() / yrs)),
        "total_usd": round(float(net.sum())),
        "maxdd_usd": round(float((cum - cum.cummax()).min())),
        "roundtrips_yr": round(float(turn.sum() / 2 / yrs), 1),
        "hit_rate": round(float((net > 0).mean() * 100), 1),
        "time_in_mkt": round(float(exposed.mean() * 100), 1),
        "n_days": int(len(net)),
    }


# --------------------------------------------------------------------------- #
# the full strategy universe, all defined against quincena.frame()
# --------------------------------------------------------------------------- #
def pos_flat_long(df):
    """Benchmark: always long USD (carry the colon's drift against you)."""
    return np.ones(len(df))


def pos_flat_short(df):
    """Benchmark: always short USD (the colon's appreciation crawl)."""
    return -np.ones(len(df))


def pos_orderflow(df):
    """backtest.sig_orderflow: fade the volume z-score (heavy volume = USD supply).

    vol_z uses a 20-session rolling mean/std INCLUDING session t; the position is
    taken at the end of t and earns t->t+1, so it is causal.
    """
    v = df.volume_musd
    z = (v - v.rolling(20).mean()) / v.rolling(20).std()
    return -np.sign(z).fillna(0).values


def pos_donchian(df, n=40):
    """backtest.sig_donchian: breakout vs the prior-n-session high/low."""
    hi = df.close.rolling(n).max().shift(1)
    lo = df.close.rolling(n).min().shift(1)
    p = pd.Series(np.nan, index=df.index)
    p[df.close >= hi] = 1.0
    p[df.close <= lo] = -1.0
    return p.ffill().fillna(0).values


def pos_sma(df, n=120):
    """backtest.sig_sma: long USD above the n-session VWAP average."""
    return np.sign(df.vwap - df.vwap.rolling(n).mean()).fillna(0).values


def pos_volume_rule(df):
    """dynamics.rule_volume: thin market (seasonally-adjusted vol < 1) => long USD."""
    return np.where(df.vol_adj < 1.0, 1.0, -1.0)


def pos_skew_rule(df):
    """dynamics.rule_skew: VWAP above the simple average => long USD."""
    return np.where(df.vwap > df.avg_simple, 1.0, -1.0)


def pos_combined_vote(df):
    """dynamics.rule_combined: majority vote of volume, quincena and skew."""
    v = np.sign(pos_volume_rule(df) + pos_base(df) + pos_skew_rule(df))
    return np.where(v == 0, 1.0, v)


# Exit overlays are built at run time (they need the tuned bands), see build().
STATIC = [
    # (label, family, fn, colour)
    ("Always long USD", "benchmark", pos_flat_long, GREY),
    ("Always short USD", "benchmark", pos_flat_short, GREY),
    ("Order-flow daily", "flow", pos_orderflow, ORANGE),
    ("Volume rule", "flow", pos_volume_rule, ORANGE),
    ("VWAP-skew rule", "flow", pos_skew_rule, ORANGE),
    ("Combined vote", "flow", pos_combined_vote, ORANGE),
    ("Donchian-40 trend", "trend", pos_donchian, NAVY),
    ("SMA-120 trend", "trend", pos_sma, NAVY),
    ("Base quincena (<=15)", "calendar", pos_base, PURPLE),
    (f"Refined ({SHORT_START}-{SHORT_END})", "calendar", pos_refined, PURPLE),
    ("Refined + slow-vol", "calendar", pos_refined_slowvol, PURPLE),
    (f"Real calendar (<={CAL_PRE}d)", "calendar", pos_calendar, GREEN),
    ("Calendar + slow-vol", "calendar", pos_calendar_slowvol, GREEN),
]


# --------------------------------------------------------------------------- #
def build(df, k, apy):
    """Every strategy, scored on full / in-sample / out-of-sample windows."""
    dis, dos = df.iloc[:k], df.iloc[k:]
    rows = []

    def add(label, family, pos_full, colour):
        pos_full = np.asarray(pos_full, float)
        f = stats(pos_full, df, apy)
        i = stats(pos_full[:k], dis, apy)
        o = stats(pos_full[k:], dos, apy)
        if f is None or i is None or o is None:
            return
        rows.append({"label": label, "family": family, "colour": colour,
                     "full": f, "is": i, "oos": o})

    for label, family, fn, colour in STATIC:
        add(label, family, fn(df), colour)

    # --- exit overlays on the recommended calendar entry ---------------------
    # Bands are tuned on the IN-SAMPLE window only (see exits.optimise); here we
    # simply re-price the already-published choices on the common basis.
    base = pos_calendar(df)
    ex = json.loads((OUT / "exits_results.json").read_text())
    tr = ex["_meta"]["trail_bps"]
    ctr, cfl = ex["_meta"]["combo_trail_bps"], ex["_meta"]["combo_floor_bps"]
    add(f"Calendar + trail {tr}bps", "exit", apply_exit(base, df.vwap, trail=tr), RED)
    add(f"Calendar + trail {ctr} + floor {cfl}bps", "exit",
        apply_exit(base, df.vwap, trail=ctr, floor=cfl), RED)

    # The exit_lab recommendation (drawdown-aware 1-SE selection, no take-profit).
    # Optional so the ranking still runs before exit_lab has been executed once.
    lab = OUT / "exit_lab.json"
    if lab.exists():
        p = json.loads(lab.read_text())["stage2_joint"]["bps"]["parsimonious"]
        add(f"Calendar + trail {p['trail']} + floor {p['floor']} (exit_lab)", "exit",
            apply_exit(base, df.vwap, trail=p["trail"], floor=p["floor"]), RED)

    rows.sort(key=lambda r: r["is"]["sharpe"], reverse=True)
    for n, r in enumerate(rows, 1):
        r["rank_is"] = n
    return rows


def chart(rows, apy, res):
    """Ranked in-sample Sharpe with the out-of-sample value beside each bar."""
    rows = list(reversed(rows))                     # best at top
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 2.4))
    ax.barh(y - 0.19, [r["is"]["sharpe"] for r in rows], 0.38,
            color=[r["colour"] for r in rows], label="in-sample (selection)")
    ax.barh(y + 0.19, [r["oos"]["sharpe"] for r in rows], 0.38,
            color=[r["colour"] for r in rows], alpha=0.42,
            label="out-of-sample (confirmation)")
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"net Sharpe  ($1M/trade · VWAP-priced · 0.65 CRC RT · annualised on {apy:.0f} sessions/yr)")
    ax.set_title("Every strategy on one accounting basis — ranked by IN-SAMPLE Sharpe\n"
                 "(faded bar = held-out confirmation; it never feeds the ranking)",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    for i, r in enumerate(rows):
        ax.text(max(r["is"]["sharpe"], r["oos"]["sharpe"]) + 0.06, i,
                f"${r['is']['per_year_usd']/1e3:,.0f}k/yr", va="center", fontsize=8,
                color="#4a5a6a")
    fig.tight_layout()
    fig.savefig(OUT / "ranking.png", dpi=110)
    plt.close(fig)


def main():
    df = frame()
    apy = ann_days(df)
    k = int(len(df) * OOS_FRAC)
    rows = build(df, k, apy)

    res = {
        "_meta": {
            "n_days": int(len(df)),
            "date_min": str(df.date.min().date()),
            "date_max": str(df.date.max().date()),
            "sessions_per_year_actual": round(float(apy), 1),
            "sessions_per_year_legacy": SESSIONS_PER_YEAR_LEGACY,
            "per_year_usd_legacy_inflation": round(SESSIONS_PER_YEAR_LEGACY / apy - 1, 4),
            "sharpe_legacy_inflation": round(float(np.sqrt(SESSIONS_PER_YEAR_LEGACY / apy) - 1), 4),
            "notional_usd": NOTIONAL,
            "cost_crc_per_side": COST_SIDE_CRC,
            "basis": "VWAP-to-VWAP next session, net of slippage",
            "oos_frac": OOS_FRAC,
            "split_date": str(df.date.iloc[k].date()),
            "is_window": [str(df.date.iloc[0].date()), str(df.date.iloc[k - 1].date())],
            "oos_window": [str(df.date.iloc[k].date()), str(df.date.iloc[-1].date())],
            "ranked_by": "in-sample net Sharpe",
        },
        "ranking": [{key: r[key] for key in ("rank_is", "label", "family", "is", "oos", "full")}
                    for r in rows],
    }
    chart(rows, apy, res)
    Path(OUT / "ranking.json").write_text(json.dumps(res, indent=2))

    m = res["_meta"]
    print(f"USD/CRC strategy ranking — {m['n_days']} sessions, {m['date_min']} -> {m['date_max']}")
    print(f"Annualised on {m['sessions_per_year_actual']} sessions/yr "
          f"(legacy 252 inflates P&L {m['per_year_usd_legacy_inflation']:+.1%}, "
          f"Sharpe {m['sharpe_legacy_inflation']:+.1%})")
    print(f"In-sample {m['is_window'][0]} -> {m['is_window'][1]}   "
          f"| out-of-sample {m['oos_window'][0]} -> {m['oos_window'][1]}\n")
    hdr = (f"{'#':>2}  {'strategy':<36} {'family':<9} "
           f"{'IS Sh':>6} {'OOS Sh':>7} {'IS $/yr':>10} {'OOS $/yr':>10} "
           f"{'IS DD':>9} {'rt/yr':>6} {'%mkt':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['rank_is']:>2}  {r['label']:<36} {r['family']:<9} "
              f"{r['is']['sharpe']:>6.2f} {r['oos']['sharpe']:>7.2f} "
              f"{r['is']['per_year_usd']:>10,} {r['oos']['per_year_usd']:>10,} "
              f"{r['is']['maxdd_usd']:>9,} {r['is']['roundtrips_yr']:>6.1f} "
              f"{r['is']['time_in_mkt']:>5.0f}")
    print("\nWrote out/ranking.json and out/ranking.png")


if __name__ == "__main__":
    main()
