"""VWAP-priced backtest — the realistic-execution twin of ``backtest.py``.

``backtest.py`` marks every strategy to the *closing* print and books the same
0.65 CRC round-trip slippage against the close. That flatters the result: on
MONEX you cannot be sure of a fill at the last print of the session, but a real
execution desk *can* work an order to the day's **volume-weighted average price
(VWAP)** — the ``vwap`` column here. This script re-prices the identical
strategies and the identical slippage on a **VWAP-to-VWAP** basis, which is the
more realistic assumption the desk actually lives with.

What changes vs ``backtest.py`` (and *only* this):
  * Tradeable return is VWAP-to-VWAP (``vwap.pct_change``) instead of
    close-to-close. Position decided from information through session *t* still
    earns the *next* session's move, exactly as before — no look-ahead added.
  * The 0.65 CRC round-trip slippage (0.325/side) is unchanged; it is divided by
    the VWAP rather than the close to convert colones→bps.

The signals themselves are byte-for-byte the same functions imported from
``backtest.py`` — this isolates the execution-price effect so the close-vs-VWAP
comparison is apples-to-apples.

Writes bt_vwap_*.png and out/backtest_vwap_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import OUT, load
from backtest import (
    COST_CRC,
    COST_CRC_RT,
    GREEN,
    GREY,
    NAVY,
    OOS_FRAC,
    ORANGE,
    RED,
    block,
    sharpe,
    sig_donchian,
    sig_orderflow,
    sig_sma,
)


# --------------------------------------------------------------------------- #
def frame():
    """Same feature frame as backtest.frame(), but the tradeable return is VWAP-
    to-VWAP rather than close-to-close."""
    df = load().copy()
    df["ret_vw"] = df.vwap.pct_change() * 1e4              # tradeable VWAP-to-VWAP USD return (bps)
    df["r_next"] = df.ret_vw.shift(-1)                     # earned next session (at VWAP)
    v = df.volume_musd
    df["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    df["year"] = df.date.dt.year
    return df


def bt(pos, df, cost_crc=COST_CRC):
    """Net P&L (bps): position earns the next session's VWAP-to-VWAP move; the
    0.65 CRC round-trip slippage is charged on |position change|, converted to
    bps against that day's **VWAP** (the fill reference), not the close."""
    pos = pd.Series(np.asarray(pos, float)).reset_index(drop=True)
    r = df.r_next.reset_index(drop=True)
    px = df.vwap.reset_index(drop=True)
    turn = pos.diff().abs()
    turn.iloc[0] = abs(pos.iloc[0])
    cost_bps = cost_crc / px * 1e4
    net = pos * r - cost_bps * turn
    return net, turn


# --------------------------------------------------------------------------- #
def chart_net_compare(df, res, strats):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for name, pos, color in strats:
        net, turn = bt(pos, df)
        res[name] = block(net, turn, len(df))
        res[name]["gross_sharpe"] = round(float(sharpe(pd.Series(pos).reset_index(drop=True)
                                                        * df.r_next.reset_index(drop=True))), 2)
        ax.plot(df.date, np.nancumsum(net), color=color, lw=1.5,
                label=f"{name} (net Sh {res[name]['sharpe']}, {res[name]['roundtrips_yr']} rt/yr)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("VWAP-priced net cumulative P&L (bps) — 0.65 CRC round-trip, 2015-2026")
    ax.set_ylabel("cum P&L (bps)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "bt_vwap_net_compare.png", dpi=110)
    plt.close(fig)


def chart_slippage(df, res, strats):
    rt_costs = np.linspace(0, 2.4, 25)            # round-trip CRC on the x-axis
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for name, pos, color in strats:
        srs = [sharpe(bt(pos, df, c / 2)[0]) for c in rt_costs]   # per-side = rt/2
        ax.plot(rt_costs, srs, color=color, lw=1.8, label=name)
    ax.axvline(COST_CRC_RT, color=RED, ls="--", lw=1)
    ax.text(COST_CRC_RT + 0.03, ax.get_ylim()[1] * 0.85, "your 0.65 CRC RT", color=RED, fontsize=9)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_title("VWAP-priced net Sharpe vs round-trip slippage (CRC)")
    ax.set_xlabel("slippage per round trip (CRC)")
    ax.set_ylabel("net Sharpe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "bt_vwap_slippage.png", dpi=110)
    plt.close(fig)


def chart_tearsheet(df, pos, res, name="Donchian-40"):
    net, turn = bt(pos, df)
    s = pd.Series(net.values, index=df.date)
    cum = s.cumsum()
    dd = cum - cum.cummax()
    yearly = s.groupby(df.date.dt.year.values).sum()
    res[name + "_full"] = block(net, turn, len(df))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    b = res[name + "_full"]
    fig.suptitle(f"{name} long/short USD — VWAP-priced, NET of 0.65 CRC slippage  "
                 f"(Sharpe {b['sharpe']} · {b['hit']}% hit · {b['roundtrips_yr']} rt/yr · "
                 f"{b['ann_bps']:.0f} bps/yr)", fontsize=12, fontweight="bold")
    ax[0, 0].plot(cum.index, cum.values, color=GREEN, lw=1.5)
    ax[0, 0].set_title("Cumulative net P&L (bps)")
    ax[0, 1].fill_between(dd.index, dd.values, 0, color=RED, alpha=0.5)
    ax[0, 1].set_title(f"Drawdown (bps) — max {dd.min():.0f}")
    ax[1, 0].bar(yearly.index.astype(int).astype(str), yearly.values,
                 color=[GREEN if v >= 0 else RED for v in yearly.values])
    ax[1, 0].set_title("Net P&L by year (bps)")
    ax[1, 0].tick_params(axis="x", rotation=90, labelsize=8)
    ax[1, 1].hist(net.dropna().values, bins=40, color=GREEN, alpha=0.8)
    ax[1, 1].axvline(0, color="k", lw=0.8)
    ax[1, 1].set_title("Daily net P&L distribution (bps)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "bt_vwap_trend_tearsheet.png", dpi=110)
    plt.close(fig)


def chart_oos(df, pos, res, name="Donchian-40"):
    k = int(len(df) * OOS_FRAC)
    net, turn = bt(pos, df)
    res[name + "_is"] = block(net.iloc[:k])
    res[name + "_oos"] = block(net.iloc[k:])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.date, net.cumsum(), color=GREEN, lw=1.5)
    ax.axvspan(df.date.iloc[0], df.date.iloc[k], color=GREY, alpha=0.10)
    ax.axvline(df.date.iloc[k], color=RED, ls="--", lw=1)
    ax.set_title(f"{name} (VWAP-priced): in-sample Sharpe {res[name+'_is']['sharpe']} "
                 f"-> out-of-sample {res[name+'_oos']['sharpe']} (net)")
    ax.set_ylabel("cum net P&L (bps)")
    fig.tight_layout()
    fig.savefig(OUT / "bt_vwap_oos.png", dpi=110)
    plt.close(fig)


def chart_close_vs_vwap(df, res, strats):
    """Head-to-head: same strategy, same slippage, priced at the close (as in
    backtest.py) vs at the VWAP (this file). The gap is the realism haircut."""
    # rebuild the close-basis return locally so we don't mutate the vwap frame
    dfx = df.copy()
    dfx["r_next_close"] = (dfx.close.pct_change() * 1e4).shift(-1)

    def bt_close(pos):
        pos = pd.Series(np.asarray(pos, float)).reset_index(drop=True)
        r = dfx.r_next_close.reset_index(drop=True)
        px = dfx.close.reset_index(drop=True)
        turn = pos.diff().abs()
        turn.iloc[0] = abs(pos.iloc[0])
        return pos * r - (COST_CRC / px * 1e4) * turn

    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for name, pos, color in strats:
        net_c = bt_close(pos)
        net_v, _ = bt(pos, df)
        sc, sv = sharpe(net_c), sharpe(net_v)
        res.setdefault("close_vs_vwap", {})[name] = {
            "close_sharpe": round(float(sc), 2),
            "vwap_sharpe": round(float(sv), 2),
            "close_ann_bps": round(float(np.nanmean(net_c) * 252), 0),
            "vwap_ann_bps": round(float(np.nanmean(net_v) * 252), 0),
            "sharpe_haircut": round(float(sc - sv), 2),
        }
        rows.append(name)
        ax[0].plot(df.date, np.nancumsum(net_c), color=color, lw=1.4,
                   label=f"{name} (Sh {sc:.2f})")
        ax[1].plot(df.date, np.nancumsum(net_v), color=color, lw=1.4,
                   label=f"{name} (Sh {sv:.2f})")
    ax[0].set_title("Priced at CLOSE (backtest.py)")
    ax[1].set_title("Priced at VWAP (realistic)")
    for a in ax:
        a.axhline(0, color="k", lw=0.6)
        a.set_ylabel("cum net P&L (bps)")
        a.legend(loc="upper left", fontsize=8)
    fig.suptitle("Same strategies & 0.65 CRC slippage — close-print vs VWAP execution",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "bt_vwap_close_compare.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    df = frame()
    res = {"_meta": {
        "basis": "VWAP-to-VWAP (realistic execution)",
        "n_days": int(len(df)), "date_min": str(df.date.min().date()),
        "date_max": str(df.date.max().date()),
        "cost_crc_roundtrip": COST_CRC_RT, "cost_crc_per_side": COST_CRC,
        "cost_bps_rt_at_475": round(COST_CRC_RT / 475 * 1e4, 1),
        "cost_bps_side_at_475": round(COST_CRC / 475 * 1e4, 1),
        "oos_frac": OOS_FRAC,
    }}
    of = sig_orderflow(df)
    don = sig_donchian(df, 40)
    sma = sig_sma(df, 120)
    strats = [("Order-flow daily", of, ORANGE),
              ("Donchian-40 trend", don, GREEN),
              ("SMA-120 trend", sma, NAVY)]

    chart_net_compare(df, res, strats)
    chart_slippage(df, res, strats)
    chart_close_vs_vwap(df, res, strats)

    recent = (df.date >= "2019-01-01").values
    for name, pos, _ in strats:
        net, _t = bt(pos, df)
        res[name]["recent_2019"] = block(net[recent])

    chart_tearsheet(df, don, res, "Donchian-40")
    chart_oos(df, don, res, "Donchian-40")

    Path(OUT / "backtest_vwap_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("\nWrote bt_vwap_*.png and backtest_vwap_results.json")


if __name__ == "__main__":
    main()
