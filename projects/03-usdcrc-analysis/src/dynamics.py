"""Underlying-dynamics deep-dive + transparent rule definitions in DOLLARS.

Explains WHAT the flow/seasonality model exploits, and backtests three fully
transparent long/short rules (no ML) sized at USD 1,000,000 per position, net of
0.65 CRC round-trip slippage. Writes dyn_*.png and out/dynamics_results.json.

P&L convention (base = USD, USD 1M notional per full position):
  * 1 bp of next-day USD move  ->  USD 100 on a USD 1,000,000 position.
  * slippage 0.325 CRC/side    ->  USD 1M * 0.325 / spot per side (~USD 680/side).
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

plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
NAVY, GREEN, RED, GREY, ORANGE, PURPLE = "#1f3b73", "#2e8b57", "#c0392b", "#7f8c8d", "#e08a2b", "#7d3c98"
NOTIONAL = 1_000_000       # USD per full position
COST_SIDE_CRC = 0.325      # 0.65 round-trip
ANN = np.sqrt(252)
USD_PER_BP = NOTIONAL / 1e4   # = 100


def frame():
    df = load().copy()
    df["vol"] = df.volume_usd / 1e6
    df["r_now"] = (df.vwap - df.prev_vwap) / df.prev_vwap * 1e4   # same-day USD move
    df["r_next"] = df.close.pct_change().shift(-1) * 1e4          # tradeable next-day move
    df["dom"] = df.date.dt.day
    df["dow"] = df.date.dt.dayofweek
    df["year"] = df.date.dt.year
    df["skew"] = df.vwap - df.avg_simple                         # precio ponderado vs simple
    # causal seasonally-adjusted volume: today vs trailing mean of same weekday
    df["vol_dow_ref"] = df.groupby("dow").vol.transform(
        lambda s: s.shift(1).expanding(min_periods=10).mean())
    df["vol_adj"] = df.vol / df.vol_dow_ref                      # <1 => below-normal (low) volume
    return df


# --------------------------------------------------------------------------- #
# transparent rules (each returns positions in {-1,0,+1} = short/flat/long USD)
# --------------------------------------------------------------------------- #
def rule_volume(df):
    """LOW seasonally-adjusted volume -> long USD; HIGH -> short USD."""
    return np.where(df.vol_adj < 1.0, 1.0, -1.0)


def rule_quincena(df):
    """First half of month -> short USD (colón tends to strengthen); 2nd half -> long."""
    return np.where(df.dom <= 15, -1.0, 1.0)


def rule_skew(df):
    """VWAP above simple average (big tickets buying USD) -> long USD next day."""
    return np.where(df["skew"] > 0, 1.0, -1.0)


def rule_combined(df):
    """Majority vote of the three transparent signals (always ±1: three odd votes)."""
    return np.sign(rule_volume(df) + rule_quincena(df) + rule_skew(df))


def dollars(pos, df):
    pos = pd.Series(np.asarray(pos, float)).reset_index(drop=True)
    r = df.r_next.reset_index(drop=True) / 1e4
    S = df.vwap.reset_index(drop=True)
    turn = pos.diff().abs()
    turn.iloc[0] = abs(pos.iloc[0])
    pnl_usd = pos * NOTIONAL * r                              # USD P&L on 1M notional
    cost_usd = turn * NOTIONAL * COST_SIDE_CRC / S            # slippage in USD
    net = (pnl_usd - cost_usd).dropna()
    return net, turn


def rule_stats(pos, df):
    net, turn = dollars(pos, df)
    cum = net.cumsum()
    years = len(net) / 252
    trades = float(np.nansum(turn) / 2)
    bps = net / USD_PER_BP
    return {
        "total_usd": round(float(net.sum())),
        "per_year_usd": round(float(net.sum() / years)),
        "per_trade_usd": round(float(net.sum() / trades)) if trades else 0,
        "trades_total": int(round(trades)),
        "roundtrips_yr": round(float(trades / years), 1),
        "win_rate": round(float((net > 0).mean() * 100), 1),
        "sharpe": round(float(bps.mean() / bps.std() * ANN), 2) if bps.std() else 0.0,
        "maxdd_usd": round(float((cum - cum.cummax()).min())),
    }


# --------------------------------------------------------------------------- #
# dynamics charts
# --------------------------------------------------------------------------- #
def chart_mechanism(df, res):
    up = df[df.r_now > 0].vol.mean()
    dn = df[df.r_now < 0].vol.mean()
    res["vol_on_usd_up"] = round(float(up), 1)
    res["vol_on_usd_down"] = round(float(dn), 1)
    lo = df[df.vol_adj < df.vol_adj.quantile(.2)].r_next.mean()
    hi = df[df.vol_adj > df.vol_adj.quantile(.8)].r_next.mean()
    res["nextmove_after_lowvol"] = round(float(lo), 1)
    res["nextmove_after_highvol"] = round(float(hi), 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(["colón strengthens\n(USD down)", "colón weakens\n(USD up)"], [dn, up],
              color=[GREEN, RED])
    ax[0].set_ylabel("avg daily volume (US$M)")
    ax[0].set_title(f"USD sellers trade in SIZE\n{dn:.0f}M when colón up vs {up:.0f}M when colón down")
    ax[1].bar(["after LOW\nvolume day", "after HIGH\nvolume day"], [lo, hi],
              color=[GREEN if lo > 0 else RED, GREEN if hi > 0 else RED])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("next-day USD move (bps)")
    ax[1].set_title("Thin market → USD rises next day")
    fig.suptitle("The engine: exporters/inflows are the swing USD supply", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "dyn_mechanism.png", dpi=110)
    plt.close(fig)


def chart_domcycle(df, res):
    g = df.groupby("dom").agg(r_next=("r_next", "mean"), vol=("vol", "mean"))
    g = g.reindex(range(1, 32)).dropna()
    fig, ax1 = plt.subplots(figsize=(12, 4.4))
    col = [GREEN if v > 0 else RED for v in g.r_next]
    ax1.bar(g.index, g.r_next, color=col, alpha=0.85)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_ylabel("avg next-day USD move (bps)")
    ax1.set_xlabel("day of month")
    ax2 = ax1.twinx()
    ax2.plot(g.index, g.vol, color=NAVY, lw=2, marker="o", ms=3)
    ax2.set_ylabel("avg volume (US$M)", color=NAVY)
    ax2.grid(False)
    ax1.set_title("Intra-month cash-flow cycle: mid-month USD-supply surge (high vol, colón up) "
                  "then reversal")
    fig.tight_layout()
    fig.savefig(OUT / "dyn_domcycle.png", dpi=110)
    plt.close(fig)


def chart_skew(df, res):
    d = df.dropna(subset=["r_next", "skew"])
    res["skew_corr"] = round(float(d["skew"].corr(d.r_next)), 3)
    q = pd.qcut(d["skew"], 5, labels=False)
    m = d.groupby(q, observed=True).r_next.mean()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(5), m.values, color=[GREEN if v > 0 else RED for v in m.values])
    ax.set_xticks(range(5))
    ax.set_xticklabels(["VWAP «\nsimple", "", "≈", "", "VWAP »\nsimple"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("next-day USD move (bps)")
    ax.set_title(f"Precio-ponderado skew (VWAP − simple avg) predicts next day (corr {res['skew_corr']})")
    fig.tight_layout()
    fig.savefig(OUT / "dyn_skew.png", dpi=110)
    plt.close(fig)


def chart_stability(df, res):
    rows = []
    for y, g in df.groupby("year"):
        if len(g) < 50:
            continue
        c = g.vol.corr(g.r_next)
        q = g[g.dom > 15].r_next.mean() - g[g.dom <= 15].r_next.mean()
        rows.append((y, c, q))
    st = pd.DataFrame(rows, columns=["year", "vcorr", "qspread"])
    res["stability"] = {int(r.year): {"vol_corr": round(float(r.vcorr), 2),
                                      "quincena_spread": round(float(r.qspread), 1)}
                        for r in st.itertuples()}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(st.year.astype(str), st.vcorr, color=[RED if v < -0.1 else GREY for v in st.vcorr])
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("Volume → next-day move correlation by year\n(negative = low-vol→USD-up works)")
    ax[0].tick_params(axis="x", rotation=45)
    ax[1].bar(st.year.astype(str), st.qspread,
              color=[GREEN if v > 0 else RED for v in st.qspread])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("Quincena spread (2nd − 1st half, bps) by year\npositive EVERY year")
    ax[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "dyn_stability.png", dpi=110)
    plt.close(fig)


def chart_dollar_equity(df, rules, res):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for (name, pos), c in zip(rules, [GREEN, ORANGE, PURPLE, NAVY]):
        net, _ = dollars(pos, df)
        dates = df.date.iloc[net.index]
        ax.plot(dates, net.cumsum() / 1e6, color=c, lw=1.5,
                label=f"{name}: ${res[name]['total_usd']/1e6:.2f}M total, "
                      f"${res[name]['per_year_usd']/1e3:.0f}k/yr, Sh {res[name]['sharpe']}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Cumulative net P&L at USD 1M per trade (net of 0.65 CRC round-trip)")
    ax.set_ylabel("cumulative P&L (US$ millions)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "dyn_dollar_equity.png", dpi=110)
    plt.close(fig)


def main():
    df = frame()
    res = {"_meta": {"n_days": int(len(df)), "notional_usd": NOTIONAL,
                     "cost_crc_roundtrip": 0.65, "usd_per_bp": USD_PER_BP,
                     "years": round(len(df) / 252, 1)}}
    chart_mechanism(df, res)
    chart_domcycle(df, res)
    chart_skew(df, res)
    chart_stability(df, res)

    rules = [("Volume rule", rule_volume(df)),
             ("Quincena rule", rule_quincena(df)),
             ("Precio-ponderado skew rule", rule_skew(df)),
             ("Combined vote (3 signals)", rule_combined(df))]
    for name, pos in rules:
        res[name] = rule_stats(pos, df)
    chart_dollar_equity(df, rules, res)

    Path(OUT / "dynamics_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("\nWrote dyn_*.png and dynamics_results.json")


if __name__ == "__main__":
    main()
