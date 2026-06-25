"""Trading-strategy visualizations (tearsheets) for the USD/CRC analysis.

Reuses the engineered daily table from analyze.load() and builds:
  - s1_orderflow_tearsheet.png : equity, drawdown, monthly P&L, P&L distribution
  - s2_momentum_tearsheet.png  : same, for the overnight-momentum signal
  - s_mechanics.png            : the order-flow signal mechanics (why it works)
  - s_comparison.png           : Sharpe / hit-rate / avg-bps across all signals

All P&L in basis points, gross of costs. A 'position' is held one session and
P&L is measured VWAP-to-VWAP.
"""
from __future__ import annotations


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import OUT, load

plt.rcParams.update({
    "figure.facecolor": "white", "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
})
GREEN, NAVY, RED, GREY = "#2e8b57", "#1f3b73", "#c0392b", "#7f8c8d"


def _sharpe(x):
    x = pd.Series(x).dropna()
    return np.nan if x.std() == 0 or len(x) < 5 else x.mean() / x.std() * np.sqrt(252)


def _maxdd(cum):
    return (cum - np.maximum.accumulate(cum)).min()


def build_signals(df):
    """Return dict of name -> daily P&L series (indexed by date) for each signal."""
    df = df.copy()
    df["ret_next"] = df.ret_vwap_bps.shift(-1)
    sig = {}

    # Order-flow continuation: short USD after heavy day, long after light.
    df["vol_z"] = (df.volume_musd - df.volume_musd.rolling(20).mean()) / \
                  df.volume_musd.rolling(20).std()
    of = df.dropna(subset=["vol_z", "ret_next"])
    sig["Order-flow continuation"] = pd.Series(
        (-np.sign(of.vol_z) * of.ret_next).values, index=of.date)

    # Overnight momentum on >1sigma session moves.
    z = (df.ret_vwap_bps - df.ret_vwap_bps.mean()) / df.ret_vwap_bps.std()
    mom = df[z.abs() >= 1.0].dropna(subset=["ret_next"])
    sig["Overnight momentum"] = pd.Series(
        (np.sign(mom.ret_vwap_bps) * mom.ret_next).values, index=mom.date)

    # Opening-gap fade (intraday) -- kept for comparison, flagged as inflated.
    gp = df.dropna(subset=["open_gap_bps", "intraday_bps"])
    gp = gp[gp.open_gap_bps.abs() >= 5]
    sig["Opening-gap fade"] = pd.Series(
        (-np.sign(gp.open_gap_bps) * gp.intraday_bps).values, index=gp.date)
    return sig, df


def tearsheet(name, pnl, fname, color):
    pnl = pnl.dropna().sort_index()
    cum = pnl.cumsum()
    dd = cum - np.maximum.accumulate(cum)
    monthly = pnl.groupby(pnl.index.to_period("M")).sum()

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"{name} — strategy tearsheet  "
                 f"(Sharpe {_sharpe(pnl):.2f} · {(pnl>0).mean()*100:.0f}% hit · "
                 f"{pnl.mean():+.1f} bps/day · n={len(pnl)})",
                 fontsize=13, fontweight="bold")

    ax[0, 0].plot(cum.index, cum.values, color=color, lw=1.6)
    ax[0, 0].fill_between(cum.index, cum.values, color=color, alpha=0.12)
    ax[0, 0].set_title("Cumulative P&L (bps)")

    ax[0, 1].fill_between(dd.index, dd.values, 0, color=RED, alpha=0.5)
    ax[0, 1].set_title(f"Drawdown (bps) — max {_maxdd(cum.values):.0f}")

    mcol = [GREEN if v >= 0 else RED for v in monthly.values]
    ax[1, 0].bar([str(p) for p in monthly.index], monthly.values, color=mcol)
    ax[1, 0].set_title("P&L by month (bps)")
    ax[1, 0].tick_params(axis="x", rotation=90, labelsize=7)

    ax[1, 1].hist(pnl.values, bins=30, color=color, alpha=0.8)
    ax[1, 1].axvline(pnl.mean(), color="k", lw=1.2, label=f"mean {pnl.mean():+.1f}")
    ax[1, 1].axvline(0, color=GREY, lw=0.8)
    ax[1, 1].set_title("Daily P&L distribution (bps)")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / fname, dpi=110)
    plt.close(fig)


def mechanics(df):
    """Why the order-flow signal works: volume z-score vs next-session move."""
    d = df.dropna(subset=["vol_z", "ret_next"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    ax[0].scatter(d.vol_z, d.ret_next, s=12, alpha=0.4, color=NAVY)
    b, a = np.polyfit(d.vol_z, d.ret_next, 1)
    xx = np.linspace(d.vol_z.min(), d.vol_z.max(), 50)
    ax[0].plot(xx, a + b * xx, color=RED, lw=1.6)
    ax[0].axhline(0, color=GREY, lw=0.6)
    ax[0].set_title(f"Volume z-score (today) vs NEXT-session move\nslope {b:+.1f} bps per z")
    ax[0].set_xlabel("volume z-score")
    ax[0].set_ylabel("next-session move (bps)")

    buckets = pd.cut(d.vol_z, [-9, -1, -0.33, 0.33, 1, 9],
                     labels=["very light", "light", "normal", "heavy", "very heavy"])
    g = d.groupby(buckets, observed=True).ret_next.mean()
    bcol = [RED if v > 0 else GREEN for v in g.values]
    ax[1].bar(range(len(g)), g.values, color=bcol)
    ax[1].set_xticks(range(len(g)))
    ax[1].set_xticklabels(g.index, rotation=20, fontsize=9)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("Avg next-session move by today's volume bucket")
    ax[1].set_ylabel("next-session move (bps)")
    fig.tight_layout()
    fig.savefig(OUT / "s_mechanics.png", dpi=110)
    plt.close(fig)


def comparison(sig):
    rows = []
    for name, pnl in sig.items():
        pnl = pnl.dropna()
        rows.append((name, _sharpe(pnl), (pnl > 0).mean() * 100, pnl.mean(), len(pnl)))
    comp = pd.DataFrame(rows, columns=["name", "sharpe", "hit", "avg", "n"])

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    cols = [GREEN, NAVY, GREY]
    ax[0].barh(comp.name, comp.sharpe, color=cols)
    ax[0].set_title("Sharpe (annualized)")
    ax[0].invert_yaxis()
    ax[1].barh(comp.name, comp.hit, color=cols)
    ax[1].axvline(50, color=RED, lw=1, ls="--")
    ax[1].set_title("Hit rate (%)")
    ax[1].invert_yaxis()
    ax[1].set_yticklabels([])
    ax[2].barh(comp.name, comp.avg, color=cols)
    ax[2].axvline(0, color="k", lw=0.8)
    ax[2].set_title("Avg P&L (bps/day)")
    ax[2].invert_yaxis()
    ax[2].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(OUT / "s_comparison.png", dpi=110)
    plt.close(fig)
    print(comp.round(2).to_string(index=False))


def main():
    df = load()
    sig, df = build_signals(df)
    tearsheet("Order-flow continuation", sig["Order-flow continuation"],
              "s1_orderflow_tearsheet.png", GREEN)
    tearsheet("Overnight momentum", sig["Overnight momentum"],
              "s2_momentum_tearsheet.png", NAVY)
    mechanics(df)
    comparison(sig)
    print("\nWrote strategy charts:")
    for f in ["s1_orderflow_tearsheet.png", "s2_momentum_tearsheet.png",
              "s_mechanics.png", "s_comparison.png"]:
        print("   out/" + f)


if __name__ == "__main__":
    main()
