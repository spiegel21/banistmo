"""Full-history (2014-2026), slippage-aware backtest of LONG/SHORT USD strategies.

Reality checks layered in after seeing the 18-month sample was a favourable window:
  * 11.5 years spanning multiple regimes (2017 & 2022-23 colon sell-offs, COVID,
    the 2024-26 appreciation).
  * Realistic execution cost: 0.65 CRC of slippage PER SIDE, applied on every
    change in position (~13-14 bps at a ~475 spot).
  * Strategies are long/short USD (position in {-1,0,+1}); both sides reported.

Key question: which edges survive the full history AND the real slippage?
Spoiler: the daily order-flow signal does not; a slow trend-following long/short does.

Writes bt_*.png and out/backtest_results.json (consumed by build_report).
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

plt.rcParams.update({
    "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
NAVY, GREEN, RED, GREY, ORANGE = "#1f3b73", "#2e8b57", "#c0392b", "#7f8c8d", "#e08a2b"
ANN = np.sqrt(252)
COST_CRC = 0.65          # slippage per side, in colones, against spot
OOS_FRAC = 0.60
N_TRIALS = 8
RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------- #
def frame():
    df = load().copy()
    df["ret_cc"] = df.close.pct_change() * 1e4            # tradeable close-to-close USD return (bps)
    df["r_next"] = df.ret_cc.shift(-1)                    # earned next session
    v = df.volume_musd
    df["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    df["year"] = df.date.dt.year
    return df


def bt(pos, df, cost_crc=COST_CRC):
    """Net P&L (bps) of a position series; cost on |position change| at that day's price."""
    pos = pd.Series(np.asarray(pos, float)).reset_index(drop=True)
    r = df.r_next.reset_index(drop=True)
    px = df.close.reset_index(drop=True)
    turn = pos.diff().abs()
    turn.iloc[0] = abs(pos.iloc[0])
    cost_bps = cost_crc / px * 1e4
    net = pos * r - cost_bps * turn
    return net, turn


def sharpe(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return np.nan if len(x) < 5 or x.std() == 0 else x.mean() / x.std() * ANN


def sharpe_lo(x, lag=10):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    sr = x.mean() / x.std()
    xc = x - x.mean()
    d = (xc * xc).sum()
    q = 252
    s = sum((q - k) * (xc[k:] * xc[:-k]).sum() / d for k in range(1, lag + 1))
    return float(sr * q / np.sqrt(q + 2 * s))


def block(net, turn=None, n=None):
    x = np.asarray(net, float)
    x = x[~np.isnan(x)]
    cum = np.cumsum(x)
    out = {
        "sharpe": round(float(sharpe(x)), 2),
        "sharpe_lo": round(float(sharpe_lo(x)), 2),
        "mean_bps": round(float(x.mean()), 2),
        "ann_bps": round(float(x.mean() * 252), 0),
        "hit": round(float((x[x != 0] > 0).mean() * 100), 1) if (x != 0).any() else 0.0,
        "total_bps": round(float(x.sum()), 0),
        "maxdd_bps": round(float((cum - np.maximum.accumulate(cum)).min()), 0),
        "n": int(len(x)),
    }
    if turn is not None and n is not None:
        out["roundtrips_yr"] = round(float(turn.sum() / 2 / (n / 252)), 1)
    return out


# --------------------------------------------------------------------------- #
# strategies (all long/short USD)
# --------------------------------------------------------------------------- #
def sig_orderflow(df):
    return -np.sign(df.vol_z).fillna(0).values


def sig_donchian(df, n=40):
    hi = df.close.rolling(n).max().shift(1)
    lo = df.close.rolling(n).min().shift(1)
    pos = pd.Series(np.nan, index=df.index)
    pos[df.close >= hi] = 1.0
    pos[df.close <= lo] = -1.0
    return pos.ffill().fillna(0).values


def sig_sma(df, n=120):
    return np.sign(df.vwap - df.vwap.rolling(n).mean()).fillna(0).values


# --------------------------------------------------------------------------- #
# charts / tests
# --------------------------------------------------------------------------- #
def chart_regime_corr(df, res):
    rows = []
    for y, g in df.groupby("year"):
        gg = g.dropna(subset=["r_next", "vol_z"])
        c = np.corrcoef(gg.vol_z, gg.r_next)[0, 1] if len(gg) > 20 else np.nan
        mv = (g.vwap.iloc[-1] / g.vwap.iloc[0] - 1) * 100 if len(g) > 5 else np.nan
        rows.append((y, c, mv))
    cc = pd.DataFrame(rows, columns=["year", "corr", "move"]).dropna()
    res["regime_corr"] = {int(r.year): round(float(r.corr), 2) for r in cc.itertuples()}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    col = [RED if c < -0.15 else GREY for c in cc["corr"]]
    ax[0].bar(cc.year.astype(int).astype(str), cc["corr"], color=col)
    ax[0].axhline(-0.15, color=GREY, ls="--", lw=0.8)
    ax[0].set_title("corr(volume z, NEXT-day move) by year\nsignal only exists from ~2018 on")
    ax[0].set_ylabel("correlation")
    mcol = [GREEN if m < 0 else RED for m in cc["move"]]
    ax[1].bar(cc.year.astype(int).astype(str), cc["move"], color=mcol)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("colón annual move (%) — green = appreciation, red = depreciation")
    ax[1].set_ylabel("%")
    fig.tight_layout()
    fig.savefig(OUT / "bt_regime_corr.png", dpi=110)
    plt.close(fig)


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
    ax.set_title("Net-of-slippage cumulative P&L (bps) — 0.65 CRC per side, 2015-2026")
    ax.set_ylabel("cum P&L (bps)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "bt_net_compare.png", dpi=110)
    plt.close(fig)


def chart_slippage(df, res, strats):
    costs = np.linspace(0, 1.2, 25)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for name, pos, color in strats:
        srs = [sharpe(bt(pos, df, c)[0]) for c in costs]
        ax.plot(costs, srs, color=color, lw=1.8, label=name)
    ax.axvline(COST_CRC, color=RED, ls="--", lw=1)
    ax.text(COST_CRC + 0.02, ax.get_ylim()[1] * 0.85, "your 0.65 CRC", color=RED, fontsize=9)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_title("Net Sharpe vs slippage per side (CRC) — daily signals die fast, slow trend is robust")
    ax.set_xlabel("slippage per side (CRC)")
    ax.set_ylabel("net Sharpe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "bt_slippage.png", dpi=110)
    plt.close(fig)


def chart_tearsheet(df, pos, res, name="Trend"):
    net, turn = bt(pos, df)
    s = pd.Series(net.values, index=df.date)
    cum = s.cumsum()
    dd = cum - cum.cummax()
    yearly = s.groupby(df.date.dt.year.values).sum()
    res[name + "_full"] = block(net, turn, len(df))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    b = res[name + "_full"]
    fig.suptitle(f"{name} long/short USD — NET of 0.65 CRC slippage  "
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
    fig.savefig(OUT / "bt_trend_tearsheet.png", dpi=110)
    plt.close(fig)


def chart_longshort(df, pos, res, name="Trend"):
    pos = pd.Series(pos, index=df.index)
    net, _ = bt(pos, df)
    longs = net.where(pos > 0, 0)
    shorts = net.where(pos < 0, 0)
    res[name + "_sides"] = {
        "long_usd_total_bps": round(float(longs.sum()), 0),
        "short_usd_total_bps": round(float(shorts.sum()), 0),
        "long_days": int((pos > 0).sum()),
        "short_days": int((pos < 0).sum()),
    }
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.date, longs.cumsum(), color=RED, lw=1.4, label="Long-USD legs")
    ax.plot(df.date, shorts.cumsum(), color=GREEN, lw=1.4, label="Short-USD legs")
    ax.plot(df.date, net.cumsum(), color=NAVY, lw=1.6, label="Combined")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Long/short decomposition — both sides contribute across regimes")
    ax.set_ylabel("cum net P&L (bps)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "bt_longshort.png", dpi=110)
    plt.close(fig)


def chart_oos(df, pos, res, name="Trend"):
    k = int(len(df) * OOS_FRAC)
    net, turn = bt(pos, df)
    res[name + "_is"] = block(net.iloc[:k])
    res[name + "_oos"] = block(net.iloc[k:])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.date, net.cumsum(), color=GREEN, lw=1.5)
    ax.axvspan(df.date.iloc[0], df.date.iloc[k], color=GREY, alpha=0.10)
    ax.axvline(df.date.iloc[k], color=RED, ls="--", lw=1)
    ax.set_title(f"{name}: in-sample Sharpe {res[name+'_is']['sharpe']} "
                 f"-> out-of-sample {res[name+'_oos']['sharpe']} (net)")
    ax.set_ylabel("cum net P&L (bps)")
    ax.text(df.date.iloc[5], net.cumsum().max() * 0.9, "train", color=GREY)
    ax.text(df.date.iloc[k + 5], net.cumsum().max() * 0.9, "test (held out)", color=RED)
    fig.tight_layout()
    fig.savefig(OUT / "bt_oos.png", dpi=110)
    plt.close(fig)


def chart_sensitivity(df, res):
    Ns = [20, 30, 40, 50, 60, 80, 100, 120]
    don = [sharpe(bt(sig_donchian(df, n), df)[0]) for n in Ns]
    Ms = [40, 60, 80, 100, 120, 150, 200, 250]
    sma = [sharpe(bt(sig_sma(df, m), df)[0]) for m in Ms]
    res["sensitivity"] = {
        "donchian_min": round(float(np.nanmin(don)), 2),
        "donchian_max": round(float(np.nanmax(don)), 2),
        "sma_min": round(float(np.nanmin(sma)), 2),
        "sma_max": round(float(np.nanmax(sma)), 2),
    }
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(Ns, don, "-o", color=GREEN)
    ax[0].axhline(0, color=GREY, lw=0.8)
    ax[0].set_title("Donchian breakout: net Sharpe vs lookback")
    ax[0].set_xlabel("lookback (days)")
    ax[0].set_ylabel("net Sharpe")
    ax[1].plot(Ms, sma, "-o", color=NAVY)
    ax[1].axhline(0, color=GREY, lw=0.8)
    ax[1].set_title("SMA trend: net Sharpe vs lookback")
    ax[1].set_xlabel("lookback (days)")
    fig.tight_layout()
    fig.savefig(OUT / "bt_sensitivity.png", dpi=110)
    plt.close(fig)


def chart_permutation(df, pos, res, name="Trend", n=2000):
    net, _ = bt(pos, df)
    actual = sharpe(net)
    posv = np.asarray(pos, float)
    r = df.r_next.reset_index(drop=True)
    px = df.close.reset_index(drop=True)
    cb = COST_CRC / px * 1e4
    null = np.empty(n)
    for i in range(n):
        p = pd.Series(RNG.permutation(posv))
        t = p.diff().abs()
        t.iloc[0] = abs(p.iloc[0])
        null[i] = sharpe(p * r - cb * t)
    pval = float((null >= actual).mean())
    res[name + "_perm_p"] = round(pval, 4)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null, bins=50, color=GREEN, alpha=0.55)
    ax.axvline(actual, color=RED, lw=2, label=f"actual {actual:.2f}")
    ax.set_title(f"{name}: permutation null (p={pval:.3f}) — does the timing beat random?")
    ax.set_xlabel("net Sharpe under shuffled timing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "bt_permutation.png", dpi=110)
    plt.close(fig)


def main():
    df = frame()
    res = {"_meta": {
        "n_days": int(len(df)), "date_min": str(df.date.min().date()),
        "date_max": str(df.date.max().date()), "cost_crc_per_side": COST_CRC,
        "cost_bps_at_475": round(COST_CRC / 475 * 1e4, 1), "oos_frac": OOS_FRAC,
        "n_trials": N_TRIALS,
    }}
    of = sig_orderflow(df)
    don = sig_donchian(df, 40)
    sma = sig_sma(df, 120)
    strats = [("Order-flow daily", of, ORANGE),
              ("Donchian-40 trend", don, GREEN),
              ("SMA-120 trend", sma, NAVY)]

    chart_regime_corr(df, res)
    chart_net_compare(df, res, strats)
    chart_slippage(df, res, strats)
    chart_tearsheet(df, don, res, "Donchian-40")
    chart_longshort(df, don, res, "Donchian-40")
    chart_oos(df, don, res, "Donchian-40")
    chart_sensitivity(df, res)
    chart_permutation(df, don, res, "Donchian-40")

    Path(OUT / "backtest_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("\nWrote bt_*.png and backtest_results.json")


if __name__ == "__main__":
    main()
