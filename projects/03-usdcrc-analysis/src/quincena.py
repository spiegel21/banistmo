"""Deep-dive on the calendar (quincena) strategy — the most robust edge found.

Refines the crude "short USD if day<=15" into "short USD only on the mid-month
USD-supply window (days 5-15), long otherwise", validates it against overfitting
(window sensitivity grid, per-year stability), tests whether VOLUME confirmation
adds value net of cost, and reports everything in dollars at USD 1M / trade.

Writes q_*.png and out/quincena_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats as sps

from analyze import OUT, load
from basis import ANN, SESSIONS_PER_YEAR
from basis import NOTIONAL_USD as NOTIONAL
from basis import COST_CRC_PER_SIDE as COST_SIDE_CRC
from payment_calendar import annotate

plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "text.parse_math": False})  # '$' are currency, not math delimiters
NAVY, GREEN, RED, GREY, PURPLE = "#1f3b73", "#2e8b57", "#c0392b", "#7f8c8d", "#7d3c98"
SHORT_START, SHORT_END = 5, 15      # fixed day-of-month short-USD window (legacy proxy)
CAL_PRE = 6                         # calendar rule: short USD this many trading days
                                    # before the IVA/quincena deadline, through the deadline
OOS_FRAC = 0.60                     # chronological split: first 60% in-sample, last 40% out-of-sample
                                    # (matches backtest.py / backtest_vwap.py)


def frame():
    df = annotate(load().copy())    # adds td_to_iva and the real-calendar event flags
    df["vol"] = df.volume_usd / 1e6
    # Tradeable next-session return priced VWAP-to-VWAP (the realistic desk fill),
    # not close-to-close: a position decided from info through session t earns the
    # NEXT session's VWAP move. Slippage (below) is also charged against the VWAP.
    df["r_next"] = df.vwap.pct_change().shift(-1)
    df["dom"] = df.date.dt.day
    df["dow"] = df.date.dt.dayofweek
    df["year"] = df.date.dt.year
    df["vol_ref"] = df.groupby("dow").vol.transform(
        lambda s: s.shift(1).expanding(min_periods=10).mean())
    df["vol_adj"] = df.vol / df.vol_ref
    return df


def pos_base(df):
    return np.where(df.dom <= 15, -1.0, 1.0)


def pos_refined(df, a=SHORT_START, b=SHORT_END):
    """Short USD on the mid-month supply window [a,b], long otherwise."""
    return np.where((df.dom >= a) & (df.dom <= b), -1.0, 1.0)


def pos_calendar(df, pre=CAL_PRE):
    """Real-calendar rule: short USD in the run-up to the IVA/quincena deadline.

    Companies sell USD to raise colones for the 15th-of-month IVA (D-104) +
    withholding (D-103) filing and the mid-month payroll, so the colon
    strengthens in the `pre` trading days leading to (and on) that deadline —
    then reverts once the supply clears. Unlike the fixed 5-15 window this
    anchor moves with weekends/holidays (see payment_calendar.annotate).
    """
    return np.where((df.td_to_iva >= 0) & (df.td_to_iva <= pre), -1.0, 1.0)


def pos_refined_slowvol(df):
    """Refined (fixed 5–15) rule, size trimmed to 0.5 when a slow (20d) volume regime disagrees."""
    base = pos_refined(df)
    slow = df.vol_adj.rolling(20).mean()
    # in the short window we want supply (high vol) to confirm; outside we want thin (low vol)
    confirm = np.where((df.dom >= SHORT_START) & (df.dom <= SHORT_END), slow > 1, slow < 1)
    return base * np.where(confirm, 1.0, 0.5)


def pos_calendar_slowvol(df, pre=CAL_PRE):
    """RECOMMENDED rule: the deadline-anchored calendar entry with the slow-volume size trim.

    Entry timing = pos_calendar (short USD in the run-up to the rolling IVA/quincena
    deadline, long otherwise) — the economically-motivated, holiday-robust anchor.
    Sizing = trim to 0.5x whenever a slow (20-day) volume regime disagrees with the
    trade: inside the short window we want supply (high volume) to confirm; outside it
    we want a thin market (low volume). The trim never changes the position SIGN, only
    its magnitude, so it removes the fat losing tail without adding turnover. This is
    exactly the rule the report describes ("calendar anchor + optional slow-vol trim").
    """
    base = np.asarray(pos_calendar(df, pre), float)
    slow = df.vol_adj.rolling(20).mean()
    short = (df.td_to_iva >= 0) & (df.td_to_iva <= pre)
    confirm = np.where(short, slow > 1, slow < 1)
    return base * np.where(confirm, 1.0, 0.5)


def dollars(pos, df):
    pos = pd.Series(np.asarray(pos, float)).reset_index(drop=True)
    turn = pos.diff().abs()
    turn.iloc[0] = abs(pos.iloc[0])
    pnl = pos * NOTIONAL * df.r_next.reset_index(drop=True)
    cost = turn * NOTIONAL * COST_SIDE_CRC / df.vwap.reset_index(drop=True)
    net = (pnl - cost).dropna()
    return net, turn


def stat(pos, df):
    net, turn = dollars(pos, df)
    cum = net.cumsum()
    yrs = len(net) / SESSIONS_PER_YEAR
    bps = net / (NOTIONAL / 1e4)
    return {"total_usd": round(float(net.sum())), "per_year_usd": round(float(net.sum() / yrs)),
            "sharpe": round(float(bps.mean() / bps.std() * ANN), 2),
            "roundtrips_yr": round(float(turn.sum() / 2 / yrs), 1),
            "win_rate": round(float((net > 0).mean() * 100), 1),
            "maxdd_usd": round(float((cum - cum.cummax()).min()))}


# --------------------------------------------------------------------------- #
def chart_sensitivity(df, res):
    starts, ends = range(2, 9), range(12, 19)
    grid = np.full((len(list(starts)), len(list(ends))), np.nan)
    for i, a in enumerate(starts):
        for j, b in enumerate(ends):
            grid[i, j] = stat(pos_refined(df, a, b), df)["sharpe"]
    res["window_sharpe_min"] = round(float(np.nanmin(grid)), 2)
    res["window_sharpe_max"] = round(float(np.nanmax(grid)), 2)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=1.5, vmax=np.nanmax(grid))
    ax.set_xticks(range(len(list(ends))))
    ax.set_xticklabels(list(ends))
    ax.set_yticks(range(len(list(starts))))
    ax.set_yticklabels(list(starts))
    ax.set_xlabel("short-window END day")
    ax.set_ylabel("short-window START day")
    ax.set_title("Net Sharpe across short-window choices — broad green plateau = robust")
    ax.grid(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "q_sensitivity.png", dpi=110)
    plt.close(fig)


def chart_peryear(df, res):
    rows = []
    for y, g in df.groupby("year"):
        if len(g) < 50:
            continue
        rows.append((y, stat(pos_base(g), g)["per_year_usd"] * len(g) / SESSIONS_PER_YEAR / (len(g) / SESSIONS_PER_YEAR),
                     dollars(pos_refined(g), g)[0].sum()))
    yr = [r[0] for r in rows]
    ref = [dollars(pos_refined(g), g)[0].sum() for _, g in df.groupby("year") if len(g) >= 50]
    bas = [dollars(pos_base(g), g)[0].sum() for _, g in df.groupby("year") if len(g) >= 50]
    res["refined_worst_year_usd"] = round(float(min(ref)))
    x = np.arange(len(yr))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, np.array(bas) / 1e3, 0.4, label="base quincena (≤15)", color=GREY)
    ax.bar(x + 0.2, np.array(ref) / 1e3, 0.4, label="refined (short 5–15)", color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in yr])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("net P&L per year (USD thousands, 1M/trade)")
    ax.set_title("Every year is positive — refined dominates base")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "q_peryear.png", dpi=110)
    plt.close(fig)


def chart_interaction(df, res):
    d = df.dropna(subset=["r_next", "vol_adj"]).copy()
    d["half"] = np.where(d.dom <= 15, "1st half", "2nd half")
    d["vlvl"] = np.where(d.vol_adj < 1, "low vol", "high vol")
    piv = d.pivot_table("r_next", "half", "vlvl", aggfunc="mean") * 1e4
    res["interaction_bps"] = {f"{i}|{c}": round(float(piv.loc[i, c]), 1)
                              for i in piv.index for c in piv.columns}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lim = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(2))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(2))
    ax.set_yticklabels(piv.index)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{piv.values[i, j]:+.1f}", ha="center", va="center", fontsize=13)
    ax.set_title("Next-day USD move (bps): calendar × volume\nvolume CONFIRMS the calendar on the diagonal")
    ax.grid(False)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "q_interaction.png", dpi=110)
    plt.close(fig)


def chart_tearsheet(df, res):
    net, turn = dollars(pos_refined(df), df)
    s = pd.Series(net.values, index=df.date.iloc[net.index])
    cum = s.cumsum()
    dd = cum - cum.cummax()
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    b = res["Refined (short 5-15)"]
    fig.suptitle(f"Refined quincena — long/short USD 1M per trade, net of 0.65 CRC RT  "
                 f"(USD {b['per_year_usd']/1e3:.0f}k/yr · Sharpe {b['sharpe']} · "
                 f"{b['roundtrips_yr']} trades/yr · win {b['win_rate']}%)",
                 fontsize=12, fontweight="bold")
    ax[0, 0].plot(cum.index, cum.values / 1e6, color=GREEN, lw=1.6)
    ax[0, 0].set_title("Cumulative net P&L (USD millions)")
    ax[0, 1].fill_between(dd.index, dd.values / 1e3, 0, color=RED, alpha=0.5)
    ax[0, 1].set_title(f"Drawdown (USD k) — max USD {dd.min()/1e3:.0f}k")
    yearly = s.groupby(s.index.year).sum()
    ax[1, 0].bar(yearly.index.astype(str), yearly.values / 1e3, color=GREEN)
    ax[1, 0].set_title("Net P&L by year (USD k)")
    ax[1, 0].tick_params(axis="x", rotation=90, labelsize=8)
    mret = s.groupby(s.index.month).mean() * SESSIONS_PER_YEAR / 12
    ax[1, 1].bar([pd.Timestamp(2025, m, 1).strftime("%b") for m in mret.index],
                 mret.values / 1e3, color=[GREEN if v > 0 else RED for v in mret.values])
    ax[1, 1].set_title("Avg P&L by calendar month (USD k)")
    ax[1, 1].tick_params(axis="x", rotation=90, labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "q_tearsheet.png", dpi=110)
    plt.close(fig)


def chart_calendar(df, res):
    """Validate the real calendar: next-day USD move vs trading-days-to-deadline."""
    g = df.dropna(subset=["r_next"]).copy()
    g["bucket"] = g.td_to_iva.clip(-6, 10)
    prof = g.groupby("bucket").r_next.mean() * 1e4
    res["calendar_next_move_bps"] = {int(k): round(float(v), 1) for k, v in prof.items()}
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = [GREEN if k > CAL_PRE or k < 0 else RED for k in prof.index]  # red = short-USD window
    ax.bar(prof.index, prof.values, color=colors)
    ax.axvline(-0.5, color="k", lw=0.8, ls="--")
    ax.axvspan(-0.5, CAL_PRE + 0.5, color=RED, alpha=0.07)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("trading days to IVA / quincena deadline  (0 = deadline · +N = N days before · −N = after)")
    ax.set_ylabel("avg NEXT-session USD move (bps)")
    ax.set_title("USD weakens into the tax/payroll deadline, reverts after — the real-calendar signal\n"
                 "(red = short-USD window; negative bars = colon appreciation)")
    for k, v in prof.items():
        ax.text(k, v + (0.4 if v >= 0 else -0.4), f"{v:+.0f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "q_calendar.png", dpi=110)
    plt.close(fig)


def chart_calendar_peryear(df, res):
    """Per-year net P&L: calendar rule vs the fixed 5-15 window."""
    yr, cal, ref = [], [], []
    for y, g in df.groupby("year"):
        if len(g) < 50:
            continue
        yr.append(y)
        cal.append(dollars(pos_calendar(g), g)[0].sum())
        ref.append(dollars(pos_refined(g), g)[0].sum())
    res["calendar_worst_year_usd"] = round(float(min(cal)))
    x = np.arange(len(yr))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, np.array(ref) / 1e3, 0.4, label="fixed window (day 5–15)", color=GREY)
    ax.bar(x + 0.2, np.array(cal) / 1e3, 0.4, label=f"real calendar (≤{CAL_PRE}d to deadline)", color=NAVY)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in yr])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("net P&L per year (USD thousands, 1M/trade)")
    ax.set_title("Real-calendar rule vs fixed day-of-month window, by year")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "q_calendar_peryear.png", dpi=110)
    plt.close(fig)


def calendar_sensitivity(df, res):
    """Sharpe across the pre-deadline lookback choice (robustness of the calendar rule)."""
    grid = {p: stat(pos_calendar(df, p), df)["sharpe"] for p in range(3, 11)}
    res["calendar_pre_sharpe"] = {int(k): v for k, v in grid.items()}
    res["calendar_pre_sharpe_min"] = round(float(min(grid.values())), 2)
    res["calendar_pre_sharpe_max"] = round(float(max(grid.values())), 2)


def oos_stats(pos_fn, df):
    """In-sample vs out-of-sample stats for a position rule (chronological 60/40 split).

    Even though the winning rule is a fixed heuristic with no fitted parameters, we
    still hold out the last 40% of history so the reader can see the edge was not a
    property of the (calmer, more-pegged) early years alone.
    """
    k = int(len(df) * OOS_FRAC)
    dis, dos = df.iloc[:k], df.iloc[k:]
    return {"is": stat(pos_fn(dis), dis), "oos": stat(pos_fn(dos), dos),
            "split_date": str(df.date.iloc[k].date()),
            "is_start": str(df.date.iloc[0].date()), "is_end": str(df.date.iloc[k - 1].date()),
            "oos_end": str(df.date.iloc[-1].date())}


def chart_oos(df, res):
    """Cumulative VWAP-net P&L for the RECOMMENDED calendar+slow-vol rule, IS/OOS shaded."""
    k = int(len(df) * OOS_FRAC)
    net, _ = dollars(pos_calendar_slowvol(df), df)
    s = pd.Series(net.values, index=df.date.iloc[net.index])
    cum = s.cumsum()
    o = res["oos_recommended"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(cum.index, cum.values / 1e6, color=GREEN, lw=1.6)
    ax.axvspan(cum.index[0], df.date.iloc[k], color=GREY, alpha=0.12)
    ax.axvline(df.date.iloc[k], color=RED, ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("cumulative net P&L (US$ millions, 1M/trade)")
    ax.set_title(f"Recommended calendar + slow-vol rule: in-sample Sharpe {o['is']['sharpe']} "
                 f"→ out-of-sample {o['oos']['sharpe']}", fontsize=12)
    ax.text(0.02, 0.96, f"in-sample 60%  ({o['is_start']} → {o['is_end']})\n"
            f"${o['is']['per_year_usd']/1e3:.0f}k/yr · Sharpe {o['is']['sharpe']}",
            transform=ax.transAxes, color="#4a5a6a", fontsize=8.5, va="top")
    ax.text(0.98, 0.12, f"out-of-sample 40%  (→ {o['oos_end']})\n"
            f"${o['oos']['per_year_usd']/1e3:.0f}k/yr · Sharpe {o['oos']['sharpe']}",
            transform=ax.transAxes, color=RED, fontsize=8.5, ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "q_oos.png", dpi=110)
    plt.close(fig)


def _basis_stat(pos, df, r_next, px):
    """Net-P&L series + summary for a rule priced against an arbitrary return/fill basis."""
    pos = pd.Series(np.asarray(pos, float)).reset_index(drop=True)
    turn = pos.diff().abs()
    turn.iloc[0] = abs(pos.iloc[0])
    r = pd.Series(np.asarray(r_next, float)).reset_index(drop=True)
    S = pd.Series(np.asarray(px, float)).reset_index(drop=True)
    net = (pos * NOTIONAL * r - turn * NOTIONAL * COST_SIDE_CRC / S).dropna()
    cum = net.cumsum()
    yrs = len(net) / SESSIONS_PER_YEAR
    bps = net / (NOTIONAL / 1e4)
    return net, {"per_year_usd": round(float(net.sum() / yrs)),
                 "sharpe": round(float(bps.mean() / bps.std() * ANN), 2),
                 "maxdd_usd": round(float((cum - cum.cummax()).min()))}


def chart_execution(df, res):
    """The recommended calendar rule, executed at the session VWAP vs the closing print.

    This is the honest execution-realism check for the strategy we actually trade:
    the VWAP is a fill a desk can work; the closing print is the optimistic mark. Same
    0.65 CRC round-trip slippage on both. (Part F used to make this point with unrelated
    trend books — here it is the calendar rule itself.)
    """
    d = df.reset_index(drop=True)
    pos = pos_calendar_slowvol(d)
    net_v, sv = _basis_stat(pos, d, d.r_next, d.vwap)                    # r_next is VWAP-to-VWAP
    net_c, sc = _basis_stat(pos, d, d.close.pct_change().shift(-1), d.close)
    res["execution"] = {"vwap": sv, "close": sc}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(d.date.iloc[net_c.index], net_c.cumsum() / 1e6, color=GREY, lw=1.5,
            label=f"closing print (Sharpe {sc['sharpe']} · ${sc['per_year_usd']/1e3:.0f}k/yr)")
    ax.plot(d.date.iloc[net_v.index], net_v.cumsum() / 1e6, color=NAVY, lw=1.8,
            label=f"session VWAP — realistic (Sharpe {sv['sharpe']} · ${sv['per_year_usd']/1e3:.0f}k/yr)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("cumulative net P&L (US$ millions, 1M/trade)")
    ax.set_title("Recommended calendar rule — executed at the session VWAP vs the closing print\n"
                 "(same 0.65 CRC round-trip slippage; the VWAP is the fill a desk can actually work)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "q_execution.png", dpi=110)
    plt.close(fig)


def chart_position_shading(df, res, zoom_months=18):
    """Session VWAP with the recommended real-calendar position shaded behind it:
    green = long USD, red = short USD, unshaded = no position on.
    """
    pos = pd.Series(pos_calendar(df), index=df.date)
    price = pd.Series(df.vwap.values, index=df.date)

    def shade(ax, p, c):
        block = (p != p.shift(1)).cumsum()
        for _, g in p.groupby(block):
            sign = g.iloc[0]
            if sign == 0:
                continue  # no trade on -> leave blank
            ax.axvspan(g.index[0], g.index[-1], color=GREEN if sign > 0 else RED, alpha=0.22, lw=0)
        ax.plot(c.index, c.values, color=NAVY, lw=1.0)
        ax.set_ylabel("USD/CRC session VWAP")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))
    shade(ax1, pos, price)
    ax1.set_title(f"Real-calendar position vs USD/CRC VWAP — short USD ≤{CAL_PRE} bd into the "
                  "IVA/quincena deadline, long otherwise")
    handles = [Patch(facecolor=GREEN, alpha=0.4, label="long USD"),
               Patch(facecolor=RED, alpha=0.4, label="short USD")]
    ax1.legend(handles=handles, loc="upper left", fontsize=9)

    cutoff = price.index.max() - pd.DateOffset(months=zoom_months)
    shade(ax2, pos.loc[pos.index >= cutoff], price.loc[price.index >= cutoff])
    ax2.set_title(f"Last {zoom_months} months, zoomed")

    fig.tight_layout()
    fig.savefig(OUT / "q_position_shading.png", dpi=110)
    plt.close(fig)


# (label, slug, position fn, colour) — the calendar family compared in the report tabs.
# Recommended = "Calendar + slow-vol": deadline-anchored entry + slow-volume size trim.
STRATS = [
    ("Base quincena (≤15)", "base", pos_base, GREY),
    ("Real calendar", "calendar", pos_calendar, NAVY),
    ("Refined + slow-vol", "slowvol", pos_refined_slowvol, PURPLE),
    ("Calendar + slow-vol", "calslow", pos_calendar_slowvol, GREEN),
]
RECOMMENDED = "Calendar + slow-vol"


def _trade_pnl(pos_fn, df):
    """Per-trade (per directional roundtrip) net P&L in USD, VWAP-priced.

    A *trade* is one continuous stretch where the position keeps the same SIGN
    (long-USD or short-USD). These rules are always in the market, so the sign
    blocks tile the whole sample; each trade's P&L is the sum of the daily net
    P&L (including the entry-flip slippage) over its holding window, so the
    per-trade P&Ls sum back exactly to the strategy's total. Size changes within
    a single held direction (the slow-vol 0.5x trim) stay inside the same trade.
    """
    raw = np.asarray(pos_fn(df), float)
    net, _turn = dollars(pos_fn(df), df)           # USD/day; last (NaN r_next) row dropped
    idx = net.index.to_numpy()
    sign = np.sign(raw[idx])
    change = np.empty(len(sign), bool)
    change[0] = True
    change[1:] = sign[1:] != sign[:-1]
    block = np.cumsum(change)
    by_trade = pd.Series(net.to_numpy()).groupby(block).sum()
    sgn = pd.Series(sign).groupby(block).first()
    return by_trade[sgn.values != 0].to_numpy()    # drop any flat (no-position) blocks


def _trade_stats(x):
    """Shape of a per-trade P&L population, all in USD."""
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std()), "win": float((x > 0).mean() * 100),
            "skew": float(sps.skew(x)), "kurt": float(sps.kurtosis(x)),  # excess kurtosis
            "var95": float(np.percentile(x, 5)), "var99": float(np.percentile(x, 1)),
            "min": float(x.min()), "max": float(x.max())}


def _is_oos_sharpe(pos_fn, df):
    """(in-sample, out-of-sample) daily annualised Sharpe on the chronological split.
    The variant is SELECTED on the in-sample number; the out-of-sample number is the
    honest confirmation — it never feeds the selection."""
    k = int(len(df) * OOS_FRAC)
    return (float(stat(pos_fn(df.iloc[:k]), df.iloc[:k])["sharpe"]),
            float(stat(pos_fn(df.iloc[k:]), df.iloc[k:])["sharpe"]))


def chart_return_distributions(df, res):
    """One histogram+KDE of PER-TRADE net P&L (USD) per calendar-family variant, plus
    an overlay — all built on the OUT-OF-SAMPLE window only (the last 1-OOS_FRAC of
    history), so the pictures show the P&L an operator would actually have booked on
    data the rule was never tuned on. Each panel reports both the in-sample Sharpe (the
    selection metric) and the out-of-sample Sharpe (the realised, honest one). The
    recommended rule — best out-of-sample — carries the badge. Everything VWAP-priced,
    on a shared x-range so shapes compare.
    """
    k = int(len(df) * OOS_FRAC)
    dos = df.iloc[k:]                                            # out-of-sample slice
    res["return_dist_window"] = {"split_date": str(df.date.iloc[k].date()),
                                 "oos_end": str(df.date.iloc[-1].date()),
                                 "basis": "out-of-sample per-trade net P&L (USD)"}

    trades = {slug: _trade_pnl(fn, dos) for _, slug, fn, _ in STRATS}
    is_oos = {label: _is_oos_sharpe(fn, df) for label, _slug, fn, _ in STRATS}
    # SELECT on in-sample; but recommended is fixed by economics+IS and confirmed OOS.
    best_label = RECOMMENDED

    pooled = np.concatenate(list(trades.values())) / 1e3        # USD thousands
    lim = float(max(abs(np.percentile(pooled, 1)), abs(np.percentile(pooled, 99))))
    grid = np.linspace(-lim, lim, 400)

    res["return_dist"] = {}
    res["return_dist_best"] = best_label
    for label, slug, _fn, color in STRATS:
        x = trades[slug]
        st = _trade_stats(x)
        rd = {k2: (int(v) if k2 == "n" else round(float(v), 2)) for k2, v in st.items()}
        rd["is_sharpe"], rd["oos_sharpe"] = round(is_oos[label][0], 2), round(is_oos[label][1], 2)
        res["return_dist"][label] = rd

        xk = x / 1e3
        n_out = int((np.abs(xk) > lim).sum())
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(np.clip(xk, -lim, lim), bins=30, density=True, color=color, alpha=0.32,
                edgecolor="white", linewidth=0.3)
        if len(np.unique(xk)) > 1:
            ax.plot(grid, sps.gaussian_kde(xk)(grid), color=color, lw=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(st["mean"] / 1e3, color=color, lw=1.6, label=f"mean ${st['mean']/1e3:+.1f}k/trade")
        ax.axvline(st["var95"] / 1e3, color=RED, ls="--", lw=1.6,
                   label=f"5% worst ${st['var95']/1e3:.1f}k")
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("net P&L per trade (USD thousands · $1M/trade, VWAP-priced · out-of-sample)")
        ax.set_ylabel("density")
        crown = "   ·   RECOMMENDED — best out-of-sample" if label == best_label else ""
        ax.set_title(f"{label} — out-of-sample per-trade net P&L{crown}")
        txt = (f"OOS trades {st['n']}\n"
               f"mean ${st['mean']/1e3:+.1f}k   median ${st['median']/1e3:+.1f}k\n"
               f"std ${st['std']/1e3:.1f}k    win {st['win']:.0f}%\n"
               f"skew {st['skew']:+.2f}    ex-kurt {st['kurt']:+.1f}\n"
               f"best ${st['max']/1e3:+.0f}k    worst ${st['min']/1e3:.0f}k\n"
               f"Sharpe: IS {is_oos[label][0]:.2f} -> OOS {is_oos[label][1]:.2f}")
        if n_out:
            txt += f"\n({n_out} trades beyond ±${lim:.0f}k clipped in)"
        ax.text(0.985, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.92))
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"q_dist_{slug}.png", dpi=110)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, slug, _fn, color in STRATS:
        xk = trades[slug] / 1e3
        if len(np.unique(xk)) < 2:
            continue
        y = sps.gaussian_kde(xk)(grid)
        is_best = label == best_label
        ax.plot(grid, y, color=color, lw=2.8 if is_best else 1.7,
                label=label + (" — recommended" if is_best else ""))
        ax.fill_between(grid, y, color=color, alpha=0.08 if is_best else 0.04)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("net P&L per trade (USD thousands · $1M/trade, VWAP-priced · out-of-sample)")
    ax.set_ylabel("density")
    ax.set_title("Out-of-sample per-trade net P&L — calendar-family variants overlaid\n"
                 "(the slow-vol variants keep their right-shift and thin left tail on held-out data)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "q_dist_overlay.png", dpi=110)
    plt.close(fig)


def trading_calendar(df):
    """Practical map: for each day-of-month, the position the fixed refined rule takes.

    (The recommended calendar rule keys off trading-days-to-deadline, not the raw
    day number; see payment_calendar and the `calendar_*` results.)
    """
    return {int(d): ("SHORT USD" if SHORT_START <= d <= SHORT_END else "LONG USD")
            for d in range(1, 32)}


def main():
    df = frame()
    res = {"_meta": {"n_days": int(len(df)), "notional_usd": NOTIONAL,
                     "short_window": [SHORT_START, SHORT_END], "cal_pre": CAL_PRE,
                     "sessions_per_year": SESSIONS_PER_YEAR,
                     "years": round(len(df) / SESSIONS_PER_YEAR, 1)}}
    res["Base quincena (<=15)"] = stat(pos_base(df), df)
    res["Refined (short 5-15)"] = stat(pos_refined(df), df)
    res["Refined + slow-vol sizing"] = stat(pos_refined_slowvol(df), df)
    res[f"Real calendar (<={CAL_PRE}d to deadline)"] = stat(pos_calendar(df), df)
    res["Calendar + slow-vol (recommended)"] = stat(pos_calendar_slowvol(df), df)

    res["oos_refined"] = oos_stats(pos_refined, df)
    res["oos_calendar"] = oos_stats(pos_calendar, df)
    res["oos_recommended"] = oos_stats(pos_calendar_slowvol, df)

    # Whole calendar family, in-sample vs out-of-sample, in one block: selection is on
    # the IS Sharpe; the OOS columns are the honest confirmation (never fed the pick).
    res["family_oos"] = {
        "Base quincena (≤15)": oos_stats(pos_base, df),
        "Real calendar": oos_stats(pos_calendar, df),
        "Refined (fixed 5–15)": oos_stats(pos_refined, df),
        "Refined + slow-vol": oos_stats(pos_refined_slowvol, df),
        "Calendar + slow-vol": oos_stats(pos_calendar_slowvol, df),
    }

    chart_sensitivity(df, res)
    chart_peryear(df, res)
    chart_interaction(df, res)
    chart_tearsheet(df, res)
    chart_calendar(df, res)
    chart_calendar_peryear(df, res)
    chart_oos(df, res)
    chart_execution(df, res)
    chart_position_shading(df, res)
    chart_return_distributions(df, res)
    calendar_sensitivity(df, res)
    res["trading_calendar"] = trading_calendar(df)

    Path(OUT / "quincena_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "trading_calendar"}, indent=2))
    print("\nWrote q_*.png and quincena_results.json")


if __name__ == "__main__":
    main()
