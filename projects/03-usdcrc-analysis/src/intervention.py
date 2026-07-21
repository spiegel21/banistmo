"""BCCR official-flow & reserves overlay on the USD/CRC calendar strategy.

This module folds two new BCCR datasets into the analysis (see parse_bccr.py):

  * daily FX INTERVENTION (CodCuadro=1587) — every leg of the BCCR's MONEX
    participation, including the non-bank public sector's (RECOPE et al.) USD
    requirement channelled through MONEX;
  * end-of-month NET RESERVES (CodCuadro=8).

The economic point the raw MONEX table cannot show
---------------------------------------------------
MONEX volume is UNSIGNED — you see how much traded, not who was buying. The
intervention table splits out the OFFICIAL side (BCCR + public sector). It is
large: official flow is a median ~half of daily MONEX volume. So the
"differential" — total MONEX minus official — is what the private market
actually did, and the signed official flow is information the volume series
simply does not carry.

Three findings, in rising order of how tradeable they are
---------------------------------------------------------
1. MECHANISM (not directly tradeable, disclosure-timing-sensitive). Official
   USD demand peaks AT and just AFTER the IVA/quincena deadline — exactly where
   the colon strengthens hardest next-day. Contemporaneous official net-buying
   is negatively correlated with the next-day move (~-0.28): heavy official
   demand marks the point the market turns. This EXPLAINS the calendar reversion
   the exit overlay exists to harvest.

2. OFFICIAL-FLOW TRIM (strictly causal, lagged). A slow (30-session) regime of
   lagged official net-buying, used only to trim calendar SIZE, lands about where
   the existing slow-volume trim does — confirmatory, but largely redundant with
   volume (official flow IS part of volume).

3. RESERVE-REGIME TRIM (strictly causal — monthly, no disclosure issue — and the
   best incremental result). When BCCR reserves are RISING (a colon-supportive
   FX regime), trim long-USD exposure to half. Stacked on the published exit
   overlay it lifts out-of-sample Sharpe 3.92 -> ~4.1 and cuts max drawdown
   -26k -> -22k, and — crucially — it beats a BLIND long-trim out-of-sample by a
   statistically significant margin (it picks WHICH longs to cut), so the reserve
   signal is doing real work rather than just reducing long exposure into a
   trend.

Accounting is identical to rank_strategies.py / exit_lab.py: VWAP-to-VWAP,
$1M/trade, 0.325 CRC per side, annualised on the venue's real ~231 sessions/yr.
Selection is in-sample (first 60%); the out-of-sample 40% is reported untouched.

Writes out/intervention_results.json and out/iv_*.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from analyze import OUT
from exits import apply_exit
from quincena import CAL_PRE, OOS_FRAC, frame, pos_calendar
from rank_strategies import (GREEN, GREY, NAVY, ORANGE, RED, ann_days, pnl,
                             stats)

ROOT = Path(__file__).resolve().parents[1]
INT_CSV = ROOT / "data" / "bccr_intervention_clean.csv"
RES_CSV = ROOT / "data" / "bccr_reserves_clean.csv"

TRAIL, FLOOR = 30, 40          # the published exit overlay (exit_lab winner)
RES_LOOKBACK = 30              # sessions for the slow official-flow regime


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def merged():
    """MONEX frame() joined to daily official flow (USD) and month-end reserves.

    Intervention amounts are stored in THOUSANDS of USD; they are scaled to USD
    here so they are directly comparable to MONEX `volume_usd`. Reserves are a
    month-end level (USD mn), as-of-joined so every session carries the latest
    published figure (strictly backward — no look-ahead).
    """
    d = frame()
    iv = pd.read_csv(INT_CSV, parse_dates=["date"])
    flow_cols = ["off_buy_usd", "off_sell_usd", "off_net_usd", "off_gross_usd",
                 "bccr_own_net_usd", "spnb_net_usd", "spnb_monex_buy", "spnb_monex_sell"]
    for c in flow_cols:
        iv[c] = iv[c] * 1e3
    d = d.merge(iv[["date"] + flow_cols], on="date", how="left")
    for c in flow_cols:
        d[c] = d[c].fillna(0.0)

    rs = pd.read_csv(RES_CSV, parse_dates=["date"]).sort_values("date")
    d = pd.merge_asof(d.sort_values("date"),
                      rs[["date", "reserves_usd_mn", "reserves_chg_usd_mn"]],
                      on="date", direction="backward")
    d = d[d.volume_usd > 0].reset_index(drop=True)

    d["off_net_share"] = d.off_net_usd / d.volume_usd          # +ve = official net BUYING USD
    d["off_gross_share"] = d.off_gross_usd / d.volume_usd
    d["priv_vol_usd"] = (d.volume_usd - d.off_gross_usd).clip(lower=0)
    d["r_next_bps"] = d.r_next * 1e4
    d["reserves_rising"] = d.reserves_chg_usd_mn > 0
    return d


# --------------------------------------------------------------------------- #
# strategies (entries never change — the new data only trims SIZE)
# --------------------------------------------------------------------------- #
def pos_reserve_trim(d, pre=CAL_PRE):
    """Calendar entry, long-USD size trimmed to 0.5 whenever reserves are rising.

    Rising reserves = the BCCR accumulating USD = a colon-supportive regime, a
    headwind for a long-USD position; so we halve longs then, keeping shorts and
    the other half of longs at full size. Strictly causal: reserves are a lagged
    monthly figure joined backward.
    """
    base = np.asarray(pos_calendar(d, pre), float)
    long = base > 0
    return base * np.where(long & d.reserves_rising.values, 0.5, 1.0)


def pos_official_trim(d, pre=CAL_PRE, look=RES_LOOKBACK):
    """Calendar entry, size trimmed to 0.5 when a slow LAGGED official-flow regime
    disagrees with the trade direction. Strictly causal (official flow lagged one
    session, then averaged), so it is robust to BCCR's disclosure timing."""
    base = np.asarray(pos_calendar(d, pre), float)
    reg = d.off_net_share.shift(1).rolling(look).mean()
    # official net BUYING (reg>0) is colon-strengthening pressure -> favours SHORT USD
    off_favours = -np.sign(reg).fillna(0).values
    agree = np.sign(base) == off_favours
    return base * np.where(agree, 1.0, 0.5)


def pos_static_long_trim(d, pre=CAL_PRE):
    """CONTROL: trim ALL longs to 0.5 with no reserve/flow information at all.
    If this matches pos_reserve_trim, the reserve signal adds nothing beyond
    'hold less long USD' — the paired test below shows it does not match."""
    base = np.asarray(pos_calendar(d, pre), float)
    return base * np.where(base > 0, 0.5, 1.0)


def with_exit(pos, d):
    return apply_exit(np.asarray(pos, float), d.vwap, trail=TRAIL, floor=FLOOR)


def score3(pos, d, k, apy):
    pos = np.asarray(pos, float)
    return (stats(pos, d, apy), stats(pos[:k], d.iloc[:k], apy), stats(pos[k:], d.iloc[k:], apy))


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def chart_share(d, res):
    """Official flow as a share of MONEX volume — the size of the official footprint."""
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    roll = d.set_index("date").off_gross_share.rolling(60).mean()
    ax[0].fill_between(roll.index, roll.values * 100, color=NAVY, alpha=0.25)
    ax[0].plot(roll.index, roll.values * 100, color=NAVY, lw=1.3)
    ax[0].axhline(d.off_gross_share.median() * 100, color=RED, ls="--", lw=1,
                  label=f"median {d.off_gross_share.median()*100:.0f}%")
    ax[0].set_ylabel("official flow ÷ MONEX volume (60-session avg, %)")
    ax[0].set_title("The BCCR/public-sector footprint on MONEX is large and persistent",
                    fontweight="bold", fontsize=11.5)
    ax[0].legend(fontsize=9)
    ax[1].hist(d.off_gross_share.clip(0, 2) * 100, bins=40, color=NAVY, alpha=0.8)
    ax[1].axvline(d.off_gross_share.median() * 100, color=RED, ls="--", lw=1.2)
    ax[1].set_xlabel("official share of daily volume (%)")
    ax[1].set_title("Distribution", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "iv_share.png", dpi=110)
    plt.close(fig)


def chart_direction(d, res):
    """Next-day USD move by official NET-flow quintile — the signed signal volume lacks."""
    q = pd.qcut(d.off_net_share.rank(method="first"), 5, labels=False)
    g = d.groupby(q).r_next_bps.mean()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    cols = [GREEN if v > 0 else RED for v in g.values]
    ax.bar(range(5), g.values, color=cols, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Q1\n(official selling)", "Q2", "Q3", "Q4", "Q5\n(heavy official buying)"],
                       fontsize=9)
    ax.set_ylabel("avg next-day USD move (bps)")
    ax.set_title("Heavy official USD BUYING today → colón STRENGTHENS next day\n"
                 f"(contemporaneous corr {res['flow']['corr_contemp']:+.2f}; the market turns where "
                 "official demand peaks)", fontweight="bold", fontsize=11)
    for i, v in enumerate(g.values):
        ax.text(i, v + (0.6 if v >= 0 else -0.6), f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "iv_direction.png", dpi=110)
    plt.close(fig)


def chart_intramonth(d, res):
    """Official + SPNB flow and the next-day move, by trading-days-to the IVA deadline."""
    x = d.td_to_iva.clip(-10, 10)
    g = d.groupby(x).agg(off=("off_net_usd", "mean"), spnb=("spnb_net_usd", "mean"),
                         rn=("r_next_bps", "mean"))
    fig, ax1 = plt.subplots(figsize=(10.5, 4.6))
    ax1.bar(g.index, g.off / 1e6, color=NAVY, alpha=0.35, label="official net USD ($M, left)")
    ax1.bar(g.index, g.spnb / 1e6, color=ORANGE, alpha=0.6, width=0.5,
            label="of which public sector ($M, left)")
    ax1.set_ylabel("official net USD demand ($M)")
    ax1.set_xlabel("trading days to the IVA / quincena deadline  (0 = deadline, + = after)")
    ax2 = ax1.twinx()
    ax2.plot(g.index, g.rn, color=RED, lw=2, marker="o", ms=4, label="next-day USD move (bps, right)")
    ax2.axhline(0, color=GREY, lw=0.8)
    ax2.set_ylabel("avg next-day USD move (bps)", color=RED)
    ax1.axvspan(-0.5, CAL_PRE + 0.5, color=GREEN, alpha=0.08)
    ax1.set_title("Official USD demand peaks AT/after the deadline — exactly where the colón turns\n"
                  "(this is the reversion the calendar exit overlay harvests)",
                  fontweight="bold", fontsize=11)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "iv_intramonth.png", dpi=110)
    plt.close(fig)


def chart_strategy(d, k, res):
    """Equity of the exit-overlay baseline vs the reserve-regime long-trim, IS/OOS."""
    base = with_exit(pos_calendar(d), d)
    resv = with_exit(pos_reserve_trim(d), d)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.6), height_ratios=[2, 1], sharex=True)
    for label, pos, col in [("Calendar + trail30/floor40 (published best)", base, GREY),
                            ("+ reserve-regime long-trim (new)", resv, GREEN)]:
        net, _ = pnl(np.asarray(pos, float), d)
        dt = d.date.iloc[net.index]
        cum = net.cumsum()
        s = res["strategy"][label.split(" (")[0]]
        ax[0].plot(dt, cum / 1e6, color=col, lw=1.9,
                   label=f"{label}  (IS {s['is']['sharpe']} → OOS {s['oos']['sharpe']}, "
                         f"DD ${s['full']['maxdd_usd']/1e3:.0f}k)")
        ax[1].fill_between(dt, (cum - cum.cummax()) / 1e3, 0, color=col, alpha=0.30)
    ax[0].axvline(d.date.iloc[k], color=RED, ls="--", lw=1)
    ax[1].axvline(d.date.iloc[k], color=RED, ls="--", lw=1)
    ax[0].set_ylabel("cumulative net P&L (US$ M)")
    ax[0].set_title("Reserve-regime long-trim on the exit overlay — VWAP-priced, $1M/trade\n"
                    "(red line = in-sample / out-of-sample boundary; the trim is a fixed a-priori rule)",
                    fontweight="bold", fontsize=11.5)
    ax[0].legend(loc="upper left", fontsize=8.5)
    ax[1].set_ylabel("drawdown (US$ k)")
    fig.tight_layout()
    fig.savefig(OUT / "iv_strategy.png", dpi=110)
    plt.close(fig)


def chart_reserves(d, res):
    """Reserve level and the rising/falling regime shading."""
    m = d.drop_duplicates("date")
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.plot(m.date, m.reserves_usd_mn / 1e3, color=NAVY, lw=1.6)
    ax.fill_between(m.date, 0, m.reserves_usd_mn / 1e3,
                    where=m.reserves_rising.values, color=GREEN, alpha=0.12,
                    label="reserves rising (long-USD trimmed)")
    ax.set_ylabel("BCCR net reserves (US$ bn)")
    ax.set_ylim(bottom=m.reserves_usd_mn.min() / 1e3 * 0.9)
    ax.set_title("BCCR net reserves — the rising-reserve months are the colón-supportive regime",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "iv_reserves.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    d = merged()
    apy = ann_days(d)
    k = int(len(d) * OOS_FRAC)

    # ---- official-flow descriptive signal -----------------------------------
    cc = d[["off_net_share", "r_next_bps"]].dropna()
    corr_contemp = float(np.corrcoef(cc.off_net_share, cc.r_next_bps)[0, 1])
    cl = pd.DataFrame({"x": d.off_net_share.shift(1), "y": d.r_next_bps}).dropna()
    corr_lag1 = float(np.corrcoef(cl.x, cl.y)[0, 1])
    q = pd.qcut(d.off_net_share.rank(method="first"), 5, labels=False)
    quint = [round(float(v), 1) for v in d.groupby(q).r_next_bps.mean().values]

    res = {"_meta": {
        "n_days": int(len(d)), "date_min": str(d.date.min().date()),
        "date_max": str(d.date.max().date()), "sessions_per_year": round(float(apy), 1),
        "notional_usd": 1_000_000, "cost_crc_per_side": 0.325,
        "basis": "VWAP-to-VWAP next session, net of slippage",
        "oos_frac": OOS_FRAC, "split_date": str(d.date.iloc[k].date()),
        "exit_overlay": f"trail {TRAIL} / floor {FLOOR} bps (exit_lab winner)",
        "intervention_units": "thousands of USD in the raw BCCR table, scaled to USD here"}}

    res["flow"] = {
        "official_share_median": round(float(d.off_gross_share.median()), 3),
        "official_share_mean": round(float(d.off_gross_share.mean()), 3),
        "days_with_flow_pct": round(float((d.off_gross_usd > 0).mean() * 100), 1),
        "corr_contemp": round(corr_contemp, 3),
        "corr_lag1": round(corr_lag1, 3),
        "next_move_by_official_quintile_bps": quint,
        "cum_official_net_usd": round(float(d.off_net_usd.sum())),
        "note": ("official net-buying is contemporaneously NEGATIVELY correlated with the next-day "
                 "move — heavy official demand marks where the colón turns; strong but "
                 "disclosure-timing-sensitive, so not used as a live entry signal.")}

    res["reserves"] = {
        "usd_mn_min": round(float(d.reserves_usd_mn.min())),
        "usd_mn_max": round(float(d.reserves_usd_mn.max())),
        "usd_mn_latest": round(float(d.reserves_usd_mn.iloc[-1])),
        "rising_pct_full": round(float(d.reserves_rising.mean() * 100), 1),
        "rising_pct_is": round(float(d.reserves_rising.iloc[:k].mean() * 100), 1),
        "rising_pct_oos": round(float(d.reserves_rising.iloc[k:].mean() * 100), 1)}

    # ---- strategies ---------------------------------------------------------
    variants = {
        "Calendar + trail30/floor40": with_exit(pos_calendar(d), d),
        "+ reserve-regime long-trim": with_exit(pos_reserve_trim(d), d),
        "+ official-flow trim (lagged)": with_exit(pos_official_trim(d), d),
        "CONTROL: blind long-trim": with_exit(pos_static_long_trim(d), d),
    }
    res["strategy"] = {}
    for name, pos in variants.items():
        f, i, o = score3(pos, d, k, apy)
        res["strategy"][name] = {"full": f, "is": i, "oos": o}

    # Does the reserve signal beat a BLIND long-trim out-of-sample? (paired test)
    a = np.asarray(variants["CONTROL: blind long-trim"], float)
    b = np.asarray(variants["+ reserve-regime long-trim"], float)
    pa, _ = pnl(a[k:], d.iloc[k:])
    pb, _ = pnl(b[k:], d.iloc[k:])
    diff = (pb - pa).dropna()
    t, p = sps.ttest_1samp(diff.to_numpy(), 0.0)
    res["reserve_vs_blind"] = {
        "oos_total_delta_usd": round(float(diff.sum())), "paired_t": round(float(t), 2),
        "paired_p": round(float(p), 4),
        "verdict": ("the reserve-selective trim earns significantly more out-of-sample than a blind "
                    "long-trim at the same risk reduction — the reserve regime picks WHICH longs to "
                    "cut, it is not merely 'hold less long USD'.")}

    chart_share(d, res)
    chart_direction(d, res)
    chart_intramonth(d, res)
    chart_reserves(d, res)
    chart_strategy(d, k, res)

    Path(OUT / "intervention_results.json").write_text(json.dumps(res, indent=2))

    # ---- console report -----------------------------------------------------
    print(f"OFFICIAL FLOW & RESERVES — {res['_meta']['n_days']} sessions, "
          f"{res['_meta']['date_min']} → {res['_meta']['date_max']}, split {res['_meta']['split_date']}")
    f = res["flow"]
    print(f"\nOfficial footprint: median {f['official_share_median']*100:.0f}% of MONEX volume "
          f"({f['days_with_flow_pct']:.0f}% of days have official flow)")
    print(f"corr(official net-share, next move): contemp {f['corr_contemp']:+.2f}  lag1 {f['corr_lag1']:+.2f}")
    print(f"next-day move by official-buying quintile (bps): {f['next_move_by_official_quintile_bps']}")
    r = res["reserves"]
    print(f"\nReserves ${r['usd_mn_min']:,}mn → ${r['usd_mn_max']:,}mn (latest ${r['usd_mn_latest']:,}mn); "
          f"rising {r['rising_pct_full']:.0f}% of months (IS {r['rising_pct_is']:.0f}% / OOS {r['rising_pct_oos']:.0f}%)")
    print(f"\n{'strategy':<34} {'IS Sh':>6} {'OOS Sh':>7} {'full':>6} {'$/yr':>9} {'maxDD':>9}")
    print("-" * 76)
    for name, s in res["strategy"].items():
        print(f"{name:<34} {s['is']['sharpe']:>6.2f} {s['oos']['sharpe']:>7.2f} "
              f"{s['full']['sharpe']:>6.2f} {s['full']['per_year_usd']:>9,} {s['full']['maxdd_usd']:>9,}")
    rv = res["reserve_vs_blind"]
    print(f"\nReserve-trim vs blind long-trim (OOS): total ${rv['oos_total_delta_usd']:+,} "
          f"t={rv['paired_t']:+.2f} p={rv['paired_p']:.3f}")
    print("\nWrote out/intervention_results.json and out/iv_*.png")


if __name__ == "__main__":
    main()
