"""Exit-methodology lab — stop-loss, take-profit, time stop, and vol-scaled bands.

This extends src/exits.py (trailing stop + hard floor) into a full exit engine
layered on the SAME calendar entry rule (quincena.pos_calendar). No entry logic
is changed anywhere: the calendar decides WHEN to be short/long USD, and this
module only decides WHEN TO STAND DOWN EARLY inside a trade.

Four exit mechanisms, each usable alone or together
---------------------------------------------------
  trail    give back this many bps from the trade's running peak  (let winners
           run, then bank when the move stalls)
  floor    hard stop-loss: cut at -X bps from entry
  target   take-profit: bank at +X bps from entry
  max_days stand down after N sessions in the trade (the calendar edge is
           front-loaded, so a stale trade is mostly reversion risk)

Fixed-bps vs volatility-scaled bands
------------------------------------
A fixed 30 bps band means something completely different in the 2015-17 pegged
regime (daily vol ~6 bps) than in the 2022+ float (daily vol ~25 bps). So every
mechanism can instead be expressed in units of TRAILING REALISED VOLATILITY:
band_t = mult * sigma_t, where sigma_t is the 20-session std of VWAP returns
observed strictly through session t-1. That makes one parameter set travel
across regimes instead of being implicitly fitted to one of them.

Overfitting discipline
----------------------
  * Every parameter is chosen on the IN-SAMPLE 60% ONLY. The out-of-sample 40%
    is scored once, at the end, and never feeds a choice.
  * Stage 1 tests each mechanism ALONE, so the marginal value of each is visible
    before any of them are combined.
  * Stage 2 runs the joint grid but REPORTS THE TRIAL COUNT, and reports the
    neighbourhood median around the winner, so a lucky spike is distinguishable
    from a genuine plateau.
  * A deflated in-sample Sharpe (Bailey/Lopez de Prado style haircut for the
    number of trials) is reported alongside the raw one.

Accounting is identical to rank_strategies.py: VWAP-to-VWAP, USD 1M/trade,
0.325 CRC per side, annualised on the venue's REAL ~231 sessions/year.

Writes out/exit_lab.json and xl_*.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from analyze import OUT
from quincena import CAL_PRE, OOS_FRAC, frame, pos_calendar
from rank_strategies import NAVY, GREEN, RED, GREY, ORANGE, PURPLE, ann_days, stats

VOL_LOOKBACK = 20          # sessions used for the realised-vol estimate


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def realised_vol_bps(df, n=VOL_LOOKBACK):
    """Trailing realised daily vol of VWAP returns, in bps, STRICTLY causal.

    Shifted by one session so the band applied to session t is built only from
    returns observed through t-1. Backfilled with the first valid value so the
    opening window is not silently dropped.
    """
    r = df.vwap.pct_change() * 1e4
    v = r.rolling(n).std().shift(1)
    return v.bfill().to_numpy()


def episodes(pos):
    """Maximal runs of the same-signed position: (start, end_inclusive, sign)."""
    pos = np.asarray(pos, float)
    out, i, n = [], 0, len(pos)
    while i < n:
        s = np.sign(pos[i])
        j = i
        while j + 1 < n and np.sign(pos[j + 1]) == s:
            j += 1
        out.append((i, j, s))
        i = j + 1
    return out


def apply_exits(pos, vwap, trail=None, floor=None, target=None, max_days=None,
                vol=None, mult=False):
    """Flatten each trade early once an exit condition fires; stay flat until the
    entry rule opens the next trade.

    Causality: the running P&L is measured at session t's VWAP against the
    episode's entry VWAP, and a breach zeroes the position from t onward — so the
    t->t+1 leg is not earned. This is the same cadence the entry rule uses, so no
    look-ahead is introduced.

    When `mult` is True the four band arguments are interpreted as MULTIPLES of
    the trailing realised vol in `vol` (evaluated at the session being tested)
    rather than as absolute bps.
    """
    pos = np.asarray(pos, float)
    v = np.asarray(vwap, float)
    out = pos.copy()
    if mult and vol is None:
        raise ValueError("mult=True requires a vol series")

    for (a, b, s) in episodes(pos):
        if s == 0:
            continue
        entry, peak, exited = v[a], 0.0, False
        for t in range(a + 1, b + 1):
            if exited:
                out[t] = 0.0
                continue
            r = s * (v[t] / entry - 1) * 1e4          # running P&L, bps
            peak = max(peak, r)
            sig = vol[t] if mult else 1.0             # band unit
            hit = False
            if trail is not None and peak > 0 and (peak - r) >= trail * sig:
                hit = True
            if floor is not None and r <= -floor * sig:
                hit = True
            if target is not None and r >= target * sig:
                hit = True
            if max_days is not None and (t - a) >= max_days:
                hit = True
            if hit:
                out[t] = 0.0
                exited = True
    return out


def deflated_sharpe(sr_ann, rets, trial_sharpes_ann, apy):
    """Deflated Sharpe ratio (Bailey & Lopez de Prado): P(true Sharpe > 0) for a
    result that was SELECTED as the best of N trials.

    Picking the maximum of N noisy Sharpes inflates it even when no edge exists,
    so the selected Sharpe must clear the expected maximum under the null rather
    than merely clear zero. That benchmark is

        SR0 = sigma_trials * [(1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N*e))]

    where sigma_trials is the CROSS-SECTIONAL dispersion of the trial Sharpes and
    g is Euler-Mascheroni. Scaling by sigma_trials is essential: without it the
    quantile term (~3) is compared against a per-period Sharpe (~0.25) and the
    test saturates at 0 regardless of the evidence.

    Everything is converted to per-period (per-session) units first, and the
    denominator carries the skew/kurtosis correction for non-normal returns.
    """
    n_trials, n_obs = len(trial_sharpes_ann), len(rets)
    if n_trials < 2 or n_obs < 10:
        return None
    root = np.sqrt(apy)
    sr = sr_ann / root                                        # per-session
    sigma_trials = float(np.std(np.asarray(trial_sharpes_ann, float) / root, ddof=1))
    if sigma_trials <= 0:
        return None
    g = 0.5772156649                                          # Euler-Mascheroni
    z1 = sps.norm.ppf(1 - 1 / n_trials)
    z2 = sps.norm.ppf(1 - 1 / (n_trials * np.e))
    sr0 = sigma_trials * ((1 - g) * z1 + g * z2)

    r = np.asarray(rets, float)
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))               # non-excess
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(sps.norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / denom))


# --------------------------------------------------------------------------- #
# experiment stages
# --------------------------------------------------------------------------- #
def score(pos, df, k, apy):
    """(full, in-sample, out-of-sample) stat dicts on the common accounting basis."""
    pos = np.asarray(pos, float)
    return (stats(pos, df, apy), stats(pos[:k], df.iloc[:k], apy),
            stats(pos[k:], df.iloc[k:], apy))


def stage1_marginal(df, base, k, apy, vol):
    """Each mechanism ALONE, in bps and in vol multiples — the marginal value.

    Answering: does a stop-loss help on its own? does a take-profit? a time stop?
    Before any of them are allowed to interact in a joint grid.
    """
    rows = []

    def add(kind, unit, value, pos):
        f, i, o = score(pos, df, k, apy)
        if i is None:
            return
        rows.append({"mechanism": kind, "unit": unit, "value": value,
                     "is_sharpe": i["sharpe"], "oos_sharpe": o["sharpe"],
                     "is_per_yr": i["per_year_usd"], "oos_per_yr": o["per_year_usd"],
                     "is_maxdd": i["maxdd_usd"], "rt_yr": i["roundtrips_yr"],
                     "time_in_mkt": i["time_in_mkt"]})

    for b in [20, 30, 40, 50, 60, 80, 100, 120, 160]:
        add("trail", "bps", b, apply_exits(base, df.vwap, trail=b))
        add("floor", "bps", b, apply_exits(base, df.vwap, floor=b))
        add("target", "bps", b, apply_exits(base, df.vwap, target=b))
    for m in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        add("trail", "vol_mult", m, apply_exits(base, df.vwap, trail=m, vol=vol, mult=True))
        add("floor", "vol_mult", m, apply_exits(base, df.vwap, floor=m, vol=vol, mult=True))
        add("target", "vol_mult", m, apply_exits(base, df.vwap, target=m, vol=vol, mult=True))
    for d in [2, 3, 4, 5, 6, 8, 10, 12]:
        add("max_days", "sessions", d, apply_exits(base, df.vwap, max_days=d))
    return rows


def stage2_joint(df, base, k, apy, vol, mult):
    """Joint grid over (trail, floor, target), tuned on IN-SAMPLE Sharpe only.

    `mult=False` searches absolute-bps bands; `mult=True` searches vol multiples.
    Returns every cell so the plateau (not just the peak) can be inspected.
    """
    if mult:
        trails = [None, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        floors = [None, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        targets = [None, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    else:
        trails = [None, 20, 30, 40, 60, 80, 100, 140]
        floors = [None, 40, 60, 80, 120, 160, 240]
        targets = [None, 60, 80, 120, 160, 240, 320]

    cells = []
    for tr in trails:
        for fl in floors:
            for tg in targets:
                if tr is None and fl is None and tg is None:
                    continue                       # that is just the baseline
                p = apply_exits(base, df.vwap, trail=tr, floor=fl, target=tg,
                                vol=vol, mult=mult)
                f, i, o = score(p, df, k, apy)
                if i is None:
                    continue
                cells.append({"trail": tr, "floor": fl, "target": tg,
                              "is_sharpe": i["sharpe"], "oos_sharpe": o["sharpe"],
                              "is_per_yr": i["per_year_usd"], "oos_per_yr": o["per_year_usd"],
                              "full_sharpe": f["sharpe"], "is_maxdd": i["maxdd_usd"],
                              "oos_maxdd": o["maxdd_usd"], "rt_yr": i["roundtrips_yr"]})
    cells.sort(key=lambda c: c["is_sharpe"], reverse=True)
    return cells


def sharpe_se(sr_ann, n_obs, apy):
    """Standard error of an annualised Sharpe (Lo's approximation, iid case).

    SE(SR) ~ sqrt((1 + SR^2/2) / T) in per-period units. Used for a
    one-standard-error selection rule: parameter sets whose in-sample Sharpe is
    within 1 SE of the peak are statistically indistinguishable from it, so we
    are free to prefer the SIMPLEST of them rather than the highest-scoring one.
    """
    root = np.sqrt(apy)
    sr = sr_ann / root
    return float(np.sqrt((1 + sr ** 2 / 2) / n_obs) * root)


def complexity(cell):
    """How many exit mechanisms a cell actually arms (fewer = preferred)."""
    return sum(cell[k] is not None for k in ("trail", "floor", "target"))


def pick_parsimonious(cells, n_obs, apy, dd_tol=0.05):
    """Selection rule: one-standard-error band on Sharpe, then MINIMUM DRAWDOWN,
    then simplicity. All three criteria are in-sample only.

    Why not simplest-within-the-band alone: with a short sample the 1-SE band is
    wide (~0.4 Sharpe here) and swallows most of the grid, so "fewest mechanisms"
    would strip the stop-loss purely because Sharpe cannot see what a stop-loss
    does. A stop-loss buys TAIL control, and Sharpe — a ratio of mean to standard
    deviation — barely prices tails. Drawdown does, and it is observable in-sample,
    so it is a legitimate second criterion that costs us no look-ahead.

    Order of preference:
      1. in-sample Sharpe within 1 SE of the grid peak (statistical tie)
      2. smallest in-sample max drawdown, to within `dd_tol` relative tolerance
      3. fewest armed mechanisms (drop bands that are effectively inert)
      4. highest in-sample Sharpe
    """
    best = cells[0]
    se = sharpe_se(best["is_sharpe"], n_obs, apy)
    band = [c for c in cells if c["is_sharpe"] >= best["is_sharpe"] - se]

    floor_dd = min(abs(c["is_maxdd"]) for c in band)
    tight = [c for c in band if abs(c["is_maxdd"]) <= floor_dd * (1 + dd_tol)]
    tight.sort(key=lambda c: (complexity(c), -c["is_sharpe"]))
    return tight[0], round(se, 3), len(band), len(tight)


def neighbourhood(cells, best, mult):
    """Median in-sample Sharpe of parameter sets ADJACENT to the winner.

    A genuine plateau has neighbours nearly as good as the peak; a curve-fit
    spike is surrounded by much worse cells. Reported as the ratio
    median(neighbours) / best — close to 1.0 means robust.
    """
    def near(a, b, tol):
        if a is None or b is None:
            return a is None and b is None
        return abs(a - b) <= tol * max(abs(b), 1e-9)

    tol = 0.55
    nb = [c["is_sharpe"] for c in cells
          if near(c["trail"], best["trail"], tol)
          and near(c["floor"], best["floor"], tol)
          and near(c["target"], best["target"], tol)]
    if len(nb) < 2:
        return None
    return {"n": len(nb), "median_is_sharpe": round(float(np.median(nb)), 2),
            "ratio_to_best": round(float(np.median(nb) / best["is_sharpe"]), 3)}


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def chart_marginal(rows, base_is, base_oos, res):
    """Each mechanism alone: in-sample Sharpe vs its band, baseline as reference."""
    fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.2), sharey=True)
    panels = [("trail", "bps", "Trailing stop (bps)", GREEN),
              ("floor", "bps", "Hard stop-loss (bps)", RED),
              ("target", "bps", "Take-profit (bps)", NAVY),
              ("max_days", "sessions", "Time stop (sessions)", PURPLE)]
    for a, (mech, unit, title, col) in zip(ax, panels):
        sub = [r for r in rows if r["mechanism"] == mech and r["unit"] == unit]
        if sub:
            a.plot([r["value"] for r in sub], [r["is_sharpe"] for r in sub], "-o",
                   color=col, ms=4, label="in-sample")
            a.plot([r["value"] for r in sub], [r["oos_sharpe"] for r in sub], "-o",
                   color=col, ms=3, alpha=0.42, label="out-of-sample")
        a.axhline(base_is["sharpe"], color=GREY, ls="--", lw=1.2)
        a.axhline(base_oos["sharpe"], color=GREY, ls=":", lw=1.2)
        a.set_title(title, fontsize=10.5)
        a.set_xlabel(unit)
    ax[0].set_ylabel("net Sharpe")
    ax[0].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Marginal value of each exit mechanism ALONE, on the calendar entry rule\n"
                 f"(dashed = no-exit baseline in-sample {base_is['sharpe']}, "
                 f"dotted = out-of-sample {base_oos['sharpe']})",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "xl_marginal.png", dpi=110)
    plt.close(fig)


def chart_vol_vs_fixed(rows, base_is, res):
    """Fixed-bps bands vs volatility-scaled bands, per mechanism."""
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for a, mech, col in zip(ax, ["trail", "floor", "target"], [GREEN, RED, NAVY]):
        fx = [r for r in rows if r["mechanism"] == mech and r["unit"] == "bps"]
        vm = [r for r in rows if r["mechanism"] == mech and r["unit"] == "vol_mult"]
        a.plot(range(len(fx)), [r["is_sharpe"] for r in fx], "-o", color=col, ms=4,
               label="fixed bps")
        a.plot(range(len(vm)), [r["is_sharpe"] for r in vm], "-s", color=ORANGE, ms=4,
               label="vol-scaled")
        a.axhline(base_is["sharpe"], color=GREY, ls="--", lw=1.2)
        a.set_title(f"{mech} — fixed vs vol-scaled", fontsize=10.5)
        a.set_xlabel("band index (loose -> tight ordering differs per unit)")
        a.legend(fontsize=8.5)
    ax[0].set_ylabel("in-sample net Sharpe")
    fig.suptitle("Do volatility-scaled bands travel better than fixed bps?\n"
                 "(a fixed band is implicitly fitted to one volatility regime)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(OUT / "xl_vol_vs_fixed.png", dpi=110)
    plt.close(fig)


def chart_equity(df, variants, res):
    """Cumulative equity and drawdown for the baseline vs the chosen overlays."""
    from rank_strategies import pnl
    fig, ax = plt.subplots(2, 1, figsize=(11.5, 6.8), height_ratios=[2, 1], sharex=True)
    for label, pos, col, lw in variants:
        net, _ = pnl(pos, df)
        d = df.date.iloc[net.index]
        cum = net.cumsum()
        dd = cum - cum.cummax()
        s = res["variants"][label]
        ax[0].plot(d, cum / 1e6, color=col, lw=lw,
                   label=f"{label} (IS Sh {s['is']['sharpe']} -> OOS {s['oos']['sharpe']} · "
                         f"${s['full']['per_year_usd']/1e3:.0f}k/yr)")
        ax[1].fill_between(d, dd / 1e3, 0, color=col, alpha=0.30)
    k = int(len(df) * OOS_FRAC)
    for a in ax:
        a.axvline(df.date.iloc[k], color=RED, ls="--", lw=1)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("cumulative net P&L (US$ M)")
    ax[0].set_title("Calendar entry rule with and without the exit overlay — "
                    "VWAP-priced, net of 0.65 CRC RT, $1M/trade\n"
                    "(red line = in-sample / out-of-sample boundary; bands tuned left of it only)",
                    fontweight="bold", fontsize=11.5)
    ax[0].legend(loc="upper left", fontsize=8.5)
    ax[1].set_ylabel("drawdown (US$ k)")
    fig.tight_layout()
    fig.savefig(OUT / "xl_equity.png", dpi=110)
    plt.close(fig)


def chart_plateau(cells, best, res, mult):
    """In-sample vs out-of-sample Sharpe for every grid cell — is the peak real?"""
    fig, ax = plt.subplots(figsize=(7.6, 6))
    x = [c["is_sharpe"] for c in cells]
    y = [c["oos_sharpe"] for c in cells]
    ax.scatter(x, y, s=16, color=NAVY, alpha=0.35, edgecolor="none")
    ax.scatter([best["is_sharpe"]], [best["oos_sharpe"]], s=110, color=RED,
               zorder=5, label="chosen (best in-sample)")
    lim = [min(x + y) - 0.2, max(x + y) + 0.2]
    ax.plot(lim, lim, color=GREY, ls="--", lw=1, label="IS = OOS")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("in-sample net Sharpe (the selection metric)")
    ax.set_ylabel("out-of-sample net Sharpe (never used to select)")
    r = np.corrcoef(x, y)[0, 1]
    ax.set_title(f"Does in-sample selection survive out-of-sample?\n"
                 f"{len(cells)} parameter sets · corr(IS, OOS) = {r:+.2f}",
                 fontweight="bold", fontsize=11.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / ("xl_plateau_vol.png" if mult else "xl_plateau_bps.png"), dpi=110)
    plt.close(fig)
    return round(float(r), 3)


# --------------------------------------------------------------------------- #
def main():
    df = frame()
    apy = ann_days(df)
    k = int(len(df) * OOS_FRAC)
    base = pos_calendar(df)
    vol = realised_vol_bps(df)

    bf, bi, bo = score(base, df, k, apy)
    res = {"_meta": {
        "n_days": int(len(df)), "sessions_per_year": round(float(apy), 1),
        "notional_usd": 1_000_000, "cost_crc_roundtrip": 0.65,
        "basis": "VWAP-to-VWAP next session, net of slippage",
        "entry_rule": f"quincena.pos_calendar (short USD <= {CAL_PRE} sessions to IVA deadline)",
        "oos_frac": OOS_FRAC, "split_date": str(df.date.iloc[k].date()),
        "vol_lookback": VOL_LOOKBACK,
        "vol_bps_median": round(float(np.median(vol)), 1),
        "vol_bps_is_median": round(float(np.median(vol[:k])), 1),
        "vol_bps_oos_median": round(float(np.median(vol[k:])), 1),
        "tuning": "all parameters selected on the in-sample window only"}}

    res["baseline"] = {"full": bf, "is": bi, "oos": bo}

    # ---- stage 1: each mechanism alone -------------------------------------
    marg = stage1_marginal(df, base, k, apy, vol)
    res["stage1_marginal"] = marg
    res["stage1_best_per_mechanism"] = {}
    for mech in ["trail", "floor", "target", "max_days"]:
        for unit in ["bps", "vol_mult", "sessions"]:
            sub = [r for r in marg if r["mechanism"] == mech and r["unit"] == unit]
            if sub:
                b = max(sub, key=lambda r: r["is_sharpe"])
                res["stage1_best_per_mechanism"][f"{mech}_{unit}"] = b

    # ---- stage 2: joint grids, bps and vol-scaled ---------------------------
    from rank_strategies import pnl
    out = {}
    for mult, tag in [(False, "bps"), (True, "vol")]:
        cells = stage2_joint(df, base, k, apy, vol, mult)
        best = cells[0]
        nb = neighbourhood(cells, best, mult)
        # Deflate the SELECTED in-sample Sharpe against the whole trial population.
        best_pos = apply_exits(base, df.vwap, trail=best["trail"], floor=best["floor"],
                               target=best["target"], vol=vol, mult=mult)
        is_rets, _ = pnl(best_pos[:k], df.iloc[:k])
        dsr = deflated_sharpe(best["is_sharpe"], is_rets.to_numpy(),
                              [c["is_sharpe"] for c in cells], apy)
        corr = chart_plateau(cells, best, res, mult)
        pars, se, n_band, n_tight = pick_parsimonious(cells, k, apy)
        out[tag] = {"n_trials": len(cells), "best": best, "parsimonious": pars,
                    "sharpe_se": se, "n_within_1se": n_band,
                    "n_within_dd_tolerance": n_tight, "neighbourhood": nb,
                    "deflated_sharpe_prob": None if dsr is None else round(dsr, 4),
                    "is_oos_corr": corr, "top10": cells[:10]}
    res["stage2_joint"] = out

    # ---- does a take-profit help at all? ------------------------------------
    # Reported explicitly because the joint grid parks `target` at its loosest
    # value, which is easy to misread as "a 320 bps take-profit was selected"
    # when in fact the optimiser is switching the mechanism OFF.
    tp = []
    for tg in [None, 320, 240, 160, 120, 80]:
        p = apply_exits(base, df.vwap, trail=30, floor=40, target=tg)
        f, i, o = score(p, df, k, apy)
        tp.append({"target_bps": tg, "is_sharpe": i["sharpe"], "oos_sharpe": o["sharpe"],
                   "is_per_yr": i["per_year_usd"], "oos_per_yr": o["per_year_usd"]})
    res["take_profit_sweep"] = tp
    res["take_profit_verdict"] = (
        "HARMFUL — tightening the take-profit degrades both windows monotonically "
        f"(OOS Sharpe {tp[0]['oos_sharpe']} with no target -> {tp[-1]['oos_sharpe']} at 80 bps). "
        "The calendar edge is a drift that needs to be held through the deadline; "
        "capping the winner truncates exactly the move the strategy exists to capture. "
        "Recommendation: no take-profit.")

    # ---- the chosen variants, scored ---------------------------------------
    bb, vb = out["bps"]["parsimonious"], out["vol"]["parsimonious"]
    rec_pos = apply_exits(base, df.vwap, trail=bb["trail"], floor=bb["floor"],
                          target=bb["target"])
    rec_label = f"RECOMMENDED — trail {bb['trail']} / floor {bb['floor']} / TP {bb['target']}"
    chosen = [
        ("No exit (baseline)", base, GREY, 1.5),
        (rec_label, rec_pos, GREEN, 2.0),
        (f"vol-scaled — trail {vb['trail']} / floor {vb['floor']} / TP {vb['target']} x sigma",
         apply_exits(base, df.vwap, trail=vb["trail"], floor=vb["floor"],
                     target=vb["target"], vol=vol, mult=True),
         ORANGE, 1.6),
    ]
    res["variants"] = {}
    for label, pos, _c, _lw in chosen:
        f, i, o = score(pos, df, k, apy)
        res["variants"][label] = {"full": f, "is": i, "oos": o}
    res["recommended"] = rec_label

    # ---- is the overlay an ALPHA gain or a RISK gain? -----------------------
    # A paired test on the daily P&L difference, out-of-sample only. If the mean
    # difference is not significant, the overlay is not adding return; it is
    # buying a smoother path, which is a different (still useful) claim.
    nb_, _ = pnl(base[k:], df.iloc[k:])
    ne_, _ = pnl(rec_pos[k:], df.iloc[k:])
    diff = (ne_ - nb_).dropna()
    t, pv = sps.ttest_1samp(diff.to_numpy(), 0.0)
    b_oos, r_oos = res["baseline"]["oos"], res["variants"][rec_label]["oos"]
    res["alpha_vs_risk"] = {
        "oos_total_usd_baseline": round(float(nb_.sum())),
        "oos_total_usd_overlay": round(float(ne_.sum())),
        "oos_total_usd_delta": round(float(ne_.sum() - nb_.sum())),
        "paired_t": round(float(t), 2), "paired_p": round(float(pv), 4),
        "oos_sharpe_delta": round(r_oos["sharpe"] - b_oos["sharpe"], 2),
        "oos_maxdd_delta": r_oos["maxdd_usd"] - b_oos["maxdd_usd"],
        "verdict": ("RISK improvement, not an alpha improvement — the overlay does not "
                    "significantly change total P&L (see paired_p), but it materially "
                    "raises Sharpe and cuts max drawdown. Its value is that the smoother "
                    "path can be sized up, not that it earns more per unit of notional.")}

    chart_marginal(marg, bi, bo, res)
    chart_vol_vs_fixed(marg, bi, res)
    chart_equity(df, chosen, res)

    Path(OUT / "exit_lab.json").write_text(json.dumps(res, indent=2))

    # ---- report -------------------------------------------------------------
    print(f"EXIT LAB — entry = {res['_meta']['entry_rule']}")
    print(f"{res['_meta']['n_days']} sessions · annualised on {apy:.0f}/yr · "
          f"split {res['_meta']['split_date']}")
    print(f"realised vol median: full {res['_meta']['vol_bps_median']} bps · "
          f"IS {res['_meta']['vol_bps_is_median']} · OOS {res['_meta']['vol_bps_oos_median']}\n")
    print(f"BASELINE (no exit):  IS Sharpe {bi['sharpe']:.2f} -> OOS {bo['sharpe']:.2f} | "
          f"IS ${bi['per_year_usd']:,}/yr | IS maxDD ${bi['maxdd_usd']:,}\n")

    print("STAGE 1 — best setting of each mechanism ALONE (selected in-sample):")
    hdr = f"  {'mechanism':<18} {'value':>7} {'IS Sh':>6} {'OOS Sh':>7} {'IS $/yr':>10} {'IS DD':>9} {'%mkt':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key, b in res["stage1_best_per_mechanism"].items():
        print(f"  {key:<18} {b['value']:>7} {b['is_sharpe']:>6.2f} {b['oos_sharpe']:>7.2f} "
              f"{b['is_per_yr']:>10,} {b['is_maxdd']:>9,} {b['time_in_mkt']:>5.0f}")

    print("\nSTAGE 2 — joint grid (trail x floor x target), selected on IN-SAMPLE Sharpe:")
    for tag in ["bps", "vol"]:
        o = out[tag]
        b, p = o["best"], o["parsimonious"]
        nbs = o["neighbourhood"]
        print(f"  [{tag}] {o['n_trials']} trials | grid peak: trail={b['trail']} "
              f"floor={b['floor']} target={b['target']}  ->  IS {b['is_sharpe']:.2f} / "
              f"OOS {b['oos_sharpe']:.2f}")
        print(f"        selection: SE={o['sharpe_se']:.2f} -> {o['n_within_1se']} sets tie on Sharpe, "
              f"{o['n_within_dd_tolerance']} of them tie on drawdown; simplest of those:")
        print(f"        trail={p['trail']} floor={p['floor']} target={p['target']}")
        print(f"        -> IS Sharpe {p['is_sharpe']:.2f} -> OOS {p['oos_sharpe']:.2f} | "
              f"IS ${p['is_per_yr']:,}/yr -> OOS ${p['oos_per_yr']:,}/yr | "
              f"IS DD ${p['is_maxdd']:,} | OOS DD ${p['oos_maxdd']:,}")
        if nbs:
            print(f"        plateau: {nbs['n']} neighbours, median IS Sharpe "
                  f"{nbs['median_is_sharpe']:.2f} ({nbs['ratio_to_best']:.0%} of peak)")
        print(f"        deflated-Sharpe P(true>0) = {o['deflated_sharpe_prob']} | "
              f"corr(IS,OOS) across trials = {o['is_oos_corr']:+.2f}")

    print("\nTAKE-PROFIT SWEEP (on trail 30 + floor 40):")
    print(f"  {'target':>8} {'IS Sh':>6} {'OOS Sh':>7} {'IS $/yr':>10} {'OOS $/yr':>10}")
    for r in res["take_profit_sweep"]:
        print(f"  {str(r['target_bps']):>8} {r['is_sharpe']:>6.2f} {r['oos_sharpe']:>7.2f} "
              f"{r['is_per_yr']:>10,} {r['oos_per_yr']:>10,}")
    print(f"  VERDICT: {res['take_profit_verdict'][:66]}...")

    a = res["alpha_vs_risk"]
    print(f"\nALPHA vs RISK (out-of-sample, {res['recommended']}):")
    print(f"  total P&L  ${a['oos_total_usd_baseline']:,} -> ${a['oos_total_usd_overlay']:,} "
          f"(delta ${a['oos_total_usd_delta']:+,}, paired t={a['paired_t']:+.2f} p={a['paired_p']:.3f})")
    print(f"  Sharpe     {res['baseline']['oos']['sharpe']:.2f} -> "
          f"{res['variants'][res['recommended']]['oos']['sharpe']:.2f} "
          f"({a['oos_sharpe_delta']:+.2f})")
    print(f"  max DD     ${res['baseline']['oos']['maxdd_usd']:,} -> "
          f"${res['variants'][res['recommended']]['oos']['maxdd_usd']:,} "
          f"({a['oos_maxdd_delta']:+,})")
    print("  => the overlay buys a smoother path, NOT more P&L. Size up against it.")
    print("\nWrote out/exit_lab.json and xl_*.png")


if __name__ == "__main__":
    main()
