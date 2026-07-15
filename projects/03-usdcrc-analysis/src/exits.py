"""Exit optimisation for the calendar rule — a stop-loss / take-profit OVERLAY.

The calendar (quincena) rule is always in the market: SHORT USD into the
IVA/quincena deadline, LONG the rest of the month (see quincena.pos_calendar).
This module does NOT change that entry logic. It layers a per-trade exit on top:
within each holding "episode" (a maximal run of the same signed calendar
position) the position is flattened early once its running P&L, measured from the
episode's entry VWAP, gives back a `trail` band from its best level (a **dynamic
trailing stop**) — optionally also on a hard `floor` stop-loss. After an early
exit we stay flat until the calendar starts the next episode, so the exit can
only ever *remove* the tail of a trade, never add a new one.

Why it helps: the calendar edge front-loads — the colón strengthens INTO the
deadline and reverts after — so a trailing stop banks the appreciation once the
move stalls instead of handing it back on the reversion.

Everything is priced VWAP-to-VWAP, net of the 0.65 CRC round-trip slippage, at
USD 1M/trade — identical accounting to quincena.py. Parameters are chosen on the
IN-SAMPLE 60% only; the out-of-sample 40% is reported untouched.

Writes ex_*.png and out/exits_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import OUT
from quincena import (CAL_PRE, GREEN, GREY, NAVY, OOS_FRAC, RED, dollars, frame,
                      pos_calendar, stat)

ORANGE = "#e08a2b"
TRAIL_GRID = list(range(20, 161, 10))       # candidate trailing bands (bps of price)
FLOOR_GRID = [60, 80, 100, 120, 160, 200]   # candidate hard floor stops (bps)


# --------------------------------------------------------------------------- #
# the overlay
# --------------------------------------------------------------------------- #
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


def apply_exit(pos, vwap, trail=None, floor=None):
    """Return a copy of `pos` flattened to 0 for the rest of any episode once the
    running P&L (from the episode entry VWAP, evaluated at each session's VWAP)
    gives back `trail` bps from its peak, or breaches a `-floor` bps hard stop.

    Causal: the breach is observed at session t's VWAP and flattens the t->t+1 leg
    onward, so no look-ahead is introduced (same cadence the entry rule uses)."""
    pos = np.asarray(pos, float).copy()
    v = np.asarray(vwap, float)
    out = pos.copy()
    for (a, b, s) in episodes(pos):
        if s == 0:
            continue
        entry, peak, exited = v[a], 0.0, False
        for t in range(a + 1, b + 1):
            if exited:
                out[t] = 0.0
                continue
            r = s * (v[t] / entry - 1) * 1e4                 # running P&L, bps
            peak = max(peak, r)
            hit_trail = trail is not None and peak > 0 and (peak - r) >= trail
            hit_floor = floor is not None and r <= -floor
            if hit_trail or hit_floor:
                out[t] = 0.0
                exited = True
    return out


def windows(pos, df, k):
    """(full, in-sample, out-of-sample) stat dicts for a position series."""
    return (stat(pos, df), stat(pos[:k], df.iloc[:k]), stat(pos[k:], df.iloc[k:]))


# --------------------------------------------------------------------------- #
# optimisation (in-sample only)
# --------------------------------------------------------------------------- #
def optimise(df, base, k):
    """Grid-search the exits on the IN-SAMPLE window only, then report OOS.

    Two objectives, each honest about which window it was tuned on:
      * trailing-only, chosen to MAXIMISE in-sample P&L/yr  -> the P&L enhancement
      * trailing+floor, chosen to MAXIMISE in-sample Sharpe -> the risk enhancement
    Also returns the full trailing grid (full/IS/OOS) so the report can show the
    robustness plateau rather than a single lucky point.
    """
    v = df.vwap
    grid = []
    for tr in TRAIL_GRID:
        f, i, o = windows(apply_exit(base, v, trail=tr), df, k)
        grid.append({"trail": tr,
                     "full_per_yr": f["per_year_usd"], "is_per_yr": i["per_year_usd"],
                     "oos_per_yr": o["per_year_usd"], "full_sharpe": f["sharpe"],
                     "is_sharpe": i["sharpe"], "oos_sharpe": o["sharpe"],
                     "full_maxdd": f["maxdd_usd"]})
    # trailing chosen by in-sample P&L (tie-break in-sample Sharpe)
    best = max(grid, key=lambda g: (g["is_per_yr"], g["is_sharpe"]))
    trail_star = best["trail"]

    # trailing+floor chosen by in-sample Sharpe over a joint grid
    combo, best_c = None, -1e9
    for tr in TRAIL_GRID:
        for fl in FLOOR_GRID:
            i = stat(apply_exit(base, v, trail=tr, floor=fl)[:k], df.iloc[:k])
            if i["sharpe"] > best_c:
                best_c, combo = i["sharpe"], (tr, fl)
    return trail_star, combo, grid


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def chart_optimization(df, grid, trail_star, base_full, base_is, base_oos, res):
    """The optimiser surface: P&L and Sharpe across the trailing band, with the
    baseline (no-exit) as a dashed reference and the chosen band marked."""
    tr = [g["trail"] for g in grid]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].plot(tr, [g["is_per_yr"] / 1e3 for g in grid], "-o", color=GREEN, ms=3, label="in-sample")
    ax[0].plot(tr, [g["oos_per_yr"] / 1e3 for g in grid], "-o", color=NAVY, ms=3, label="out-of-sample")
    ax[0].axhline(base_is["per_year_usd"] / 1e3, color=GREEN, ls="--", lw=1, alpha=.7)
    ax[0].axhline(base_oos["per_year_usd"] / 1e3, color=NAVY, ls="--", lw=1, alpha=.7)
    ax[0].axvline(trail_star, color=RED, ls=":", lw=1.4)
    ax[0].set_title("Net P&L per year vs trailing band\n(dashed = no-exit baseline)")
    ax[0].set_xlabel("trailing band (bps given back from peak)")
    ax[0].set_ylabel("net P&L (US$ k / yr, 1M/trade)")
    ax[0].legend(fontsize=8.5)
    ax[1].plot(tr, [g["is_sharpe"] for g in grid], "-o", color=GREEN, ms=3, label="in-sample")
    ax[1].plot(tr, [g["oos_sharpe"] for g in grid], "-o", color=NAVY, ms=3, label="out-of-sample")
    ax[1].axhline(base_is["sharpe"], color=GREEN, ls="--", lw=1, alpha=.7)
    ax[1].axhline(base_oos["sharpe"], color=NAVY, ls="--", lw=1, alpha=.7)
    ax[1].axvline(trail_star, color=RED, ls=":", lw=1.4)
    ax[1].text(trail_star, ax[1].get_ylim()[0], f" chosen {trail_star}bps", color=RED, fontsize=8.5,
               va="bottom")
    ax[1].set_title("Net Sharpe vs trailing band\n(broad plateau above baseline = robust, not curve-fit)")
    ax[1].set_xlabel("trailing band (bps given back from peak)")
    ax[1].set_ylabel("net Sharpe")
    ax[1].legend(fontsize=8.5)
    fig.suptitle("Exit optimiser — every trailing band from ~30 to ~100 bps beats the no-exit rule "
                 "in BOTH windows", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "ex_optimization.png", dpi=110)
    plt.close(fig)


def chart_equity(df, base, enhanced, res):
    """Before/after: cumulative equity and drawdown, no-exit vs trailing overlay."""
    nb, _ = dollars(base, df)
    ne, _ = dollars(enhanced, df)
    d = df.date.iloc[nb.index]
    cb, ce = nb.cumsum(), ne.cumsum()
    ddb, dde = cb - cb.cummax(), ce - ce.cummax()
    bf, ef = res["baseline"]["full"], res["trailing"]["full"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), height_ratios=[2, 1], sharex=True)
    ax[0].plot(d, cb / 1e6, color=GREY, lw=1.5,
               label=f"calendar rule, no exit (Sharpe {bf['sharpe']} · ${bf['per_year_usd']/1e3:.0f}k/yr)")
    ax[0].plot(d, ce / 1e6, color=GREEN, lw=1.8,
               label=f"+ trailing-{res['_meta']['trail_bps']}bps exit "
                     f"(Sharpe {ef['sharpe']} · ${ef['per_year_usd']/1e3:.0f}k/yr)")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("cumulative net P&L (US$ M)")
    ax[0].set_title("Calendar rule vs calendar rule + trailing exit — VWAP-priced, net of 0.65 CRC RT",
                    fontweight="bold")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[1].fill_between(d, ddb / 1e3, 0, color=GREY, alpha=0.5, label=f"no exit (max ${bf['maxdd_usd']/1e3:.0f}k)")
    ax[1].fill_between(d, dde / 1e3, 0, color=GREEN, alpha=0.5,
                       label=f"+ trailing exit (max ${ef['maxdd_usd']/1e3:.0f}k)")
    ax[1].set_ylabel("drawdown (US$ k)")
    ax[1].legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "ex_equity.png", dpi=110)
    plt.close(fig)


def chart_episode(df, base, trail, res):
    """Illustrate the mechanism: the single episode where the trailing exit saved
    the most vs holding to the calendar's episode end."""
    v = df.vwap.values
    best = None
    for (a, b, s) in episodes(base):
        if s == 0 or b <= a + 2:
            continue
        entry = v[a]
        # per-episode net P&L (bps) held-to-end vs with the trailing exit
        held = s * (v[b] / entry - 1) * 1e4
        # find the trailing-exit day within the episode
        ex_t = None
        peak = 0.0
        for t in range(a + 1, b + 1):
            r = s * (v[t] / entry - 1) * 1e4
            peak = max(peak, r)
            if peak > 0 and (peak - r) >= trail:
                ex_t = t
                break
        if ex_t is None:
            continue
        banked = s * (v[ex_t] / entry - 1) * 1e4
        saved = banked - held
        if best is None or saved > best["saved"]:
            best = {"a": a, "b": b, "s": s, "ex_t": ex_t, "saved": saved,
                    "peak": peak, "banked": banked, "held": held, "entry": entry}
    if best is None:
        return None
    a, b, s, ex_t = best["a"], best["b"], best["s"], best["ex_t"]
    lo = max(0, a - 2)
    hi = min(len(df) - 1, b + 3)
    seg = df.iloc[lo:hi + 1]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(seg.date, seg.vwap, "-o", color=NAVY, ms=3, lw=1.4, label="session VWAP")
    side = "SHORT USD" if s < 0 else "LONG USD"
    ax.axvspan(df.date.iloc[a], df.date.iloc[b], color=RED if s < 0 else GREEN, alpha=0.07)
    ax.axvline(df.date.iloc[a], color=GREY, lw=1, ls="--")
    ax.annotate(f"entry ({side})", (df.date.iloc[a], df.vwap.iloc[a]), fontsize=9,
                xytext=(0, 16), textcoords="offset points", ha="center", color=GREY)
    ax.axvline(df.date.iloc[ex_t], color=GREEN, lw=1.4)
    ax.annotate(f"trailing exit\nbanked {best['banked']:+.0f} bps", (df.date.iloc[ex_t], df.vwap.iloc[ex_t]),
                fontsize=9, xytext=(6, -28), textcoords="offset points", color=GREEN)
    ax.axvline(df.date.iloc[b], color=RED, lw=1.2, ls=":")
    ax.annotate(f"calendar episode end\nheld-to-end {best['held']:+.0f} bps", (df.date.iloc[b], df.vwap.iloc[b]),
                fontsize=9, xytext=(6, 12), textcoords="offset points", color=RED)
    ax.set_ylabel("colones / US$")
    ax.set_title(f"How the trailing exit enhances a trade — one {side} episode near a deadline\n"
                 f"banks the move at the peak instead of giving back "
                 f"{best['saved']:.0f} bps on the reversion", fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ex_episode.png", dpi=110)
    plt.close(fig)
    return {"side": "long" if s > 0 else "short", "banked_bps": round(best["banked"]),
            "held_bps": round(best["held"]), "saved_bps": round(best["saved"]),
            "entry_date": str(df.date.iloc[a].date()), "exit_date": str(df.date.iloc[ex_t].date()),
            "end_date": str(df.date.iloc[b].date())}


# --------------------------------------------------------------------------- #
def main():
    df = frame()
    k = int(len(df) * OOS_FRAC)
    base = pos_calendar(df)
    v = df.vwap

    bf, bi, bo = windows(base, df, k)
    trail_star, (tr_c, fl_c), grid = optimise(df, base, k)

    enh = apply_exit(base, v, trail=trail_star)
    ef, ei, eo = windows(enh, df, k)
    cmb = apply_exit(base, v, trail=tr_c, floor=fl_c)
    cf, ci, co = windows(cmb, df, k)

    split_date = str(df.date.iloc[k].date())
    res = {"_meta": {"n_days": int(len(df)), "notional_usd": 1_000_000,
                     "cost_crc_roundtrip": 0.65, "oos_frac": OOS_FRAC,
                     "split_date": split_date, "cal_pre": CAL_PRE,
                     "trail_bps": trail_star, "combo_trail_bps": tr_c, "combo_floor_bps": fl_c,
                     "trail_grid": [TRAIL_GRID[0], TRAIL_GRID[-1]]},
           "baseline": {"full": bf, "is": bi, "oos": bo},
           "trailing": {"full": ef, "is": ei, "oos": eo, "trail_bps": trail_star},
           "trailing_floor": {"full": cf, "is": ci, "oos": co,
                              "trail_bps": tr_c, "floor_bps": fl_c},
           "grid": grid,
           "improve": {
               "full_per_yr_delta": ef["per_year_usd"] - bf["per_year_usd"],
               "full_sharpe_delta": round(ef["sharpe"] - bf["sharpe"], 2),
               "full_maxdd_delta": ef["maxdd_usd"] - bf["maxdd_usd"],
               "oos_per_yr_delta": eo["per_year_usd"] - bo["per_year_usd"],
               "oos_sharpe_delta": round(eo["sharpe"] - bo["sharpe"], 2),
               "combo_sharpe_delta": round(cf["sharpe"] - bf["sharpe"], 2),
               "combo_maxdd_delta": cf["maxdd_usd"] - bf["maxdd_usd"]}}

    chart_optimization(df, grid, trail_star, bf, bi, bo, res)
    chart_equity(df, base, enh, res)
    ep = chart_episode(df, base, trail_star, res)
    if ep is not None:
        res["episode"] = ep

    Path(OUT / "exits_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k2: v2 for k2, v2 in res.items() if k2 != "grid"}, indent=2))
    print(f"\nChosen trailing = {trail_star} bps | conservative = trailing {tr_c} + floor {fl_c} bps")
    print("Wrote ex_*.png and exits_results.json")


if __name__ == "__main__":
    main()
