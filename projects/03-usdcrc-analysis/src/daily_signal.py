"""One-page daily signal sheet for the recommended refined-quincena strategy.

Given a date (default: the day after the last row in the data, i.e. the next
tradeable session), it prints the position the refined-quincena rule takes plus
the slow-volume size multiplier, in plain language and in dollars at USD 1M/trade.

The logic mirrors quincena.py EXACTLY (pos_refined + slow-volume sizing) so the
sheet can never drift from the backtested strategy:

    day 1-4    -> LONG USD   (start-of-month, USD tends up)
    day 5-15   -> SHORT USD  (mid-month USD-supply surge, colon strengthens)
    day 16-end -> LONG USD   (supply fades, USD drifts up)

    size 1.0 when the slow (20d) seasonally-adjusted volume regime CONFIRMS,
    else 0.5 (trim when a slow volume regime disagrees with the calendar).

Usage:
    python src/daily_signal.py                 # signal for the next session
    python src/daily_signal.py 2026-06-10      # signal for a specific date
    python src/daily_signal.py --png           # also render out/daily_signal.png

Writes out/daily_signal.txt (always) and out/daily_signal.png (with --png).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import OUT, load
from quincena import SHORT_END, SHORT_START, NOTIONAL

DOW_NAME = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def build_history() -> pd.DataFrame:
    """Trading-day history with the same volume features quincena.py uses."""
    df = load().copy()
    df["vol"] = df.volume_usd / 1e6
    df["dom"] = df.date.dt.day
    df["dow"] = df.date.dt.dayofweek
    # causal seasonally-adjusted volume: today vs trailing mean of same weekday
    df["vol_ref"] = df.groupby("dow").vol.transform(
        lambda s: s.shift(1).expanding(min_periods=10).mean())
    df["vol_adj"] = df.vol / df.vol_ref
    df["slow"] = df.vol_adj.rolling(20).mean()
    return df


def refined_position(dom: int) -> float:
    """+1 long USD / -1 short USD from the refined-quincena calendar."""
    return -1.0 if SHORT_START <= dom <= SHORT_END else 1.0


def size_multiplier(dom: int, slow: float) -> float:
    """1.0 when the slow 20d volume regime confirms the calendar, else 0.5.

    In the short-USD window we want supply present (high slow volume) to confirm;
    outside it we want a thin market (low slow volume). Disagreement -> half size.
    """
    if slow is None or np.isnan(slow):
        return 1.0
    in_short = SHORT_START <= dom <= SHORT_END
    confirms = (slow > 1) if in_short else (slow < 1)
    return 1.0 if confirms else 0.5


def signal_for(target: pd.Timestamp, hist: pd.DataFrame) -> dict:
    """Compute the signal for `target` using only data strictly before it."""
    past = hist[hist.date < target]
    if past.empty:
        raise SystemExit(f"No history before {target.date()}.")
    last = past.iloc[-1]
    slow = float(last.slow) if pd.notna(last.slow) else float("nan")

    dom = int(target.day)
    pos = refined_position(dom)
    size = size_multiplier(dom, slow)
    signed = pos * size
    in_short = SHORT_START <= dom <= SHORT_END

    return {
        "date": target,
        "dom": dom,
        "dow_name": DOW_NAME[target.dayofweek],
        "direction": "SHORT USD (sell USD / buy colon)" if pos < 0
        else "LONG USD (buy USD / sell colon)",
        "window": (f"mid-month supply window (days {SHORT_START}-{SHORT_END})" if in_short
                   else "outside the short window"),
        "size": size,
        "signed": signed,
        "notional_usd": int(round(NOTIONAL * abs(signed))),
        "slow_vol": slow,
        "slow_state": ("above normal (supply present)" if slow > 1
                       else "below normal (thin market)") if pd.notna(slow) else "n/a",
        "confirms": size == 1.0,
        "as_of": pd.Timestamp(last.date),
        "last_close": float(last.close),
        "last_vwap": float(last.vwap),
    }


def render_text(s: dict) -> str:
    bar = "=" * 60
    dir_short = "SHORT USD" if s["signed"] < 0 else "LONG USD"
    lines = [
        bar,
        "  USD/CRC DAILY SIGNAL  -  refined quincena (recommended)",
        bar,
        f"  Session date     : {s['date'].date()}  ({s['dow_name']})",
        f"  Day of month     : {s['dom']}  ->  {s['window']}",
        "",
        f"  POSITION         : {dir_short}",
        f"  Size             : {s['size']:.1f}x  "
        f"({'full - volume regime confirms' if s['confirms'] else 'half - volume regime disagrees'})",
        f"  Notional         : USD {s['notional_usd']:,}  (of USD {NOTIONAL:,} full)",
        "",
        f"  Slow 20d volume  : {s['slow_vol']:.2f}x normal  ({s['slow_state']})",
        f"  As of last data  : {s['as_of'].date()}  "
        f"(close {s['last_close']:.2f}, VWAP {s['last_vwap']:.2f})",
        bar,
        "  Rule: days 1-4 LONG USD | days 5-15 SHORT USD | days 16+ LONG USD;",
        "  half size when the slow 20d volume regime disagrees with the calendar.",
        bar,
    ]
    return "\n".join(lines)


def render_png(s: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = s["signed"] < 0
    accent = "#c0392b" if short else "#2e8b57"
    dir_short = "SHORT USD" if short else "LONG USD"
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.add_patch(plt.Rectangle((0, 0.82), 1, 0.18, transform=ax.transAxes,
                               color="#1f3b73", zorder=0))
    ax.text(0.5, 0.905, "USD/CRC DAILY SIGNAL", ha="center", va="center",
            color="white", fontsize=15, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.855, f"refined quincena  |  {s['date'].date()} ({s['dow_name']})",
            ha="center", va="center", color="#dfe6f2", fontsize=9.5, transform=ax.transAxes)

    ax.text(0.5, 0.66, dir_short, ha="center", va="center", color=accent,
            fontsize=34, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.55, f"{s['size']:.1f}x  ->  USD {s['notional_usd']:,}",
            ha="center", va="center", color="#333", fontsize=13, transform=ax.transAxes)

    rows = [
        ("Day of month", f"{s['dom']}  ({s['window']})"),
        ("Size reason", "volume regime confirms" if s["confirms"]
         else "volume regime disagrees -> half size"),
        ("Slow 20d volume", f"{s['slow_vol']:.2f}x normal  ({s['slow_state']})"),
        ("As of last data", f"{s['as_of'].date()}  close {s['last_close']:.2f}"),
    ]
    y = 0.40
    for k, v in rows:
        ax.text(0.06, y, k, ha="left", va="center", color="#7f8c8d",
                fontsize=9.5, transform=ax.transAxes)
        ax.text(0.40, y, v, ha="left", va="center", color="#222",
                fontsize=9.5, transform=ax.transAxes)
        y -= 0.085
    ax.text(0.5, 0.03,
            "days 1-4 LONG | 5-15 SHORT | 16+ LONG  -  half size when slow volume disagrees",
            ha="center", va="center", color="#7f8c8d", fontsize=7.5, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = [a for a in sys.argv[1:]]
    want_png = "--png" in args
    date_args = [a for a in args if not a.startswith("--")]

    hist = build_history()
    if date_args:
        target = pd.Timestamp(date_args[0])
    else:
        # next weekday after the last data row (next candidate MONEX session)
        target = pd.Timestamp(hist.date.iloc[-1]) + pd.Timedelta(days=1)
        while target.dayofweek >= 5:            # skip Sat/Sun
            target += pd.Timedelta(days=1)

    s = signal_for(target, hist)
    txt = render_text(s)
    print(txt)
    (OUT / "daily_signal.txt").write_text(txt + "\n")
    if want_png:
        render_png(s, OUT / "daily_signal.png")
        print(f"\nWrote {OUT / 'daily_signal.png'}")


if __name__ == "__main__":
    main()
