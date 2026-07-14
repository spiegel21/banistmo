"""One-page daily signal sheet for the recommended calendar (quincena) strategy.

Given a date (default: the day after the last row in the data, i.e. the next
tradeable session), it prints the position the real-calendar rule takes plus
the slow-volume size multiplier, in plain language and in dollars at USD 1M/trade.

The logic mirrors quincena.py EXACTLY (pos_calendar + slow-volume sizing) so the
sheet can never drift from the backtested strategy. The position keys off the
business-day distance to the Costa Rica IVA (D-104) / mid-month quincena deadline
(the 15th, rolled to the next business day), NOT the raw day number:

    <= CAL_PRE business days before the deadline (through it) -> SHORT USD
        (firms sell USD to raise colones for the tax + payroll; colon strengthens)
    otherwise                                                 -> LONG USD
        (supply fades, USD drifts up)

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
from quincena import CAL_PRE, NOTIONAL
from payment_calendar import td_to_iva_for

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


def in_short_window(td_to_iva: int) -> bool:
    """True when `td_to_iva` business days to the deadline falls in the short-USD window."""
    return 0 <= td_to_iva <= CAL_PRE


def calendar_position(td_to_iva: int) -> float:
    """+1 long USD / -1 short USD from the real-calendar (deadline-anchored) rule."""
    return -1.0 if in_short_window(td_to_iva) else 1.0


def size_multiplier(td_to_iva: int, slow: float) -> float:
    """1.0 when the slow 20d volume regime confirms the calendar, else 0.5.

    In the short-USD window we want supply present (high slow volume) to confirm;
    outside it we want a thin market (low slow volume). Disagreement -> half size.
    """
    if slow is None or np.isnan(slow):
        return 1.0
    confirms = (slow > 1) if in_short_window(td_to_iva) else (slow < 1)
    return 1.0 if confirms else 0.5


def signal_for(target: pd.Timestamp, hist: pd.DataFrame) -> dict:
    """Compute the signal for `target` using only data strictly before it."""
    past = hist[hist.date < target]
    if past.empty:
        raise SystemExit(f"No history before {target.date()}.")
    last = past.iloc[-1]
    slow = float(last.slow) if pd.notna(last.slow) else float("nan")

    dom = int(target.day)
    td = td_to_iva_for(target)
    pos = calendar_position(td)
    size = size_multiplier(td, slow)
    signed = pos * size
    in_short = in_short_window(td)
    if in_short:
        window = f"{td} business day(s) to IVA/quincena deadline (short-USD window)"
    elif td < 0:
        window = f"{-td} business day(s) past the deadline (supply cleared)"
    else:
        window = f"{td} business days to the deadline (too early — long-USD)"

    return {
        "date": target,
        "dom": dom,
        "td_to_iva": td,
        "dow_name": DOW_NAME[target.dayofweek],
        "direction": "SHORT USD (sell USD / buy colon)" if pos < 0
        else "LONG USD (buy USD / sell colon)",
        "window": window,
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
        "  USD/CRC DAILY SIGNAL  -  real calendar quincena (recommended)",
        bar,
        f"  Session date     : {s['date'].date()}  ({s['dow_name']})",
        f"  Days to deadline : {s['td_to_iva']:+d} bd  ->  {s['window']}",
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
        f"  Rule: SHORT USD within {CAL_PRE} business days of the IVA/quincena",
        "  deadline (15th, rolled fwd), LONG otherwise; half size when the slow",
        "  20d volume regime disagrees with the calendar.",
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
    ax.text(0.5, 0.855, f"real calendar quincena  |  {s['date'].date()} ({s['dow_name']})",
            ha="center", va="center", color="#dfe6f2", fontsize=9.5, transform=ax.transAxes)

    ax.text(0.5, 0.66, dir_short, ha="center", va="center", color=accent,
            fontsize=34, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.55, f"{s['size']:.1f}x  ->  USD {s['notional_usd']:,}",
            ha="center", va="center", color="#333", fontsize=13, transform=ax.transAxes)

    rows = [
        ("Days to deadline", f"{s['td_to_iva']:+d} bd  ({s['window']})"),
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
            f"SHORT USD within {CAL_PRE} bd of the IVA/quincena deadline, LONG otherwise  -  "
            "half size when slow volume disagrees",
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
