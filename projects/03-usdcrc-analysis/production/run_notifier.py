"""Daily USD/CRC signal notifier — the single entry point Task Scheduler runs.

Pipeline (all steps best-effort and logged; a scheduled run never dies silently):

  1. optionally refresh MONEX data from BCCR (refresh_monex),
  2. compute the next-session signal by reusing daily_signal.signal_for /
     render_text verbatim, so the notification can never drift from the
     backtested rule,
  3. write out/daily_signal.txt (+ optional PNG),
  4. deliver it via Outlook and/or a popup per config.NOTIFY_CHANNEL,
  5. append a one-line result to production/logs/notifier.log.

Exit code 0 on success, 1 if the signal could not be computed at all.

    python production/run_notifier.py                 # next session
    python production/run_notifier.py 2026-06-10       # a specific date
    python production/run_notifier.py --no-refresh     # skip the BCCR fetch
    python production/run_notifier.py --channel popup   # override config for one run
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

import config

sys.path.insert(0, str(config.SRC_DIR))
import daily_signal  # noqa: E402

log = logging.getLogger("usdcrc.notifier")


def _setup_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-4s  %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(config.LOG_DIR / "notifier.log", maxBytes=512_000, backupCount=3)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def _pick_target(hist: pd.DataFrame, date_arg: str | None) -> pd.Timestamp:
    if date_arg:
        return pd.Timestamp(date_arg)
    # next weekday after the last data row (mirrors daily_signal.main)
    target = pd.Timestamp(hist.date.iloc[-1]) + pd.Timedelta(days=1)
    while target.dayofweek >= 5:
        target += pd.Timedelta(days=1)
    return target


def _subject(s: dict) -> str:
    dir_short = "SHORT USD" if s["signed"] < 0 else "LONG USD"
    return (f"{config.EMAIL_SUBJECT_PREFIX} {s['date'].date()}: "
            f"{dir_short} {s['size']:.1f}x (USD {s['notional_usd']:,})")


def _deliver(channel: str, subject: str, body: str, png: Path | None) -> None:
    import notify

    channel = channel.lower()
    attach = png if (png and png.exists()) else None

    if channel == "none":
        log.info("channel=none: notification skipped (artifacts + log only)")
        return

    did_outlook = False
    if channel in ("outlook", "both", "auto"):
        ok, msg = notify.send_outlook(subject, body, to=config.EMAIL_TO, attachment=attach)
        log.info("outlook: %s", msg)
        did_outlook = ok
        if channel == "auto" and ok:
            return  # auto = Outlook if it worked, else fall through to popup

    if channel == "popup" or channel == "both" or (channel == "auto" and not did_outlook):
        ok, msg = notify.show_popup(subject, body)
        log.info("popup: %s", msg)


def main() -> int:
    _setup_logging()
    args = sys.argv[1:]
    channel = config.NOTIFY_CHANNEL
    do_refresh = config.REFRESH_BEFORE_RUN
    date_arg = None
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == "--no-refresh":
            do_refresh = False
        elif a.startswith("--channel"):
            if "=" in a:
                channel = a.split("=", 1)[1]
            elif i + 1 < len(args):
                channel = args[i + 1]
                skip_next = True  # don't re-read the value as a date
        elif not a.startswith("--"):
            date_arg = a

    log.info("=== run start (channel=%s, refresh=%s) ===", channel, do_refresh)

    if do_refresh:
        try:
            import refresh_monex
            r = refresh_monex.refresh()
            log.info("refresh: %s%s", r.message,
                     f" -> last {r.last_date} (+{r.n_new_rows})" if r.ok else " [using local data]")
        except Exception as exc:
            log.warning("refresh crashed, using local data: %r", exc)

    # --- compute the signal (the one thing that must succeed) ---------------- #
    try:
        hist = daily_signal.build_history()
        target = _pick_target(hist, date_arg)
        s = daily_signal.signal_for(target, hist)
        body = daily_signal.render_text(s)
    except Exception as exc:
        log.error("SIGNAL COMPUTATION FAILED: %r", exc, exc_info=True)
        return 1

    print(body)
    (config.OUT_DIR / "daily_signal.txt").write_text(body + "\n")

    png: Path | None = None
    if config.ATTACH_PNG:
        try:
            png = config.OUT_DIR / "daily_signal.png"
            daily_signal.render_png(s, png)
        except Exception as exc:
            log.info("png skipped (matplotlib not installed?): %r", exc)
            png = None

    _deliver(channel, _subject(s), body, png)
    log.info("=== run ok: %s %s %.1fx USD %s ===",
             s["date"].date(), "SHORT" if s["signed"] < 0 else "LONG",
             s["size"], f"{s['notional_usd']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
