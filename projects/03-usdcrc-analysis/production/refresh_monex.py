"""Best-effort daily refresh of MONEX price/volume data from BCCR.

Fetches the "Resumen de negociación en MONEX" export (CodCuadro=770 — the exact
HTML-as-XLS matrix ``parse_monex.py`` reads), parses a trailing window, and
UPSERTS the rows into ``data/monex_clean.csv`` by date. The deep history the
backtests rely on is preserved: only recent rows are added or overwritten.

Designed to never raise into the caller. If anything goes wrong (offline, VPN
blocks BCCR, the site is down, a weekend with no new session), it returns a
``RefreshResult`` with ``ok=False`` and a human-readable message, and the
notifier falls back to whatever ``monex_clean.csv`` already holds — the calendar
direction is unaffected; at worst the size multiplier is one session stale.

Uses only the standard library for the download (``urllib``), so the production
install stays at pandas+numpy — no ``requests`` dependency.
"""
from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from datetime import date

import pandas as pd

import config

sys.path.insert(0, str(config.SRC_DIR))
import parse_monex  # noqa: E402  (needs SRC_DIR on the path first)


@dataclass
class RefreshResult:
    ok: bool
    message: str
    n_new_rows: int = 0
    last_date: str | None = None


_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) usdcrc-daily-signal/1.0"


def _fetch(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted host)
        return resp.read()


def _existing() -> pd.DataFrame | None:
    clean = config.DATA_DIR / "monex_clean.csv"
    if not clean.exists():
        return None
    return pd.read_csv(clean, parse_dates=["date"])


def refresh() -> RefreshResult:
    """Fetch the recent MONEX window and upsert it into monex_clean.csv."""
    clean_path = config.DATA_DIR / "monex_clean.csv"
    existing = _existing()

    # Start a little before the last row we have (to catch late revisions), or at
    # the series origin if there is no local file yet.
    if existing is not None and not existing.empty:
        start = (existing["date"].max() - pd.Timedelta(days=config.REFRESH_LOOKBACK_DAYS)).date()
    else:
        start = date(2014, 12, 1)
    end = date.today()

    url = config.MONEX_EXPORT_URL.format(
        start=start.strftime("%Y/%m/%d"), end=end.strftime("%Y/%m/%d")
    )

    try:
        raw = _fetch(url, config.HTTP_TIMEOUT_SECONDS)
    except Exception as exc:  # network/timeout/HTTP — degrade gracefully
        return RefreshResult(False, f"download failed: {exc!r}")

    # BCCR serves windows-1252; the parser re-reads with that codec, so write bytes.
    tmp = config.DATA_DIR / "monex_raw_prod.xls"
    try:
        tmp.write_bytes(raw)
        fetched = parse_monex.parse(tmp)
    except Exception as exc:
        return RefreshResult(False, f"parse failed: {exc!r}")
    finally:
        tmp.unlink(missing_ok=True)

    fetched = fetched.dropna(subset=["date"])
    if fetched.empty:
        return RefreshResult(False, "export contained no parseable rows")

    if existing is None:
        merged = fetched
    else:
        # Upsert: fetched rows win over stale local rows for the same date.
        old = existing[~existing["date"].isin(set(fetched["date"]))]
        merged = pd.concat([old, fetched], ignore_index=True)
    merged = merged.sort_values("date").reset_index(drop=True)

    prev_last = existing["date"].max() if existing is not None and not existing.empty else None
    merged.to_csv(clean_path, index=False)

    new_last = merged["date"].max()
    n_new = int((merged["date"] > prev_last).sum()) if prev_last is not None else len(merged)
    return RefreshResult(
        True,
        f"refreshed via BCCR CodCuadro=770 ({start}..{end})",
        n_new_rows=n_new,
        last_date=str(pd.Timestamp(new_last).date()),
    )


if __name__ == "__main__":
    r = refresh()
    print(("OK  " if r.ok else "WARN") + f"  {r.message}")
    if r.ok:
        print(f"     last session in data: {r.last_date}  (+{r.n_new_rows} new rows)")
