"""
Data-quality / reconciliation engine — powers the Debug ▸ Needs-Attention view.

The goal is total transparency: every trade, bond, and price that is broken or
incomplete is surfaced in one place with a severity, the exact field at fault,
and a suggested fix. Nothing is silently coerced away.

A *finding* is one dict with this shape:

    severity      "error" | "warning" | "needs_input"
    category      "Trade" | "Bond Static" | "Price"
    source        which file the problem lives in (e.g. "trades.csv")
    key           the row's identifier (cusip, or "row N")
    field         the offending column ("" if row-level)
    issue         human-readable description of what's wrong
    suggested_fix what to do about it, and where

`run_all_checks()` returns a single tidy DataFrame of all findings plus a
summary count by severity. Checks accept their inputs as arguments (with loader
defaults) so they are pure and unit-testable.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

import config
from classification import classify, UNKNOWN
from models import BondStatic

# ── tuning knobs ──────────────────────────────────────────────────────────────

# A bond clean price normally trades roughly within these bounds (% of par).
# Outside → flagged as "implausible / check price".
PRICE_MIN_PLAUSIBLE = 20.0
PRICE_MAX_PLAUSIBLE = 200.0

# Relative tolerance for the trade economic recompute checks.
ECON_REL_TOL = 0.01      # 1%
ECON_ABS_TOL = 1.0       # or within $1, whichever is looser

# A price repeated unchanged across at least this many recent dates is flagged
# as possibly stale (Bloomberg not updating / weekend carried forward).
STALE_REPEAT_DAYS = 5

FINDING_COLUMNS = ["severity", "category", "source", "key", "field", "issue", "suggested_fix"]


def _finding(severity, category, source, key, field, issue, suggested_fix) -> dict:
    return {
        "severity": severity, "category": category, "source": source,
        "key": str(key), "field": field, "issue": issue, "suggested_fix": suggested_fix,
    }


# ── trade checks ──────────────────────────────────────────────────────────────

def check_trades(
    raw_trades: pd.DataFrame,
    bonds_static: dict[str, BondStatic],
    known_cusips: set[str] | None = None,
) -> list[dict]:
    """Validate the raw trades CSV: structure, parseability, and economics.

    Operates on the *raw* frame (strings as written) so parse failures are
    caught rather than hidden by a tolerant loader.

    ``known_cusips`` is the set of CUSIPs that *physically appear* in
    bonds_static.csv (before parsing/validation). It lets us tell a genuinely
    missing bond apart from one whose row exists but failed to load (e.g. a
    blank ``maturity_date``) — the two need different fixes, so conflating them
    sends the user to add a bond that is already there.
    """
    findings: list[dict] = []
    if raw_trades is None or raw_trades.empty:
        return findings

    for i, row in raw_trades.iterrows():
        n = i + 1
        key = f"row {n}"
        cusip = str(row.get("cusip", "")).strip()

        if not cusip or cusip.lower() == "nan":
            findings.append(_finding(
                "error", "Trade", "trades.csv", key, "cusip",
                "CUSIP is blank", "Add the 9-char CUSIP in Data Editor ▸ Trades."))
        elif cusip not in bonds_static:
            if known_cusips is not None and cusip in known_cusips:
                # Row is present but did not load — almost always a missing
                # maturity_date (the one field with no fallback in the loader).
                findings.append(_finding(
                    "warning", "Trade", "trades.csv", cusip, "cusip",
                    f"Bond static row for {cusip} exists but is incomplete "
                    f"(it failed to load — usually a missing maturity_date)",
                    "Complete the bond's static fields in Data Editor ▸ Bond Static "
                    "(maturity_date is required)."))
            else:
                findings.append(_finding(
                    "warning", "Trade", "trades.csv", cusip, "cusip",
                    f"Trade references CUSIP {cusip} with no bond static row",
                    "Add the bond in Data Editor ▸ Bond Static (or run Bloomberg import)."))

        side = str(row.get("side", "")).strip().lower()
        if side not in ("buy", "sell"):
            findings.append(_finding(
                "error", "Trade", "trades.csv", key, "side",
                f"side {row.get('side')!r} is not buy/sell",
                "Set Side to 'buy' or 'sell'."))

        # numeric parseability
        nums: dict[str, float] = {}
        for fld in ("nominal", "price", "principal", "net", "accrued"):
            raw_val = row.get(fld)
            try:
                nums[fld] = float(raw_val)
            except (TypeError, ValueError):
                if fld in ("nominal", "price"):
                    findings.append(_finding(
                        "error", "Trade", "trades.csv", key, fld,
                        f"{fld} {raw_val!r} is not numeric", f"Enter a number for {fld}."))

        # dates
        td = pd.to_datetime(row.get("trade_date"), errors="coerce")
        sd = pd.to_datetime(row.get("settle_date"), errors="coerce")
        if pd.isna(td):
            findings.append(_finding(
                "error", "Trade", "trades.csv", key, "trade_date",
                f"trade_date {row.get('trade_date')!r} is not a valid date",
                "Use mm/dd/yy or ISO YYYY-MM-DD."))
        if pd.isna(sd):
            findings.append(_finding(
                "error", "Trade", "trades.csv", key, "settle_date",
                f"settle_date {row.get('settle_date')!r} is not a valid date",
                "Use mm/dd/yy or ISO YYYY-MM-DD."))
        if not pd.isna(td) and not pd.isna(sd) and sd < td:
            findings.append(_finding(
                "warning", "Trade", "trades.csv", key, "settle_date",
                f"settle_date ({sd.date()}) is before trade_date ({td.date()})",
                "Check the settlement date (usually T+1/T+2)."))

        # economic recompute: principal ≈ price × |nominal| / 100
        if "price" in nums and "nominal" in nums and "principal" in nums:
            expected_principal = nums["price"] * abs(nums["nominal"]) / 100
            if not _close(nums["principal"], expected_principal):
                findings.append(_finding(
                    "warning", "Trade", "trades.csv", key, "principal",
                    f"principal {nums['principal']:,.2f} ≠ price×nominal/100 "
                    f"({expected_principal:,.2f})",
                    "Re-check the parsed price/nominal/principal for this clip."))

        # |net| ≈ principal + accrued. net is stored UNSIGNED in the CSV (the
        # sign is applied on load by side, like nominal), so only the magnitude
        # is validated here — a positive net on a buy is correct, not an error.
        if "principal" in nums and "net" in nums:
            accr = nums.get("accrued", 0.0)
            expected_net_abs = abs(nums["principal"]) + accr
            if not _close(abs(nums["net"]), expected_net_abs):
                findings.append(_finding(
                    "warning", "Trade", "trades.csv", key, "net",
                    f"|net| {abs(nums['net']):,.2f} ≠ principal+accrued "
                    f"({expected_net_abs:,.2f})",
                    "Verify accrued and net cash on the confirmation."))

    return findings


# ── bond static checks ────────────────────────────────────────────────────────

# Classification dimensions checked by default. ``market`` is intentionally
# excluded: it is derived heuristically (domestic-vs-hard currency) and is often
# left blank on the Bloomberg Static sheet, so flagging it is noise. Credit
# ratings are likewise optional (see ``require_ratings``). Both can be re-enabled
# by the caller when a stricter, fully-classified book is required.
_REQUIRED_CLASSIFICATION = ("instrument_type", "country_of_risk", "sector")


def check_bonds(
    bonds_static: dict[str, BondStatic],
    held_cusips: set[str] | None = None,
    *,
    require_ratings: bool = False,
    require_market: bool = False,
) -> list[dict]:
    """Flag bonds missing required or classification fields (needs manual input).

    If ``held_cusips`` is given, missing fields on held bonds are escalated
    (they directly affect live P&L/exposure); reference-only bonds stay low-key.

    ``require_ratings`` / ``require_market`` are off by default: credit ratings
    and the Local/Global market dimension are treated as optional reference data
    (commonly blank on the Bloomberg Static sheet) and are not flagged unless the
    caller opts in.
    """
    findings: list[dict] = []
    held = held_cusips or set()

    fields = list(_REQUIRED_CLASSIFICATION)
    if require_market:
        fields.append("market")

    for cusip, bond in bonds_static.items():
        # Name missing — the most common "bond doesn't show a name" case.
        if not (bond.name or "").strip():
            findings.append(_finding(
                "needs_input", "Bond Static", "bonds_static.csv", cusip, "name",
                "Bond has no name", "Add the security name (or run Bloomberg import)."))

        # day-count present but maturity is implicitly required at load; here we
        # also surface a coupon-bearing bond with no coupon rate.
        if bond.coupon_frequency and bond.coupon_rate == 0.0:
            findings.append(_finding(
                "warning", "Bond Static", "bonds_static.csv", cusip, "coupon_rate",
                "Coupon frequency set but coupon rate is 0 — accrual will be zero",
                "Enter the coupon rate (decimal, e.g. 0.05)."))

        # classification dimensions (Unknown after derivation → needs input)
        derived = classify(bond)
        for fld in fields:
            if derived.get(fld) == UNKNOWN:
                sev = "needs_input" if cusip in held else "warning"
                findings.append(_finding(
                    sev, "Bond Static", "bonds_static.csv", cusip, fld,
                    f"{fld.replace('_', ' ')} could not be determined",
                    f"Set {fld} in Data Editor ▸ Bond Static (or fetch from Bloomberg)."))

        if require_ratings and not any(
            getattr(bond, r, "") for r in ("rating_sp", "rating_moody", "rating_fitch")
        ):
            findings.append(_finding(
                "needs_input", "Bond Static", "bonds_static.csv", cusip, "rating",
                "No credit rating on file (S&P / Moody's / Fitch)",
                "Add at least one agency rating for risk reporting."))

    return findings


# ── price checks ──────────────────────────────────────────────────────────────

def check_prices(
    held_cusips: set[str],
    current_prices: dict[str, float],
    price_history: pd.DataFrame,
    bonds_static: dict[str, BondStatic] | None = None,
    as_of: date | None = None,
) -> list[dict]:
    """Flag missing, implausible ('weird'), and stale prices for held bonds.

    Bonds that have matured on/before ``as_of`` are skipped: a redeemed/expired
    bond has no live market price (Bloomberg returns #N/A for it), so flagging a
    "missing price" for it would be a false positive.
    """
    findings: list[dict] = []
    bonds_static = bonds_static or {}
    as_of = as_of or date.today()

    def _name(c: str) -> str:
        b = bonds_static.get(c)
        return (b.name if b and b.name else c)

    for cusip in sorted(held_cusips):
        bond = bonds_static.get(cusip)
        if bond is not None and bond.is_matured(as_of):
            continue  # matured/expired → no live price expected
        px = current_prices.get(cusip)
        if px is None:
            findings.append(_finding(
                "warning", "Price", "current prices", cusip, "px_last",
                f"No current price for held bond {_name(cusip)}",
                "Run Bloomberg ▸ Import, or add a manual price in Data Editor."))
            continue
        try:
            pxf = float(px)
        except (TypeError, ValueError):
            findings.append(_finding(
                "error", "Price", "current prices", cusip, "px_last",
                f"Price {px!r} is not numeric", "Fix the price value."))
            continue

        if pxf <= 0:
            findings.append(_finding(
                "error", "Price", "current prices", cusip, "px_last",
                f"Price {pxf} is non-positive", "Check the price source — 0/negative is invalid."))
        elif pxf < PRICE_MIN_PLAUSIBLE or pxf > PRICE_MAX_PLAUSIBLE:
            findings.append(_finding(
                "warning", "Price", "current prices", cusip, "px_last",
                f"Price {pxf:g} is outside the plausible range "
                f"[{PRICE_MIN_PLAUSIBLE:g}, {PRICE_MAX_PLAUSIBLE:g}] for a bond",
                "Verify — could be a yield/spread mis-mapped into the price cell."))

    # stale detection from price history: same px repeated across recent dates.
    # Matured bonds are excluded — a redeemed bond's last price legitimately
    # carries forward and is not "stale".
    live_cusips = {
        c for c in held_cusips
        if not (bonds_static.get(c) and bonds_static[c].is_matured(as_of))
    }
    findings.extend(_check_stale(live_cusips, price_history, bonds_static))
    return findings


def _check_stale(held_cusips, price_history, bonds_static) -> list[dict]:
    out: list[dict] = []
    if price_history is None or price_history.empty:
        return out
    ph = price_history.copy()
    ph["date"] = pd.to_datetime(ph["date"], errors="coerce")
    for cusip in sorted(held_cusips):
        grp = ph[ph["cusip"] == cusip].sort_values("date")
        if len(grp) < STALE_REPEAT_DAYS:
            continue
        recent = grp["px_last"].tail(STALE_REPEAT_DAYS).round(6)
        if recent.nunique() == 1:
            name = bonds_static.get(cusip).name if bonds_static.get(cusip) else cusip
            out.append(_finding(
                "warning", "Price", "price_history.csv", cusip, "px_last",
                f"Price unchanged at {recent.iloc[-1]:g} for the last "
                f"{STALE_REPEAT_DAYS} records ({name}) — possibly stale",
                "Confirm the bond is still actively priced by Bloomberg."))
    return out


# ── orchestration ─────────────────────────────────────────────────────────────

def run_all_checks(
    raw_trades: pd.DataFrame | None = None,
    bonds_static: dict[str, BondStatic] | None = None,
    held_cusips: set[str] | None = None,
    current_prices: dict[str, float] | None = None,
    price_history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Run every check and return (findings_df, summary_counts_by_severity).

    All inputs are optional; missing ones are loaded from disk so the function
    works both standalone (CLI) and wired into the dashboard with preloaded data.
    """
    if bonds_static is None:
        from accruals import load_bonds_static
        bonds_static = load_bonds_static()
    if raw_trades is None:
        raw_trades = _read_raw_trades()
    if held_cusips is None:
        from position_manager import load_all_trades, compute_positions
        positions = compute_positions(load_all_trades())
        held_cusips = {c for c, p in positions.items() if p.net_nominal != 0}
    if current_prices is None:
        from bloomberg import load_latest_prices
        current_prices = load_latest_prices()
    if price_history is None:
        from bloomberg import load_price_history
        price_history = load_price_history()

    known_cusips = _read_known_cusips()

    findings: list[dict] = []
    findings += check_trades(raw_trades, bonds_static, known_cusips=known_cusips)
    findings += check_bonds(bonds_static, held_cusips)
    findings += check_prices(held_cusips, current_prices, price_history, bonds_static)

    df = pd.DataFrame(findings, columns=FINDING_COLUMNS)
    summary = {sev: int((df["severity"] == sev).sum()) for sev in ("error", "warning", "needs_input")}
    summary["total"] = len(df)
    return df, summary


def _read_raw_trades(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else config.TRADES_PATH
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"cusip": str})


def _read_known_cusips(path: Path | None = None) -> set[str]:
    """CUSIPs physically present in bonds_static.csv, regardless of whether the
    row parses into a valid BondStatic. Used to distinguish a missing bond from
    an incomplete one in the trade checks."""
    path = Path(path) if path is not None else config.BONDS_STATIC_PATH
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, dtype={"cusip": str})
    if "cusip" not in df.columns:
        return set()
    return {str(c).strip() for c in df["cusip"].dropna() if str(c).strip()}


def _close(actual: float, expected: float) -> bool:
    """True when actual ≈ expected within the configured rel/abs tolerance."""
    return abs(actual - expected) <= max(ECON_ABS_TOL, ECON_REL_TOL * abs(expected))
