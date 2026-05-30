"""
File-writing layer for user edits made from the dashboard.

Every public `save_*` makes a timestamped backup of the target file *before*
overwriting it, so a bad edit is always recoverable. Validation runs before any
backup or write, so a rejected save leaves the file untouched.

On-disk conventions (must match the readers in position_manager / bloomberg):
  - trades.csv          : nominal stored UNSIGNED; sign applied on load by side
  - dates               : written ISO (YYYY-MM-DD)
  - cusip               : string (leading zeros preserved)
"""
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from config import get_logger
from models import BondStatic

log = get_logger(__name__)

TRADES_COLUMNS = [
    "Timestamp", "cusip", "side", "nominal", "principal", "net", "accrued",
    "price", "yield_closed", "trade_date", "settle_date", "trader", "portfolio",
]
BONDS_COLUMNS = [
    "cusip", "name", "currency", "country", "coupon_rate", "coupon_frequency",
    "day_count_convention", "maturity_date", "first_coupon_date",
]
INITIAL_COLUMNS = ["portfolio", "cusip", "nominal", "price", "book_value", "inception_date"]
MANUAL_PRICES_COLUMNS = ["cusip", "px_last", "date"]
MANUAL_HISTORY_COLUMNS = ["date", "cusip", "px_last"]


# ── backup ──────────────────────────────────────────────────────────────────

def backup_file(path: Path) -> Path | None:
    """Copy `path` to data/backups/<stem>_<YYYYmmdd_HHMMSS><suffix>; no-op if absent."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    config.BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = config.BACKUPS_DIR / f"{path.stem}_{ts}{path.suffix}"
    shutil.copy2(path, dest)
    log.info("Backed up %s → %s", path.name, dest)
    return dest


# ── date helpers ──────────────────────────────────────────────────────────────

def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse mm/dd/yy first, fall back to ISO/default for anything else."""
    parsed = pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    unresolved = parsed.isna() & series.notna() & (series.astype(str).str.strip() != "")
    if unresolved.any():
        parsed[unresolved] = pd.to_datetime(series[unresolved], errors="coerce")
    return parsed


def _iso(series: pd.Series) -> pd.Series:
    return _parse_dates(series).dt.strftime("%Y-%m-%d")


# ── validation ──────────────────────────────────────────────────────────────

def validate_trades(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems; empty list means OK to save."""
    problems: list[str] = []
    if df.empty:
        return problems
    for i, row in df.iterrows():
        n = i + 1
        cusip = str(row.get("cusip", "")).strip()
        if not cusip or cusip.lower() == "nan":
            problems.append(f"Row {n}: cusip is blank")
        side = str(row.get("side", "")).strip().lower()
        if side not in {"buy", "sell"}:
            problems.append(f"Row {n}: side {row.get('side')!r} must be 'buy' or 'sell'")
        for fld in ("nominal", "price"):
            try:
                float(row.get(fld))
            except (TypeError, ValueError):
                problems.append(f"Row {n}: {fld} {row.get(fld)!r} is not numeric")
        for fld in ("trade_date", "settle_date"):
            if pd.isna(pd.to_datetime(_one(row.get(fld)), errors="coerce")):
                problems.append(f"Row {n}: {fld} {row.get(fld)!r} is not a valid date")
    return problems


def _one(value):
    """Parse a single date-like value via the mm/dd/yy-first rule."""
    s = pd.Series([value])
    return _parse_dates(s).iloc[0]


def _validate_bonds(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    for i, row in df.iterrows():
        n = i + 1
        try:
            BondStatic(
                cusip=str(row["cusip"]),
                name=str(row.get("name", "")),
                currency=str(row.get("currency", "")),
                country=str(row.get("country", "")),
                coupon_rate=float(row["coupon_rate"]),
                coupon_frequency=int(row["coupon_frequency"]),
                day_count_convention=str(row["day_count_convention"]),
                maturity_date=_one(row["maturity_date"]).date(),
                first_coupon_date=_one(row["first_coupon_date"]).date(),
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            problems.append(f"Row {n} ({row.get('cusip')}): {exc}")
    return problems


# ── savers ──────────────────────────────────────────────────────────────────

def save_trades(df: pd.DataFrame) -> Path | None:
    """Validate, back up, then write trades.csv in on-disk convention."""
    problems = validate_trades(df)
    if problems:
        raise ValueError("Cannot save trades:\n" + "\n".join(problems))

    out = df.copy()
    out["cusip"] = out["cusip"].astype(str)
    out["side"] = out["side"].astype(str).str.lower().str.strip()
    out["nominal"] = pd.to_numeric(out["nominal"], errors="coerce").abs()  # store unsigned
    for col in ("trade_date", "settle_date"):
        out[col] = _iso(out[col])
    for col in TRADES_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[TRADES_COLUMNS]

    backup = backup_file(config.TRADES_PATH)
    config.TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.TRADES_PATH, index=False)
    log.info("Wrote %d trades to %s", len(out), config.TRADES_PATH)
    return backup


def save_bonds_static(df: pd.DataFrame) -> Path | None:
    """Validate every row via BondStatic, then back up and write bonds_static.csv."""
    problems = _validate_bonds(df)
    if problems:
        raise ValueError("Cannot save bond static:\n" + "\n".join(problems))

    out = df.copy()
    out["cusip"] = out["cusip"].astype(str)
    for col in ("maturity_date", "first_coupon_date"):
        out[col] = _iso(out[col])
    out = out[[c for c in BONDS_COLUMNS if c in out.columns]]

    backup = backup_file(config.BONDS_STATIC_PATH)
    out.to_csv(config.BONDS_STATIC_PATH, index=False)
    log.info("Wrote %d bonds to %s", len(out), config.BONDS_STATIC_PATH)
    return backup


def save_initial_positions(df: pd.DataFrame) -> Path | None:
    out = df.copy()
    out["cusip"] = out["cusip"].astype(str)
    if "inception_date" in out.columns:
        out["inception_date"] = _iso(out["inception_date"])
    out = out[[c for c in INITIAL_COLUMNS if c in out.columns]]

    backup = backup_file(config.INITIAL_POSITIONS_PATH)
    out.to_csv(config.INITIAL_POSITIONS_PATH, index=False)
    log.info("Wrote %d initial positions to %s", len(out), config.INITIAL_POSITIONS_PATH)
    return backup


def save_manual_prices(df: pd.DataFrame) -> Path | None:
    out = df.copy()
    out["cusip"] = out["cusip"].astype(str)
    if "date" in out.columns:
        out["date"] = _iso(out["date"])
    out = out[[c for c in MANUAL_PRICES_COLUMNS if c in out.columns]]

    config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    backup = backup_file(config.MANUAL_PRICES_PATH)
    out.to_csv(config.MANUAL_PRICES_PATH, index=False)
    log.info("Wrote %d manual prices to %s", len(out), config.MANUAL_PRICES_PATH)
    return backup


def save_manual_price_history(df: pd.DataFrame) -> Path | None:
    out = df.copy()
    out["cusip"] = out["cusip"].astype(str)
    if "date" in out.columns:
        out["date"] = _iso(out["date"])
    out = out[[c for c in MANUAL_HISTORY_COLUMNS if c in out.columns]]

    config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    backup = backup_file(config.MANUAL_HISTORY_PATH)
    out.to_csv(config.MANUAL_HISTORY_PATH, index=False)
    log.info("Wrote %d manual price-history rows to %s", len(out), config.MANUAL_HISTORY_PATH)
    return backup
