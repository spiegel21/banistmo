"""
File-writing layer for user edits made from the dashboard.

Every public `save_*` makes a timestamped backup of the target file *before*
overwriting it, so a bad edit is always recoverable. Validation runs before any
backup or write, so a rejected save leaves the file untouched.

On-disk conventions (must match the readers in position_manager / bloomberg):
  - trades.csv          : nominal and net stored UNSIGNED; sign applied on load
                          by side (buy: nominal +, net −; sell: nominal −, net +)
  - dates               : written ISO (YYYY-MM-DD)
  - cusip               : string (leading zeros preserved)
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from config import get_logger
from models import BondStatic, normalise_day_count, normalise_instrument_type

log = get_logger(__name__)

TRADES_COLUMNS = [
    "Timestamp", "cusip", "side", "nominal", "principal", "net", "accrued",
    "price", "yield_closed", "trade_date", "settle_date", "trader", "portfolio",
]
BONDS_COLUMNS = [
    "cusip", "name", "currency", "country", "coupon_rate", "coupon_frequency",
    "day_count_convention", "maturity_date", "first_coupon_date", "bbg_ticker",
    "isin", "instrument_type", "issuer", "country_of_risk", "sector", "seniority",
    "market", "rating_sp", "rating_moody", "rating_fitch",
]
INITIAL_COLUMNS = ["portfolio", "cusip", "nominal", "price", "book_value", "inception_date"]
MANUAL_PRICES_COLUMNS = ["cusip", "px_last", "date"]
MANUAL_HISTORY_COLUMNS = ["date", "cusip", "px_last"]
BOND_PORTFOLIO_MAP_COLUMNS = ["cusip", "portfolio"]


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


def parse_bond_static_row(row) -> BondStatic:
    """Parse one bonds_static row into a validated ``BondStatic``.

    Single source of the field fallbacks shared by the dashboard loader
    (``accruals.load_bonds_static``) and the editor validator
    (``_validate_bonds_detailed``), so both treat blank / None / ``#N/A``
    values identically:

      - coupon_frequency blank/#N/A     → 0 (non-coupon-bearing)
      - coupon_rate blank/None/#N/A     → 0.0; a 0 rate forces frequency 0
      - first_coupon_date blank/#N/A    → maturity_date
      - day_count_convention blank/None → "30/360"

    ``maturity_date`` is the one field with no fallback; a missing value raises
    ``ValueError`` so the caller can skip / report the row.
    """
    if pd.isna(row.get("maturity_date")):
        raise ValueError("missing required field 'maturity_date'")
    maturity = _one(row.get("maturity_date")).date()

    # coupon_frequency: "#N/A" / blank → 0
    try:
        coupon_freq = int(round(float(row.get("coupon_frequency"))))
    except (ValueError, TypeError):
        coupon_freq = 0

    # coupon_rate: blank / None / "#N/A" → 0.0 (zero-coupon); a 0 rate forces
    # frequency 0 so no phantom accrual is computed for a non-coupon bond.
    coupon_rate = pd.to_numeric(row.get("coupon_rate"), errors="coerce")
    coupon_rate = 0.0 if pd.isna(coupon_rate) else float(coupon_rate)
    if coupon_rate == 0.0:
        coupon_freq = 0

    # first_coupon_date: blank / "#N/A" → maturity_date
    fcd = _one(row.get("first_coupon_date"))
    first_coupon = maturity if pd.isna(fcd) else fcd.date()

    # day_count_convention: blank / None → 30/360 (irrelevant for zero-coupon)
    dcc_raw = row.get("day_count_convention")
    dcc_str = str(dcc_raw).strip() if pd.notna(dcc_raw) else ""
    dcc = normalise_day_count(dcc_str) if dcc_str else "30/360"

    def _s(field: str) -> str:
        """Blank-safe string read for an optional static column."""
        val = row.get(field)
        return "" if pd.isna(val) else str(val).strip()

    return BondStatic(
        cusip=str(row.get("cusip", "")),
        name="" if pd.isna(row.get("name")) else str(row.get("name", "")).strip(),
        currency="" if pd.isna(row.get("currency")) else str(row.get("currency", "")).strip(),
        country="" if pd.isna(row.get("country")) else str(row.get("country", "")).strip(),
        coupon_rate=coupon_rate,
        coupon_frequency=coupon_freq,
        day_count_convention=dcc,
        maturity_date=maturity,
        first_coupon_date=first_coupon,
        bbg_ticker="" if pd.isna(row.get("bbg_ticker")) else str(row.get("bbg_ticker", "")),
        isin=_s("isin").upper(),
        instrument_type=normalise_instrument_type(_s("instrument_type")),
        issuer=_s("issuer"),
        country_of_risk=_s("country_of_risk"),
        sector=_s("sector"),
        seniority=_s("seniority"),
        market=_s("market"),
        rating_sp=_s("rating_sp"),
        rating_moody=_s("rating_moody"),
        rating_fitch=_s("rating_fitch"),
    )


def _validate_bonds(df: pd.DataFrame) -> list[str]:
    """Return human-readable problems; empty list means every row is valid."""
    problems, _ = _validate_bonds_detailed(df)
    return problems


def _validate_bonds_detailed(df: pd.DataFrame) -> tuple[list[str], list]:
    """Like _validate_bonds but also returns the DataFrame index labels of bad rows."""
    problems: list[str] = []
    bad_index: list = []
    for i, row in df.iterrows():
        n = i + 1
        try:
            parse_bond_static_row(row)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            problems.append(f"Row {n} ({row.get('cusip')}): {exc}")
            bad_index.append(i)
    return problems, bad_index


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
    out["net"] = pd.to_numeric(out["net"], errors="coerce").abs()          # store unsigned (sign applied on load)
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


def save_bonds_static(df: pd.DataFrame, skip_invalid: bool = False) -> Path | None:
    """Validate every row via BondStatic, then back up and write bonds_static.csv.

    By default a single invalid row aborts the whole write (strict mode used by
    the Data Editor). With ``skip_invalid=True`` invalid rows are logged and kept
    in the file as-is rather than raising, so one unfetchable bond (e.g. a govt
    security where the default "CUSIP Corp" ticker returns #N/A) cannot block the
    static refresh of every other bond. Rows still missing required fields are
    skipped gracefully at load time by ``load_bonds_static``.
    """
    problems, bad_index = _validate_bonds_detailed(df)
    if problems:
        if not skip_invalid:
            raise ValueError("Cannot save bond static:\n" + "\n".join(problems))
        log.warning(
            "Writing bonds_static.csv with %d row(s) still missing required "
            "static data (kept as placeholders):\n%s",
            len(bad_index), "\n".join(problems),
        )

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


def save_bond_portfolio_map(df: pd.DataFrame) -> Path | None:
    """Write bond_portfolio_map.csv (cusip,portfolio).

    Only rows with both a cusip and a non-blank portfolio are kept — blank
    portfolios mean "let automatic resolution decide" and must not be persisted
    as pins. The last entry wins if a cusip is listed twice.
    """
    out = df.copy()
    if "cusip" not in out.columns:
        out["cusip"] = ""
    if "portfolio" not in out.columns:
        out["portfolio"] = ""
    # fillna("") first so empty selectbox cells (None/NaN) don't stringify to
    # the literal "None"/"nan" and survive the blank filter as bogus pins.
    out["cusip"] = out["cusip"].fillna("").astype(str).str.strip()
    out["portfolio"] = out["portfolio"].fillna("").astype(str).str.strip()
    _blank = {"", "nan", "none"}
    out = out[~out["cusip"].str.lower().isin(_blank)
              & ~out["portfolio"].str.lower().isin(_blank)]
    out = out[BOND_PORTFOLIO_MAP_COLUMNS].drop_duplicates(subset=["cusip"], keep="last")

    config.BOND_PORTFOLIO_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_file(config.BOND_PORTFOLIO_MAP_PATH)
    out.to_csv(config.BOND_PORTFOLIO_MAP_PATH, index=False)
    log.info("Wrote %d bond→portfolio pin(s) to %s", len(out), config.BOND_PORTFOLIO_MAP_PATH)
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
