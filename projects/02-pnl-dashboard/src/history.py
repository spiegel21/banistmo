"""
Historical daily P&L using the dirty-price change method.

For each business day in a range, per-CUSIP P&L is computed as:

    price_pnl  = (clean_today  - clean_prev ) × net_nominal_sod / 100
    accrual    = (accrued_today - accrued_prev) × net_nominal_sod / 100
    realized   = WAVG gains from trades that closed on this exact day
    total_pnl  = price_pnl + accrual + realized

net_nominal_sod is the start-of-day (= prior business day end-of-day) position.
Using SOD nominal means bonds sold today are included in the daily price move;
bonds bought today start contributing the following day (standard convention).

Previous-day clean price is sourced from price_history.csv. For the first day
of the range, if no prior-day price is in price_history, the `price` field from
initial_positions.csv is used as the Day-0 (e.g. April 30th) reference.

Cumulating daily rows produces the running P&L curve from inception to today.
Results are cached in pnl_history.csv.
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import config
from config import ALL_PORTFOLIOS, get_logger
from position_manager import load_all_trades, compute_positions
from accruals import load_bonds_static, accrued_interest, last_coupon_date, days_accrued
from trading_gains import realized_pnl
from mtm import mark_to_market
from bloomberg import load_price_history

log = get_logger(__name__)

_COLUMNS = [
    "date", "portfolio", "cusip", "net_nominal",
    "px_last", "price_pnl", "accrued", "realized_gain", "total_pnl",
]


def business_days(start: date, end: date) -> list[date]:
    """All Mon–Fri dates between start and end inclusive."""
    days, current = [], start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _prev_business_day(d: date) -> date:
    """Return the previous Mon–Fri date (Monday goes back to Friday)."""
    delta = 3 if d.weekday() == 0 else 1
    return d - timedelta(days=delta)


def _load_inception_prices(portfolio: str | None = None) -> dict[str, float]:
    """
    Clean prices from initial_positions.csv — used as the Day-0 price reference
    when price_history.csv has no entry for the previous business day.
    """
    path = config.INITIAL_POSITIONS_PATH
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path, dtype={"cusip": str, "portfolio": str})
    if df.empty or "price" not in df.columns:
        return {}
    if portfolio is not None and "portfolio" in df.columns:
        df = df[df["portfolio"] == portfolio]
    return dict(zip(df["cusip"], df["price"].astype(float)))


def compute_daily_pnl(
    start_date: date,
    end_date: date,
    portfolio: str | None = None,
    prices_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Daily P&L for every business day in [start_date, end_date].

    Writes results to pnl_history.csv (replacing rows for the same portfolio
    scope) and returns them. Cumulate the returned rows to get the running curve.
    """
    all_trades = load_all_trades()
    if portfolio is not None and not all_trades.empty:
        all_trades = all_trades[all_trades["portfolio"] == portfolio]

    bonds_static = load_bonds_static()
    if prices_df is None:
        prices_df = load_price_history()

    inception_prices = _load_inception_prices(portfolio)
    realized_detail = realized_pnl(all_trades)
    scope = portfolio or ALL_PORTFOLIOS

    # Pre-index price history by date for O(1) lookup
    def _px_on(d: date) -> dict[str, float]:
        if prices_df.empty:
            return {}
        row = prices_df[prices_df["date"] == pd.Timestamp(d)]
        return dict(zip(row["cusip"], row["px_last"])) if not row.empty else {}

    records = []
    for day in business_days(start_date, end_date):
        prev_day = _prev_business_day(day)

        # SOD = end-of-previous-business-day positions (the ones that earned today's P&L)
        positions_sod = compute_positions(all_trades, as_of=prev_day)

        px_today = _px_on(day)
        px_prev_hist = _px_on(prev_day)

        def prev_price(cusip: str) -> float | None:
            if cusip in px_prev_hist:
                return px_prev_hist[cusip]
            return inception_prices.get(cusip)  # Day-0 fallback

        # Daily realized only (not cumulative)
        realized_today: dict[str, float] = {}
        if not realized_detail.empty:
            mask = realized_detail["close_date"] == pd.Timestamp(day)
            if mask.any():
                realized_today = (
                    realized_detail[mask].groupby("cusip")["realized_gain"].sum().to_dict()
                )

        # Include SOD positions + any CUSIP with realized booked today
        cusips = set(positions_sod.keys()) | set(realized_today.keys())

        for cusip in cusips:
            sod_pos = positions_sod.get(cusip)
            net_nominal = sod_pos.net_nominal if sod_pos else 0.0
            bond = bonds_static.get(cusip)
            clean_t = px_today.get(cusip)
            clean_prev = prev_price(cusip)

            price_pnl = None
            accrued = None
            if net_nominal != 0 and clean_t is not None and clean_prev is not None:
                acc_t = accrued_interest(100, bond, day) if bond else 0.0
                acc_prev = accrued_interest(100, bond, prev_day) if bond else 0.0
                price_pnl = round(net_nominal * (clean_t - clean_prev) / 100, 2)
                accrued = round(net_nominal * (acc_t - acc_prev) / 100, 2)

            realized_gain = round(realized_today.get(cusip, 0.0), 2)
            total = (price_pnl or 0.0) + (accrued or 0.0) + realized_gain

            records.append({
                "date": day.isoformat(),
                "portfolio": scope,
                "cusip": cusip,
                "net_nominal": round(net_nominal, 2),
                "px_last": clean_t,
                "price_pnl": price_pnl,
                "accrued": accrued,
                "realized_gain": realized_gain,
                "total_pnl": round(total, 2),
            })

    df = pd.DataFrame(records, columns=_COLUMNS)
    _write_pnl_history(df, scope)
    log.info("Computed daily P&L for %s: %d rows over %s → %s",
             scope, len(df), start_date, end_date)
    return df


def _write_pnl_history(new_df: pd.DataFrame, scope: str, path: Path | None = None) -> None:
    """Merge new_df into pnl_history.csv, replacing rows for the same scope."""
    path = Path(path) if path is not None else config.PNL_HISTORY_PATH
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path, dtype={"cusip": str})
        existing = existing[existing["portfolio"] != scope]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.sort_values(["date", "portfolio", "cusip"]).to_csv(path, index=False)


def load_pnl_history(
    portfolio: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """
    Load pnl_history.csv. When portfolio is None, returns the ALL_PORTFOLIOS scope
    (not every scope concatenated) so totals are never double counted.

    Rows are DAILY values — cumsum them in the caller to get a running P&L curve.
    """
    path = Path(path) if path is not None else config.PNL_HISTORY_PATH
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=_COLUMNS)

    df = pd.read_csv(path, dtype={"cusip": str})
    df["date"] = pd.to_datetime(df["date"])

    scope = portfolio if portfolio is not None else ALL_PORTFOLIOS
    df = df[df["portfolio"] == scope]

    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    return df.reset_index(drop=True)


# ── daily transparency views ──────────────────────────────────────────────────

_SNAPSHOT_COLUMNS = [
    "cusip", "net_nominal", "clean_px", "accrued_pct", "dirty_px",
    "mtm_value", "book_value", "price_pnl", "accrued_pnl", "note",
]


def daily_snapshot(day: date, portfolio: str | None = None) -> pd.DataFrame:
    """
    One row per CUSIP held on `day`: position size, clean & dirty price, MTM value,
    and the price/accrued P&L split — all marked as of `day`.

    Prices come from price_history.csv for that exact day. Reuses mark_to_market
    so the numbers reconcile with the rest of the app.
    """
    positions = compute_positions(load_all_trades(), as_of=day, portfolio=portfolio)
    if not positions:
        return pd.DataFrame(columns=_SNAPSHOT_COLUMNS)

    bonds_static = load_bonds_static()
    prices_df = load_price_history()
    day_px = prices_df[prices_df["date"] == pd.Timestamp(day)] if not prices_df.empty else prices_df
    prices = dict(zip(day_px["cusip"], day_px["px_last"])) if not day_px.empty else {}

    mtm = mark_to_market(positions, prices, bonds_static, as_of=day)
    if mtm.empty:
        return pd.DataFrame(columns=_SNAPSHOT_COLUMNS)

    out = mtm.rename(columns={"accrued_today_pct": "accrued_pct"})
    return out[_SNAPSHOT_COLUMNS].reset_index(drop=True)


def position_timeseries(
    start: date, end: date, portfolio: str | None = None
) -> pd.DataFrame:
    """`daily_snapshot` stacked for every business day in [start, end]."""
    frames = []
    for day in business_days(start, end):
        snap = daily_snapshot(day, portfolio)
        if not snap.empty:
            snap.insert(0, "date", day.isoformat())
            frames.append(snap)
    if not frames:
        return pd.DataFrame(columns=["date"] + _SNAPSHOT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


_ACCRUAL_COLUMNS = [
    "cusip", "day_count_convention", "last_coupon_date", "days_accrued",
    "accrued_per_100", "accrued_total", "note",
]


def accrual_breakdown(positions, bonds_static, as_of: date) -> pd.DataFrame:
    """
    Per-CUSIP accrual transparency: last coupon date, days accrued, accrued per 100
    par, and the position-scaled accrued total. Mirrors accruals.accrued_interest.
    """
    rows = []
    for cusip, pos in positions.items():
        if pos.net_nominal == 0:
            continue
        bond = bonds_static.get(cusip)
        if bond is None:
            rows.append({
                "cusip": cusip, "day_count_convention": "", "last_coupon_date": None,
                "days_accrued": None, "accrued_per_100": None, "accrued_total": None,
                "note": "missing bond static",
            })
            continue
        rows.append({
            "cusip": cusip,
            "day_count_convention": bond.day_count_convention,
            "last_coupon_date": last_coupon_date(bond, as_of).isoformat(),
            "days_accrued": days_accrued(bond, as_of),
            "accrued_per_100": round(accrued_interest(100, bond, as_of), 6),
            "accrued_total": round(accrued_interest(pos.net_nominal, bond, as_of), 2),
            "note": "",
        })
    return pd.DataFrame(rows, columns=_ACCRUAL_COLUMNS)
