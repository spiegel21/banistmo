"""
Historical daily P&L.

For each business day in a range, compute per-CUSIP:
  - price_pnl  : unrealized P&L from clean-price moves
  - accrued    : accrued interest (carry) on the current position
  - realized   : cumulative realized P&L from closes up to that day
  - total_pnl  : price_pnl + accrued + realized   (no double counting)

Realized P&L persists for fully-closed positions (they no longer have a live
position, but their booked gains remain in the daily total).

Results are cached in pnl_history.csv.
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import config
from config import ALL_PORTFOLIOS, get_logger
from position_manager import load_all_trades, compute_positions
from accruals import load_bonds_static, accrued_interest
from trading_gains import realized_pnl
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


def compute_daily_pnl(
    start_date: date,
    end_date: date,
    portfolio: str | None = None,
    prices_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Daily P&L for every business day in [start_date, end_date].

    Writes results to pnl_history.csv (replacing rows for the same portfolio
    scope) and returns them.
    """
    all_trades = load_all_trades()
    if portfolio is not None and not all_trades.empty:
        all_trades = all_trades[all_trades["portfolio"] == portfolio]

    bonds_static = load_bonds_static()
    if prices_df is None:
        prices_df = load_price_history()

    realized_detail = realized_pnl(all_trades)
    scope = portfolio or ALL_PORTFOLIOS

    records = []
    for day in business_days(start_date, end_date):
        ts = pd.Timestamp(day)
        positions = compute_positions(all_trades, as_of=day)

        day_px = prices_df[prices_df["date"] == ts] if not prices_df.empty else prices_df
        px_lookup = dict(zip(day_px["cusip"], day_px["px_last"])) if not day_px.empty else {}

        realized_to_day: dict[str, float] = {}
        if not realized_detail.empty:
            mask = realized_detail["close_date"] <= ts
            realized_to_day = (
                realized_detail[mask].groupby("cusip")["realized_gain"].sum().to_dict()
            )

        # Union: live positions + any cusip with realized P&L booked by this day.
        cusips = set(positions.keys()) | set(realized_to_day.keys())
        for cusip in cusips:
            pos = positions.get(cusip)
            net_nominal = pos.net_nominal if pos else 0.0
            px = px_lookup.get(cusip)
            bond = bonds_static.get(cusip)

            price_pnl = None
            accrued = None
            if pos and net_nominal != 0 and px is not None:
                accrued_pct = accrued_interest(100, bond, day) if bond else 0.0
                dirty = px + accrued_pct
                mtm_gain = net_nominal * dirty / 100 + pos.book_value
                accrued = round(net_nominal * accrued_pct / 100, 2)
                price_pnl = round(mtm_gain - accrued, 2)

            realized_gain = round(realized_to_day.get(cusip, 0.0), 2)
            total = (price_pnl or 0.0) + (accrued or 0.0) + realized_gain

            records.append({
                "date": day.isoformat(),
                "portfolio": scope,
                "cusip": cusip,
                "net_nominal": round(net_nominal, 2),
                "px_last": px,
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
