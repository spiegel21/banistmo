"""
Historical P&L computation.

For each business day from start_date to end_date, compute:
  - Position as of that date (initial_positions + trades up to that date)
  - MTM gain (using price_history.csv)
  - Accrued interest earned
  - Cumulative realized gains

Writes results to data/pnl_history.csv.
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from position_manager import load_trades, load_initial_positions, compute_positions
from accruals import load_bonds_static, accrued_interest
from trading_gains import realized_pnl
from bloomberg import load_price_history

DATA_DIR = Path(__file__).parent.parent / "data"
PNL_HISTORY_PATH = DATA_DIR / "pnl_history.csv"


def business_days(start: date, end: date) -> list[date]:
    """Return all Mon–Fri dates between start and end inclusive."""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # 0=Mon … 4=Fri
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
    Build a daily P&L table for every business day in [start_date, end_date].

    Columns returned:
        date, portfolio, cusip, net_nominal, px_last,
        mtm_gain, accrued, realized_gain, total_pnl

    If prices_df is None, loads from price_history.csv.
    Results are also written to pnl_history.csv (full overwrite for the given
    portfolio / date range).
    """
    trades = load_trades()
    initial = load_initial_positions()

    all_trades = (
        pd.concat([initial, trades], ignore_index=True).sort_values("trade_date")
        if not initial.empty else trades
    )

    if portfolio is not None:
        all_trades = all_trades[all_trades["portfolio"] == portfolio].copy()

    bonds_static = load_bonds_static()

    if prices_df is None:
        prices_df = load_price_history()

    # Pre-compute all realised P&L sell events once (for the entire history)
    realized_detail = realized_pnl(all_trades)  # cusip, sell_date, realized_gain

    days = business_days(start_date, end_date)
    records = []

    for day in days:
        positions = compute_positions(all_trades, as_of=day)

        # Price lookup for this day
        day_px_df = prices_df[prices_df["date"] == pd.Timestamp(day)]
        px_lookup: dict[str, float] = dict(zip(day_px_df["cusip"], day_px_df["px_last"]))

        # Cumulative realized gains up to this day
        realized_to_day = {}
        if not realized_detail.empty:
            realized_to_day = (
                realized_detail[realized_detail["sell_date"] <= pd.Timestamp(day)]
                .groupby("cusip")["realized_gain"]
                .sum()
                .to_dict()
            )

        for cusip, pos in positions.items():
            if pos.net_nominal == 0:
                continue

            px = px_lookup.get(cusip)
            bond = bonds_static.get(cusip)

            # MTM gain vs book
            mtm_gain = None
            if px is not None:
                accrued_pct = accrued_interest(100, bond, day) if bond else 0.0
                dirty = px + accrued_pct
                mtm_gain = round(pos.net_nominal * dirty / 100 + pos.book_value, 2)

            # Accrued interest on current position size
            accrued_today = None
            if bond:
                ai = accrued_interest(abs(pos.net_nominal), bond, day)
                accrued_today = round(-ai if pos.net_nominal < 0 else ai, 2)

            realized_gain = round(realized_to_day.get(cusip, 0.0), 2)
            total = (mtm_gain or 0.0) + (accrued_today or 0.0) + realized_gain

            records.append({
                "date": day.isoformat(),
                "portfolio": portfolio or "all",
                "cusip": cusip,
                "net_nominal": round(pos.net_nominal, 2),
                "px_last": px,
                "mtm_gain": mtm_gain,
                "accrued": accrued_today,
                "realized_gain": realized_gain,
                "total_pnl": round(total, 2),
            })

    df = pd.DataFrame(records)
    _write_pnl_history(df, portfolio)
    return df


def _write_pnl_history(new_df: pd.DataFrame, portfolio: str | None) -> None:
    """Merge new_df into pnl_history.csv, replacing rows for the same portfolio."""
    path = PNL_HISTORY_PATH
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        # Drop rows that belong to the portfolio being recomputed
        if portfolio is not None:
            existing = existing[existing["portfolio"] != portfolio]
        else:
            existing = existing[existing["portfolio"] != "all"]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.sort_values(["date", "portfolio", "cusip"]).to_csv(path, index=False)


def load_pnl_history(
    portfolio: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    path: Path = PNL_HISTORY_PATH,
) -> pd.DataFrame:
    """Load pnl_history.csv with optional filters."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=[
            "date", "portfolio", "cusip", "net_nominal",
            "px_last", "mtm_gain", "accrued", "realized_gain", "total_pnl",
        ])

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    if portfolio is not None:
        df = df[df["portfolio"] == portfolio]
    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    return df.reset_index(drop=True)
