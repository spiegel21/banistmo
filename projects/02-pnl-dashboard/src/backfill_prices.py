"""
Bulk historical price backfill via Bloomberg BDH.

Run once to populate price_history.csv from portfolio inception to today.

Usage:
    python backfill_prices.py --start 2025-01-01
    python backfill_prices.py --start 2025-01-01 --end 2025-06-01 --portfolio "HY Book"
    python backfill_prices.py --start 2025-01-01 --manual path/to/prices.csv
"""
import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))

from position_manager import load_trades, load_initial_positions, compute_positions
from bloomberg import get_historical_prices_bdh, _append_df_to_price_history, TEMPLATE_PATH


def _get_cusips(portfolio: str | None) -> list[str]:
    trades = load_trades()
    initial = load_initial_positions()
    all_trades = (
        pd.concat([initial, trades], ignore_index=True)
        if not initial.empty else trades
    )
    if portfolio:
        all_trades = all_trades[all_trades["portfolio"] == portfolio]
    return sorted(all_trades["cusip"].dropna().unique().tolist())


def main():
    parser = argparse.ArgumentParser(description="Backfill Bloomberg historical prices")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--portfolio", default=None, help="Filter to a specific portfolio")
    parser.add_argument(
        "--manual",
        default=None,
        help="Path to a manual price CSV (date, cusip, px_last) to import instead of Bloomberg",
    )
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()

    if args.manual:
        # Import a manually prepared CSV directly into price_history.csv
        df = pd.read_csv(args.manual)
        required = {"date", "cusip", "px_last"}
        if not required.issubset(df.columns):
            print(f"Manual CSV must have columns: {required}. Found: {set(df.columns)}")
            sys.exit(1)
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        df = df[mask]
        if args.portfolio:
            print(f"Importing manual prices for date range {start} → {end}")
        _append_df_to_price_history(df)
        print(f"Imported {len(df)} rows into price_history.csv")
        return

    cusips = _get_cusips(args.portfolio)
    if not cusips:
        print("No CUSIPs found. Check trades.csv / initial_positions.csv.")
        sys.exit(1)

    print(f"Fetching BDH prices for {len(cusips)} CUSIPs from {start} to {end}...")
    df = get_historical_prices_bdh(cusips, start, end, TEMPLATE_PATH)

    if df.empty:
        print("No prices returned. Check Bloomberg Terminal is running and ISINs are valid.")
        sys.exit(1)

    print(f"Fetched {len(df)} price records. Saved to price_history.csv.")


if __name__ == "__main__":
    main()
