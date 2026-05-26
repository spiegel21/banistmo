"""
Realized and unrealized P&L calculation.

Realized P&L uses FIFO matching: sells are matched against the earliest buys.
Unrealized P&L compares current mark-to-market dirty price vs. book dirty price.
"""
from collections import deque
from datetime import date
from pathlib import Path

import pandas as pd

from models import Position
from accruals import accrued_interest, load_bonds_static

BONDS_STATIC_PATH = Path(__file__).parent.parent / "data" / "bonds_static.csv"


def realized_pnl(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    FIFO P&L for all ISINs.

    Returns a DataFrame with one row per sell trade:
    isin, sell_date, sold_nominal, proceeds, cost_basis, realized_gain
    """
    results = []

    for isin, group in trades_df.groupby("isin"):
        group = group.sort_values("trade_date").reset_index(drop=True)
        buy_queue: deque[tuple[float, float]] = deque()  # (nominal, net_proceeds_per_unit)

        for _, row in group.iterrows():
            if row["nominal"] > 0:
                # buy: push (nominal, proceeds_per_unit) onto queue
                buy_queue.append((row["nominal"], row["net_proceeds"] / row["nominal"]))
            else:
                # sell: match against earliest buys
                sell_nominal = abs(row["nominal"])
                sell_proceeds = abs(row["net_proceeds"])
                remaining = sell_nominal
                cost_basis = 0.0

                while remaining > 0 and buy_queue:
                    buy_nom, buy_unit_cost = buy_queue[0]
                    matched = min(remaining, buy_nom)
                    cost_basis += matched * abs(buy_unit_cost)
                    remaining -= matched
                    if matched == buy_nom:
                        buy_queue.popleft()
                    else:
                        buy_queue[0] = (buy_nom - matched, buy_unit_cost)

                results.append({
                    "isin": isin,
                    "sell_date": row["trade_date"],
                    "sold_nominal": sell_nominal,
                    "proceeds": sell_proceeds,
                    "cost_basis": cost_basis,
                    "realized_gain": sell_proceeds - cost_basis,
                })

    if not results:
        return pd.DataFrame(columns=[
            "isin", "sell_date", "sold_nominal", "proceeds", "cost_basis", "realized_gain"
        ])
    return pd.DataFrame(results)


def total_realized_pnl(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realized P&L by ISIN."""
    detail = realized_pnl(trades_df)
    if detail.empty:
        return detail
    return detail.groupby("isin", as_index=False)["realized_gain"].sum()


def unrealized_pnl(
    positions: dict[str, Position],
    current_prices: dict[str, float],  # {isin: clean_price}
    bonds_static_path: Path = BONDS_STATIC_PATH,
    as_of: date | None = None,
) -> pd.DataFrame:
    """
    Unrealized P&L = (current dirty price - book dirty price) × net_nominal / 100

    current_prices are clean prices (% of par). We add today's accrued to get dirty.
    Book dirty price = wavg_clean_price + accrued_at_trade (approximated from bonds_static).
    """
    if as_of is None:
        as_of = date.today()

    bonds_static = load_bonds_static(bonds_static_path)
    rows = []

    for isin, pos in positions.items():
        if pos.net_nominal == 0:
            continue

        clean_px = current_prices.get(isin)
        if clean_px is None:
            rows.append({"isin": isin, "unrealized_gain": None, "note": "no price"})
            continue

        bond = bonds_static.get(isin)
        if bond is None:
            rows.append({"isin": isin, "unrealized_gain": None, "note": "missing bond static"})
            continue

        current_accrued_pct = accrued_interest(100, bond, as_of)  # per 100 nominal
        current_dirty = clean_px + current_accrued_pct

        book_accrued_pct = accrued_interest(100, bond, pos.last_settle)
        book_dirty = pos.wavg_clean_price + book_accrued_pct

        unrealized = (current_dirty - book_dirty) * pos.net_nominal / 100
        rows.append({
            "isin": isin,
            "net_nominal": pos.net_nominal,
            "book_clean_px": pos.wavg_clean_price,
            "current_clean_px": clean_px,
            "unrealized_gain": round(unrealized, 2),
            "note": "",
        })

    return pd.DataFrame(rows)
