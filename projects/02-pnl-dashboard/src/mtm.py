"""
Mark-to-market valuation and unrealized-P&L decomposition.

For a position, total unrealized P&L is:
    mtm_gain = net_nominal * dirty_px / 100  +  book_value      (book_value < 0 for longs)

This decomposes cleanly into a price component and an accrued (carry) component:
    accrued_pnl = net_nominal * accrued_today_pct / 100         (== total_portfolio_accruals)
    price_pnl   = mtm_gain - accrued_pnl

so total P&L = realized + price_pnl + accrued_pnl with no double counting.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from accruals import accrued_interest, load_bonds_static
from models import BondStatic, Position

_COLUMNS = [
    "cusip", "net_nominal", "clean_px", "accrued_today_pct", "dirty_px",
    "mtm_value", "book_value", "mtm_gain", "accrued_pnl", "price_pnl", "note",
]


def mark_to_market(
    positions: dict[str, Position],
    current_prices: dict[str, float],          # {cusip: clean_price % of par}
    bonds_static: dict[str, BondStatic] | None = None,
    as_of: date | None = None,
) -> pd.DataFrame:
    """
    Mark each position to market and split unrealized P&L into price vs accrued.

    bonds_static may be a preloaded dict (preferred, avoids repeated disk reads)
    or None to load from the default path.
    """
    if as_of is None:
        as_of = date.today()
    if bonds_static is None:
        bonds_static = load_bonds_static()

    rows = []
    for cusip, pos in positions.items():
        if pos.net_nominal == 0:
            continue

        clean_px = current_prices.get(cusip)
        if clean_px is None:
            rows.append({
                "cusip": cusip, "net_nominal": pos.net_nominal, "clean_px": None,
                "accrued_today_pct": None, "dirty_px": None, "mtm_value": None,
                "book_value": round(pos.book_value, 2), "mtm_gain": None,
                "accrued_pnl": None, "price_pnl": None, "note": "no price",
            })
            continue

        bond = bonds_static.get(cusip)
        accrued_pct = accrued_interest(100, bond, as_of) if bond else 0.0
        dirty_px = clean_px + accrued_pct
        mtm_value = pos.net_nominal * dirty_px / 100
        mtm_gain = mtm_value + pos.book_value           # total unrealized (price + carry)
        accrued_pnl = pos.net_nominal * accrued_pct / 100
        price_pnl = mtm_gain - accrued_pnl

        rows.append({
            "cusip": cusip,
            "net_nominal": pos.net_nominal,
            "clean_px": clean_px,
            "accrued_today_pct": round(accrued_pct, 6),
            "dirty_px": round(dirty_px, 6),
            "mtm_value": round(mtm_value, 2),
            "book_value": round(pos.book_value, 2),
            "mtm_gain": round(mtm_gain, 2),
            "accrued_pnl": round(accrued_pnl, 2),
            "price_pnl": round(price_pnl, 2),
            "note": "" if bond else "missing bond static (no accrual)",
        })

    return pd.DataFrame(rows, columns=_COLUMNS)
