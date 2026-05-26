"""
Mark-to-market valuation.

MTM value = net_nominal × current_dirty_price / 100
MTM gain  = MTM value - book value (sum of net_proceeds)
"""
from datetime import date
from pathlib import Path

import pandas as pd

from models import Position
from accruals import accrued_interest, load_bonds_static

BONDS_STATIC_PATH = Path(__file__).parent.parent / "data" / "bonds_static.csv"


def mark_to_market(
    positions: dict[str, Position],
    current_prices: dict[str, float],  # {isin: clean_price % of par}
    bonds_static_path: Path = BONDS_STATIC_PATH,
    as_of: date | None = None,
) -> pd.DataFrame:
    """
    Compute MTM for each position.

    Columns returned:
      isin, net_nominal, clean_px, accrued_today_pct, dirty_px,
      mtm_value, book_value, mtm_gain
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
            rows.append({
                "isin": isin,
                "net_nominal": pos.net_nominal,
                "clean_px": None,
                "accrued_today_pct": None,
                "dirty_px": None,
                "mtm_value": None,
                "book_value": pos.book_value,
                "mtm_gain": None,
                "note": "no price",
            })
            continue

        bond = bonds_static.get(isin)
        accrued_pct = accrued_interest(100, bond, as_of) if bond else 0.0
        dirty_px = clean_px + accrued_pct
        mtm_value = pos.net_nominal * dirty_px / 100
        # book_value is sum of net_proceeds: negative for long positions (cash out)
        # MTM gain for a long = current value minus the cash we paid
        mtm_gain = mtm_value + pos.book_value  # book_value is negative for longs

        rows.append({
            "isin": isin,
            "net_nominal": pos.net_nominal,
            "clean_px": clean_px,
            "accrued_today_pct": round(accrued_pct, 4),
            "dirty_px": round(dirty_px, 4),
            "mtm_value": round(mtm_value, 2),
            "book_value": round(pos.book_value, 2),
            "mtm_gain": round(mtm_gain, 2),
            "note": "" if bond else "missing bond static (no accrual)",
        })

    return pd.DataFrame(rows)
