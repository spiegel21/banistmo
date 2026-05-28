"""
Realized P&L via signed-lot FIFO matching.

Handles longs, shorts, and position flips symmetrically: a closing trade is
matched FIFO against open lots of the opposite sign; any excess opens a new lot
in the new direction. Realized gain is booked on the date of the closing trade.

Unrealized P&L lives in mtm.py (it requires current prices), keeping this module
dependent only on the trade blotter.
"""
from collections import deque
from dataclasses import dataclass

import pandas as pd

_QTY_EPS = 1e-6  # nominal smaller than this is treated as fully closed


@dataclass
class _Lot:
    qty: float        # signed: >0 long, <0 short
    unit_cash: float  # absolute cash per unit of nominal (always positive)


def _grouping_keys(trades_df: pd.DataFrame) -> list[str]:
    """Match FIFO within (portfolio, cusip) when portfolio is present, else cusip."""
    return ["portfolio", "cusip"] if "portfolio" in trades_df.columns else ["cusip"]


def realized_pnl(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    FIFO realized P&L, one row per closing event.

    Returns columns: portfolio (if present), cusip, close_date,
    closed_nominal, realized_gain.
    """
    keys = _grouping_keys(trades_df)
    out_cols = keys + ["close_date", "closed_nominal", "realized_gain"]

    if trades_df.empty:
        return pd.DataFrame(columns=out_cols)

    results = []
    for key_vals, group in trades_df.groupby(keys):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        group = group.sort_values("trade_date").reset_index(drop=True)
        lots: deque[_Lot] = deque()

        for _, row in group.iterrows():
            q = float(row["nominal"])
            if q == 0:
                continue
            unit = abs(float(row["net"])) / abs(q)
            direction = 1 if q > 0 else -1
            remaining = q
            closed_here = 0.0
            gain_here = 0.0

            # Close against opposite-sign lots FIFO.
            while abs(remaining) > _QTY_EPS and lots and (lots[0].qty > 0) == (direction < 0):
                lot = lots[0]
                m = min(abs(remaining), abs(lot.qty))  # units matched (positive)
                if lot.qty > 0:
                    # closing a long with a sell: gain = (sell - buy) per unit
                    gain_here += m * (unit - lot.unit_cash)
                else:
                    # closing a short with a buy: gain = (short sale - buy cover) per unit
                    gain_here += m * (lot.unit_cash - unit)
                closed_here += m
                remaining -= direction * m   # move remaining toward zero
                lot.qty += direction * m     # move lot toward zero
                if abs(lot.qty) <= _QTY_EPS:
                    lots.popleft()

            # Any leftover opens a new lot in the trade's direction.
            if abs(remaining) > _QTY_EPS:
                lots.append(_Lot(qty=remaining, unit_cash=unit))

            if closed_here > _QTY_EPS:
                results.append({
                    **dict(zip(keys, key_vals)),
                    "close_date": row["trade_date"],
                    "closed_nominal": round(closed_here, 2),
                    "realized_gain": round(gain_here, 2),
                })

    return pd.DataFrame(results, columns=out_cols)


def total_realized_pnl(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realized P&L by cusip (summed across the grouping keys)."""
    detail = realized_pnl(trades_df)
    if detail.empty:
        return pd.DataFrame(columns=["cusip", "realized_gain"])
    return detail.groupby("cusip", as_index=False)["realized_gain"].sum()
