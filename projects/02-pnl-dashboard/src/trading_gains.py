"""
Realized P&L via weighted-average cost (WAVG) matching.

For each (portfolio, cusip) pair, trades are processed in trade_date order.
A running (qty, wavg_cost) tracks the open position. When a trade is in the
opposite direction it closes (or reduces) the open position, booking realized
gain at that moment. If it crosses zero, the gain on the full old size is
booked first and the remainder opens a new position at the new trade's price.

WAVG is the standard methodology for fixed-income portfolio books and matches
the default in Bloomberg PRISM, Advent Geneva, and FactSet.
"""
import pandas as pd

_QTY_EPS = 1e-6  # nominal smaller than this is treated as fully closed


def _grouping_keys(trades_df: pd.DataFrame) -> list[str]:
    """Group within (portfolio, cusip) when portfolio column is present."""
    return ["portfolio", "cusip"] if "portfolio" in trades_df.columns else ["cusip"]


def realized_pnl(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    WAVG realized P&L, one row per closing event.

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

        qty = 0.0    # signed: >0 long, <0 short
        wavg = 0.0   # always positive: unit cash per unit of nominal

        for _, row in group.iterrows():
            q = float(row["nominal"])   # signed: >0 buy, <0 sell
            if abs(q) < _QTY_EPS:
                continue
            # Clean-price cost basis (% of par as a fraction). Using `net` (dirty:
            # principal + accrued) would fold the accrued paid/received into the
            # "trading" gain, double-attributing carry that the accrual engine
            # already books. Realized trading P&L is a clean-price concept.
            unit = float(row["price"]) / 100.0   # always positive

            if abs(qty) < _QTY_EPS:
                qty, wavg = q, unit
                continue

            if (q > 0) == (qty > 0):
                # Adding to position — blend cost.
                wavg = (abs(qty) * wavg + abs(q) * unit) / (abs(qty) + abs(q))
                qty += q
                continue

            # Closing trade: opposite direction to current position.
            close_qty = min(abs(q), abs(qty))
            gain = close_qty * (unit - wavg) if qty > 0 else close_qty * (wavg - unit)

            results.append({
                **dict(zip(keys, key_vals)),
                "close_date": row["trade_date"],
                "closed_nominal": round(close_qty, 2),
                "realized_gain": round(gain, 2),
            })

            leftover = abs(q) - close_qty
            if abs(qty) - close_qty < _QTY_EPS:
                # Fully closed or flipped — leftover (if any) opens opposite side.
                qty = (1 if q > 0 else -1) * leftover
                wavg = unit if leftover > _QTY_EPS else 0.0
            else:
                qty += q   # partial close: reduce position

    return pd.DataFrame(results, columns=out_cols)


def total_realized_pnl(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realized P&L by cusip (summed across grouping keys)."""
    detail = realized_pnl(trades_df)
    if detail.empty:
        return pd.DataFrame(columns=["cusip", "realized_gain"])
    return detail.groupby("cusip", as_index=False)["realized_gain"].sum()
