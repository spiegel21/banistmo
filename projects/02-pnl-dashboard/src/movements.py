"""
Per-bond movement lineage — the full audit trail behind every position.

For a CUSIP, walk its trades in date order and emit one row per trade showing
the running net nominal, running WAVG cost basis, cash flow, and the realized
gain booked at that moment. This makes a position's entire evolution — from the
inception clip to today — explainable and reconcilable.

The running-WAVG bookkeeping mirrors trading_gains.realized_pnl exactly, so the
per-trade realized column here sums to the same total that the rest of the app
reports. Grouping is by (portfolio, cusip): the same bond in two books keeps
independent cost bases (standard for separately-managed portfolios).
"""
from __future__ import annotations

import pandas as pd

_QTY_EPS = 1e-6

MOVEMENT_COLUMNS = [
    "trade_date", "portfolio", "cusip", "trader", "side", "nominal", "price",
    "cash_flow", "running_nominal", "running_wavg_cost", "realized_gain",
    "cumulative_cash", "cumulative_realized",
]


def position_movements(
    trades_df: pd.DataFrame,
    cusip: str | None = None,
    portfolio: str | None = None,
) -> pd.DataFrame:
    """Chronological sub-ledger of every trade and its effect on the position.

    cusip / portfolio: optional filters. With neither, returns movements for the
    entire book (every CUSIP), already sorted by trade date.
    """
    cols = MOVEMENT_COLUMNS
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=cols)

    df = trades_df.copy()
    if cusip is not None:
        df = df[df["cusip"] == cusip]
    if portfolio is not None and "portfolio" in df.columns:
        df = df[df["portfolio"] == portfolio]
    if df.empty:
        return pd.DataFrame(columns=cols)

    keys = ["portfolio", "cusip"] if "portfolio" in df.columns else ["cusip"]
    rows: list[dict] = []

    for key_vals, group in df.groupby(keys):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_map = dict(zip(keys, key_vals))
        group = group.sort_values("trade_date").reset_index(drop=True)

        qty = 0.0        # signed running nominal
        wavg = 0.0       # positive unit cost (cash per unit nominal)
        cum_cash = 0.0
        cum_realized = 0.0

        for _, r in group.iterrows():
            q = float(r["nominal"])               # signed
            net = float(r.get("net", 0.0))        # signed cash (buys negative)
            cum_cash += net
            realized_this = 0.0

            if abs(q) >= _QTY_EPS:
                unit = abs(net) / abs(q) if net else float(r.get("price", 0.0))
                if abs(qty) < _QTY_EPS:
                    qty, wavg = q, unit
                elif (q > 0) == (qty > 0):
                    # adding to position — blend cost
                    wavg = (abs(qty) * wavg + abs(q) * unit) / (abs(qty) + abs(q))
                    qty += q
                else:
                    # closing / reducing / flipping
                    close_qty = min(abs(q), abs(qty))
                    realized_this = (
                        close_qty * (unit - wavg) if qty > 0
                        else close_qty * (wavg - unit)
                    )
                    leftover = abs(q) - close_qty
                    if abs(qty) - close_qty < _QTY_EPS:
                        qty = (1 if q > 0 else -1) * leftover
                        wavg = unit if leftover > _QTY_EPS else 0.0
                    else:
                        qty += q

            cum_realized += realized_this
            rows.append({
                "trade_date": r["trade_date"],
                "portfolio": key_map.get("portfolio", ""),
                "cusip": key_map.get("cusip", r.get("cusip", "")),
                "trader": r.get("trader", ""),
                "side": "buy" if q > 0 else "sell",
                "nominal": round(q, 2),
                "price": round(float(r.get("price", 0.0)), 6),
                "cash_flow": round(net, 2),
                "running_nominal": round(qty, 2),
                "running_wavg_cost": round(wavg, 6),
                "realized_gain": round(realized_this, 2),
                "cumulative_cash": round(cum_cash, 2),
                "cumulative_realized": round(cum_realized, 2),
            })

    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["trade_date", "cusip", "portfolio"]).reset_index(drop=True)
