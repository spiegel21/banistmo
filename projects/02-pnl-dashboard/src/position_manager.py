"""
Compute net positions from full trade history and persist to portfolio.csv.

trades.csv   — append-only source of truth written by the email parser
portfolio.csv — computed positions (recomputed from trades every time; not hand-edited)
"""
from datetime import date
from pathlib import Path

import pandas as pd

from models import Position

TRADES_PATH = Path(__file__).parent.parent / "data" / "trades.csv"
PORTFOLIO_PATH = Path(__file__).parent.parent / "data" / "portfolio.csv"

_DTYPES = {
    "cusip": str,
    "side": str,
    "nominal": float,
    "principal": float,
    "net": float,
    "accrued": float,
    "price": float,
    "trader": str,
}


def load_trades(trades_path: Path = TRADES_PATH) -> pd.DataFrame:
    """Read trades.csv; return empty DataFrame if file doesn't exist yet."""
    path = Path(trades_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(_DTYPES.keys()) + ["yield_closed", "trade_date", "settle_date"])

    df = pd.read_csv(
        path,
        dtype=_DTYPES,
        parse_dates=["trade_date", "settle_date"],
    )

    # sign nominal by side so downstream code can use nominal > 0 for buys
    df["side"] = df["side"].str.lower().str.strip()
    df["nominal"] = df.apply(
        lambda r: r["nominal"] if r["side"] == "buy" else -abs(r["nominal"]),
        axis=1,
    )

    # yield_closed may be None/NaN — leave as float column with NaN
    if "yield_closed" in df.columns:
        df["yield_closed"] = pd.to_numeric(df["yield_closed"], errors="coerce")

    df = df.drop_duplicates(subset=["cusip", "trade_date", "trader", "nominal"])
    return df.sort_values("trade_date").reset_index(drop=True)


def compute_positions(trades_df: pd.DataFrame) -> dict[str, Position]:
    """Group trades by CUSIP → net position with weighted-average price."""
    if trades_df.empty:
        return {}

    positions: dict[str, Position] = {}
    for cusip, group in trades_df.groupby("cusip"):
        net_nominal = group["nominal"].sum()
        book_value = group["net"].sum()

        buys = group[group["nominal"] > 0]
        wavg = (
            (buys["price"] * buys["nominal"]).sum() / buys["nominal"].sum()
            if not buys.empty else 0.0
        )
        last_settle = group["settle_date"].max().date()

        positions[cusip] = Position(
            cusip=cusip,
            net_nominal=net_nominal,
            wavg_price=wavg,
            book_value=book_value,
            last_settle=last_settle,
        )

    return positions


def update_portfolio(positions: dict[str, Position], portfolio_path: Path = PORTFOLIO_PATH) -> None:
    """Rewrite portfolio.csv with current positions (replaces previous contents)."""
    rows = [vars(p) for p in positions.values()]
    pd.DataFrame(rows).to_csv(portfolio_path, index=False)


def load_portfolio(portfolio_path: Path = PORTFOLIO_PATH) -> dict[str, Position]:
    """Read portfolio.csv; returns empty dict if file doesn't exist."""
    path = Path(portfolio_path)
    if not path.exists():
        return {}

    df = pd.read_csv(path, parse_dates=["last_settle"])
    positions = {}
    for _, row in df.iterrows():
        p = Position(
            cusip=row["cusip"],
            net_nominal=float(row["net_nominal"]),
            wavg_price=float(row["wavg_price"]),
            book_value=float(row["book_value"]),
            last_settle=row["last_settle"].date(),
        )
        positions[p.cusip] = p
    return positions


def refresh_portfolio() -> dict[str, Position]:
    """Recompute positions from trades.csv and write portfolio.csv. Main entry point."""
    trades = load_trades()
    positions = compute_positions(trades)
    update_portfolio(positions)
    return positions


if __name__ == "__main__":
    positions = refresh_portfolio()
    for cusip, pos in positions.items():
        print(
            f"{cusip}  nominal={pos.net_nominal:,.0f}"
            f"  wavg_px={pos.wavg_price:.4f}"
            f"  book={pos.book_value:,.2f}"
        )
