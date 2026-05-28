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
    "isin": str,
    "nominal": float,
    "principal": float,
    "net_proceeds": float,
    "accrued_at_trade": float,
    "clean_price": float,
    "yield_pct": float,
    "trader": str,
}


def load_trades(trades_path: Path = TRADES_PATH) -> pd.DataFrame:
    """Read trades.csv; return empty DataFrame if file doesn't exist yet."""
    path = Path(trades_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(_DTYPES.keys()) + ["trade_date", "settle_date"])

    df = pd.read_csv(path, dtype=_DTYPES, parse_dates=["trade_date", "settle_date"])
    df = df.drop_duplicates(subset=["isin", "trade_date", "trader", "nominal"])
    return df.sort_values("trade_date").reset_index(drop=True)


def compute_positions(trades_df: pd.DataFrame) -> dict[str, Position]:
    """Group trades by ISIN → net position with weighted-average clean price."""
    if trades_df.empty:
        return {}

    positions: dict[str, Position] = {}
    for isin, group in trades_df.groupby("isin"):
        net_nominal = group["nominal"].sum()
        book_value = group["net_proceeds"].sum()

        buys = group[group["nominal"] > 0]
        wavg = (
            (buys["clean_price"] * buys["nominal"]).sum() / buys["nominal"].sum()
            if not buys.empty else 0.0
        )
        last_settle = group["settle_date"].max().date()

        positions[isin] = Position(
            isin=isin,
            net_nominal=net_nominal,
            wavg_clean_price=wavg,
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
            isin=row["isin"],
            net_nominal=float(row["net_nominal"]),
            wavg_clean_price=float(row["wavg_clean_price"]),
            book_value=float(row["book_value"]),
            last_settle=row["last_settle"].date(),
        )
        positions[p.isin] = p
    return positions


def refresh_portfolio() -> dict[str, Position]:
    """Recompute positions from trades.csv and write portfolio.csv. Main entry point."""
    trades = load_trades()
    positions = compute_positions(trades)
    update_portfolio(positions)
    return positions


if __name__ == "__main__":
    positions = refresh_portfolio()
    for isin, pos in positions.items():
        print(
            f"{isin}  nominal={pos.net_nominal:,.0f}"
            f"  wavg_px={pos.wavg_clean_price:.4f}"
            f"  book={pos.book_value:,.2f}"
        )
