"""
Compute net positions from full trade history.

Positions are always recomputed from scratch — snapshots are cache only.
"""
import os
from datetime import datetime, date
from pathlib import Path

import pandas as pd

from models import Trade, Position

TRADES_DIR = Path(__file__).parent.parent / "data" / "trades"
POSITIONS_DIR = Path(__file__).parent.parent / "data" / "positions"

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


def load_trades(trades_dir: Path = TRADES_DIR) -> pd.DataFrame:
    """Read all CSVs in trades_dir, deduplicate, return sorted DataFrame."""
    files = sorted(Path(trades_dir).glob("*_trades.csv"))
    if not files:
        return pd.DataFrame(columns=list(_DTYPES.keys()) + ["trade_date", "settle_date"])

    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=_DTYPES, parse_dates=["trade_date", "settle_date"])
        frames.append(df)

    all_trades = pd.concat(frames, ignore_index=True)
    # deduplicate: same isin + trade_date + trader + nominal is the same confirmation
    all_trades = all_trades.drop_duplicates(
        subset=["isin", "trade_date", "trader", "nominal"]
    )
    return all_trades.sort_values("trade_date").reset_index(drop=True)


def compute_positions(trades_df: pd.DataFrame) -> dict[str, Position]:
    """Group trades by ISIN → net position with weighted-average clean price."""
    if trades_df.empty:
        return {}

    positions: dict[str, Position] = {}
    for isin, group in trades_df.groupby("isin"):
        net_nominal = group["nominal"].sum()
        book_value = group["net_proceeds"].sum()

        buys = group[group["nominal"] > 0]
        if not buys.empty:
            wavg = (buys["clean_price"] * buys["nominal"]).sum() / buys["nominal"].sum()
        else:
            wavg = 0.0

        last_settle = group["settle_date"].max().date()

        positions[isin] = Position(
            isin=isin,
            net_nominal=net_nominal,
            wavg_clean_price=wavg,
            book_value=book_value,
            last_settle=last_settle,
        )

    return positions


def save_snapshot(positions: dict[str, Position], positions_dir: Path = POSITIONS_DIR) -> Path:
    """Write a timestamped parquet snapshot of current positions."""
    Path(positions_dir).mkdir(parents=True, exist_ok=True)
    rows = [vars(p) for p in positions.values()]
    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(positions_dir) / f"snapshot_{ts}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def load_latest_snapshot(positions_dir: Path = POSITIONS_DIR) -> dict[str, Position] | None:
    """Load the most recent snapshot; returns None if none exists."""
    snapshots = sorted(Path(positions_dir).glob("snapshot_*.parquet"))
    if not snapshots:
        return None

    df = pd.read_parquet(snapshots[-1])
    positions = {}
    for _, row in df.iterrows():
        p = Position(
            isin=row["isin"],
            net_nominal=row["net_nominal"],
            wavg_clean_price=row["wavg_clean_price"],
            book_value=row["book_value"],
            last_settle=row["last_settle"] if isinstance(row["last_settle"], date)
                        else row["last_settle"].date(),
        )
        positions[p.isin] = p
    return positions


def get_positions(use_cache: bool = False) -> dict[str, Position]:
    """Main entry point: load trades and return current positions."""
    if use_cache:
        cached = load_latest_snapshot()
        if cached is not None:
            return cached

    trades = load_trades()
    return compute_positions(trades)


if __name__ == "__main__":
    positions = get_positions()
    for isin, pos in positions.items():
        print(
            f"{isin}  nominal={pos.net_nominal:,.0f}"
            f"  wavg_px={pos.wavg_clean_price:.4f}"
            f"  book={pos.book_value:,.2f}"
        )
