"""
Compute net positions from full trade history and persist to portfolio.csv.

trades.csv            — append-only source written by the email parser
initial_positions.csv — seed positions at portfolio inception (user-entered once)
portfolio.csv         — computed positions (recomputed from trades; not hand-edited)
"""
from datetime import date
from pathlib import Path

import pandas as pd

from models import Position

TRADES_PATH = Path(__file__).parent.parent / "data" / "trades.csv"
INITIAL_PATH = Path(__file__).parent.parent / "data" / "initial_positions.csv"
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
    "portfolio": str,
}


def load_trades(trades_path: Path = TRADES_PATH) -> pd.DataFrame:
    """Read trades.csv; return empty DataFrame if file doesn't exist yet."""
    path = Path(trades_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(_DTYPES.keys()) + ["yield_closed", "trade_date", "settle_date"])

    df = pd.read_csv(path, dtype=_DTYPES, parse_dates=["trade_date", "settle_date"])

    # sign nominal by side so downstream code can use nominal > 0 for buys
    df["side"] = df["side"].str.lower().str.strip()
    df["nominal"] = df.apply(
        lambda r: r["nominal"] if r["side"] == "buy" else -abs(r["nominal"]),
        axis=1,
    )

    # yield_closed may be None/NaN
    if "yield_closed" in df.columns:
        df["yield_closed"] = pd.to_numeric(df["yield_closed"], errors="coerce")

    # portfolio defaults to "default" if not provided
    if "portfolio" not in df.columns:
        df["portfolio"] = "default"
    else:
        df["portfolio"] = df["portfolio"].fillna("default")

    df = df.drop_duplicates(subset=["cusip", "trade_date", "trader", "nominal"])
    return df.sort_values("trade_date").reset_index(drop=True)


def load_initial_positions(initial_path: Path = INITIAL_PATH) -> pd.DataFrame:
    """
    Read initial_positions.csv and return rows shaped like a trades DataFrame.

    initial_positions.csv columns:
        portfolio, cusip, nominal, price, book_value, inception_date

    nominal is signed (positive = long). These are treated as synthetic buys/sells
    on inception_date so they can be combined with live trades in compute_positions().
    """
    path = Path(initial_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    df = pd.read_csv(
        path,
        dtype={"portfolio": str, "cusip": str, "nominal": float, "price": float, "book_value": float},
        parse_dates=["inception_date"],
    )
    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        signed_nominal = float(row["nominal"])
        rows.append({
            "cusip": row["cusip"],
            "side": "buy" if signed_nominal >= 0 else "sell",
            "nominal": signed_nominal,
            "principal": abs(signed_nominal) * float(row["price"]) / 100,
            "net": float(row["book_value"]),
            "accrued": 0.0,
            "price": float(row["price"]),
            "yield_closed": float("nan"),
            "trade_date": row["inception_date"],
            "settle_date": row["inception_date"],
            "trader": "inception",
            "portfolio": row["portfolio"],
        })

    result = pd.DataFrame(rows)
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result["settle_date"] = pd.to_datetime(result["settle_date"])
    return result


def _merge_with_initial(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Prepend initial positions to the trades DataFrame."""
    initial = load_initial_positions()
    if initial.empty:
        return trades_df
    combined = pd.concat([initial, trades_df], ignore_index=True)
    return combined.sort_values("trade_date").reset_index(drop=True)


def compute_positions(
    trades_df: pd.DataFrame,
    as_of: date | None = None,
    portfolio: str | None = None,
) -> dict[str, Position]:
    """
    Group trades by CUSIP → net position with weighted-average price.

    as_of:     if given, only include trades with trade_date <= as_of
    portfolio: if given, only include trades for that portfolio
    """
    df = trades_df.copy()

    if as_of is not None:
        df = df[df["trade_date"] <= pd.Timestamp(as_of)]

    if portfolio is not None:
        df = df[df["portfolio"] == portfolio]

    if df.empty:
        return {}

    positions: dict[str, Position] = {}
    for cusip, group in df.groupby("cusip"):
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


def get_positions_as_of(
    as_of_date: date,
    portfolio: str | None = None,
) -> dict[str, Position]:
    """Compute positions as of a specific date, including initial positions."""
    trades = load_trades()
    all_trades = _merge_with_initial(trades)
    return compute_positions(all_trades, as_of=as_of_date, portfolio=portfolio)


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


def refresh_portfolio(portfolio: str | None = None) -> dict[str, Position]:
    """Recompute positions from full trade history and write portfolio.csv."""
    trades = load_trades()
    all_trades = _merge_with_initial(trades)
    positions = compute_positions(all_trades, portfolio=portfolio)
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
