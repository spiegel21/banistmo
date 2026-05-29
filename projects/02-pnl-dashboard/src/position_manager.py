"""
Compute net positions from full trade history and persist to portfolio.csv.

trades.csv            — append-only source written by the email parser
initial_positions.csv — seed positions at portfolio inception (user-entered once)
portfolio.csv         — computed positions (recomputed from trades; not hand-edited)
"""
from datetime import date
from pathlib import Path

import pandas as pd

import config
from config import DEFAULT_PORTFOLIO, get_logger
from models import Position

log = get_logger(__name__)

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

# Columns used to detect a re-appended (duplicate) confirmation. Includes
# Timestamp + the economics so two genuine same-day clips are NOT collapsed.
_DEDUP_KEYS = ["Timestamp", "cusip", "side", "nominal", "net", "price", "trade_date", "trader"]


def load_trades(trades_path: Path | None = None) -> pd.DataFrame:
    """Read trades.csv; return empty DataFrame if file doesn't exist yet."""
    path = Path(trades_path) if trades_path is not None else config.TRADES_PATH
    empty_cols = list(_DTYPES.keys()) + ["yield_closed", "trade_date", "settle_date", "Timestamp"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=empty_cols)

    df = pd.read_csv(path, dtype=_DTYPES)
    for col in ("trade_date", "settle_date"):
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], format="%m/%d/%y", errors="coerce")
        # Fall back to ISO / default parsing for any values that didn't match mm/dd/yy
        unresolved = parsed.isna() & df[col].notna()
        if unresolved.any():
            parsed[unresolved] = pd.to_datetime(df.loc[unresolved, col], errors="coerce")
        df[col] = parsed

    # normalise side, then sign nominal so nominal > 0 = buy, < 0 = sell
    df["side"] = df["side"].str.lower().str.strip()
    invalid = ~df["side"].isin(["buy", "sell"])
    if invalid.any():
        log.warning("%d trade(s) with side not in {buy, sell}; treated as sell", invalid.sum())
    df["nominal"] = df["nominal"].abs() * df["side"].map({"buy": 1}).fillna(-1)

    if "yield_closed" in df.columns:
        df["yield_closed"] = pd.to_numeric(df["yield_closed"], errors="coerce")

    if "portfolio" not in df.columns:
        df["portfolio"] = DEFAULT_PORTFOLIO
    else:
        df["portfolio"] = df["portfolio"].fillna(DEFAULT_PORTFOLIO)

    # Drop only exact re-appends (same confirmation parsed twice).
    dedup_keys = [k for k in _DEDUP_KEYS if k in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_keys)
    if len(df) < before:
        log.info("Dropped %d duplicate trade row(s)", before - len(df))

    return df.sort_values("trade_date").reset_index(drop=True)


def load_initial_positions(initial_path: Path | None = None) -> pd.DataFrame:
    """
    Read initial_positions.csv and return rows shaped like a trades DataFrame.

    initial_positions.csv columns:
        portfolio, cusip, nominal, price, book_value, inception_date

    nominal is signed (positive = long). Treated as synthetic trades on
    inception_date so they combine with live trades in compute_positions().
    """
    path = Path(initial_path) if initial_path is not None else config.INITIAL_POSITIONS_PATH
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
            "cusip": str(row["cusip"]),
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
            "portfolio": str(row["portfolio"]),
        })

    result = pd.DataFrame(rows)
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result["settle_date"] = pd.to_datetime(result["settle_date"])
    return result


def load_all_trades(
    trades_path: Path | None = None,
    initial_path: Path | None = None,
) -> pd.DataFrame:
    """Initial positions + live trades, sorted by trade_date. Single source of truth."""
    trades = load_trades(trades_path)
    initial = load_initial_positions(initial_path)
    if initial.empty:
        return trades
    if trades.empty:
        return initial.sort_values("trade_date").reset_index(drop=True)
    combined = pd.concat([initial, trades], ignore_index=True)
    return combined.sort_values("trade_date").reset_index(drop=True)


def compute_positions(
    trades_df: pd.DataFrame,
    as_of: date | None = None,
    portfolio: str | None = None,
) -> dict[str, Position]:
    """
    Group trades by CUSIP → net position with weighted-average price.

    as_of:     include only trades with trade_date <= as_of
    portfolio: include only trades for that portfolio
    """
    if trades_df.empty:
        return {}

    df = trades_df
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

        positions[str(cusip)] = Position(
            cusip=str(cusip),
            net_nominal=float(net_nominal),
            wavg_price=float(wavg),
            book_value=float(book_value),
            last_settle=last_settle,
        )
    return positions


def get_positions_as_of(as_of_date: date, portfolio: str | None = None) -> dict[str, Position]:
    """Positions as of a date, including initial positions."""
    return compute_positions(load_all_trades(), as_of=as_of_date, portfolio=portfolio)


def update_portfolio(positions: dict[str, Position], portfolio_path: Path | None = None) -> None:
    """Rewrite portfolio.csv with current positions (replaces previous contents)."""
    path = Path(portfolio_path) if portfolio_path is not None else config.PORTFOLIO_PATH
    rows = [vars(p) for p in positions.values()]
    cols = ["cusip", "net_nominal", "wavg_price", "book_value", "last_settle"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def load_portfolio(portfolio_path: Path | None = None) -> dict[str, Position]:
    """Read portfolio.csv; returns empty dict if file doesn't exist."""
    path = Path(portfolio_path) if portfolio_path is not None else config.PORTFOLIO_PATH
    if not path.exists() or path.stat().st_size == 0:
        return {}

    df = pd.read_csv(path, dtype={"cusip": str}, parse_dates=["last_settle"])
    positions = {}
    for _, row in df.iterrows():
        p = Position(
            cusip=str(row["cusip"]),
            net_nominal=float(row["net_nominal"]),
            wavg_price=float(row["wavg_price"]),
            book_value=float(row["book_value"]),
            last_settle=row["last_settle"].date(),
        )
        positions[p.cusip] = p
    return positions


def refresh_portfolio(portfolio: str | None = None) -> dict[str, Position]:
    """Recompute positions from full trade history and write portfolio.csv."""
    positions = compute_positions(load_all_trades(), portfolio=portfolio)
    update_portfolio(positions)
    return positions


if __name__ == "__main__":
    for cusip, pos in refresh_portfolio().items():
        print(f"{cusip}  nominal={pos.net_nominal:,.0f}  "
              f"wavg_px={pos.wavg_price:.4f}  book={pos.book_value:,.2f}")
