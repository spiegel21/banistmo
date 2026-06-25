"""
Compute net positions from full trade history and persist to portfolio.csv.

trades.csv            — append-only source written by the email parser
initial_positions.csv — seed positions at portfolio inception (user-entered once)
portfolio.csv         — computed positions (recomputed from trades; not hand-edited)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

import config
from config import DEFAULT_PORTFOLIO, get_logger
from models import BondStatic, Position
from security_id import alias_map, canonicalize_series

log = get_logger(__name__)

_QTY_EPS = 1e-6  # net nominal smaller than this is treated as flat (fully closed)
REDEMPTION_TRADER = "redemption"  # marker on synthetic par-redemption trades

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
    # Collapse any ISIN-keyed clip onto its canonical CUSIP so the same bond
    # never splits into two positions (see security_id.py).
    df["cusip"] = canonicalize_series(df["cusip"], alias_map())
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
    # net cash is stored UNSIGNED in the CSV (same as nominal); apply the sign by
    # side here so buys are negative (cash out) and sells positive (cash in).
    # abs() first makes this idempotent for rows that are already signed.
    df["net"] = df["net"].abs() * df["side"].map({"buy": -1}).fillna(1)

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

    df["cusip"] = canonicalize_series(df["cusip"], alias_map())
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


def synthesize_redemptions(
    trades_df: pd.DataFrame,
    bonds_static: dict[str, BondStatic] | None = None,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Par-redemption trades for bonds that have matured on/before ``as_of``.

    A bond redeems at par (100) on its maturity date, but that redemption is
    never sent as a trade confirmation, so a position would otherwise live on
    forever. We synthesize the redemption as a sell-at-100 of the full net
    nominal held at maturity (a buy-at-100 to cover a short), dated on the
    maturity date. Injected into the trade stream it:

      - closes the position (net nominal → 0), so it drops out of MTM/exposure;
      - realizes the pull-to-par P&L through the normal WAVG machinery; and
      - books the face value as cash leaving the bond — it is now cash on the
        balance sheet, not a position.

    One redemption per (portfolio, cusip) holding a non-zero net nominal at
    maturity. Bonds with no/blank maturity, or maturing after ``as_of``, are
    left open. Returns an empty frame (same columns as ``trades_df``) when
    nothing has matured.
    """
    empty = pd.DataFrame(columns=trades_df.columns)
    if trades_df is None or trades_df.empty:
        return empty
    if bonds_static is None:
        from accruals import load_bonds_static
        bonds_static = load_bonds_static()
    as_of = as_of or date.today()

    keys = ["portfolio", "cusip"] if "portfolio" in trades_df.columns else ["cusip"]
    rows = []
    for key_vals, group in trades_df.groupby(keys):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_map = dict(zip(keys, key_vals))
        cusip = str(key_map["cusip"])

        bond = bonds_static.get(cusip)
        if bond is None or bond.maturity_date is None or not bond.is_matured(as_of):
            continue

        mat = pd.Timestamp(bond.maturity_date)
        # Net nominal still held at maturity (only trades dated on/before it).
        net_nominal = float(group.loc[group["trade_date"] <= mat, "nominal"].sum())
        if abs(net_nominal) < _QTY_EPS:
            continue

        face = abs(net_nominal)                       # principal at par = face value
        is_long = net_nominal > 0
        rows.append({
            "Timestamp": "",
            "cusip": cusip,
            "side": "sell" if is_long else "buy",     # close the position
            "nominal": -net_nominal,                  # signed, flattens the book
            "principal": face,                        # price 100 → 100 * face / 100
            "net": face if is_long else -face,        # cash in for a long redemption
            "accrued": 0.0,                           # carry already stopped at maturity
            "price": 100.0,                           # redeemed at par
            "yield_closed": float("nan"),
            "trade_date": mat,
            "settle_date": mat,
            "trader": REDEMPTION_TRADER,
            "portfolio": key_map.get("portfolio", DEFAULT_PORTFOLIO),
        })

    if not rows:
        return empty
    out = pd.DataFrame(rows)
    # Align to the incoming frame's columns (fill any extras pandas added on concat).
    for col in trades_df.columns:
        if col not in out.columns:
            out[col] = ""
    return out[list(trades_df.columns)]


def apply_portfolio_assignment(
    combined: pd.DataFrame,
    initial: pd.DataFrame,
) -> pd.DataFrame:
    """Override each trade's ``portfolio`` with its bond's resolved portfolio.

    A bond's portfolio is decided once (manual pin → initial position → issuer
    inference; see ``bond_portfolio``) and applied to all of its trades so the
    book stays consistent. Bonds that can't be resolved confidently keep the
    portfolio their trades already carry. No-op on an empty frame.
    """
    if combined.empty or "cusip" not in combined.columns:
        return combined
    from accruals import load_bonds_static
    from bond_portfolio import resolve_portfolios, apply_to_trades, load_manual_map

    bonds_static = load_bonds_static()
    assignment = resolve_portfolios(
        initial, bonds_static,
        combined["cusip"].dropna().unique().tolist(),
        load_manual_map(),
    )
    return apply_to_trades(combined, assignment)


def load_all_trades(
    trades_path: Path | None = None,
    initial_path: Path | None = None,
    redeem_matured: bool = True,
    as_of: date | None = None,
    assign_portfolios: bool = True,
) -> pd.DataFrame:
    """Initial positions + live trades, sorted by trade_date. Single source of truth.

    When ``redeem_matured`` is True (default), par-redemption trades are appended
    for every bond that has matured on/before ``as_of`` (today if None), so
    matured positions close themselves and their proceeds move off the book as
    cash. Set it False to get the raw booked stream only.

    When ``assign_portfolios`` is True (default), each trade's ``portfolio`` is
    overridden with its bond's resolved portfolio (see ``bond_portfolio``), so a
    bond's whole trade history sits in one book regardless of what the email
    parser tagged each clip with. Set it False to keep the raw per-trade tags.
    """
    trades = load_trades(trades_path)
    initial = load_initial_positions(initial_path)
    if initial.empty:
        combined = trades
    elif trades.empty:
        combined = initial.sort_values("trade_date").reset_index(drop=True)
    else:
        combined = pd.concat([initial, trades], ignore_index=True)
        combined = combined.sort_values("trade_date").reset_index(drop=True)

    if assign_portfolios and not combined.empty:
        combined = apply_portfolio_assignment(combined, initial)

    if redeem_matured and not combined.empty:
        redemptions = synthesize_redemptions(combined, as_of=as_of)
        if not redemptions.empty:
            combined = pd.concat([combined, redemptions], ignore_index=True)
            combined = combined.sort_values("trade_date").reset_index(drop=True)

    return combined


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
