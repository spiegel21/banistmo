"""
bond_portfolio.py — deterministic CUSIP → portfolio assignment.

The portfolio a bond belongs to is decided once, *per bond*, and then applied to
every trade of that bond so the book stays internally consistent. The email
parser's per-trade ``portfolio`` field is advisory only: once a bond has a
resolved portfolio, all of its trades (past and future) inherit it.

Resolution precedence, highest first:

  1. Manual override   — ``data/bond_portfolio_map.csv`` (``cusip,portfolio``),
     edited through the **Portfolio Assignment** view in the dashboard. This is
     where a bond an issuer can't resolve automatically gets pinned by hand.
  2. Initial position  — a bond seeded in ``initial_positions.csv`` keeps the
     portfolio it was booked under.
  3. Issuer inference  — a bond inherits its issuer's portfolio when every
     initial-position bond from that issuer sits in ONE portfolio. When an
     issuer's initial bonds span MORE THAN ONE portfolio the inference is
     ambiguous and the bond is left UNASSIGNED, to be pinned manually.

Issuer identity is the ``issuer`` field from ``bonds_static.csv`` crossed with
the bond's **currency**, falling back to the 6-digit CUSIP issuer prefix (the
same proxy the app's "Issuer (6-digit CUSIP)" rollup uses) when no issuer name
is known. The issuer-name field is preferred because two different issuers can
share a 6-digit prefix (e.g. one underwriter's range) yet belong to different
books. Currency is part of the bucket so the same obligor in two currencies
(Mex in USD vs Mex in EUR) resolves independently: if every Mex-USD initial
bond sits in one book, a new Mex-USD bond inherits it even when Mex-EUR bonds
live in a different book.

Only *confidently* resolved bonds (manual / initial / issuer) are returned in
``assigned`` — that is the map applied to trades, so an unresolved bond keeps
whatever portfolio its trades already carry rather than being silently moved to
the default book.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config
from config import DEFAULT_PORTFOLIO
from security_id import alias_map, canonical_id

SOURCE_MANUAL = "manual"
SOURCE_INITIAL = "initial position"
SOURCE_ISSUER = "issuer"
SOURCE_UNASSIGNED = "unassigned"


def issuer_key(cusip: str, bonds_static: dict | None) -> tuple[str, str, str]:
    """Return a hashable issuer identity for ``cusip``: (kind, value, currency).

    The issuer bucket is the **issuer crossed with currency**, so the same
    obligor in two currencies (e.g. Mex in USD vs Mex in EUR) is treated as two
    distinct issuers — each can resolve independently. ``value`` prefers the
    explicit ``issuer`` field; it falls back to the 6-digit CUSIP issuer prefix
    when no issuer name is known. ``kind`` marks which rule fired so an issuer
    literally named "25714P" can never collide with the prefix 25714P.
    """
    b = bonds_static.get(cusip) if bonds_static else None
    ccy = str(getattr(b, "currency", "") or "").strip().upper() if b is not None else ""
    if b is not None and str(getattr(b, "issuer", "") or "").strip():
        return ("issuer", str(b.issuer).strip(), ccy)
    return ("cusip6", str(cusip or "")[:6], ccy)


def issuer_label(key: tuple[str, str, str]) -> str:
    """Human-readable label for an issuer_key tuple, e.g. ``Mexico (USD)``."""
    kind, value, ccy = key
    base = value if kind == "issuer" else f"CUSIP {value}*"
    return f"{base} ({ccy})" if ccy else base


@dataclass
class PortfolioAssignment:
    """Resolved bond→portfolio assignment and the diagnostics behind it."""

    portfolio: dict[str, str]              # cusip → resolved portfolio (DEFAULT fallback for unassigned)
    source: dict[str, str]                 # cusip → SOURCE_*
    assigned: dict[str, str]               # cusip → portfolio, ONLY confidently resolved (manual/initial/issuer)
    issuer_label: dict[str, str]           # cusip → issuer label used for inference
    ambiguous_issuers: dict[str, list[str]]  # issuer label → sorted portfolios in conflict
    unassigned: list[str] = field(default_factory=list)  # cusips needing a manual pin

    def portfolio_of(self, cusip: str) -> str:
        return self.portfolio.get(cusip, DEFAULT_PORTFOLIO)


def load_manual_map(path: Path | None = None) -> dict[str, str]:
    """Read ``bond_portfolio_map.csv`` → {canonical cusip: portfolio}.

    Returns an empty dict when the file is absent/empty. Blank cusip/portfolio
    rows are ignored. CUSIPs are canonicalised so a manual pin keyed by an ISIN
    still lands on the bond's canonical identifier.
    """
    path = Path(path) if path is not None else config.BOND_PORTFOLIO_MAP_PATH
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    amap = alias_map()
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        cusip = str(row.get("cusip", "")).strip()
        portfolio = str(row.get("portfolio", "")).strip()
        if cusip and portfolio:
            out[canonical_id(cusip, amap)] = portfolio
    return out


def _direct_from_initial(initial_df: pd.DataFrame | None) -> dict[str, str]:
    """{cusip: portfolio} from the initial-positions frame (portfolio + cusip cols)."""
    direct: dict[str, str] = {}
    if initial_df is None or initial_df.empty:
        return direct
    if "cusip" not in initial_df.columns or "portfolio" not in initial_df.columns:
        return direct
    for _, row in initial_df.iterrows():
        cusip = str(row["cusip"]).strip()
        portfolio = str(row["portfolio"]).strip()
        if cusip:
            direct[cusip] = portfolio or DEFAULT_PORTFOLIO
    return direct


def resolve_portfolios(
    initial_df: pd.DataFrame | None,
    bonds_static: dict | None,
    cusips: list[str] | set[str] | None = None,
    manual_map: dict[str, str] | None = None,
) -> PortfolioAssignment:
    """Resolve every bond's portfolio. See module docstring for the precedence.

    ``cusips`` is the universe to resolve (typically every CUSIP that appears in
    trades); initial-position and manual-map CUSIPs are always included so they
    surface even before they trade.
    """
    manual_map = manual_map or {}
    direct = _direct_from_initial(initial_df)

    # Issuer → set of portfolios, learned only from initial positions.
    issuer_ports: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cusip, portfolio in direct.items():
        issuer_ports[issuer_key(cusip, bonds_static)].add(portfolio)
    issuer_map = {k: next(iter(v)) for k, v in issuer_ports.items() if len(v) == 1}
    ambiguous = {
        issuer_label(k): sorted(v) for k, v in issuer_ports.items() if len(v) > 1
    }

    universe = set(cusips or []) | set(direct) | set(manual_map)

    portfolio: dict[str, str] = {}
    source: dict[str, str] = {}
    assigned: dict[str, str] = {}
    labels: dict[str, str] = {}
    unassigned: list[str] = []

    for cusip in universe:
        key = issuer_key(cusip, bonds_static)
        labels[cusip] = issuer_label(key)
        if manual_map.get(cusip):
            portfolio[cusip] = manual_map[cusip]
            source[cusip] = SOURCE_MANUAL
            assigned[cusip] = manual_map[cusip]
        elif cusip in direct:
            portfolio[cusip] = direct[cusip]
            source[cusip] = SOURCE_INITIAL
            assigned[cusip] = direct[cusip]
        elif key in issuer_map:
            portfolio[cusip] = issuer_map[key]
            source[cusip] = SOURCE_ISSUER
            assigned[cusip] = issuer_map[key]
        else:
            portfolio[cusip] = DEFAULT_PORTFOLIO
            source[cusip] = SOURCE_UNASSIGNED
            unassigned.append(cusip)

    return PortfolioAssignment(
        portfolio=portfolio,
        source=source,
        assigned=assigned,
        issuer_label=labels,
        ambiguous_issuers=ambiguous,
        unassigned=sorted(unassigned),
    )


def apply_to_trades(trades_df: pd.DataFrame, assignment: PortfolioAssignment) -> pd.DataFrame:
    """Override the ``portfolio`` column of ``trades_df`` for confidently-resolved
    bonds; leave unresolved bonds' trades untouched. Returns a new frame."""
    if trades_df.empty or "cusip" not in trades_df.columns or not assignment.assigned:
        return trades_df
    out = trades_df.copy()
    mapped = out["cusip"].map(assignment.assigned)
    out["portfolio"] = mapped.fillna(out["portfolio"])
    return out
