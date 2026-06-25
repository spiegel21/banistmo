"""
Security-identifier normalisation: collapse CUSIP / ISIN aliases for the *same*
bond onto a single canonical key so positions, prices, and P&L never split.

The problem
-----------
Trade confirmations and price feeds reference a bond by either its 9-char CUSIP
or its 12-char ISIN.  The rest of the system keys everything on one ``cusip``
field, so the same physical bond shows up twice — once under its CUSIP, once
under its ISIN — fragmenting net position, cost basis, MTM, and accruals.

Methodology (three layers, most-trusted first)
-----------------------------------------------
1. **Structural collapse — deterministic, risk-free.**
   A US/Canada ISIN embeds its CUSIP verbatim::

       ISIN = <2-char country><9-char NSIN><1 check digit>

   For country prefixes whose national number (NSIN) *is* the CUSIP (US, CA),
   the CUSIP is characters 3..11.  We verify the ISIN check digit *and* the
   embedded CUSIP check digit before collapsing, so a malformed or coincidental
   string is never silently mis-mapped.  No external data needed.

2. **Explicit crosswalk.**
   An ``alias -> canonical-cusip`` map built from the ``isin`` column of
   ``bonds_static.csv`` (populated by the Bloomberg ``ID_ISIN`` field) and/or a
   standalone ``data/id_map.csv`` (``alias,cusip``).  Covers non-US ISINs
   (XS / DE / FR / …) that carry no embedded CUSIP, and any manual override.

3. **Name-based suggestion (caller-side).**
   ``reconciliation.check_identifier_aliases`` flags two identifiers that
   resolve to the same security *name* as a merge candidate.  This is surfaced
   in the Debug ▸ Needs-Attention view for one click to add a crosswalk entry —
   never auto-merged, because two bonds from one issuer can share a near-name.

``canonical_id(raw, alias_map)`` is the single entry point; apply it to the
``cusip`` column at every data-load edge and the merge becomes automatic and
consistent everywhere downstream.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

import config

# ── shape detection ───────────────────────────────────────────────────────────

# CUSIP: 9 chars, alphanumeric (plus the rare *, @, # in the body), last is the
# check digit.  ISIN: 2 country letters + 9 alphanumeric NSIN + 1 numeric check.
_CUSIP_RE = re.compile(r"^[0-9A-Z*@#]{9}$")
_ISIN_RE = re.compile(r"^[A-Z]{2}[0-9A-Z]{9}[0-9]$")

# ISO country prefixes whose national security number IS the CUSIP, so the CUSIP
# can be lifted straight out of the ISIN body (chars 3..11).
_CUSIP_ISIN_PREFIXES = ("US", "CA")


def _clean(raw) -> str:
    """Upper-case, strip surrounding whitespace; '' for null/blank/'nan'."""
    if raw is None:
        return ""
    s = str(raw).strip().upper()
    return "" if s in ("", "NAN", "NONE") else s


def is_cusip_shaped(s: str) -> bool:
    return bool(_CUSIP_RE.match(_clean(s)))


def is_isin_shaped(s: str) -> bool:
    return bool(_ISIN_RE.match(_clean(s)))


# ── check-digit math ──────────────────────────────────────────────────────────

def _char_value(c: str) -> int:
    """Single-char value: digits 0-9 → 0-9, A-Z → 10-35, * @ # → 36-38."""
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - ord("A") + 10
    return {"*": 36, "@": 37, "#": 38}.get(c, 0)


def cusip_check_digit(body8: str) -> str:
    """Return the CUSIP check digit for the first 8 characters."""
    total = 0
    for i, c in enumerate(body8[:8]):
        v = _char_value(c)
        if i % 2 == 1:          # double every second char (1-indexed even)
            v *= 2
        total += v // 10 + v % 10
    return str((10 - (total % 10)) % 10)


def is_valid_cusip(s: str) -> bool:
    """True only when ``s`` is 9 chars and its 9th char is the correct check digit."""
    s = _clean(s)
    if not _CUSIP_RE.match(s):
        return False
    return cusip_check_digit(s[:8]) == s[8]


def isin_check_digit(body11: str) -> str:
    """Return the ISIN check digit (Luhn mod 10) for the first 11 characters."""
    # Expand letters to two-digit numbers, then run Luhn over the digit string.
    digits = "".join(
        c if c.isdigit() else str(_char_value(c)) for c in body11[:11]
    )
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:          # rightmost digit is doubled (check digit follows)
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - (total % 10)) % 10)


def is_valid_isin(s: str) -> bool:
    """True only when ``s`` is a 12-char ISIN with a correct check digit."""
    s = _clean(s)
    if not _ISIN_RE.match(s):
        return False
    return isin_check_digit(s[:11]) == s[11]


def isin_to_cusip(isin) -> str | None:
    """Lift the embedded CUSIP out of a US/CA ISIN, or return None.

    Returns the 9-char CUSIP only when (a) the ISIN check digit is valid,
    (b) the country prefix is one whose NSIN is a CUSIP (US/CA), and (c) the
    extracted 9 chars are themselves a check-digit-valid CUSIP.  Any failure
    yields None so the caller leaves the identifier untouched rather than
    inventing a wrong mapping.
    """
    s = _clean(isin)
    if not is_valid_isin(s):
        return None
    if s[:2] not in _CUSIP_ISIN_PREFIXES:
        return None
    cusip = s[2:11]
    return cusip if is_valid_cusip(cusip) else None


# ── canonicalisation ──────────────────────────────────────────────────────────

def canonical_id(raw, alias_map: dict[str, str] | None = None) -> str:
    """Return the canonical identifier for any CUSIP/ISIN input.

    Resolution order:
      1. explicit crosswalk hit (``alias_map``) — covers non-US ISINs and
         manual overrides;
      2. structural ISIN→CUSIP collapse for US/CA ISINs (deterministic);
      3. otherwise the cleaned (upper-cased, stripped) identifier as-is.

    Idempotent: ``canonical_id(canonical_id(x)) == canonical_id(x)``.  Blank /
    null input is returned stripped (never upper-cased to a sentinel), so empty
    cells stay empty for the validators to flag.
    """
    s = _clean(raw)
    if not s:
        return "" if raw is None else str(raw).strip()

    if alias_map:
        mapped = alias_map.get(s)
        if mapped:
            return mapped

    if len(s) == 12:
        cusip = isin_to_cusip(s)
        if cusip:
            # The embedded CUSIP may itself be remapped by the crosswalk.
            return (alias_map or {}).get(cusip, cusip)

    return s


def canonicalize_series(series: pd.Series, alias_map: dict[str, str] | None = None) -> pd.Series:
    """Vectorised ``canonical_id`` over a pandas Series, preserving nulls."""
    return series.map(
        lambda v: canonical_id(v, alias_map) if pd.notna(v) else v
    )


# ── alias-map loader ──────────────────────────────────────────────────────────

def _pick_canonical_cusip(cusips: list[str]) -> str:
    """Choose the canonical id for a set of identifiers that are the same bond.

    Prefer a real, check-digit-valid CUSIP, then any CUSIP-shaped id, falling
    back to the first id when every member is ISIN-shaped.  Deterministic
    (sorted) within each tier so the choice never depends on row order.
    """
    ordered = sorted(set(cusips))
    valid = [c for c in ordered if is_valid_cusip(c)]
    if valid:
        return valid[0]
    cusip_shaped = [c for c in ordered if is_cusip_shaped(c)]
    if cusip_shaped:
        return cusip_shaped[0]
    return ordered[0]


def load_alias_map(
    bonds_static_path: Path | None = None,
    id_map_path: Path | None = None,
) -> dict[str, str]:
    """Build the ``alias -> canonical-cusip`` crosswalk from disk.

    Two sources, both optional:
      - ``bonds_static.csv`` ``isin`` column → ``{isin: cusip}`` (the Bloomberg
        ID_ISIN-populated field).  When two rows carry the *same* ISIN they are
        the same bond fetched under two identifiers, so every member CUSIP (and
        the shared ISIN) is collapsed onto one canonical CUSIP automatically —
        no reconciliation warning, no manual crosswalk entry needed.
      - ``data/id_map.csv`` (``alias,cusip``) → manual overrides for non-US
        ISINs or any identifier the structural rule can't reach.

    The canonical side is itself structurally normalised, so a bonds_static row
    that is (legacy) keyed by an ISIN still collapses to its CUSIP.  Returns an
    empty dict when neither file exists.
    """
    bonds_static_path = Path(bonds_static_path) if bonds_static_path is not None else config.BONDS_STATIC_PATH
    id_map_path = Path(id_map_path) if id_map_path is not None else config.ID_MAP_PATH

    amap: dict[str, str] = {}

    if bonds_static_path.exists() and bonds_static_path.stat().st_size > 0:
        df = pd.read_csv(bonds_static_path, dtype=str)
        if "cusip" in df.columns and "isin" in df.columns:
            # Group every row's CUSIP by the ISIN it carries.  A shared ISIN is a
            # deterministic "same bond" signal, so all CUSIPs in a group collapse
            # onto one canonical id (auto-merge); a singleton group is just the
            # plain isin→cusip crosswalk.
            by_isin: dict[str, list[str]] = {}
            for _, row in df.iterrows():
                canon = canonical_id(row.get("cusip"))
                alias = _clean(row.get("isin"))
                if canon and alias:
                    by_isin.setdefault(alias, []).append(canon)
            for isin, cusips in by_isin.items():
                canonical = _pick_canonical_cusip(cusips)
                for c in set(cusips):
                    if c != canonical:
                        amap[c] = canonical
                if isin != canonical:
                    amap[isin] = canonical

    if id_map_path.exists() and id_map_path.stat().st_size > 0:
        df = pd.read_csv(id_map_path, dtype=str)
        if "alias" in df.columns and "cusip" in df.columns:
            for _, row in df.iterrows():
                canon = canonical_id(row.get("cusip"))
                alias = _clean(row.get("alias"))
                if canon and alias and alias != canon:
                    amap[alias] = canon

    return amap


@lru_cache(maxsize=8)
def _cached_alias_map(bonds_path: str, bonds_mtime: float,
                      id_path: str, id_mtime: float) -> dict[str, str]:
    return load_alias_map(Path(bonds_path), Path(id_path))


def alias_map() -> dict[str, str]:
    """Process-cached alias map, invalidated when either source file changes.

    Loaders call this on every read; keying the cache on each file's path *and*
    mtime keeps a long-running Streamlit session in sync after a Data Editor /
    Bloomberg import edits the crosswalk (and keeps test temp-dirs isolated),
    without re-reading CSVs on every single row.
    """
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    bs, idm = config.BONDS_STATIC_PATH, config.ID_MAP_PATH
    return _cached_alias_map(str(bs), _mtime(bs), str(idm), _mtime(idm))
