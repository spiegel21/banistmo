"""
Derivation helpers for the enterprise classification dimensions:

    - instrument type   (Sovereign / Corporate / Agency / Supranational / …)
    - country of risk    (falls back to issuer country)
    - market             (Local vs Global — domestic-currency vs hard-currency)

Every value is *derived*, never invented: when nothing resolves, the result is
``"Unknown"`` so the Debug / Needs-Attention view can flag it for manual input.

Precedence is always: explicit field on BondStatic  →  Bloomberg-extracted value
→  heuristic  →  ``"Unknown"``.
"""
from __future__ import annotations

from models import BondStatic, normalise_instrument_type

UNKNOWN = "Unknown"

# Domestic currency by ISO-2 country of risk. Used only for the Local/Global
# heuristic when the explicit `market` field is blank. Intentionally focused on
# the issuer universe a LatAm credit book is most likely to hold; unmapped
# countries fall through to "Unknown" market rather than a wrong guess.
_COUNTRY_DOMESTIC_CCY: dict[str, str] = {
    "US": "USD", "CO": "COP", "MX": "MXN", "BR": "BRL", "CL": "CLP",
    "PE": "PEN", "AR": "ARS", "PA": "USD", "DO": "DOP", "GT": "GTQ",
    "CR": "CRC", "UY": "UYU", "PY": "PYG", "EC": "USD", "BO": "BOB",
    "DE": "EUR", "FR": "EUR", "ES": "EUR", "IT": "EUR", "NL": "EUR",
    "GB": "GBP", "JP": "JPY", "CA": "CAD", "CH": "CHF", "AU": "AUD",
    "CN": "CNY",
}


def instrument_type(bond: BondStatic) -> str:
    """Canonical Sovereign/Corporate/Agency/… for a bond.

    Explicit ``instrument_type`` wins; otherwise infer from the Bloomberg ticker
    suffix (`<id> Govt` → Sovereign, `<id> Corp` → Corporate). Returns
    ``"Unknown"`` when nothing resolves.
    """
    explicit = normalise_instrument_type(bond.instrument_type)
    if explicit:
        return explicit

    ticker = (bond.bbg_ticker or "").strip().lower()
    if ticker:
        # Bloomberg yellow-key sits at the end of the ticker string.
        suffix = ticker.rsplit(" ", 1)[-1]
        inferred = normalise_instrument_type(suffix)
        if inferred:
            return inferred
    return UNKNOWN


def country_of_risk(bond: BondStatic) -> str:
    """Country of risk, falling back to the issuer ``country`` field."""
    cor = (bond.country_of_risk or "").strip()
    if cor:
        return cor.upper()
    country = (bond.country or "").strip()
    return country.upper() if country else UNKNOWN


def market_scope(bond: BondStatic) -> str:
    """Classify a bond as ``Local`` or ``Global``.

    ``Local``  = issued in the domestic currency of its country of risk
                 (a domestic-market bond).
    ``Global`` = issued in a non-domestic / hard currency (eurobond, e.g. a
                 Colombian issuer selling USD paper).

    Explicit ``market`` overrides the heuristic. Returns ``"Unknown"`` when the
    currency or country mapping is unavailable.
    """
    explicit = (bond.market or "").strip().lower()
    if explicit in ("local", "domestic"):
        return "Local"
    if explicit in ("global", "external", "hard", "hard currency", "eurobond"):
        return "Global"

    ccy = (bond.currency or "").strip().upper()
    cor = country_of_risk(bond)
    if not ccy or cor == UNKNOWN:
        return UNKNOWN
    domestic = _COUNTRY_DOMESTIC_CCY.get(cor)
    if domestic is None:
        return UNKNOWN
    return "Local" if ccy == domestic else "Global"


def is_sovereign(bond: BondStatic) -> bool:
    """True for Sovereign or Agency (quasi-sovereign) instruments."""
    return instrument_type(bond) in ("Sovereign", "Agency", "Supranational")


def classify(bond: BondStatic) -> dict[str, str]:
    """Return all derived classification dimensions for a bond as a flat dict.

    Convenience for building DataFrames in the dashboard/exposure layers.
    """
    return {
        "instrument_type": instrument_type(bond),
        "country_of_risk": country_of_risk(bond),
        "market": market_scope(bond),
        "sector": (bond.sector or "").strip() or UNKNOWN,
        "seniority": (bond.seniority or "").strip() or UNKNOWN,
        "issuer": (bond.issuer or "").strip() or UNKNOWN,
        "rating_sp": (bond.rating_sp or "").strip() or UNKNOWN,
        "rating_moody": (bond.rating_moody or "").strip() or UNKNOWN,
        "rating_fitch": (bond.rating_fitch or "").strip() or UNKNOWN,
    }
