"""
Exposure, concentration, and maturity-ladder analytics.

Transparency-grade risk reporting that needs no pricing/yield model: it
aggregates *nominal* and *market value* (MTM) across the classification
dimensions (country of risk, sector, issuer, currency, sovereign/corp,
local/global) and buckets the book by remaining tenor.

Everything is built on the existing mark_to_market output so the numbers
reconcile with the rest of the app; classification dimensions come from
classification.classify so derivation rules stay in one place.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from classification import classify, UNKNOWN
from mtm import mark_to_market
from models import BondStatic, Position

# Dimension key → BondStatic-derived column produced by exposure_base().
DIMENSIONS = {
    "Country of Risk": "country_of_risk",
    "Sector": "sector",
    "Issuer": "issuer",
    "Currency": "currency",
    "Sovereign / Corp": "instrument_type",
    "Local / Global": "market",
    "Seniority": "seniority",
    "Portfolio": "portfolio",
}

# Remaining-tenor buckets (years), ordered.
_TENOR_BANDS = [
    (0.0, 1.0, "0-1y"),
    (1.0, 3.0, "1-3y"),
    (3.0, 5.0, "3-5y"),
    (5.0, 7.0, "5-7y"),
    (7.0, 10.0, "7-10y"),
    (10.0, float("inf"), "10y+"),
]


def exposure_base(
    positions: dict[str, Position],
    bonds_static: dict[str, BondStatic],
    current_prices: dict[str, float],
    as_of: date | None = None,
) -> pd.DataFrame:
    """One row per held CUSIP with MTM value + every classification dimension.

    This is the join-ready foundation for all the aggregations below.
    """
    as_of = as_of or date.today()
    mtm = mark_to_market(positions, current_prices, bonds_static, as_of)
    if mtm.empty:
        return pd.DataFrame()

    rows = []
    for _, m in mtm.iterrows():
        cusip = m["cusip"]
        bond = bonds_static.get(cusip)
        dims = classify(bond) if bond else {k: UNKNOWN for k in
                                            ("instrument_type", "country_of_risk",
                                             "market", "sector", "seniority", "issuer")}
        mv = m.get("mtm_value")
        rows.append({
            "cusip": cusip,
            "name": (bond.name if bond and bond.name else cusip),
            "net_nominal": m.get("net_nominal", 0.0),
            "mtm_value": float(mv) if mv is not None and not pd.isna(mv) else 0.0,
            "currency": (bond.currency if bond and bond.currency else UNKNOWN),
            "maturity_date": (bond.maturity_date if bond else None),
            **{k: dims.get(k, UNKNOWN) for k in
               ("instrument_type", "country_of_risk", "market", "sector",
                "seniority", "issuer")},
        })
    return pd.DataFrame(rows)


def aggregate_exposure(base: pd.DataFrame, dimension_col: str) -> pd.DataFrame:
    """Aggregate nominal & MTM by a dimension, with % of book and counts.

    Sorted by absolute MTM value descending (largest exposures first).
    """
    cols = [dimension_col, "net_nominal", "mtm_value", "pct_nominal", "pct_mtm", "n_bonds"]
    if base is None or base.empty or dimension_col not in base.columns:
        return pd.DataFrame(columns=cols)

    grp = (
        base.groupby(dimension_col)
        .agg(net_nominal=("net_nominal", "sum"),
             mtm_value=("mtm_value", "sum"),
             n_bonds=("cusip", "nunique"))
        .reset_index()
    )
    total_nom = grp["net_nominal"].abs().sum()
    total_mtm = grp["mtm_value"].abs().sum()
    grp["pct_nominal"] = (grp["net_nominal"].abs() / total_nom * 100).round(2) if total_nom else 0.0
    grp["pct_mtm"] = (grp["mtm_value"].abs() / total_mtm * 100).round(2) if total_mtm else 0.0
    grp = grp.sort_values("mtm_value", key=lambda s: s.abs(), ascending=False)
    return grp[cols].reset_index(drop=True)


def concentration(base: pd.DataFrame, dimension_col: str, top_n: int = 5) -> dict:
    """Top-N concentration summary for a dimension.

    Returns dict with the top-N share of MTM and the largest single bucket.
    """
    agg = aggregate_exposure(base, dimension_col)
    if agg.empty:
        return {"top_n": top_n, "top_n_pct": 0.0, "largest": None, "largest_pct": 0.0}
    top = agg.head(top_n)
    return {
        "top_n": top_n,
        "top_n_pct": round(float(top["pct_mtm"].sum()), 2),
        "largest": agg.iloc[0][dimension_col],
        "largest_pct": round(float(agg.iloc[0]["pct_mtm"]), 2),
    }


def maturity_ladder(base: pd.DataFrame, as_of: date | None = None) -> pd.DataFrame:
    """Bucket nominal & MTM by remaining tenor band.

    Bonds with no maturity date land in an "Unknown" bucket so they are never
    silently dropped.
    """
    cols = ["bucket", "net_nominal", "mtm_value", "n_bonds"]
    if base is None or base.empty:
        return pd.DataFrame(columns=cols)
    as_of = as_of or date.today()

    def _band(mat) -> str:
        if mat is None or pd.isna(mat):
            return UNKNOWN
        years = (pd.Timestamp(mat) - pd.Timestamp(as_of)).days / 365.25
        if years < 0:
            return "Matured"
        for lo, hi, label in _TENOR_BANDS:
            if lo <= years < hi:
                return label
        return UNKNOWN

    b = base.copy()
    b["bucket"] = b["maturity_date"].map(_band)
    grp = (
        b.groupby("bucket")
        .agg(net_nominal=("net_nominal", "sum"),
             mtm_value=("mtm_value", "sum"),
             n_bonds=("cusip", "nunique"))
        .reset_index()
    )
    # order buckets sensibly
    order = ["Matured"] + [lbl for _, _, lbl in _TENOR_BANDS] + [UNKNOWN]
    grp["__o"] = grp["bucket"].map({lbl: i for i, lbl in enumerate(order)}).fillna(99)
    return grp.sort_values("__o").drop(columns="__o")[cols].reset_index(drop=True)
