"""
Analytical risk engine for fixed-rate bonds — YTM, duration, DV01, convexity.

Pure and framework-agnostic. The valuation convention is the standard street
approach: cash flows are discounted at the yield compounded `coupon_frequency`
times per year, with time measured in Act/365 years from the valuation date.
Duration / DV01 / convexity are computed by central finite differences off the
same `price_from_yield`, so they are internally consistent and free of
closed-form convention bugs.

Inputs come from the existing data model:
  - cash flows  ← accruals._coupon_dates / coupon_rate / maturity
  - accrued     ← accruals.accrued_interest (clean ↔ dirty)
  - clean price ← current_prices (Bloomberg / manual)

Portfolio aggregation is market-value weighted (the standard for duration/
convexity); DV01 aggregates additively in cash terms.

No external yield curve is used, so G-/Z-spread are out of scope here; VaR is
provided historically from realised daily P&L (see var_historical).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from accruals import _coupon_dates, accrued_interest
from models import BondStatic, Position

REDEMPTION = 100.0          # bullet redemption per 100 par
_DEFAULT_FREQ = 2           # discount frequency for non-coupon bonds (semi street)
_DY = 1e-5                  # finite-difference bump in yield (≈0.1bp)


def _discount_freq(bond: BondStatic) -> int:
    return bond.coupon_frequency if bond.coupon_frequency else _DEFAULT_FREQ


def remaining_cashflows(bond: BondStatic, as_of: date) -> list[tuple[date, float]]:
    """Remaining cash flows per 100 par strictly after `as_of`.

    Each future coupon pays `100 * coupon_rate / frequency`; the maturity date
    additionally redeems 100. Zero-coupon bonds return a single redemption flow.
    """
    flows: dict[date, float] = {}
    if bond.coupon_frequency:
        per = REDEMPTION * bond.coupon_rate / bond.coupon_frequency
        for d in _coupon_dates(bond):
            if d > as_of:
                flows[d] = flows.get(d, 0.0) + per
    # redemption at maturity (only if still in the future)
    if bond.maturity_date > as_of:
        flows[bond.maturity_date] = flows.get(bond.maturity_date, 0.0) + REDEMPTION
    return sorted(flows.items())


def price_from_yield(bond: BondStatic, as_of: date, y: float) -> float:
    """Dirty price per 100 par at yield `y` (decimal, e.g. 0.05)."""
    f = _discount_freq(bond)
    total = 0.0
    for d, amt in remaining_cashflows(bond, as_of):
        t = (d - as_of).days / 365.0
        if t <= 0:
            continue
        total += amt / (1.0 + y / f) ** (f * t)
    return total


def yield_to_maturity(
    bond: BondStatic, as_of: date, clean_price: float,
    lo: float = -0.50, hi: float = 2.00, tol: float = 1e-8, max_iter: int = 200,
) -> float | None:
    """Solve for the yield that reprices the bond to `clean_price`.

    Bisection on the (monotonically decreasing) dirty-price function. Returns
    None if there are no remaining cash flows or the target is unreachable.
    """
    flows = remaining_cashflows(bond, as_of)
    if not flows:
        return None
    accrued = accrued_interest(REDEMPTION, bond, as_of)
    target_dirty = clean_price + accrued

    f_lo = price_from_yield(bond, as_of, lo) - target_dirty
    f_hi = price_from_yield(bond, as_of, hi) - target_dirty
    if f_lo * f_hi > 0:
        return None  # target outside the bracket — no solution in [lo, hi]

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = price_from_yield(bond, as_of, mid) - target_dirty
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def risk_measures(bond: BondStatic, as_of: date, clean_price: float) -> dict:
    """Full risk profile for one bond at a given clean price.

    Returns ytm, macaulay/modified duration (years), dv01_per_100 (price points
    per 1bp), convexity, and the dirty price used. NaN fields when the yield
    cannot be solved.
    """
    nan = float("nan")
    out = {
        "clean_px": clean_price, "dirty_px": nan, "accrued": nan, "ytm": nan,
        "mac_duration": nan, "mod_duration": nan, "dv01_per_100": nan, "convexity": nan,
    }
    y = yield_to_maturity(bond, as_of, clean_price)
    if y is None:
        return out

    accrued = accrued_interest(REDEMPTION, bond, as_of)
    p = price_from_yield(bond, as_of, y)
    p_up = price_from_yield(bond, as_of, y + _DY)
    p_dn = price_from_yield(bond, as_of, y - _DY)
    if p <= 0:
        return out

    d_p = (p_up - p_dn) / (2 * _DY)            # dP/dy
    mod_dur = -d_p / p
    f = _discount_freq(bond)
    mac_dur = mod_dur * (1 + y / f)
    dv01_per_100 = -d_p * 1e-4                  # price move per 1bp, positive
    convexity = (p_up + p_dn - 2 * p) / (p * _DY ** 2)

    out.update({
        "dirty_px": round(p, 6), "accrued": round(accrued, 6), "ytm": round(y, 8),
        "mac_duration": round(mac_dur, 6), "mod_duration": round(mod_dur, 6),
        "dv01_per_100": round(dv01_per_100, 8), "convexity": round(convexity, 4),
    })
    return out


# ── portfolio aggregation ─────────────────────────────────────────────────────

_BOND_COLS = [
    "cusip", "name", "net_nominal", "clean_px", "dirty_px", "mtm_value",
    "ytm", "mac_duration", "mod_duration", "dv01_dollar", "convexity", "note",
]


def portfolio_risk(
    positions: dict[str, Position],
    bonds_static: dict[str, BondStatic],
    current_prices: dict[str, float],
    as_of: date | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Per-bond risk table + market-value-weighted portfolio summary.

    Portfolio modified duration / convexity / YTM are |MV|-weighted; dollar
    DV01 sums additively (signed with the position). Bonds with no price or no
    solvable yield are listed with a note and excluded from the weighting.
    """
    as_of = as_of or date.today()
    rows = []
    for cusip, pos in positions.items():
        if pos.net_nominal == 0:
            continue
        bond = bonds_static.get(cusip)
        name = (bond.name if bond and bond.name else cusip)
        clean = current_prices.get(cusip)
        if bond is None or clean is None:
            rows.append({**{c: None for c in _BOND_COLS}, "cusip": cusip,
                         "name": name, "net_nominal": pos.net_nominal,
                         "clean_px": clean,
                         "note": "no price" if bond else "missing bond static"})
            continue

        rm = risk_measures(bond, as_of, float(clean))
        dirty = rm["dirty_px"]
        mv = pos.net_nominal * dirty / 100 if dirty == dirty else None  # NaN-safe
        dv01_dollar = (rm["dv01_per_100"] * pos.net_nominal / 100
                       if rm["dv01_per_100"] == rm["dv01_per_100"] else None)
        rows.append({
            "cusip": cusip, "name": name, "net_nominal": pos.net_nominal,
            "clean_px": float(clean), "dirty_px": dirty,
            "mtm_value": round(mv, 2) if mv is not None else None,
            "ytm": rm["ytm"], "mac_duration": rm["mac_duration"],
            "mod_duration": rm["mod_duration"],
            "dv01_dollar": round(dv01_dollar, 2) if dv01_dollar is not None else None,
            "convexity": rm["convexity"],
            "note": "" if rm["ytm"] == rm["ytm"] else "no yield solution",
        })

    df = pd.DataFrame(rows, columns=_BOND_COLS)
    summary = _summarise(df)
    return df, summary


def _summarise(df: pd.DataFrame) -> dict:
    """|MV|-weighted portfolio risk summary from the per-bond table."""
    empty = {"mtm_value": 0.0, "dv01_dollar": 0.0, "mod_duration": float("nan"),
             "mac_duration": float("nan"), "convexity": float("nan"),
             "ytm": float("nan"), "n_priced": 0}
    if df.empty:
        return empty
    valid = df[df["mod_duration"].notna() & df["mtm_value"].notna()].copy()
    if valid.empty:
        return empty
    w = valid["mtm_value"].abs()
    wsum = w.sum()
    if wsum == 0:
        return empty
    return {
        "mtm_value": round(float(df["mtm_value"].sum(min_count=1) or 0.0), 2),
        "dv01_dollar": round(float(df["dv01_dollar"].sum(min_count=1) or 0.0), 2),
        "mod_duration": round(float((valid["mod_duration"] * w).sum() / wsum), 4),
        "mac_duration": round(float((valid["mac_duration"] * w).sum() / wsum), 4),
        "convexity": round(float((valid["convexity"] * w).sum() / wsum), 4),
        "ytm": round(float((valid["ytm"] * w).sum() / wsum), 6),
        "n_priced": int(len(valid)),
    }


# ── historical VaR / Expected Shortfall ───────────────────────────────────────

def var_historical(daily_pnl: pd.Series, confidence: float = 0.99) -> dict:
    """Historical VaR and Expected Shortfall from a daily P&L series.

    VaR is reported as a positive loss number (the worst (1-confidence) tail).
    ES (a.k.a. CVaR) is the mean loss beyond VaR. Returns zeros when there is
    insufficient data.
    """
    out = {"confidence": confidence, "var": 0.0, "es": 0.0, "n_obs": 0,
           "worst_day": 0.0, "best_day": 0.0}
    s = pd.to_numeric(daily_pnl, errors="coerce").dropna()
    out["n_obs"] = int(len(s))
    if len(s) < 2:
        return out
    q = s.quantile(1 - confidence)            # left-tail P&L (negative)
    tail = s[s <= q]
    out["var"] = round(float(-q), 2)
    out["es"] = round(float(-tail.mean()), 2) if not tail.empty else round(float(-q), 2)
    out["worst_day"] = round(float(s.min()), 2)
    out["best_day"] = round(float(s.max()), 2)
    return out
