"""Real Costa Rica payment & tax calendar for the USD/CRC flow model.

The crude quincena rule in `quincena.py` shorts USD on a *fixed* day-of-month
window (days 5-15). That window is only a proxy for the actual cash-flow events
that drive CRC demand (companies sell USD to raise colones to pay them):

  * IVA (D-104) + income-tax withholdings (D-103): statutory deadline the **15th
    of the following month**, rolled to the next business day
    (Hacienda / TRIBU-CR fiscal calendar).
  * Quincena — first fortnight payroll: paid mid-month (~the 15th).
  * Quincena — second fortnight payroll: paid at month-end.
  * CCSS planilla (social security): due by the **4th business day** of the month.
  * Pagos parciales de renta (partial income tax): quarterly, end of
    Mar / Jun / Sep / Dec.

Rather than carry a separate holiday file, we anchor every statutory date to
**MONEX's own trading calendar** (the set of dates that actually appear in the
data): the "15th" becomes the first trading day on/after the 15th, the "4th
business day" is the 4th trading day of the month, "month-end" is the last
trading day, etc. That makes the calendar business-day- and holiday-correct by
construction and perfectly consistent with the price series it is scored on.

`build_calendar(dates)` returns the per-month event dates; `annotate(df)` adds,
for every trading day, the signed trading-day distance to the mid-month
IVA/quincena anchor plus the event-window flags used by the calendar strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Statutory day-of-month targets (rolled onto the trading calendar in build_calendar).
IVA_DAY = 15            # D-104 IVA + D-103 retenciones: 15th of the following month
CCSS_BUSINESS_DAY = 4   # CCSS planilla: 4th business day of the month
RENTA_MONTHS = (3, 6, 9, 12)   # pagos parciales de renta: quarter-ends


def _first_trading_on_or_after(day_idx: pd.DatetimeIndex, y: int, m: int, dom: int):
    """First trading date in month (y, m) with day-of-month >= dom (or None)."""
    target = pd.Timestamp(y, m, 1) + pd.offsets.Day(dom - 1)
    cand = day_idx[(day_idx.year == y) & (day_idx.month == m) & (day_idx >= target)]
    return cand.min() if len(cand) else None


def build_calendar(dates) -> pd.DataFrame:
    """Per (year, month) statutory event dates anchored on the trading calendar.

    Columns: year, month, iva (mid-month IVA/quincena anchor), ccss (4th trading
    day), quincena_end (last trading day), renta (quarter-end renta or NaT).
    """
    day_idx = pd.DatetimeIndex(pd.to_datetime(sorted(pd.unique(dates)))).normalize()
    rows = []
    months = sorted({(d.year, d.month) for d in day_idx})
    for y, m in months:
        month_days = day_idx[(day_idx.year == y) & (day_idx.month == m)].sort_values()
        if len(month_days) == 0:
            continue
        iva = _first_trading_on_or_after(day_idx, y, m, IVA_DAY)
        if iva is None:                       # no trading day on/after the 15th: use last
            iva = month_days.max()
        ccss = month_days[min(CCSS_BUSINESS_DAY - 1, len(month_days) - 1)]
        quincena_end = month_days.max()
        renta = quincena_end if m in RENTA_MONTHS else pd.NaT
        rows.append({"year": y, "month": m, "iva": iva, "ccss": ccss,
                     "quincena_end": quincena_end, "renta": renta})
    return pd.DataFrame(rows)


def iva_anchor(year: int, month: int) -> np.datetime64:
    """Business-day-adjusted IVA/quincena deadline: first weekday on/after the 15th.

    Forward-looking companion to build_calendar for dates that have no price data
    yet (e.g. tomorrow's session). Uses a Mon-Fri week mask; MONEX market holidays
    are few, so this matches the trading-calendar anchor in practice.
    """
    d = np.datetime64(f"{year:04d}-{month:02d}-{IVA_DAY:02d}")
    return np.busday_offset(d, 0, roll="forward")


def td_to_iva_for(target) -> int:
    """Signed business-day distance from `target` to its month's IVA anchor.

    >0 before the deadline, 0 on it, <0 after — same convention as annotate()'s
    `td_to_iva`. Works for future dates the price series doesn't cover yet.
    """
    target = pd.Timestamp(target)
    t = np.datetime64(target.date())
    return int(np.busday_count(t, iva_anchor(target.year, target.month)))


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add real-calendar features to a daily frame that has a `date` column.

    Adds:
      td_to_iva      signed trading-day distance to THIS month's IVA/quincena
                     anchor (>0 before the anchor, 0 on it, <0 after).
      td_from_ccss   trading days since the 4th-business-day CCSS date (>=0).
      td_to_meend    trading days until the month-end quincena payday (>=0).
      is_iva / is_ccss / is_meend / is_renta : event-day flags.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    cal = build_calendar(df["date"]).set_index(["year", "month"])
    tdate = df["date"].dt.normalize()
    # trading-day ordinal for every date -> lets us measure distances in *sessions*
    order = {d: i for i, d in enumerate(tdate)}
    idx_of = lambda d: order.get(pd.Timestamp(d), np.nan)

    ym = list(zip(tdate.dt.year, tdate.dt.month))
    iva_idx = np.array([idx_of(cal.loc[k, "iva"]) if k in cal.index else np.nan for k in ym])
    ccss_idx = np.array([idx_of(cal.loc[k, "ccss"]) if k in cal.index else np.nan for k in ym])
    meend_idx = np.array([idx_of(cal.loc[k, "quincena_end"]) if k in cal.index else np.nan for k in ym])
    self_idx = np.arange(len(df), dtype=float)

    df["td_to_iva"] = iva_idx - self_idx        # +ve: anchor still ahead this month
    df["td_from_ccss"] = self_idx - ccss_idx     # +ve: after CCSS date
    df["td_to_meend"] = meend_idx - self_idx     # +ve: month-end payday ahead
    df["is_iva"] = df["td_to_iva"] == 0
    df["is_ccss"] = df["td_from_ccss"] == 0
    df["is_meend"] = df["td_to_meend"] == 0
    renta_dates = set(cal["renta"].dropna())
    df["is_renta"] = tdate.isin(renta_dates)
    return df


if __name__ == "__main__":
    from analyze import load
    d = annotate(load())
    cal = build_calendar(d["date"])
    print(f"Built calendar for {len(cal)} months "
          f"({d.date.min().date()} -> {d.date.max().date()})\n")
    print("Sample of recent monthly event dates (trading-day anchored):")
    print(cal.tail(6).to_string(index=False))
    print("\nMid-month IVA/quincena anchor — day-of-month distribution (rolled to "
          "next trading day):")
    print(cal["iva"].dt.day.value_counts().sort_index().to_string())
