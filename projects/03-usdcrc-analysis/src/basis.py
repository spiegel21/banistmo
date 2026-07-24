"""Single source of truth for the accounting basis used across the whole project.

Every module — legacy or new — annualises P&L and Sharpe, charges slippage, and
sizes trades against the constants defined here, so no two tables in the report can
silently disagree on the basis.

Basis (fixed a-priori, never fitted):
  * Notional          USD 1,000,000 per full position (``-1`` = short USD 1M).
  * Slippage          0.325 CRC per side  ==  0.65 CRC per round trip.
  * Pricing           VWAP-to-VWAP next session (the realistic desk fill).
  * Annualisation     231 sessions/year — the density MONEX actually trades
                      (2,663 sessions over 11.53 calendar years ≈ 231.0), NOT the
                      252 equity-market convention. Using the real density is why
                      the numbers here are ~9% lower in $/yr and ~4.5% lower in
                      Sharpe than an earlier 252-annualised draft.

``ann_days(df)`` returns the same figure measured directly from a frame's own date
span; the modern ranking/exit/intervention modules use it so their annualisation is
literally derived from the data. It evaluates to ~231.0 on the full analysis frame,
identical to ``SESSIONS_PER_YEAR`` at every displayed precision.
"""
from __future__ import annotations

import math

# --- the basis ------------------------------------------------------------- #
NOTIONAL_USD = 1_000_000          # USD per full position
COST_CRC_PER_SIDE = 0.325         # slippage charged on every change in position
COST_CRC_ROUNDTRIP = 0.65         # == 2 * COST_CRC_PER_SIDE
SESSIONS_PER_YEAR = 231.0         # MONEX's real trading density (not 252)
SESSIONS_PER_YEAR_LEGACY = 252    # the equity convention this project deliberately drops

# Calendar (quincena) rule window — the single tunable of the recommended strategy:
# SHORT USD when within CAL_PRE business days of the IVA/quincena deadline (through
# it), LONG otherwise. Lives here (the basis) rather than in quincena.py so the
# lightweight daily-signal path can read it without importing the research stack;
# quincena.py re-exports it, so its dependents are unaffected.
CAL_PRE = 6

# Sharpe annualisation factor (daily Sharpe * ANN = annualised Sharpe).
ANN = math.sqrt(SESSIONS_PER_YEAR)

# How much the dropped 252 convention overstated the legacy figures.
PER_YEAR_LEGACY_INFLATION = SESSIONS_PER_YEAR_LEGACY / SESSIONS_PER_YEAR - 1     # ~ +9.1%
SHARPE_LEGACY_INFLATION = math.sqrt(SESSIONS_PER_YEAR_LEGACY / SESSIONS_PER_YEAR) - 1  # ~ +4.5%


def ann_days(df):
    """Sessions per year measured from the frame's own date span (~231 for MONEX).

    ``span_yrs`` uses calendar days / 365.25 so leap years are handled; dividing the
    row count by it gives the venue's realised session density. Kept as a function
    (not just the constant) so the ranking/exit/intervention modules annualise on a
    figure derived directly from the data they price.
    """
    span_yrs = (df.date.max() - df.date.min()).days / 365.25
    return len(df) / span_yrs
