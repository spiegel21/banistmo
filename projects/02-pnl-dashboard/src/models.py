from __future__ import annotations

from dataclasses import dataclass
from datetime import date

VALID_DAY_COUNTS = {"Act/360", "Act/365", "30/360", "Act/Act"}
VALID_FREQUENCIES = {0, 1, 2, 4, 12}   # 0 = zero-coupon / non-coupon bearing

# Normalization map: any Bloomberg or user-typed variant → internal canonical form.
# Keys are lower-cased; the lookup in normalise_day_count() strips and lower-cases first.
_DAY_COUNT_ALIASES: dict[str, str] = {
    # ACT/360 family
    "act/360": "Act/360",
    "actual/360": "Act/360",
    "a/360": "Act/360",
    # ACT/365 family
    "act/365": "Act/365",
    "actual/365": "Act/365",
    "a/365": "Act/365",
    # ACT/ACT family (ISMA / ICMA)
    "act/act": "Act/Act",
    "actual/actual": "Act/Act",
    "a/a": "Act/Act",
    "act/act (isma)": "Act/Act",
    "act/act (icma)": "Act/Act",
    "actual/actual (isma)": "Act/Act",
    "actual/actual (icma)": "Act/Act",
    # 30/360 family
    "30/360": "30/360",
    "30/360 bond": "30/360",
    "30/360 isda": "30/360",
    "isma-30/360": "30/360",
    "isma-30/360 noneom": "30/360",
    "isma 30/360": "30/360",
    "30e/360": "30/360",
    "bond basis": "30/360",
}


def normalise_day_count(raw: str) -> str:
    """Convert any known day-count string to internal canonical form.

    Already-canonical values are returned unchanged. Unknown strings are
    returned as-is (BondStatic.__post_init__ will reject them with a clear
    error if they are not in VALID_DAY_COUNTS).
    """
    if not raw:
        return raw
    return _DAY_COUNT_ALIASES.get(raw.strip().lower(), raw.strip())


# Canonical instrument-type buckets used everywhere in the app for the
# sovereign-vs-corp dimension. Empty string == unknown (never guessed silently).
VALID_INSTRUMENT_TYPES = {
    "Sovereign", "Corporate", "Agency", "Supranational", "Municipal", "Other", ""
}

# Any Bloomberg MARKET_SECTOR_DES / user-typed variant → canonical bucket.
# Keys are lower-cased; the lookup strips and lower-cases first.
_INSTRUMENT_TYPE_ALIASES: dict[str, str] = {
    # sovereign / government
    "sovereign": "Sovereign",
    "govt": "Sovereign",
    "government": "Sovereign",
    "treasury": "Sovereign",
    "sov": "Sovereign",
    # corporate
    "corp": "Corporate",
    "corporate": "Corporate",
    "credit": "Corporate",
    # agency / quasi-sovereign
    "agency": "Agency",
    "agncy": "Agency",
    "quasi": "Agency",
    "quasi-sovereign": "Agency",
    # supranational
    "supra": "Supranational",
    "supranational": "Supranational",
    # municipal
    "muni": "Municipal",
    "municipal": "Municipal",
    "mun": "Municipal",
    # other / pass-through
    "other": "Other",
}


def normalise_instrument_type(raw: str) -> str:
    """Map any Bloomberg/user instrument-type string to a canonical bucket.

    Unknown non-empty strings fall through to ``"Other"``; blank stays blank
    (== unknown) so the debug view can flag it as needing manual input.
    """
    if not raw or not str(raw).strip():
        return ""
    key = str(raw).strip().lower()
    if key in _INSTRUMENT_TYPE_ALIASES:
        return _INSTRUMENT_TYPE_ALIASES[key]
    # Already-canonical (case-insensitive) values pass through unchanged.
    for canon in VALID_INSTRUMENT_TYPES:
        if canon and key == canon.lower():
            return canon
    return "Other"


@dataclass
class Trade:
    cusip: str
    side: str               # "buy" or "sell" (normalised to lowercase)
    nominal: float          # signed: positive = buy, negative = sell
    principal: float        # price * |nominal| / 100
    net: float              # cash exchanged; negative for buys
    accrued: float          # accrued interest at settle date
    price: float            # quoted clean price (% of par)
    yield_closed: float | None  # yield to maturity at trade time; may be None
    trade_date: date
    settle_date: date
    trader: str
    portfolio: str          # book/portfolio name, e.g. "HY Book"


@dataclass
class Position:
    cusip: str
    net_nominal: float      # sum of signed nominals
    wavg_price: float       # weighted-average clean price (cost basis)
    book_value: float       # sum of net (negative = net long position)
    last_settle: date


@dataclass
class BondStatic:
    cusip: str
    name: str
    currency: str
    country: str                # issuer country, e.g. "US", "DE"
    coupon_rate: float          # decimal, e.g. 0.045 for 4.5%
    coupon_frequency: int       # 1=annual, 2=semi-annual, 4=quarterly, 12=monthly
    day_count_convention: str   # "Act/360", "Act/365", "30/360"
    maturity_date: date
    first_coupon_date: date
    bbg_ticker: str = ""        # full Bloomberg ticker e.g. "912828Z78 Govt"; blank → cusip + " Corp"
    isin: str = ""              # ISIN crosswalk (Bloomberg ID_ISIN); "" == unknown. Used to merge
                                # ISIN-keyed trades/prices onto this bond's canonical CUSIP.
    instrument_type: str = ""   # canonical: Sovereign/Corporate/Agency/Supranational/Municipal/Other/"" (unknown)
    # ── enterprise classification dimensions (all optional; "" == unknown) ──────
    issuer: str = ""            # issuer / obligor name (Bloomberg ISSUER)
    country_of_risk: str = ""   # country of risk (Bloomberg CNTRY_OF_RISK); may differ from `country`
    sector: str = ""            # industry sector (Bloomberg INDUSTRY_SECTOR / BICS level 1)
    seniority: str = ""         # payment rank / seniority (Bloomberg PAYMENT_RANK)
    market: str = ""            # "Local" or "Global" (domestic-ccy vs hard-ccy / eurobond); "" == derive
    rating_sp: str = ""         # S&P rating (Bloomberg RTG_SP)
    rating_moody: str = ""      # Moody's rating (Bloomberg RTG_MOODY)
    rating_fitch: str = ""      # Fitch rating (Bloomberg RTG_FITCH)

    def __post_init__(self):
        if self.coupon_frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"{self.cusip}: coupon_frequency {self.coupon_frequency!r} not in {VALID_FREQUENCIES}"
            )
        if self.day_count_convention not in VALID_DAY_COUNTS:
            raise ValueError(
                f"{self.cusip}: day_count_convention {self.day_count_convention!r} "
                f"not in {VALID_DAY_COUNTS}"
            )
        # Allow first_coupon_date == maturity_date for zero-coupon / bullet bonds
        # where Bloomberg sets first_coupon_date = maturity_date.
        if self.maturity_date < self.first_coupon_date:
            raise ValueError(
                f"{self.cusip}: maturity_date ({self.maturity_date}) must not be before "
                f"first_coupon_date ({self.first_coupon_date})"
            )
        if not 0 <= self.coupon_rate < 1:
            raise ValueError(
                f"{self.cusip}: coupon_rate {self.coupon_rate!r} should be a decimal "
                f"(e.g. 0.05 for 5%), got a value outside [0, 1)"
            )

    def is_matured(self, as_of: date) -> bool:
        """True once the bond has reached or passed maturity (redeemed at par).

        On and after the maturity date the bond no longer accrues interest and
        has no remaining cash flows; callers use this to stop carry/MTM and to
        flag still-open positions that should have been redeemed. A missing
        maturity_date is treated as "not matured" (unknown, never guessed).
        """
        return self.maturity_date is not None and as_of >= self.maturity_date
