from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

VALID_DAY_COUNTS = {"Act/360", "Act/365", "30/360"}
VALID_FREQUENCIES = {1, 2, 4, 12}


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
        if self.maturity_date <= self.first_coupon_date:
            raise ValueError(
                f"{self.cusip}: maturity_date ({self.maturity_date}) must be after "
                f"first_coupon_date ({self.first_coupon_date})"
            )
        if not 0 <= self.coupon_rate < 1:
            raise ValueError(
                f"{self.cusip}: coupon_rate {self.coupon_rate!r} should be a decimal "
                f"(e.g. 0.05 for 5%), got a value outside [0, 1)"
            )
