from dataclasses import dataclass
from datetime import date


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
    country: str            # issuer country, e.g. "US", "DE"
    coupon_rate: float          # decimal, e.g. 0.045 for 4.5%
    coupon_frequency: int       # 1=annual, 2=semi-annual, 4=quarterly
    day_count_convention: str   # "Act/360", "Act/365", "30/360"
    maturity_date: date
    first_coupon_date: date
