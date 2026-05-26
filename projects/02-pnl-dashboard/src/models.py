from dataclasses import dataclass
from datetime import date


@dataclass
class Trade:
    isin: str
    nominal: float          # signed: positive = buy, negative = sell
    principal: float        # clean_price * |nominal| / 100
    net_proceeds: float     # cash exchanged; negative for buys
    accrued_at_trade: float # accrued interest at settle date
    clean_price: float      # quoted clean price (% of par)
    yield_pct: float        # yield to maturity at trade time
    trade_date: date
    trader: str
    settle_date: date


@dataclass
class Position:
    isin: str
    net_nominal: float      # sum of signed nominals
    wavg_clean_price: float # weighted-average clean price (cost basis)
    book_value: float       # sum of net_proceeds (negative = net long position)
    last_settle: date


@dataclass
class BondStatic:
    isin: str
    name: str
    currency: str
    coupon_rate: float          # decimal, e.g. 0.045 for 4.5%
    coupon_frequency: int       # 1=annual, 2=semi-annual, 4=quarterly
    day_count_convention: str   # "Act/360", "Act/365", "30/360"
    maturity_date: date
    first_coupon_date: date
