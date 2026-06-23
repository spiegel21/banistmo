from datetime import date

import pytest

from models import BondStatic
from accruals import accrued_interest, days_accrued, last_coupon_date, load_bonds_static


def _bond(dcc, freq=2):
    return BondStatic(
        cusip="T", name="t", currency="USD", country="US",
        coupon_rate=0.05, coupon_frequency=freq, day_count_convention=dcc,
        maturity_date=date(2030, 6, 15), first_coupon_date=date(2024, 6, 15),
    )


def test_act360_half_year():
    b = _bond("Act/360")
    # from 2025-06-15 to 2025-09-13 is 90 days
    ai = accrued_interest(1_000_000, b, date(2025, 9, 13))
    assert ai == pytest.approx(1_000_000 * 0.05 * 90 / 360)


def test_30_360_clean_months():
    b = _bond("30/360")
    # last coupon 2025-06-15 → 2025-09-15 is exactly 90 days (30/360)
    assert days_accrued(b, date(2025, 9, 15)) == 90
    ai = accrued_interest(1_000_000, b, date(2025, 9, 15))
    assert ai == pytest.approx(1_000_000 * 0.05 * 90 / 360)


def test_last_coupon_date_on_coupon():
    b = _bond("Act/360")
    assert last_coupon_date(b, date(2025, 6, 15)) == date(2025, 6, 15)


def test_zero_at_coupon_date():
    b = _bond("Act/360")
    assert accrued_interest(1_000_000, b, date(2025, 6, 15)) == pytest.approx(0.0)


def test_invalid_frequency_rejected():
    with pytest.raises(ValueError):
        _bond("Act/360", freq=3)


def test_invalid_daycount_rejected():
    with pytest.raises(ValueError):
        BondStatic(cusip="T", name="t", currency="USD", country="US",
                   coupon_rate=0.05, coupon_frequency=2, day_count_convention="Foo/Bar",
                   maturity_date=date(2030, 6, 15), first_coupon_date=date(2024, 6, 15))


def test_load_skips_invalid_rows(tmp_path):
    import pandas as pd
    p = tmp_path / "bonds.csv"
    pd.DataFrame([
        dict(cusip="GOOD", name="g", currency="USD", country="US", coupon_rate=0.04,
             coupon_frequency=2, day_count_convention="Act/360",
             maturity_date="2030-01-01", first_coupon_date="2024-01-01"),
        dict(cusip="BAD", name="b", currency="USD", country="US", coupon_rate=0.04,
             coupon_frequency=3, day_count_convention="Act/360",   # freq 3 → not in {0,1,2,4,12}
             maturity_date="2030-01-01", first_coupon_date="2024-01-01"),
    ]).to_csv(p, index=False)
    bonds = load_bonds_static(p)
    assert "GOOD" in bonds and "BAD" not in bonds


def test_load_blank_coupon_loads_as_zero_coupon(tmp_path):
    """A bond with a missing/blank coupon must still load (as zero-coupon),
    not vanish from the dashboard."""
    import pandas as pd
    p = tmp_path / "bonds.csv"
    pd.DataFrame([
        # blank coupon_rate, blank coupon_frequency, blank day_count
        dict(cusip="ZC", name="zero", currency="USD", country="US", coupon_rate="",
             coupon_frequency="", day_count_convention="",
             maturity_date="2030-01-01", first_coupon_date=""),
    ]).to_csv(p, index=False)
    bonds = load_bonds_static(p)
    assert "ZC" in bonds
    b = bonds["ZC"]
    assert b.coupon_rate == 0.0
    assert b.coupon_frequency == 0
    # zero-coupon → no accrual regardless of as-of date
    assert accrued_interest(1_000_000, b, date(2025, 6, 1)) == 0.0


def test_load_zero_coupon_rate_forces_zero_frequency(tmp_path):
    """coupon_rate 0 with a stated frequency is normalised to a non-coupon bond."""
    import pandas as pd
    p = tmp_path / "bonds.csv"
    pd.DataFrame([
        dict(cusip="ZC2", name="z", currency="USD", country="US", coupon_rate=0,
             coupon_frequency=2, day_count_convention="Act/360",
             maturity_date="2030-01-01", first_coupon_date="2024-01-01"),
    ]).to_csv(p, index=False)
    b = load_bonds_static(p)["ZC2"]
    assert b.coupon_rate == 0.0 and b.coupon_frequency == 0
