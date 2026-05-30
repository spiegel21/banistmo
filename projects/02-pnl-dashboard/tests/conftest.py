"""
Test fixtures: build an isolated data directory and point the app at it via
PNL_DATA_DIR *before* importing any src module (config reads the env at import).
"""
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

# Dedicated data dir for the whole test session.
_DATA_DIR = Path(tempfile.mkdtemp(prefix="pnl_test_data_"))
(_DATA_DIR / "prices").mkdir(parents=True, exist_ok=True)
os.environ["PNL_DATA_DIR"] = str(_DATA_DIR)


def _write_fixtures():
    pd.DataFrame([
        dict(Timestamp="2025-01-05T09:00", cusip="037833100", side="buy",  nominal=1_000_000, principal=985000, net=-986000, accrued=1000, price=98.5, yield_closed=5.1, trade_date="2025-01-03", settle_date="2025-01-07", trader="ALICE", portfolio="HY"),
        dict(Timestamp="2025-02-05T09:00", cusip="037833100", side="buy",  nominal=500_000,  principal=497500, net=-498000, accrued=500,  price=99.5, yield_closed=4.9, trade_date="2025-02-03", settle_date="2025-02-05", trader="ALICE", portfolio="HY"),
        dict(Timestamp="2025-03-05T09:00", cusip="037833100", side="sell", nominal=600_000,  principal=600000, net=601000,  accrued=1000, price=100.0,yield_closed=4.5, trade_date="2025-03-03", settle_date="2025-03-05", trader="BOB",   portfolio="HY"),
        dict(Timestamp="2025-02-15T09:00", cusip="912828XY9", side="buy",  nominal=2_000_000, principal=1900000,net=-1902000,accrued=2000, price=95.0, yield_closed=3.2, trade_date="2025-02-10", settle_date="2025-02-12", trader="BOB",   portfolio="IG"),
    ]).to_csv(_DATA_DIR / "trades.csv", index=False)

    pd.DataFrame([
        dict(cusip="037833100", name="APPLE 5% 2030", currency="USD", country="US", coupon_rate=0.05, coupon_frequency=2, day_count_convention="30/360", maturity_date="2030-06-15", first_coupon_date="2024-06-15"),
        dict(cusip="912828XY9", name="UST 3% 2029",  currency="USD", country="US", coupon_rate=0.03, coupon_frequency=2, day_count_convention="Act/360", maturity_date="2029-09-20", first_coupon_date="2024-09-20"),
    ]).to_csv(_DATA_DIR / "bonds_static.csv", index=False)

    pd.DataFrame([
        dict(portfolio="HY", cusip="037833100", nominal=300_000, price=97.0, book_value=-291000, inception_date="2025-01-01"),
    ]).to_csv(_DATA_DIR / "initial_positions.csv", index=False)

    pd.DataFrame([
        dict(date="2025-01-03", cusip="037833100", px_last=97.5),   # for inception-fallback test
        dict(date="2025-02-28", cusip="037833100", px_last=100.3),  # prev-day for Mar 3
        dict(date="2025-02-28", cusip="912828XY9", px_last=95.8),   # prev-day for Mar 3
        dict(date="2025-03-03", cusip="037833100", px_last=100.5),
        dict(date="2025-03-03", cusip="912828XY9", px_last=96.0),
        dict(date="2025-03-04", cusip="037833100", px_last=100.6),
        dict(date="2025-03-04", cusip="912828XY9", px_last=96.1),
    ]).to_csv(_DATA_DIR / "price_history.csv", index=False)


_write_fixtures()


@pytest.fixture(autouse=True)
def _reset_fixtures():
    """Restore the input CSVs before every test so write-tests stay isolated."""
    _write_fixtures()
    yield


@pytest.fixture
def data_dir() -> Path:
    return _DATA_DIR


@pytest.fixture
def prices() -> dict[str, float]:
    return {"037833100": 100.5, "912828XY9": 96.0}
