import pandas as pd
import pytest

import config
import data_io
from position_manager import load_trades
from accruals import load_bonds_static


def test_save_trades_roundtrips_and_signs_nominal():
    raw = pd.DataFrame([dict(
        Timestamp="2025-06-01T09:00", cusip="037833100", side="sell", nominal=400_000,
        principal=400_000, net=401_000, accrued=1000, price=100.25, yield_closed=4.5,
        trade_date="2025-06-01", settle_date="2025-06-03", trader="CAROL", portfolio="HY",
    )])
    data_io.save_trades(raw)

    loaded = load_trades()
    row = loaded[loaded["trader"] == "CAROL"].iloc[0]
    assert row["cusip"] == "037833100"          # leading zero preserved
    assert row["nominal"] == pytest.approx(-400_000)  # sell → negative on load


def test_save_trades_creates_backup():
    # conftest already wrote trades.csv, so a backup must be produced.
    raw = pd.read_csv(config.TRADES_PATH, dtype={"cusip": str})
    data_io.save_trades(raw)
    backups = list(config.BACKUPS_DIR.glob("trades_*.csv"))
    assert backups, "expected a timestamped backup under data/backups/"


def test_save_bonds_rejects_invalid_and_leaves_file_unchanged():
    df = pd.read_csv(config.BONDS_STATIC_PATH, dtype={"cusip": str})
    df.loc[0, "coupon_frequency"] = 3  # invalid: not in {1,2,4,12}
    with pytest.raises(ValueError):
        data_io.save_bonds_static(df)
    # original file untouched
    assert load_bonds_static()["037833100"].coupon_frequency == 2


def test_save_bonds_skip_invalid_keeps_good_rows():
    # One unfetchable bond (blank maturity_date — e.g. a govt security whose
    # default "CUSIP Corp" ticker returned #N/A) must NOT block saving the
    # other bonds that resolved correctly.
    df = pd.read_csv(config.BONDS_STATIC_PATH, dtype={"cusip": str})
    assert len(df) >= 2, "fixture needs at least two bonds"
    df.loc[0, "maturity_date"] = None  # simulate Bloomberg #N/A for one bond

    # Strict mode still rejects the whole write.
    with pytest.raises(ValueError):
        data_io.save_bonds_static(df)

    # Resilient mode writes everything; good rows load, the blank one is skipped.
    data_io.save_bonds_static(df, skip_invalid=True)
    loaded = load_bonds_static()
    good_cusip = str(df.loc[1, "cusip"])
    bad_cusip = str(df.loc[0, "cusip"])
    assert good_cusip in loaded            # resolved bond survived
    assert bad_cusip not in loaded         # blank-maturity bond skipped at load
    # placeholder row is still on disk so it can be fixed and re-imported
    on_disk = pd.read_csv(config.BONDS_STATIC_PATH, dtype={"cusip": str})
    assert bad_cusip in on_disk["cusip"].astype(str).values


def test_validate_trades_flags_bad_side_and_date():
    bad = pd.DataFrame([dict(
        cusip="X", side="hold", nominal="abc", price=100,
        trade_date="not-a-date", settle_date="2025-01-02",
    )])
    problems = data_io.validate_trades(bad)
    assert any("side" in p for p in problems)
    assert any("trade_date" in p for p in problems)
    assert any("nominal" in p for p in problems)
