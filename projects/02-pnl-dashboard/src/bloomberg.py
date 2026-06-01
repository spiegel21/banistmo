"""
Bloomberg pricing bridge via xlwings.

BDP (current prices): writes CUSIPs to Sheet1, triggers BDP formula refresh,
  reads back clean prices. Appends results to price_history.csv.

BDH (historical prices): writes one CUSIP at a time to the History sheet,
  triggers BDH formula, reads back the date/price time series.

Requirements:
  - Windows OS
  - Bloomberg Terminal running and logged in
  - xlwings + pywin32 installed
  - templates/bloomberg_prices.xlsx with Sheet1 and History sheets

Fallback: if Excel/Bloomberg is unavailable, reads from data/prices/manual_prices.csv
  or data/prices/manual_price_history.csv for historical data.
"""
import csv
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

import config
from config import get_logger

log = get_logger(__name__)

TEMPLATE_PATH = config.BLOOMBERG_TEMPLATE_PATH
PRICES_DIR = config.PRICES_DIR
MANUAL_PRICES_PATH = config.MANUAL_PRICES_PATH
MANUAL_HISTORY_PATH = config.MANUAL_HISTORY_PATH
PRICE_HISTORY_PATH = config.PRICE_HISTORY_PATH

_BDP_TIMEOUT = 30   # seconds
_BDH_TIMEOUT = 60   # seconds (BDH is slower)
_POLL_INTERVAL = 2


def _try_xlwings_import():
    try:
        import xlwings as xw
        return xw
    except ImportError:
        return None


# ── BDP (current prices) ─────────────────────────────────────────────────────

def _write_cusips(cusips: list[str], wb) -> None:
    """Write CUSIPs into Sheet1 column A starting at row 2, forcing text format."""
    sheet = wb.sheets["Sheet1"]
    last_row = sheet.range("A2").end("down").row
    if last_row >= 2:
        sheet.range(f"A2:A{max(last_row, len(cusips) + 1)}").clear_contents()
    for i, cusip in enumerate(cusips):
        cell = sheet["A" + str(i + 2)]
        cell.number_format = "@"   # force text so leading zeros are preserved
        cell.value = str(cusip)


def _all_prices_filled(sheet, n: int) -> bool:
    """Return True only when all price cells contain numeric values (not Bloomberg error strings)."""
    values = sheet.range(f"B2:B{n + 1}").value
    if values is None:
        return False
    if isinstance(values, list):
        return all(isinstance(v, (int, float)) for v in values)
    return isinstance(values, (int, float))


def _refresh_and_read_bdp(wb, cusips: list[str]) -> dict[str, dict]:
    """Trigger BDP refresh and poll until all prices are numeric, then return results.
    Uses the original cusips list as keys to avoid Excel stripping leading zeros."""
    n_cusips = len(cusips)
    sheet = wb.sheets["Sheet1"]
    wb.app.calculate()

    elapsed = 0
    while elapsed < _BDP_TIMEOUT:
        if _all_prices_filled(sheet, n_cusips):
            break
        time.sleep(_POLL_INTERVAL)
        wb.app.calculate()
        elapsed += _POLL_INTERVAL

    px_last = sheet.range(f"B2:B{n_cusips + 1}").value
    asw = sheet.range(f"C2:C{n_cusips + 1}").value
    ytm = sheet.range(f"D2:D{n_cusips + 1}").value

    if n_cusips == 1:
        px_last, asw, ytm = [px_last], [asw], [ytm]

    result = {}
    for i, cusip in enumerate(cusips):
        px = px_last[i] if isinstance(px_last[i], (int, float)) else None
        result[str(cusip)] = {
            "px_last": px,
            "asw": asw[i] if asw and isinstance(asw[i], (int, float)) else None,
            "ytm": ytm[i] if ytm and isinstance(ytm[i], (int, float)) else None,
        }
    return result


def get_prices(cusips: list[str], wb_path: Path = TEMPLATE_PATH) -> dict[str, float]:
    """
    Fetch current prices via Bloomberg BDP.

    Returns {cusip: px_last}. Appends results to price_history.csv.
    Falls back to manual_prices.csv if Bloomberg is unavailable.
    """
    xw = _try_xlwings_import()

    if xw is not None:
        try:
            app = xw.App(visible=False, add_book=False)
            try:
                wb = app.books.open(str(wb_path))
                _write_cusips(cusips, wb)
                data = _refresh_and_read_bdp(wb, cusips)
                wb.save()
                wb.close()
            finally:
                app.quit()

            prices = {c: v["px_last"] for c, v in data.items() if v["px_last"] is not None}
            _save_price_snapshot(prices)
            _append_to_price_history(prices, date.today())
            return prices
        except Exception as e:
            log.warning("Bloomberg BDP failed (%s); falling back to manual prices", e)

    return _load_manual_prices(cusips)


# ── BDH (historical prices) ──────────────────────────────────────────────────

def get_historical_prices_bdh(
    cusips: list[str],
    start_date: date,
    end_date: date,
    wb_path: Path = TEMPLATE_PATH,
) -> pd.DataFrame:
    """
    Fetch historical daily prices via Bloomberg BDH for a list of CUSIPs.

    Returns a DataFrame with columns: date, cusip, px_last.
    Appends results to price_history.csv.
    Falls back to manual_price_history.csv if Bloomberg is unavailable.

    Uses the "History" sheet in bloomberg_prices.xlsx:
        A1: ticker string (e.g. "037833100 Corp")
        B1: start date  ("YYYYMMDD")
        C1: end date    ("YYYYMMDD")
        A3: =BDH(A1,"PX_LAST",B1,C1,"Fill","0","Dates","0")
              ↑ BDH fills downward: col A = dates, col B = prices
    """
    xw = _try_xlwings_import()

    if xw is not None:
        try:
            records = _fetch_bdh_via_excel(cusips, start_date, end_date, wb_path, xw)
            if records:
                df = pd.DataFrame(records)
                _append_df_to_price_history(df)
                return df
        except Exception as e:
            log.warning("Bloomberg BDH failed (%s); falling back to manual history", e)

    return _load_manual_price_history(cusips, start_date, end_date)


def _fetch_bdh_via_excel(cusips, start_date, end_date, wb_path, xw) -> list[dict]:
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    ticker_suffix = " Corp"

    app = xw.App(visible=False, add_book=False)
    records = []
    try:
        wb = app.books.open(str(wb_path))
        sheet = wb.sheets["History"]

        for cusip in cusips:
            sheet.clear_contents()

            # Write parameters
            sheet["A1"].value = cusip + ticker_suffix
            sheet["B1"].value = start_str
            sheet["C1"].value = end_str

            # BDH formula: returns dates in col A, prices in col B, starting row 3
            sheet["A3"].value = (
                f'=BDH(A1,"PX_LAST",B1,C1,"Fill","0","Dates","1")'
            )

            # Trigger and wait
            app.calculate()
            elapsed = 0
            while elapsed < _BDH_TIMEOUT:
                val = sheet["B3"].value
                if val is not None and val != "":
                    break
                time.sleep(_POLL_INTERVAL)
                app.calculate()
                elapsed += _POLL_INTERVAL

            # Read returned time series
            data_range = sheet.range("A3").expand().value
            if data_range is None:
                continue
            # Single-row result comes back as a flat list, not a list of lists
            if not isinstance(data_range[0], list):
                data_range = [data_range]

            for row in data_range:
                if row and row[0] and row[1]:
                    row_date = row[0].date() if hasattr(row[0], "date") else row[0]
                    records.append({
                        "date": row_date,
                        "cusip": cusip,
                        "px_last": float(row[1]),
                    })

        wb.close()
    finally:
        app.quit()

    return records


# ── price_history.csv helpers ─────────────────────────────────────────────────

def _append_to_price_history(prices: dict[str, float], as_of: date) -> None:
    """Append a dict of {cusip: px_last} for a single date to price_history.csv."""
    rows = [{"date": as_of.isoformat(), "cusip": c, "px_last": px} for c, px in prices.items()]
    _append_df_to_price_history(pd.DataFrame(rows))


def _append_df_to_price_history(df: pd.DataFrame) -> None:
    """Append a DataFrame (date, cusip, px_last) to price_history.csv, deduplicating."""
    path = PRICE_HISTORY_PATH

    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path, dtype={"cusip": str})
        combined = pd.concat([existing, df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.date.astype(str)
        combined = combined.drop_duplicates(subset=["date", "cusip"], keep="last")
        combined.sort_values(["date", "cusip"]).to_csv(path, index=False)
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        df.sort_values(["date", "cusip"]).to_csv(path, index=False)


def load_price_history(path: Path = PRICE_HISTORY_PATH) -> pd.DataFrame:
    """Load full price history; returns empty DataFrame if file doesn't exist."""
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return pd.DataFrame(columns=["date", "cusip", "px_last"])
    df = pd.read_csv(path, dtype={"cusip": str})
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── snapshot helpers (daily timestamped files) ────────────────────────────────

def _save_price_snapshot(prices: dict[str, float]) -> None:
    Path(PRICES_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PRICES_DIR / f"prices_{ts}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cusip", "px_last", "date"])
        today = date.today().isoformat()
        for cusip, px in prices.items():
            writer.writerow([str(cusip), px, today])


def load_latest_prices() -> dict[str, float]:
    """Load the most recently saved BDP snapshot (Bloomberg or manual)."""
    snapshots = sorted(PRICES_DIR.glob("prices_*.csv"))
    if snapshots:
        prices = {}
        with open(snapshots[-1], newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prices[row["cusip"]] = float(row["px_last"])
        return prices
    return _load_manual_prices(cusips=[])


# ── manual fallbacks ──────────────────────────────────────────────────────────

def _load_manual_prices(cusips: list[str]) -> dict[str, float]:
    """Most recent price per CUSIP from manual_prices.csv."""
    if not MANUAL_PRICES_PATH.exists():
        return {}

    prices: dict[str, tuple[str, float]] = {}
    with open(MANUAL_PRICES_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cusip = row["cusip"]
            if not cusips or cusip in cusips:
                existing_date = prices.get(cusip, ("", 0.0))[0]
                if row["date"] >= existing_date:
                    prices[cusip] = (row["date"], float(row["px_last"]))

    return {cusip: v[1] for cusip, v in prices.items()}


def _load_manual_price_history(cusips, start_date, end_date) -> pd.DataFrame:
    if not MANUAL_HISTORY_PATH.exists():
        return pd.DataFrame(columns=["date", "cusip", "px_last"])

    df = pd.read_csv(MANUAL_HISTORY_PATH, dtype={"cusip": str})
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    if cusips:
        mask &= df["cusip"].isin(cusips)
    return df[mask].reset_index(drop=True)
