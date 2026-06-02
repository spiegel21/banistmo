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
import math
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


def _is_real_price(v) -> bool:
    """True only for a finite number — excludes None, strings, and NaN (Bloomberg's #N/A response)."""
    return isinstance(v, (int, float)) and not math.isnan(v)


def _all_prices_filled(sheet, n: int) -> bool:
    """Return True only when all price cells contain finite numeric values."""
    values = sheet.range(f"B2:B{n + 1}").value
    if values is None:
        return False
    if isinstance(values, list):
        return all(_is_real_price(v) for v in values)
    return _is_real_price(values)


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
        result[str(cusip)] = {
            "px_last": px_last[i] if _is_real_price(px_last[i]) else None,
            "asw": asw[i] if asw and _is_real_price(asw[i]) else None,
            "ytm": ytm[i] if ytm and _is_real_price(ytm[i]) else None,
        }
    return result


def prepare_template(cusips: list[str], wb_path: Path = TEMPLATE_PATH) -> Path:
    """Write CUSIPs into Sheet1 col A and save the template.

    After calling this, the user opens the file in Excel, waits for Bloomberg
    BDP formulas to populate, saves and closes. Then call read_prices_from_template().
    """
    from openpyxl import load_workbook
    wb = load_workbook(str(wb_path))
    ws = wb["Sheet1"]
    for row in range(2, 52):           # clear up to 50 rows
        ws.cell(row=row, column=1).value = None
    for i, cusip in enumerate(cusips):
        ws.cell(row=i + 2, column=1).value = str(cusip)
    wb.save(str(wb_path))
    return wb_path


def read_prices_from_template(wb_path: Path = TEMPLATE_PATH) -> dict[str, float]:
    """Read cached Bloomberg prices from a template saved after a manual Excel refresh.

    openpyxl data_only=True reads the last-calculated cell values (not formulas),
    so this only works after the user has opened the file in Excel and saved it
    with Bloomberg prices populated.
    """
    from openpyxl import load_workbook
    wb = load_workbook(str(wb_path), data_only=True)
    ws = wb["Sheet1"]
    prices = {}
    for row in range(2, 52):
        cusip = ws.cell(row=row, column=1).value
        price = ws.cell(row=row, column=2).value
        if not cusip:
            continue
        if isinstance(price, (int, float)) and not math.isnan(float(price)):
            prices[str(cusip)] = float(price)
    if prices:
        _save_price_snapshot(prices)
        _append_to_price_history(prices, date.today())
    return prices



    """
    Fetch current prices via Bloomberg BDP.

    Returns {cusip: px_last}. Appends results to price_history.csv.
    Falls back to manual_prices.csv if Bloomberg is unavailable.
    """
    xw = _try_xlwings_import()

    if xw is not None:
        try:
            app = xw.App(visible=True, add_book=False)
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


# ── history template (manual two-step BDH workflow) ──────────────────────────

_BACKFILL_STATE_PATH = PRICES_DIR / "backfill_state.json"
_TICKER_SUFFIX = " Corp"
_BDH_BLOCK_ROWS = 510   # rows reserved per CUSIP block (ample for ~500 business days)


def _load_backfill_state() -> dict:
    if _BACKFILL_STATE_PATH.exists():
        import json
        return json.loads(_BACKFILL_STATE_PATH.read_text())
    return {}


def _save_backfill_state(state: dict) -> None:
    import json
    _BACKFILL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BACKFILL_STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def last_priced_date() -> date | None:
    """Return the latest date in price_history.csv, or None if the file is empty."""
    if not PRICE_HISTORY_PATH.exists():
        return None
    try:
        df = pd.read_csv(PRICE_HISTORY_PATH, dtype={"cusip": str})
        if df.empty:
            return None
        return pd.to_datetime(df["date"]).max().date()
    except Exception:
        return None


def prepare_history_template(
    cusip_ranges: list[tuple[str, date, date]],
    wb_path: Path = TEMPLATE_PATH,
    ticker_suffix: str = _TICKER_SUFFIX,
) -> Path:
    """Write one BDH block per CUSIP into the History sheet of the template.

    Each block (offset by _BDH_BLOCK_ROWS rows per CUSIP):
      Row n  : [ticker, start_YYYYMMDD, end_YYYYMMDD]   ← BDH parameters
      Row n+1: ["Date", "PX_LAST"]                       ← column labels
      Row n+2: =BDH formula (Bloomberg fills downward)

    After this call the user opens the file in Excel, waits for Bloomberg to
    populate every block, saves and closes. Then call read_history_from_template().

    cusip_ranges: list of (cusip, start_date, end_date) — one entry per security.
    """
    from openpyxl import load_workbook
    from openpyxl.styles import Font

    wb = load_workbook(str(wb_path))
    ws = wb["History"]
    ws.delete_rows(1, max(ws.max_row, 1))   # clear existing content

    cusips_saved = []
    for block_idx, (cusip, start, end) in enumerate(cusip_ranges):
        row = block_idx * _BDH_BLOCK_ROWS + 1
        ticker = str(cusip) + ticker_suffix

        ws.cell(row=row, column=1).value = ticker
        ws.cell(row=row, column=1).font = Font(bold=True)
        ws.cell(row=row, column=2).value = start.strftime("%Y%m%d")
        ws.cell(row=row, column=3).value = end.strftime("%Y%m%d")

        ws.cell(row=row + 1, column=1).value = "Date"
        ws.cell(row=row + 1, column=2).value = "PX_LAST"

        ws.cell(row=row + 2, column=1).value = (
            f'=BDH(A{row},"PX_LAST",B{row},C{row},"Fill","0","Dates","1")'
        )
        cusips_saved.append(str(cusip))

    wb.save(str(wb_path))

    prev = _load_backfill_state()
    _save_backfill_state({
        "pending_start": str(min(s for _, s, _ in cusip_ranges)),
        "pending_end": str(max(e for _, _, e in cusip_ranges)),
        "cusips": cusips_saved,
        "prepared_at": datetime.now().isoformat(),
        "last_date": prev.get("last_date"),
        "last_run": prev.get("last_run"),
    })

    return wb_path


def read_history_from_template(wb_path: Path = TEMPLATE_PATH) -> pd.DataFrame:
    """Read BDH blocks from the History sheet saved after manual Bloomberg refresh.

    Returns a DataFrame (date, cusip, px_last). Appends to price_history.csv
    and updates backfill_state.json with the latest date successfully imported.
    """
    from openpyxl import load_workbook

    state = _load_backfill_state()
    cusips = state.get("cusips", [])
    if not cusips:
        return pd.DataFrame(columns=["date", "cusip", "px_last"])

    wb = load_workbook(str(wb_path), data_only=True)
    ws = wb["History"]

    records = []
    for block_idx, cusip in enumerate(cusips):
        data_start = block_idx * _BDH_BLOCK_ROWS + 3   # row n+2 (0-indexed +2 = row 3)

        for r in range(data_start, data_start + _BDH_BLOCK_ROWS - 2):
            date_val = ws.cell(row=r, column=1).value
            price_val = ws.cell(row=r, column=2).value
            if not date_val:
                break
            # openpyxl returns Bloomberg dates as Python datetime objects
            if hasattr(date_val, "date"):
                d = date_val.date()
            else:
                try:
                    d = datetime.strptime(str(int(float(str(date_val)))), "%Y%m%d").date()
                except (ValueError, TypeError):
                    break
            if isinstance(price_val, (int, float)) and not math.isnan(float(price_val)):
                records.append({
                    "date": d.isoformat(),
                    "cusip": str(cusip),
                    "px_last": float(price_val),
                })

    if not records:
        return pd.DataFrame(columns=["date", "cusip", "px_last"])

    df = pd.DataFrame(records)
    _append_df_to_price_history(df)

    state["last_date"] = df["date"].max()
    state["last_run"] = datetime.now().isoformat()
    _save_backfill_state(state)

    return df


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
