"""
Bloomberg pricing bridge via xlwings.

Writes CUSIPs to the Excel template, triggers Bloomberg BDP formula refresh,
reads back clean prices (and optionally ASW spread + YTM).

Requirements:
  - Windows OS
  - Bloomberg Terminal running and logged in
  - xlwings + pywin32 installed
  - templates/bloomberg_prices.xlsx open or openable by Excel

Fallback: if Excel/Bloomberg is unavailable, reads from data/prices/manual_prices.csv
"""
import csv
import time
from datetime import date, datetime
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "bloomberg_prices.xlsx"
PRICES_DIR = Path(__file__).parent.parent / "data" / "prices"
MANUAL_PRICES_PATH = PRICES_DIR / "manual_prices.csv"

_BLOOMBERG_TIMEOUT = 30  # seconds to wait for Bloomberg to populate cells
_POLL_INTERVAL = 2       # seconds between polls


def _try_xlwings_import():
    try:
        import xlwings as xw
        return xw
    except ImportError:
        return None


def write_isins(isins: list[str], wb) -> None:
    """Write ISINs into Sheet1 column A starting at row 2."""
    sheet = wb.sheets["Sheet1"]
    # clear old ISINs below header
    last_row = sheet.range("A2").end("down").row
    if last_row >= 2:
        sheet.range(f"A2:A{max(last_row, len(isins) + 1)}").clear_contents()
    for i, isin in enumerate(isins):
        sheet["A" + str(i + 2)].value = isin


def _all_prices_filled(sheet, n_isins: int) -> bool:
    """Check that column B rows 2..n+1 are all non-empty."""
    values = sheet.range(f"B2:B{n_isins + 1}").value
    if values is None:
        return False
    if isinstance(values, list):
        return all(v is not None and v != "" for v in values)
    return values is not None


def refresh_and_read(wb, n_isins: int) -> dict[str, dict]:
    """
    Trigger Bloomberg recalculation and wait for prices to populate.

    Returns {isin: {"px_last": float, "asw": float|None, "ytm": float|None}}
    """
    xw = _try_xlwings_import()
    if xw is None:
        raise RuntimeError("xlwings not installed")

    sheet = wb.sheets["Sheet1"]
    wb.app.calculate()  # forces Bloomberg BDP formula refresh

    elapsed = 0
    while elapsed < _BLOOMBERG_TIMEOUT:
        if _all_prices_filled(sheet, n_isins):
            break
        time.sleep(_POLL_INTERVAL)
        wb.app.calculate()
        elapsed += _POLL_INTERVAL

    cusips = [sheet["A" + str(i + 2)].value for i in range(n_isins)]
    px_last = sheet.range(f"B2:B{n_isins + 1}").value
    asw = sheet.range(f"C2:C{n_isins + 1}").value
    ytm = sheet.range(f"D2:D{n_isins + 1}").value

    # xlwings returns a scalar (not list) when only one row
    if n_isins == 1:
        px_last, asw, ytm = [px_last], [asw], [ytm]

    result = {}
    for i, cusip in enumerate(cusips):
        if cusip:
            result[cusip] = {
                "px_last": px_last[i],
                "asw": asw[i] if asw else None,
                "ytm": ytm[i] if ytm else None,
            }
    return result


def get_prices(
    isins: list[str],
    wb_path: Path = TEMPLATE_PATH,
) -> dict[str, float]:
    """
    Main entry point: write ISINs → refresh → return {isin: px_last}.

    Falls back to manual_prices.csv if Excel/Bloomberg is unavailable.
    """
    xw = _try_xlwings_import()

    if xw is not None:
        try:
            app = xw.App(visible=False, add_book=False)
            try:
                wb = app.books.open(str(wb_path))
                write_isins(isins, wb)
                data = refresh_and_read(wb, len(isins))
                wb.save()
                wb.close()
            finally:
                app.quit()

            prices = {cusip: v["px_last"] for cusip, v in data.items() if v["px_last"] is not None}
            _save_price_snapshot(prices)
            return prices
        except Exception as e:
            print(f"Bloomberg bridge failed ({e}), falling back to manual prices")

    return _load_manual_prices(isins)


def _save_price_snapshot(prices: dict[str, float]) -> None:
    Path(PRICES_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PRICES_DIR / f"prices_{ts}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cusip", "px_last", "date"])
        today = date.today().isoformat()
        for cusip, px in prices.items():
            writer.writerow([cusip, px, today])


def _load_manual_prices(cusips: list[str]) -> dict[str, float]:
    """Load the most recent price for each CUSIP from manual_prices.csv."""
    if not MANUAL_PRICES_PATH.exists():
        return {}

    prices: dict[str, tuple[str, float]] = {}  # {cusip: (date_str, price)}
    with open(MANUAL_PRICES_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cusip = row["cusip"]
            if not cusips or cusip in cusips:
                existing_date = prices.get(cusip, ("", 0.0))[0]
                if row["date"] >= existing_date:
                    prices[cusip] = (row["date"], float(row["px_last"]))

    return {cusip: v[1] for cusip, v in prices.items()}


def load_latest_prices() -> dict[str, float]:
    """Load the most recently saved price snapshot (Bloomberg or manual)."""
    snapshots = sorted(PRICES_DIR.glob("prices_*.csv"))
    if snapshots:
        prices = {}
        with open(snapshots[-1], newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prices[row["isin"]] = float(row["px_last"])
        return prices
    return _load_manual_prices(cusips=[])
