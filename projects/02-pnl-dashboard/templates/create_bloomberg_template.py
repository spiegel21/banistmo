"""
Run once to create bloomberg_prices.xlsx with BDP, Static, and BDH sheets.

Usage:  python templates/create_bloomberg_template.py
"""
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

OUTPUT = Path(__file__).parent / "bloomberg_prices.xlsx"

HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TICKER_SUFFIX = " Corp"   # change to " Govt" or " Mtge" if needed
N_ROWS = 50               # max number of securities on BDP / Static sheets


def _header(ws, col, text):
    cell = ws.cell(row=1, column=col, value=text)
    cell.font = HEADER_FONT
    cell.fill = HEADER_FILL
    cell.alignment = Alignment(horizontal="center")


def create_bdp_sheet(wb: Workbook) -> None:
    """Sheet1 — current prices via BDP formulas."""
    ws = wb.active
    ws.title = "Sheet1"

    for col, header in enumerate(["CUSIP", "Last Price (PX_LAST)", "ASW Spread", "YTM (Mid)"], 1):
        _header(ws, col, header)

    fields = ["PX_LAST", "YAS_ASW_SPREAD", "YLD_YTM_MID"]
    for row in range(2, N_ROWS + 2):
        for col_offset, field in enumerate(fields, 1):
            ws.cell(row=row, column=col_offset + 1,
                    value=f'=BDP(A{row}&"{TICKER_SUFFIX}","{field}")')

    ws.column_dimensions["A"].width = 16
    for col in range(2, 5):
        ws.column_dimensions[get_column_letter(col)].width = 22


def create_static_sheet(wb: Workbook) -> None:
    """Static sheet — bond reference data via BDP reference fields.

    Columns mirror bonds_static.csv exactly so the values can be imported
    directly. CUSIPs are written into column A (by the dashboard / a prepare
    step); every other column is a BDP formula keyed off A{row}.

    Notes on field mapping:
      - coupon_rate: Bloomberg CPN is a percent (e.g. 5.5); we divide by 100
        in the formula so the cell holds 0.055 (the decimal bonds_static wants).
      - day_count_convention: DAY_CNT_DES returns Bloomberg's raw string
        (e.g. "ACT/360", "30/360", "ISMA-30/360"); the importer normalises it
        to one of {Act/360, Act/365, 30/360}.
      - maturity_date / first_coupon_date: returned as Excel dates.
    """
    ws = wb.create_sheet("Static")

    # (header, BDP field expression keyed off A{row}, or None for the CUSIP col)
    columns = [
        ("cusip", None),
        ("name", 'BDP(A{r}&"{sfx}","SECURITY_DES")'),
        ("currency", 'BDP(A{r}&"{sfx}","CRNCY")'),
        ("country", 'BDP(A{r}&"{sfx}","CNTRY_OF_RISK")'),
        ("coupon_rate", 'BDP(A{r}&"{sfx}","CPN")/100'),
        ("coupon_frequency", 'BDP(A{r}&"{sfx}","CPN_FREQ")'),
        ("day_count_convention", 'BDP(A{r}&"{sfx}","DAY_CNT_DES")'),
        ("maturity_date", 'BDP(A{r}&"{sfx}","MATURITY")'),
        ("first_coupon_date", 'BDP(A{r}&"{sfx}","FIRST_CPN_DT")'),
    ]

    for col, (header, _) in enumerate(columns, 1):
        _header(ws, col, header)

    for row in range(2, N_ROWS + 2):
        for col, (_, expr) in enumerate(columns, 1):
            if expr is None:
                continue   # column A holds the CUSIP, written by the caller
            ws.cell(row=row, column=col,
                    value="=" + expr.format(r=row, sfx=TICKER_SUFFIX))

    ws.column_dimensions["A"].width = 16
    for col in range(2, len(columns) + 1):
        ws.column_dimensions[get_column_letter(col)].width = 20


def create_history_sheet(wb: Workbook) -> None:
    """History sheet — time-series prices via BDH formula (one CUSIP at a time)."""
    ws = wb.create_sheet("History")

    # Parameter cells (written by Python before each CUSIP fetch)
    ws["A1"] = "Ticker"
    ws["B1"] = "Start"
    ws["C1"] = "End"
    ws["A1"].font = Font(bold=True)
    ws["B1"].font = Font(bold=True)
    ws["C1"].font = Font(bold=True)

    # Row 2: labels for the data output below
    for col, label in enumerate(["Date", "PX_LAST"], 1):
        _header(ws, col, label)

    # Row 3: BDH formula — Python triggers calculation after writing A1/B1/C1
    ws["A3"] = f'=BDH(A1,"PX_LAST",B1,C1,"Fill","0","Dates","1")'
    ws["A3"].font = Font(italic=True, color="555555")

    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 16


def main():
    wb = Workbook()
    create_bdp_sheet(wb)
    create_static_sheet(wb)
    create_history_sheet(wb)
    wb.save(OUTPUT)
    print(f"Template saved → {OUTPUT}")
    print("Sheet1: BDP current prices (run Refresh Bloomberg Prices in dashboard)")
    print("Static:  BDP bond reference data (coupon, frequency, dates, day count)")
    print("History: BDH time-series (used by backfill_prices.py)")


if __name__ == "__main__":
    main()
