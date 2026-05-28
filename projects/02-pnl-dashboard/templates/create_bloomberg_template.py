"""
Run once to create bloomberg_prices.xlsx with BDP and BDH sheets.

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
N_ROWS = 50               # max number of securities on BDP sheet


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
    create_history_sheet(wb)
    wb.save(OUTPUT)
    print(f"Template saved → {OUTPUT}")
    print("Sheet1: BDP current prices (run Refresh Bloomberg Prices in dashboard)")
    print("History: BDH time-series (used by backfill_prices.py)")


if __name__ == "__main__":
    main()
