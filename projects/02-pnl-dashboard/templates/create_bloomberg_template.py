"""
Run this once to create the bloomberg_prices.xlsx template with BDP formulas.

Requires: openpyxl
Usage:  python templates/create_bloomberg_template.py
"""
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

OUTPUT = Path(__file__).parent / "bloomberg_prices.xlsx"

HEADERS = ["ISIN", "Last Price (PX_LAST)", "ASW Spread", "YTM (Mid)"]
BLOOMBERG_FIELDS = ["PX_LAST", "YAS_ASW_SPREAD", "YLD_YTM_MID"]
# Bloomberg ticker suffix — change to " Govt" or " Mtge" if needed
TICKER_SUFFIX = " Corp"

HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
HEADER_FONT = Font(color="FFFFFF", bold=True)


def create_template(n_rows: int = 50) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    # headers
    for col, header in enumerate(HEADERS, start=1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center")

    # BDP formulas for rows 2..n_rows+1
    for row in range(2, n_rows + 2):
        isin_ref = f"A{row}"
        for col_offset, field in enumerate(BLOOMBERG_FIELDS, start=1):
            formula = f'=BDP({isin_ref}&"{TICKER_SUFFIX}","{field}")'
            ws.cell(row=row, column=col_offset + 1, value=formula)

    # column widths
    ws.column_dimensions["A"].width = 16
    for col in range(2, len(HEADERS) + 1):
        ws.column_dimensions[get_column_letter(col)].width = 20

    wb.save(OUTPUT)
    print(f"Template saved to {OUTPUT}")


if __name__ == "__main__":
    create_template()
