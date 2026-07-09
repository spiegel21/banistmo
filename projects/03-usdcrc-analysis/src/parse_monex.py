"""Parse the BCCR/MONEX 'frmVerCatCuadro' HTML-as-XLS export into a tidy CSV.

The export is a matrix: rows are metrics, columns are calendar dates, in two
blocks:

  TIPO DE CAMBIO NEGOCIADO (colones por US$)   -- the traded FX rate
    Primero / Ultimo / Minimo / Maximo / Promedio simple /
    Promedio ponderado / Promedio ponderado sesion anterior
  MONTO NEGOCIADO (US$)                         -- the traded amount
    Primero / Ultimo / Minimo / Maximo / Promedio simple /
    Monto total / Total de calces

This script transposes that into one row per calendar date with tidy columns
and drops non-trading days (rows with no total volume).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "monex_raw.xls"
OUT = Path(__file__).resolve().parents[1] / "data" / "monex_clean.csv"

# Spanish month abbreviations used by BCCR -> month number
MESES = {
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12,
}

# Map the (block, row-label) pairs to clean column names. Order matters because
# the same labels repeat in both blocks.
RATE_ROWS = {
    "Primero": "open",
    "Ultimo": "close",
    "Minimo": "low",
    "Maximo": "high",
    "Promedio simple": "avg_simple",
    "Promedio ponderado": "vwap",
    "Promedio ponderado sesion anterior": "prev_vwap",
}
VOL_ROWS = {
    "Primero": "trade_first_usd",
    "Ultimo": "trade_last_usd",
    "Minimo": "trade_min_usd",
    "Maximo": "trade_max_usd",
    "Promedio simple": "trade_avg_usd",
    "Monto total": "volume_usd",
    "Total de calces": "n_trades",
}


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    # decode the handful of HTML entities BCCR emits
    repl = {
        "&nbsp;": "", "&Uacute;": "U", "&uacute;": "u", "&iacute;": "i",
        "&aacute;": "a", "&oacute;": "o", "&eacute;": "e", "&ntilde;": "n",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s.strip()


def _num(s: str):
    """BCCR uses comma decimal separator and dot thousands; cells may be blank."""
    s = s.strip()
    if not s:
        return None
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(s: str):
    m = re.match(r"(\d+)\s+([A-Za-z]+)\s+(\d{4})", s)
    if not m:
        return None
    day, mon, year = int(m.group(1)), m.group(2), int(m.group(3))
    if mon not in MESES:
        return None
    try:
        return pd.Timestamp(year=year, month=MESES[mon], day=day)
    except ValueError:
        # BCCR blindly enumerates calendar slots incl. invalid ones (e.g. 29 Feb 2025)
        return None


def parse() -> pd.DataFrame:
    html = RAW.read_text(encoding="windows-1252")
    trs = re.findall(r"<tr.*?</tr>", html, re.S)

    def cells(tr):
        cs = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        return [_strip_tags(c) for c in cs]

    rows = [cells(tr) for tr in trs]

    # row 0 is the date header
    dates = [_parse_date(x) for x in rows[0][1:]]

    # Walk rows, tracking which block (rate vs volume) we're in.
    block = None
    data = {}  # col_name -> list aligned to dates
    for r in rows[1:]:
        if not r:
            continue
        label = r[0]
        if label.startswith("TIPO DE CAMBIO"):
            block = "rate"
            continue
        if label.startswith("MONTO NEGOCIADO"):
            block = "vol"
            continue
        if label.startswith("MEJORES OFERTAS"):
            block = "board"
            continue
        if block == "rate" and label in RATE_ROWS:
            data[RATE_ROWS[label]] = [_num(x) for x in r[1:]]
        elif block == "vol" and label in VOL_ROWS:
            data[VOL_ROWS[label]] = [_num(x) for x in r[1:]]

    df = pd.DataFrame(data)
    df.insert(0, "date", dates[: len(df)])
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    # Trading days: those with a positive total volume.
    df["is_trading_day"] = df["volume_usd"].fillna(0) > 0
    return df


def main():
    df = parse()
    df.to_csv(OUT, index=False)
    td = df[df["is_trading_day"]]
    print(f"Parsed {len(df)} calendar days, {len(td)} trading days")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Trading-day range: {td['date'].min().date()} -> {td['date'].max().date()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Wrote {OUT}")
    print(td[["date", "open", "close", "low", "high", "vwap",
              "prev_vwap", "volume_usd", "n_trades"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
