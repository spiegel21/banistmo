"""Parse two BCCR 'frmVerCatCuadro' HTML-as-XLS exports into tidy CSVs:

  * CodCuadro=1587  "Intervención cambiaria del BCCR"  -> daily official FX flow
  * CodCuadro=8     "Reservas netas del Banco Central"  -> end-of-month reserves

Both come from the same BCCR web query tool as MONEX (see parse_monex.py) and use
the same Spanish month abbreviations, comma-decimal / dot-thousand numbers, and
windows-1252 / utf-8 HTML. Re-fetch with, e.g.:

  curl -o data/bccr_intervention_raw.html \\
    "https://gee.bccr.fi.cr/indicadoreseconomicos/Cuadros/frmVerCatCuadro.aspx?CodCuadro=1587&Idioma=1&FecInicial=2014/12/01&FecFinal=2026/06/30&Filtro=0&Exportar=True"
  curl -o data/bccr_reserves_raw.html \\
    "https://gee.bccr.fi.cr/indicadoreseconomicos/Cuadros/frmVerCatCuadro.aspx?CodCuadro=8&Idioma=1&FecInicial=2014/01/01&FecFinal=2026/06/30&Filtro=0&Exportar=True"

Intervention matrix (one row per calendar date, 10 value columns)
----------------------------------------------------------------
BCCR reports every leg of its FX participation. The header is hierarchical:

  banda (límite inferior/superior) | Operaciones de estabilización (C/V) |
  Operaciones propias (C/V) | Requerimientos del Sector Público No Bancario
                                     [ BCCR C/V ][ Monex C/V ]

The first six columns are the BCCR's OWN market interventions (defending the
band, stabilising, own account). The last four are the non-bank public sector's
(RECOPE et al.) USD requirement: the BCCR buys/sells it FROM the public sector
(the "BCCR" leg, off-market) and sources/places it ON MONEX (the "Monex" leg).
Only the MONEX legs actually hit the wholesale order book, so the market-facing
official flow is the band + estabilización + propias interventions plus the SPNB
*Monex* legs. See derive_official_flow() below.
"""
from __future__ import annotations

import html as _html
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INT_RAW = ROOT / "data" / "bccr_intervention_raw.html"
RES_RAW = ROOT / "data" / "bccr_reserves_raw.html"
INT_OUT = ROOT / "data" / "bccr_intervention_clean.csv"
RES_OUT = ROOT / "data" / "bccr_reserves_clean.csv"

MESES = {
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12,
}
MESES_FULL = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
    "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10,
    "Noviembre": 11, "Diciembre": 12,
}

# The 10 intervention value columns, left-to-right as BCCR emits them.
INT_COLS = [
    "band_buy",        # intervención límite inferior — BCCR buys USD to defend the floor
    "band_sell",       # intervención límite superior — BCCR sells USD to defend the ceiling
    "estab_buy",       # operaciones de estabilización — purchases
    "estab_sell",      # operaciones de estabilización — sales
    "propias_buy",     # operaciones propias — purchases
    "propias_sell",    # operaciones propias — sales
    "spnb_bccr_buy",   # SPNB requirement, BCCR<->public-sector leg (off-market)
    "spnb_bccr_sell",
    "spnb_monex_buy",  # SPNB requirement, BCCR<->MONEX leg (hits the order book)
    "spnb_monex_sell",
]


def _cells(tr: str) -> list[str]:
    return [_html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m)).strip())
            for m in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]


def _num(s: str):
    """BCCR uses comma decimal + dot thousands; cells may be blank."""
    s = (s or "").strip()
    if not s:
        return None
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _read(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "windows-1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_daily_date(s: str):
    m = re.match(r"(\d+)\s+([A-Za-z]{3})\s+(\d{4})", s)
    if not m:
        return None
    day, mon, year = int(m.group(1)), m.group(2), int(m.group(3))
    if mon not in MESES:
        return None
    try:
        return pd.Timestamp(year=year, month=MESES[mon], day=day)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# interventions
# --------------------------------------------------------------------------- #
def parse_intervention() -> pd.DataFrame:
    trs = re.findall(r"<tr.*?</tr>", _read(INT_RAW), re.S)
    recs = []
    for tr in trs:
        c = _cells(tr)
        if not c:
            continue
        d = _parse_daily_date(c[0])
        if d is None:
            continue
        vals = [_num(x) for x in c[1:11]]           # exactly the 10 value columns
        vals += [None] * (len(INT_COLS) - len(vals))
        recs.append([d] + vals[: len(INT_COLS)])
    df = pd.DataFrame(recs, columns=["date"] + INT_COLS).sort_values("date")
    df = df.drop_duplicates(subset="date").reset_index(drop=True)
    for c in INT_COLS:
        df[c] = df[c].fillna(0.0)
    return derive_official_flow(df)


def derive_official_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Add the market-facing official-flow aggregates used by the analysis.

    Only legs that actually cross the MONEX order book are counted: the BCCR's
    own band / estabilización / propias interventions, plus the SPNB *Monex*
    legs (the off-market BCCR<->public-sector legs are excluded, they never touch
    the wholesale price). Sign convention matches USD/CRC direction:
    official USD *buying* pushes USD up (colón down), so it is the positive side.
    """
    df = df.copy()
    df["off_buy_usd"] = df[["band_buy", "estab_buy", "propias_buy", "spnb_monex_buy"]].sum(axis=1)
    df["off_sell_usd"] = df[["band_sell", "estab_sell", "propias_sell", "spnb_monex_sell"]].sum(axis=1)
    # Net official flow in USD: +ve = BCCR/public-sector net BUYING USD on MONEX.
    df["off_net_usd"] = df["off_buy_usd"] - df["off_sell_usd"]
    df["off_gross_usd"] = df["off_buy_usd"] + df["off_sell_usd"]
    # BCCR's own discretionary interventions only (excludes the SPNB pass-through).
    df["bccr_own_buy_usd"] = df[["band_buy", "estab_buy", "propias_buy"]].sum(axis=1)
    df["bccr_own_sell_usd"] = df[["band_sell", "estab_sell", "propias_sell"]].sum(axis=1)
    df["bccr_own_net_usd"] = df["bccr_own_buy_usd"] - df["bccr_own_sell_usd"]
    # SPNB (public-sector) net USD demand on MONEX (RECOPE et al.).
    df["spnb_net_usd"] = df["spnb_monex_buy"] - df["spnb_monex_sell"]
    df["is_intervention_day"] = df["off_gross_usd"] > 0
    return df


# --------------------------------------------------------------------------- #
# reserves (month-rows x year-cols matrix -> tidy monthly series)
# --------------------------------------------------------------------------- #
def parse_reserves() -> pd.DataFrame:
    trs = re.findall(r"<tr.*?</tr>", _read(RES_RAW), re.S)
    rows = [_cells(tr) for tr in trs]
    header = None
    recs = []
    for r in rows:
        if not r:
            continue
        if header is None:
            years = [y for y in r[1:] if re.fullmatch(r"\d{4}", y)]
            if years:
                header = [int(y) for y in years]
            continue
        mon = MESES_FULL.get(r[0])
        if mon is None:
            continue
        for yr, cell in zip(header, r[1:1 + len(header)]):
            v = _num(cell)
            if v is None:
                continue
            # end-of-month date (BCCR reports the month-end reserve level, USD mn)
            eom = pd.Timestamp(year=yr, month=mon, day=1) + pd.offsets.MonthEnd(0)
            recs.append((eom, v))
    df = pd.DataFrame(recs, columns=["date", "reserves_usd_mn"]).sort_values("date")
    df = df.drop_duplicates(subset="date").reset_index(drop=True)
    df["reserves_chg_usd_mn"] = df["reserves_usd_mn"].diff()
    return df


def main():
    it = parse_intervention()
    it.to_csv(INT_OUT, index=False)
    act = it[it["is_intervention_day"]]
    print(f"INTERVENTION — {len(it)} calendar days, {len(act)} with official flow")
    print(f"  range {it.date.min().date()} -> {it.date.max().date()}")
    print(f"  official gross USD/day (active): mean {act.off_gross_usd.mean():,.0f} "
          f"max {act.off_gross_usd.max():,.0f}")
    print(f"  net official USD: {it.off_net_usd.sum():,.0f} cumulative "
          f"(buy {it.off_buy_usd.sum():,.0f} / sell {it.off_sell_usd.sum():,.0f})")
    print(it[["date", "off_buy_usd", "off_sell_usd", "off_net_usd",
              "bccr_own_net_usd", "spnb_net_usd"]].tail(6).to_string(index=False))
    print(f"  wrote {INT_OUT}")

    rs = parse_reserves()
    rs.to_csv(RES_OUT, index=False)
    print(f"\nRESERVES — {len(rs)} month-ends, {rs.date.min().date()} -> {rs.date.max().date()}")
    print(f"  latest {rs.reserves_usd_mn.iloc[-1]:,.0f} USD mn "
          f"(range {rs.reserves_usd_mn.min():,.0f}-{rs.reserves_usd_mn.max():,.0f})")
    print(rs.tail(6).to_string(index=False))
    print(f"  wrote {RES_OUT}")


if __name__ == "__main__":
    main()
