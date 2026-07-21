# 03 — USD/CRC (MONEX) pricing analysis

Reverse-engineering the price-formation model of the USD/CRC wholesale FX market
(BCCR MONEX) and screening for a tradeable edge, from BCCR `frmVerCatCuadro` exports.

**Headline.** The tradeable edge is Costa Rica's **mid-month payment calendar**: short USD
into the IVA (D-104) / quincena deadline, long the rest of the month. Priced VWAP-to-VWAP at
$1M/trade, net of 0.65 CRC round-trip, it earns **in-sample Sharpe 3.14 → out-of-sample
2.79** flat-sized, rising to **3.70 / 3.92** with the recommended trailing 30 / hard-floor
40 bps exit overlay (which also roughly halves drawdown). It is the only edge in the study
that works in both the pegged 2014–2021 regime and the 2021–2026 float; the daily
order-flow / volume / skew signals are regime-contingent and carry $240k–$400k drawdowns.
See **FINDINGS.md** for the full write-up and **out/report.html** for the visual report.

> **One accounting basis.** Every figure — every module, table, and chart — is priced
> **VWAP-to-VWAP** next session, **$1,000,000** per trade, **net of 0.65 CRC** round-trip
> (0.325/side), annualised on the **231 sessions/yr** MONEX actually trades (not the 252
> equity convention). All modules read this from one file, **`src/basis.py`**, so no two
> tables can silently disagree. The `out/*.json` files are the source of truth; `tests/`
> reconciles the report against them.

## Layout
```
data/monex_raw.xls               # original BCCR price/volume export (HTML-as-XLS)
data/monex_clean.csv             # tidy one-row-per-trading-day table (generated)
data/bccr_intervention_raw.html  # BCCR FX-intervention export (CodCuadro=1587)
data/bccr_reserves_raw.html      # BCCR net-reserves export (CodCuadro=8)
data/bccr_intervention_clean.csv # tidy daily official-flow table (generated)
data/bccr_reserves_clean.csv     # tidy month-end reserves (generated)

src/basis.py           # SINGLE SOURCE OF TRUTH for the accounting basis (231 sess/yr,
                       #   0.325 CRC/side, $1M notional, VWAP pricing) — imported everywhere
src/parse_monex.py     # parse the price/volume HTML matrix -> clean CSV
src/parse_bccr.py      # parse the BCCR intervention + reserves exports -> clean CSVs
src/payment_calendar.py # real CR IVA (D-104) / quincena / CCSS payment calendar
src/analyze.py         # descriptive analysis (price formation, seasonality, profile)
src/eda.py             # exploratory charts (returns/ACF, volume->move, calendar)
src/strategies.py      # strategy tearsheets
src/backtest.py        # overfitting-aware tests (close-priced) -> backtest_results.json
src/backtest_vwap.py   # same strategies + slippage, priced at the session VWAP (realistic)
src/volume_model.py    # conviction-sized ML model (walk-forward) -> vm_results.json
src/dynamics.py        # underlying-mechanism deep-dive + $1M/trade dollar rules
src/quincena.py        # calendar (quincena) strategy (recommended) + trading calendar
src/exits.py           # trailing-stop / floor exit OVERLAY on the calendar rule (optimised)
src/exit_lab.py        # FULL exit engine (trailing / hard stop / take-profit / time),
                       #   fixed bps or vol multiples; in-sample tuned -> exit_lab.json
src/rank_strategies.py # EVERY strategy re-priced through ONE code path, ranked by in-sample
                       #   Sharpe -> ranking.json (the master cross-strategy comparison)
src/intervention.py    # BCCR official-flow + reserves overlay -> intervention_results.json
src/daily_signal.py    # one-page daily signal sheet (position + slow-vol size)
src/build_report.py    # assembles the self-contained HTML report (reads the json)

out/                   # PNG charts + report.html + report.pdf (generated)
tests/                 # pytest suite: reconciles the report vs the JSON + basis invariants
FINDINGS.md            # written conclusions & recommendation (the presentable document)
```

## Run (full pipeline)
```bash
cd projects/03-usdcrc-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/parse_monex.py    # -> data/monex_clean.csv
python src/parse_bccr.py     # -> data/bccr_intervention_clean.csv + bccr_reserves_clean.csv
for m in analyze eda strategies backtest backtest_vwap volume_model \
        dynamics quincena exits rank_strategies exit_lab intervention daily_signal; do
  python src/$m.py
done
python src/build_report.py   # -> out/report.html
python -c "from weasyprint import HTML; HTML('out/report.html').write_pdf('out/report.pdf')"
```

## Tests
```bash
python -m pytest tests/ -q
```
The suite is **stdlib + pytest only** (no scientific stack) so it runs anywhere. It checks
(a) every module annualises on the shared 231-session basis and nothing hardcodes 252,
(b) each `out/*.json` declares the canonical basis, (c) structural invariants — the
always-long / always-short mirror, the ranking is ordered by in-sample Sharpe, IS+OOS
session counts reconcile to the full sample, (d) the headline numbers in FINDINGS.md and
report.html match `ranking.json` / `exit_lab.json` / `intervention_results.json`, and
(e) the parsers' clean-CSV output schema.

## Daily signal sheet

Print the position the recommended rule takes for the next session (or any date), with the
slow-volume size multiplier, at USD 1M/trade:

```bash
python src/daily_signal.py            # next session; writes out/daily_signal.txt
python src/daily_signal.py 2026-06-10 # a specific date
python src/daily_signal.py --png      # also render out/daily_signal.png
```

The logic mirrors `quincena.pos_calendar` exactly: **SHORT USD within 6 business days of the
IVA (D-104) / mid-month quincena deadline** (the 15th, rolled to the next business day) —
**LONG otherwise**; size trimmed to 0.5× when the slow 20-day volume regime disagrees. It
reads only data strictly before the target date, so it is safe to run live each morning.

### The payment calendar (`payment_calendar.py`)

Statutory events (Hacienda / TRIBU-CR and CCSS), each snapped to MONEX's own trading
calendar so weekend/holiday rolling is correct by construction:

| Event | Statutory date | Effect on USD/CRC |
|-------|----------------|-------------------|
| IVA (D-104) + retenciones (D-103) | 15th of following month (→ next business day) | firms sell USD → colón strengthens into it |
| Quincena — 1st fortnight payroll | ~15th | same mid-month USD-supply surge |
| Quincena — 2nd fortnight payroll | month-end | month-end payroll |
| CCSS planilla (social security) | 4th business day of month | early-month CRC demand |
| Pagos parciales de renta | end of Mar / Jun / Sep / Dec | quarter-end CRC demand |

Measured in *business days to the IVA/quincena deadline*, the colón strengthens hardest 1–5
days **before** the deadline (−9 to −16 bps next-day) and reverts after — see `out/q_calendar.png`.

### Refreshing the BCCR data

`parse_bccr.py` reads two HTML-as-XLS exports fetched from BCCR's `frmVerCatCuadro` tool
(same source as MONEX). To refresh:

```bash
curl -o data/bccr_intervention_raw.html \
  "https://gee.bccr.fi.cr/indicadoreseconomicos/Cuadros/frmVerCatCuadro.aspx?CodCuadro=1587&Idioma=1&FecInicial=2014/12/01&FecFinal=2026/06/30&Filtro=0&Exportar=True"
curl -o data/bccr_reserves_raw.html \
  "https://gee.bccr.fi.cr/indicadoreseconomicos/Cuadros/frmVerCatCuadro.aspx?CodCuadro=8&Idioma=1&FecInicial=2014/01/01&FecFinal=2026/06/30&Filtro=0&Exportar=True"
python src/parse_bccr.py
```
