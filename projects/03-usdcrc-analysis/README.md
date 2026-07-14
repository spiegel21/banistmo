# 03 — USD/CRC (MONEX) pricing analysis

Reverse-engineering the price-formation model of the USD/CRC wholesale FX market
(BCCR MONEX) and screening for trading opportunities from a single BCCR
`frmVerCatCuadro` export.

## Layout
```
data/monex_raw.xls     # original BCCR export (HTML-as-XLS)
data/monex_clean.csv   # tidy one-row-per-trading-day table (generated)
src/parse_monex.py     # parse the HTML matrix -> clean CSV
src/analyze.py         # descriptive analysis (price formation, seasonality, profile)
src/eda.py             # exploratory charts (returns/ACF, volume->move, calendar)
src/strategies.py      # strategy tearsheets
src/backtest.py        # rigorous, overfitting-aware tests -> backtest_results.json
src/dynamics.py        # underlying-mechanism deep-dive + $1M/trade dollar rules
src/payment_calendar.py # real CR IVA (D-104) / quincena / CCSS payment calendar
src/quincena.py        # calendar (quincena) strategy (recommended) + trading calendar
src/daily_signal.py    # one-page daily signal sheet (position + slow-vol size)
src/build_report.py    # assembles the self-contained HTML report (reads the json)
out/                   # PNG charts + report.html + report.pdf (generated)
FINDINGS.md            # written conclusions
```

## Run (full pipeline)
```bash
cd projects/03-usdcrc-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/parse_monex.py    # -> data/monex_clean.csv
python src/analyze.py        # descriptive stats + charts 01-04
python src/eda.py            # exploratory charts (eda_*.png)
python src/strategies.py     # strategy tearsheets (s*_*.png)
python src/backtest.py       # OOS / walk-forward / costs / permutation / DSR -> bt_*.png + json
python src/build_report.py   # -> out/report.html
python -c "from weasyprint import HTML; HTML('out/report.html').write_pdf('out/report.pdf')"
```

## Daily signal sheet

Print the position the recommended refined-quincena rule takes for the next
session (or any date), with the slow-volume size multiplier, at USD 1M/trade:

```bash
python src/daily_signal.py            # next session; writes out/daily_signal.txt
python src/daily_signal.py 2026-06-10 # a specific date
python src/daily_signal.py --png      # also render out/daily_signal.png
```

The logic mirrors `quincena.pos_calendar` exactly: **SHORT USD within `CAL_PRE`
(=6) business days of Costa Rica's IVA (D-104) / mid-month quincena deadline** —
the 15th, rolled to the next business day — **LONG otherwise**; size trimmed to
0.5× when the slow 20-day seasonally-adjusted volume regime disagrees with the
calendar. It reads only data strictly before the target date, so it is safe to
run live each morning.

### The payment calendar (`payment_calendar.py`)

The short-USD window is anchored to the actual Costa Rica cash-flow calendar
rather than a fixed day-of-month band. Statutory events (Hacienda / TRIBU-CR and
CCSS):

| Event | Statutory date | Effect on USD/CRC |
|-------|----------------|-------------------|
| IVA (D-104) + retenciones (D-103) | 15th of following month (→ next business day) | firms sell USD → colón strengthens into it |
| Quincena — 1st fortnight payroll | ~15th | same mid-month USD-supply surge |
| Quincena — 2nd fortnight payroll | month-end | month-end payroll |
| CCSS planilla (social security) | 4th business day of month | early-month CRC demand |
| Pagos parciales de renta | end of Mar / Jun / Sep / Dec | quarter-end CRC demand |

Each date is snapped to **MONEX's own trading calendar** (the dates present in the
price series), so weekend/holiday rolling is correct by construction. Measured in
*business days to the IVA/quincena deadline*, the colón strengthens hardest 1–5
days **before** the deadline (−12 to −16 bps next-day) and reverts after — see
`out/q_calendar.png`.

See **FINDINGS.md** for conclusions. Headline: the colón is in a VWAP-anchored
appreciation crawl; daily **volume is a directional order-flow signal** (heavy
days = USD supply = colón strength) that **persists overnight**. Tested
adversarially (out-of-sample, walk-forward, transaction costs, trend benchmark,
permutation null, deflated Sharpe), the **order-flow continuation** edge holds:
out-of-sample Sharpe ≥ in-sample, **+8.3 bps/day alpha vs the trend** (t≈5.8),
and a **deployable close-to-close Sharpe ≈ 2.8** (gross). The momentum runner-up
is mostly the trend in disguise. Dominant risk: managed-float peg/policy-break
tail that an 18-month sample can't price.
