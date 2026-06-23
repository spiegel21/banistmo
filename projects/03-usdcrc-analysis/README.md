# 03 — USD/CRC (MONEX) pricing analysis

Reverse-engineering the price-formation model of the USD/CRC wholesale FX market
(BCCR MONEX) and screening for trading opportunities from a single BCCR
`frmVerCatCuadro` export.

## Layout
```
data/monex_raw.xls     # original BCCR export (HTML-as-XLS)
data/monex_clean.csv   # tidy one-row-per-trading-day table (generated)
src/parse_monex.py     # parse the HTML matrix -> clean CSV
src/analyze.py         # full quantitative analysis + charts
out/                   # analysis_report.txt + PNG charts (generated)
FINDINGS.md            # written conclusions
```

## Run
```bash
cd projects/03-usdcrc-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/parse_monex.py      # writes data/monex_clean.csv
python src/analyze.py          # writes out/*.png and prints the report
```

See **FINDINGS.md** for the conclusions. Headline: the colón is in a smooth
VWAP-anchored appreciation crawl; daily **volume is a directional order-flow
signal** (heavy days = USD supply = colón strength) that **persists overnight**,
giving an order-flow-continuation strategy of ~+7 bps/day (Sharpe ≈ 4.4, gross).
