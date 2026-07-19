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
src/backtest.py        # rigorous, overfitting-aware tests (close-priced) -> backtest_results.json
src/backtest_vwap.py   # same strategies + slippage, priced at the session VWAP (realistic) -> backtest_vwap_results.json
src/dynamics.py        # underlying-mechanism deep-dive + $1M/trade dollar rules
src/payment_calendar.py # real CR IVA (D-104) / quincena / CCSS payment calendar
src/quincena.py        # calendar (quincena) strategy (recommended) + trading calendar
src/exits.py           # stop-loss / take-profit exit OVERLAY on the calendar rule (optimised) -> exits_results.json
src/exit_lab.py        # FULL exit engine: trailing / hard stop / take-profit / time stop, in
                       #   fixed bps or vol multiples; in-sample tuned with a 1-SE + drawdown
                       #   selection rule -> exit_lab.json + xl_*.png
src/rank_strategies.py # EVERY strategy re-priced on ONE basis, ranked by in-sample Sharpe;
                       #   supersedes the per-module tables -> ranking.json + ranking.png
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
python src/backtest_vwap.py  # realistic VWAP-priced re-run of the same strategies -> bt_vwap_*.png + json
python src/volume_model.py   # conviction-sized ML model (walk-forward) -> vm_*.png + vm_results.json
python src/dynamics.py       # mechanism deep-dive + $1M/trade building-block rules -> dyn_*.png + json
python src/quincena.py       # calendar (quincena) strategy -> q_*.png + quincena_results.json
python src/exits.py          # optimise a trailing-stop / floor exit overlay on it -> ex_*.png + exits_results.json
python src/exit_lab.py       # full exit engine (stop/take-profit/time, fixed & vol-scaled) -> xl_*.png + exit_lab.json
python src/rank_strategies.py # rank EVERY strategy on one basis, in-sample -> ranking.png + ranking.json
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

See **FINDINGS.md** for conclusions.

**Headline.** Ranked on one common accounting basis (`src/rank_strategies.py`), the
**calendar (quincena) family dominates** — it is the only edge that works in both the
pegged 2014–2021 regime and the 2021–2026 float. The best configuration is the calendar
entry plus a **trailing 30 bps + hard 40 bps stop** overlay: in-sample Sharpe **3.70**,
out-of-sample **3.92**, max drawdown cut ~45%. The overlay is a **risk** improvement, not
an alpha one — total P&L is statistically unchanged (paired p = 0.73). **Take-profit
actively hurts** and is not recommended.

The daily **order-flow / volume** signals are directionally real but **regime-contingent**:
in-sample Sharpe −0.26 to +0.49 against out-of-sample 2.0–3.7, with $240k–$400k drawdowns
at 80–97 round trips/yr. Their full-sample numbers average a dead regime with a live one.

Dominant risk: a BCCR re-peg / intervention that would mute both the calendar and flow
signals. The sample is 11.5 years (2,663 sessions), but the *float* regime that carries most
of the edge is only the last ~4.5 years, so the tail is thinly sampled.

> **Two figures previously quoted here are unsourced and have been removed:** "+8.3 bps/day
> alpha vs the trend (t≈5.8)" has no reproducing output in any JSON, and the claim that the
> tail risk is one "an 18-month sample can't price" contradicted this file's own 11.5-year
> sample. The gross close-to-close Sharpe ≈ 2.8 for the order-flow rule does reproduce
> (`backtest_results.json` → `Order-flow daily.gross_sharpe`), but it is **gross of
> slippage**; net of the 0.65 CRC round trip that rule does not survive.
