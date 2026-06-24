# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout

Multi-project monorepo. Each project under `projects/` is self-contained with its own `requirements.txt` and virtualenv. Do not share code across projects.

## Project 01 — forex-strategy

EUR/USD moving-average crossover strategy with sentiment overlay over 20 years of historical data (2005–2025).

### Setup & entry points

```bash
cd projects/01-forex-strategy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/plot.py           # download EURUSD via yfinance; writes data/eurusd_data_with_indicators.csv
python src/auto_trade.py     # MA crossover backtest + scipy parameter optimisation; writes out/strategy_results.csv
python src/sentiment.py      # scrape FXStreet headlines via Selenium + VADER/TextBlob scoring
```

`email_alert.py` is a helper module (Gmail SMTP); it is imported, not run standalone.

### Architecture

Scripts are independent pipeline stages, not a single application. Each reads from the previous stage's output CSV. There are no automated tests — strategy output CSVs serve as validation.

## Project 02 — pnl-dashboard

Real-time P&L dashboard for a fixed-income bond portfolio. Tracks positions, accrued interest, trading gains, and mark-to-market via Bloomberg pricing.

### Setup

```bash
cd projects/02-pnl-dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Running

```bash
streamlit run src/dashboard.py                            # web UI on localhost:8501
python src/backfill_prices.py --start 2025-01-01          # bulk-import historical Bloomberg prices
python generate_sample_data.py                            # generate demo trades + positions (no prices)
python templates/create_bloomberg_template.py             # regenerate bloomberg_prices.xlsx
```

### Tests

```bash
cd projects/02-pnl-dashboard
python -m pytest tests/                          # run all
python -m pytest tests/test_pnl.py -v           # single file
python -m pytest tests/test_pnl.py::test_name   # single test
```

`tests/conftest.py` creates an isolated temp data directory and patches `config.py` paths for every test — no real data files are needed to run the suite.

### First-run workflow

```
python generate_sample_data.py
  └─ writes: trades.csv, initial_positions.csv, bonds_static.csv (cusip-only, no prices)

streamlit run src/dashboard.py
  → positions visible; P&L shows "—" until Bloomberg data is imported

Sidebar → Bloomberg
  [gap detection runs automatically via find_price_gaps(all_trades)]
  ① Prepare & open template     → writes tickers to Live MTM + Static sheets;
                                   writes BDH blocks to History sheet (if gaps exist)
  [Bloomberg populates all sheets → Save & Close Excel]
  ③ Import all data             → bonds_static.csv filled; current prices loaded;
                                   price_history.csv appended;
                                   P&L recomputed automatically (full history, all portfolios)
                                   → pnl_history.csv written; dashboard refreshes
```

`generate_sample_data.py` produces only reality inputs (trades + positions). `price_history.csv` is written **only** by the Bloomberg import step — never by the generator. For offline/no-Bloomberg use, populate `data/prices/manual_price_history.csv` (`date, cusip, px_last`) and `data/prices/manual_prices.csv` (`cusip, px_last, date`) manually; the import step falls back to these automatically.

### Architecture

```
Email parser (external) → data/trades.csv
                                │
                    position_manager.py  → data/portfolio.csv (cache)
                       /            \
              accruals.py        trading_gains.py
            (coupon schedule,     (realized P&L,
             day-count)            WAVG cost basis)
                       \            /
                         mtm.py  ←── bloomberg.py ←── templates/bloomberg_prices.xlsx
                            │
                        dashboard.py  (Streamlit)
                            │
                        history.py   (daily P&L snapshots → data/pnl_history.csv)
```

- `src/config.py` is the single source of all file paths; override the data directory with `PNL_DATA_DIR` env var.
- `src/models.py` defines the three dataclasses (`Trade`, `Position`, `BondStatic`) that flow through the whole system, plus `normalise_day_count` / `normalise_instrument_type` canonicalisers.
- `src/data_io.py` owns all CSV read/write and backup logic; nothing else touches disk directly except `history.py` for price caching.
- `bloomberg.find_price_gaps(all_trades)` walks the signed-trade stream per CUSIP to compute hold intervals (position-by-date), then diffs against `price_history.csv` to return only the missing business-day ranges — the exact input for `prepare_history_template()`. Called on every dashboard render so gap detection is always current.

Transparency & risk modules (all pure, unit-tested, framework-agnostic):
- `src/classification.py` — derives the enterprise classification dimensions (sovereign/corp, country of risk, local/global) with precedence: explicit `BondStatic` field → Bloomberg-extracted value → heuristic → `"Unknown"`. Never invents data.
- `src/reconciliation.py` — `run_all_checks()` returns a tidy findings DataFrame powering the **Debug / Needs-Attention** tab: broken/incomplete trades, unknown bonds, missing classification, and missing/weird/stale prices, each with severity + suggested fix.
- `src/movements.py` — `position_movements()` rebuilds each bond's per-trade running position, WAVG cost, cash, and realized gain; its realized column reconciles exactly with `trading_gains`.
- `src/exposure.py` — nominal/MTM aggregation across classification dimensions, top-N concentration, and a remaining-tenor maturity ladder (no pricing model).
- `src/analytics.py` — analytical risk engine: solves YTM from the clean price, then modified/Macaulay duration, DV01, and convexity by finite differences off `price_from_yield` (street convention: discount at yield compounded `coupon_frequency`×/yr, Act/365 time). `portfolio_risk()` returns a per-bond table + MV-weighted summary; `var_historical()` gives VaR/ES from realised daily P&L. No external curve, so G-/Z-spread are out of scope.
- `src/accruals.py` also provides `next_coupon_date` / `upcoming_coupons` for the Coupon Calendar cash-flow forecast.

### Data model

Trade confirmation fields (exact headers from `data/trades.csv`, written by the email parser):

| Field | Description |
|-------|-------------|
| `Timestamp` | When the confirmation was parsed (informational only) |
| `cusip` | 9-char CUSIP identifier |
| `side` | `buy` or `sell`; `load_trades()` normalises to signed nominal |
| `nominal` | Face value (always positive in CSV; sign applied by `load_trades`) |
| `principal` | `price × \|nominal\| / 100` |
| `net` | Cash actually exchanged (`principal + accrued`); stored unsigned in CSV, signed on load (negative for buys, positive for sells) |
| `accrued` | Interest accrued from last coupon to settle date |
| `price` | Quoted clean price (% of par, e.g. `98.5`) |
| `yield_closed` | YTM at trade time; may be `None` (ignored in all calculations) |
| `trade_date` | Date trade was agreed (ISO 8601); effective date for all history |
| `settle_date` | Settlement date (typically T+2) |
| `trader` | Trader name string |
| `portfolio` | Book name (e.g. `"HY Book"`); defaults to `"default"` if blank |

Position fields computed by `position_manager.py`:

| Field | Description |
|-------|-------------|
| `net_nominal` | Sum of all signed nominals for this CUSIP |
| `wavg_price` | Weighted-average clean price (cost basis) |
| `book_value` | Sum of `net` across all trades |

`BondStatic` (in `bonds_static.csv`) carries indenture terms (`coupon_rate`,
`coupon_frequency`, `day_count_convention`, `maturity_date`, `first_coupon_date`)
plus enterprise classification fields, all optional and `""` == unknown:
`instrument_type` (canonical Sovereign/Corporate/Agency/Supranational/Municipal/
Other), `issuer`, `country_of_risk`, `sector`, `seniority`, `market`
(Local/Global), and `rating_sp` / `rating_moody` / `rating_fitch`. Bloomberg
import fills these where the terminal returns them; `classification.py` derives
sovereign/corp and local/global when they are blank.

### Key invariants

- Positions are always recomputed from full trade history. `portfolio.csv` is a cache only.
- After `load_trades()`, `nominal > 0` = long, `nominal < 0` = short/sell.
- `net` is stored unsigned in the CSV and signed on load (like `nominal`): after `load_trades()` it is negative for a buy (cash leaves) and positive for a sell.
- Bloomberg prices are clean prices (% of par). Dirty price = clean + accrued today.
- `trade_date` (not `Timestamp`) is the effective date for all portfolio history.
- Total P&L = Realized + Price MTM + Accrued (three non-overlapping components).
- Realized P&L uses WAVG cost basis matching (consistent with Bloomberg PRISM / Advent Geneva).

### Day-count conventions

- `Act/360` — actual days / 360 (common for EUR IG corps)
- `Act/365` — actual days / 365
- `30/360` — 30-day months, 360-day year (common for USD corps)

### Bloomberg bridge

Bloomberg integration requires Windows + an active Bloomberg Terminal with the xlwings add-in.

- BDP (current prices): `bloomberg.get_prices()` — uses the "Live MTM" sheet of the Excel template
- BDH (historical prices): `bloomberg.get_historical_prices_bdh()` — uses the "History" sheet
- Static data: "Static" sheet BDP formulas populate `bonds_static.csv` (name, coupon, maturity, day-count, etc.)
- Template: `templates/bloomberg_prices.xlsx` — regenerate with `templates/create_bloomberg_template.py`
- `price_history.csv` is written **only** by the dashboard import step, never directly by any generator script
- Fallback for current prices (no Bloomberg): `data/prices/manual_prices.csv` (`cusip, px_last, date`)
- Fallback for history (no Bloomberg): `data/prices/manual_price_history.csv` (`date, cusip, px_last`)

### File layout

| File | Description | Committed? |
|------|-------------|-----------|
| `data/trades.csv` | Append-only trade log from email parser | No (real data) |
| `data/initial_positions.csv` | Seed positions at portfolio inception | Yes |
| `data/bonds_static.csv` | Bond reference data: coupon, country, day-count | Yes |
| `data/portfolio.csv` | Computed current positions (rewritten each run) | No |
| `data/price_history.csv` | Accumulated daily Bloomberg prices | No |
| `data/pnl_history.csv` | Pre-computed daily P&L by CUSIP and portfolio | No |
| `data/prices/prices_*.csv` | Timestamped BDP snapshots | No |

## Conventions

- Python 3.10+
- Dataclasses for data models (no ORM)
- pandas for all tabular computation
- No global mutable state; side effects (file I/O) only at module edges
- No linting or formatting config is present — follow PEP 8
