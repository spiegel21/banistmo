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

Bloomberg integration requires Windows + an active Bloomberg Terminal with the xlwings add-in. Without it, the fallback CSV paths are used automatically.

### Running

```bash
streamlit run src/dashboard.py                            # web UI on localhost:8501
python src/backfill_prices.py --start 2025-01-01          # bulk-import historical Bloomberg prices
python generate_sample_data.py                            # generate demo CSVs (no real data needed)
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
- `src/models.py` defines the three dataclasses (`Trade`, `Position`, `BondStatic`) that flow through the whole system.
- `src/data_io.py` owns all CSV read/write and backup logic; nothing else touches disk directly except `history.py` for price caching.

### Data model

Trade confirmation fields (exact headers from `data/trades.csv`, written by the email parser):

| Field | Description |
|-------|-------------|
| `Timestamp` | When the confirmation was parsed (informational only) |
| `cusip` | 9-char CUSIP identifier |
| `side` | `buy` or `sell`; `load_trades()` normalises to signed nominal |
| `nominal` | Face value (always positive in CSV; sign applied by `load_trades`) |
| `principal` | `price × \|nominal\| / 100` |
| `net` | Cash actually exchanged (`principal + accrued`); negative for buys |
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

### Key invariants

- Positions are always recomputed from full trade history. `portfolio.csv` is a cache only.
- After `load_trades()`, `nominal > 0` = long, `nominal < 0` = short/sell.
- `net` for a buy is negative (cash leaves); for a sell it is positive.
- Bloomberg prices are clean prices (% of par). Dirty price = clean + accrued today.
- `trade_date` (not `Timestamp`) is the effective date for all portfolio history.
- Total P&L = Realized + Price MTM + Accrued (three non-overlapping components).
- Realized P&L uses WAVG cost basis matching (consistent with Bloomberg PRISM / Advent Geneva).

### Day-count conventions

- `Act/360` — actual days / 360 (common for EUR IG corps)
- `Act/365` — actual days / 365
- `30/360` — 30-day months, 360-day year (common for USD corps)

### Bloomberg bridge

- BDP (current prices): `bloomberg.get_prices()` — uses the "Live MTM" sheet of the Excel template
- BDH (historical prices): `bloomberg.get_historical_prices_bdh()` — uses the "History" sheet
- Template: `templates/bloomberg_prices.xlsx` — regenerate with `templates/create_bloomberg_template.py`
- Fallback for current prices: `data/prices/manual_prices.csv` (`cusip, px_last, date`)
- Fallback for history: `data/prices/manual_price_history.csv` (`date, cusip, px_last`)

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
