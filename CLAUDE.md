# CLAUDE.md — banistmo

Context for AI assistants working in this repo.

## Repo layout

Multi-project monorepo. Each project under `projects/` is independent — do not share code across projects.

## Project 02 — pnl-dashboard

### Data model

Trade confirmation fields (exact headers from `data/trades.csv`, written by email parser):
- `Timestamp` — when the confirmation was parsed (informational only)
- `cusip` — CUSIP identifier (9-char string, e.g. `037833100`)
- `side` — `buy` or `sell`; `load_trades()` normalises to signed nominal
- `nominal` — face value in currency units (always positive in CSV; sign applied by `load_trades`)
- `principal` — price × |nominal| / 100
- `net` — cash actually exchanged (principal + accrued); negative for buys (cash out)
- `accrued` — interest accrued from last coupon to settle date at time of trade
- `price` — quoted clean price (percent of par, e.g. 98.5)
- `yield_closed` — yield to maturity at time of trade; may be `None` (ignored in calculations)
- `trade_date` — date trade was agreed (ISO 8601)
- `settle_date` — settlement date (typically T+2 for bonds)
- `trader` — trader name string
- `portfolio` — book name (e.g. `"HY Book"`); defaults to `"default"` if blank

Position fields (computed by position_manager.py, written to `data/portfolio.csv`):
- `net_nominal` — sum of all signed nominals for this CUSIP
- `wavg_price` — weighted average clean price (cost basis)
- `book_value` — sum of `net` across all trades

### Key invariants

- Positions are always recomputed from full trade history. `portfolio.csv` is cache only.
- After `load_trades()`, `nominal > 0` = long (buy), `nominal < 0` = sell/short.
- `net` for a buy is negative (cash leaves). For a sell it is positive.
- Bloomberg prices are clean prices (percent of par). Dirty price = clean price + accrued today.
- All monetary values are in the bond's native currency (no FX conversion).
- `yield_closed` NaN values are ignored in all calculations.
- `trade_date` (not `Timestamp`) is the effective date for all portfolio history calculations.

### Day-count conventions supported

- `Act/360` — actual days / 360 (common for EUR IG corps)
- `Act/365` — actual days / 365
- `30/360` — assumes 30-day months, 360-day year (common for USD corps)

### Bloomberg bridge

- BDP (current prices): `bloomberg.get_prices()` — uses Sheet1 of the Excel template
- BDH (historical prices): `bloomberg.get_historical_prices_bdh()` — uses History sheet
- Template: `templates/bloomberg_prices.xlsx` — regenerate with `templates/create_bloomberg_template.py`
- Fallback: `data/prices/manual_prices.csv` (`cusip, px_last, date`) for current prices
- Fallback: `data/prices/manual_price_history.csv` (`date, cusip, px_last`) for historical

### File layout

| File | Description | Committed? |
|------|------------|-----------|
| `data/trades.csv` | Append-only trade log from email parser | No (real data) |
| `data/initial_positions.csv` | Seed positions at portfolio inception (user-entered once) | Yes |
| `data/bonds_static.csv` | Bond reference data: coupon, country, day-count (user-entered once per new CUSIP) | Yes |
| `data/portfolio.csv` | Computed current positions (rewritten each run) | No |
| `data/price_history.csv` | Accumulated daily Bloomberg prices (date, cusip, px_last) | No |
| `data/pnl_history.csv` | Pre-computed daily P&L by CUSIP and portfolio | No |
| `data/prices/prices_*.csv` | Timestamped BDP snapshots | No |

## Conventions

- Python 3.10+
- Dataclasses for data models (no ORM)
- pandas for all tabular computation
- No global mutable state
- Functions are pure where possible; side effects (file I/O) at the edges
