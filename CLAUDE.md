# CLAUDE.md — banistmo

Context for AI assistants working in this repo.

## Repo layout

Multi-project monorepo. Each project under `projects/` is independent — do not share code across projects.

## Project 02 — pnl-dashboard

### Data model

Trade confirmation fields (sourced from email parser, written to `data/trades/`):
- `isin` — ISIN identifier (12-char string, e.g. `XS1234567890`)
- `nominal` — face value in currency units, **signed**: positive = buy, negative = sell
- `principal` — clean price × |nominal| / 100
- `net_proceeds` — cash actually exchanged (principal + accrued_at_trade); negative for buys (cash out)
- `accrued_at_trade` — interest accrued from last coupon to settle date at time of trade
- `clean_price` — quoted price (percent of par, e.g. 98.5)
- `yield_pct` — yield to maturity at time of trade
- `trade_date` — date trade was agreed (ISO 8601)
- `trader` — trader name string
- `settle_date` — settlement date (typically T+2 for bonds)

Position fields (computed by position_manager.py):
- `net_nominal` — sum of all signed nominals for this ISIN
- `wavg_clean_price` — weighted average clean price (cost basis)
- `book_value` — sum of net_proceeds across all trades

### Key invariants

- Positions are always recomputed from full trade history. Snapshots are cache only.
- `nominal > 0` means the desk is long; `nominal < 0` means a sell/short.
- `net_proceeds` for a buy is negative (cash leaves). For a sell it is positive.
- Bloomberg prices are clean prices (percent of par). Dirty price = clean price + accrued today.
- All monetary values are in the bond's native currency (no FX conversion).

### Day-count conventions supported

- `Act/360` — actual days / 360 (common for EUR IG corps)
- `Act/365` — actual days / 365
- `30/360` — assumes 30-day months, 360-day year (common for USD corps)

### Bloomberg bridge

- Requires Windows + Bloomberg Terminal running + xlwings + pywin32
- Template: `templates/bloomberg_prices.xlsx` — column A = ISINs, B/C/D = BDP formulas
- Fallback: `data/prices/manual_prices.csv` with columns `isin, px_last, date`

### File naming

- Trade CSVs: `data/trades/YYYY-MM-DD_trades.csv` (one file per trading day; re-running is safe)
- Price snapshots: `data/prices/prices_YYYYMMDD_HHMMSS.csv`
- Position snapshots: `data/positions/snapshot_YYYYMMDD_HHMMSS.parquet`

## Conventions

- Python 3.10+
- Dataclasses for data models (no ORM)
- pandas for all tabular computation
- No global mutable state
- Functions are pure where possible; side effects (file I/O) at the edges
