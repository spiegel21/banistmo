# 02 — Fixed-Income P&L Dashboard

Real-time P&L dashboard for a fixed-income bond portfolio. Tracks positions, accrued interest, trading gains, and mark-to-market via Bloomberg pricing.

## Architecture

```
Email parser (local, not in repo)
    │ appends rows to
    ▼
data/trades.csv
    │
    ▼
src/position_manager.py  ──►  data/portfolio.csv (current positions)
    │
    ├── src/accruals.py      ◄── data/bonds_static.csv
    ├── src/trading_gains.py     (realized FIFO + unrealized)
    └── src/mtm.py           ◄── src/bloomberg.py  ◄── templates/bloomberg_prices.xlsx
                │
                ▼
        src/dashboard.py  (Streamlit, localhost:8501)
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -r requirements.txt

# 1. Place your trades.csv in data/  (email parser writes here)
# 2. Fill data/bonds_static.csv for any new ISINs
# 3. Run the dashboard
streamlit run src/dashboard.py
```

## Data inputs

### Trade log (`data/trades.csv`)

Written and appended to by the local email parser. One row per trade confirmation. Never hand-edit.

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | str | Parse timestamp (informational) |
| `cusip` | str | 9-char CUSIP, e.g. `037833100` |
| `side` | str | `buy` or `sell` |
| `nominal` | float | Face value (always positive in CSV; `load_trades` applies sign) |
| `principal` | float | price × \|nominal\| / 100 |
| `price` | float | Quoted clean price (% of par, e.g. 98.5) |
| `accrued` | float | Accrued interest at settle date |
| `net` | float | Cash exchanged (negative for buys) |
| `yield_closed` | float\|None | Yield to maturity; may be blank — ignored in calculations |
| `trade_date` | date | ISO 8601 (YYYY-MM-DD) |
| `settle_date` | date | Settlement date |
| `trader` | str | Trader name |

### Bond static data (`data/bonds_static.csv`)

Fill once per new ISIN. Never changes (it reflects bond indenture terms).

| Column | Values |
|--------|--------|
| `isin` | 12-char ISIN |
| `name` | Bond description |
| `currency` | EUR / USD / GBP |
| `coupon_rate` | Decimal (e.g. 0.045 for 4.5%) |
| `coupon_frequency` | 1 (annual) / 2 (semi) / 4 (quarterly) |
| `day_count_convention` | `Act/360`, `Act/365`, `30/360` |
| `maturity_date` | YYYY-MM-DD |
| `first_coupon_date` | YYYY-MM-DD |

## Bloomberg pricing bridge

Requires Windows + Bloomberg Terminal running.

```
templates/bloomberg_prices.xlsx
  Col A: ISIN
  Col B: =BDP(A2&" Corp","PX_LAST")        ← last clean price
  Col C: =BDP(A2&" Corp","YAS_ASW_SPREAD") ← asset-swap spread
  Col D: =BDP(A2&" Corp","YLD_YTM_MID")    ← yield to maturity
```

Python writes ISINs to column A, calls `app.calculate()` via xlwings (triggers Bloomberg refresh), then reads back columns B–D. No manual copy-paste needed.

**Fallback (Bloomberg unavailable):** drop `data/prices/manual_prices.csv` with columns `isin, px_last, date`. The dashboard picks this up automatically.

## Dashboard layout

```
┌───────────────────────────────────────────────────────┐
│  Total P&L  |  Unrealized  |  Realized  |  Accruals   │
├────────────────────┬──────────────────────────────────┤
│  Positions table   │  P&L attribution (bar chart)     │
│  ISIN | nominal |  │  trading gain / accrual / MTM    │
│  MTM  | accrual |  ├──────────────────────────────────┤
│  gain | yield   │  │  Cumulative P&L over time        │
├────────────────────┴──────────────────────────────────┤
│  [🔄 Refresh Bloomberg Prices]   Last updated: HH:MM  │
└───────────────────────────────────────────────────────┘
```

## Module overview

| File | Responsibility |
|------|---------------|
| `src/models.py` | `Trade` and `Position` dataclasses |
| `src/position_manager.py` | Load trades → compute net positions |
| `src/accruals.py` | Accrued interest (Act/360, Act/365, 30/360) |
| `src/trading_gains.py` | Realized P&L (FIFO) + unrealized |
| `src/bloomberg.py` | xlwings bridge to Bloomberg Excel add-in |
| `src/mtm.py` | Mark-to-market valuation |
| `src/dashboard.py` | Streamlit front-end |
