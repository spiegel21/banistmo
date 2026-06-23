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

Fill once per new CUSIP (auto-filled by the Bloomberg import where possible).
Reflects bond indenture terms plus enterprise classification dimensions.

| Column | Values |
|--------|--------|
| `cusip` | 9-char CUSIP |
| `name` | Bond description |
| `currency` | USD / EUR / COP / … |
| `country` | Issuer country (ISO-2) |
| `coupon_rate` | Decimal (e.g. 0.045 for 4.5%) |
| `coupon_frequency` | 1 (annual) / 2 (semi) / 4 (quarterly) / 12 / 0 (zero-coupon) |
| `day_count_convention` | `Act/360`, `Act/365`, `30/360`, `Act/Act` |
| `maturity_date` | YYYY-MM-DD |
| `first_coupon_date` | YYYY-MM-DD |
| `bbg_ticker` | Full Bloomberg ticker (e.g. `912828Z78 Govt`) |
| `instrument_type` | `Sovereign` / `Corporate` / `Agency` / `Supranational` / `Municipal` / `Other` |
| `issuer` | Issuer / obligor name |
| `country_of_risk` | Country of risk (may differ from `country`) |
| `sector` | Industry sector |
| `seniority` | Payment rank / seniority |
| `market` | `Local` (domestic-ccy) or `Global` (hard-ccy/eurobond); derived if blank |
| `rating_sp` / `rating_moody` / `rating_fitch` | Agency credit ratings |

Classification fields not supplied are **derived** (sovereign/corp from ticker
suffix; local/global from currency vs country of risk) and shown as `Unknown`
when they cannot be resolved — see the **Debug** tab for what needs input.

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

## Enterprise transparency & risk views

The dashboard is organised into tabs, several focused on auditability:

- **Overview** — daily/cumulative P&L, attribution, P&L by country, and the
  **Exposure & Concentration** panel (nominal/MTM by country of risk, sector,
  issuer, currency, sovereign/corp, local/global) plus a **maturity ladder**.
- **🔧 Debug** — *Needs-Attention* view: every trade, bond, or price that is
  broken or incomplete, with severity (error / warning / needs-input), the
  offending field, and a suggested fix. Catches missing names, weird/stale
  prices, economic mismatches, unknown bonds, and missing classification.
- **MTM Attribution** — Bond Detail, **Bond Movements** (full per-bond audit
  trail: running position, WAVG cost, cash, realized), Rollup (group by any
  classification dimension), Accrual Detail, **Coupon Calendar** (cash-flow
  forecast), and Price History.
- **Daily Ledger / Time Series / Data Editor** — drill-down and editing.

## Module overview

| File | Responsibility |
|------|---------------|
| `src/models.py` | `Trade`, `Position`, `BondStatic` dataclasses + day-count / instrument-type normalisers |
| `src/position_manager.py` | Load trades → compute net positions |
| `src/accruals.py` | Accrued interest + coupon schedule / calendar (Act/360, Act/365, 30/360, Act/Act) |
| `src/trading_gains.py` | Realized P&L (WAVG cost matching) |
| `src/mtm.py` | Mark-to-market valuation (price vs accrued split) |
| `src/history.py` | Daily P&L history, snapshots, accrual breakdown |
| `src/bloomberg.py` | xlwings bridge to Bloomberg Excel add-in |
| `src/classification.py` | Derive sovereign/corp, country of risk, local/global |
| `src/reconciliation.py` | Data-quality / Needs-Attention findings engine |
| `src/movements.py` | Per-bond movement lineage (running position & cost) |
| `src/exposure.py` | Exposure, concentration & maturity-ladder analytics |
| `src/dashboard.py` | Streamlit front-end |
