"""
P&L Dashboard — Streamlit app.

Run with:  streamlit run src/dashboard.py
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from position_manager import refresh_portfolio, load_trades, load_initial_positions, _merge_with_initial, compute_positions
from accruals import load_bonds_static, total_portfolio_accruals
from trading_gains import total_realized_pnl, unrealized_pnl
from mtm import mark_to_market
from bloomberg import get_prices, load_latest_prices
from history import compute_daily_pnl, load_pnl_history

DATA_DIR = Path(__file__).parent.parent / "data"
TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "bloomberg_prices.xlsx"


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.0f}"


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="P&L Dashboard", page_icon="📊", layout="wide")
st.title("Fixed-Income P&L Dashboard")

# ── load raw data (unfiltered) ────────────────────────────────────────────────

_raw_trades = load_trades(DATA_DIR / "trades.csv")
_initial = load_initial_positions(DATA_DIR / "initial_positions.csv")
_all_trades = (
    pd.concat([_initial, _raw_trades], ignore_index=True).sort_values("trade_date")
    if not _initial.empty else _raw_trades
)
bonds_static = load_bonds_static(DATA_DIR / "bonds_static.csv")

# ── sidebar: filters ──────────────────────────────────────────────────────────

st.sidebar.header("Filters")

# Portfolio
all_portfolios = sorted(_all_trades["portfolio"].dropna().unique().tolist()) if not _all_trades.empty else []
sel_portfolios = st.sidebar.multiselect("Portfolio", all_portfolios, default=all_portfolios)

# Country (from bonds_static)
all_countries = sorted({b.country for b in bonds_static.values() if b.country}) if bonds_static else []
sel_countries = st.sidebar.multiselect("Country", all_countries, default=all_countries)

# Trader
all_traders = sorted(_all_trades["trader"].dropna().unique().tolist()) if not _all_trades.empty else []
sel_traders = st.sidebar.multiselect("Trader", all_traders, default=all_traders)

# CUSIP / Bond
all_cusips = sorted(_all_trades["cusip"].dropna().unique().tolist()) if not _all_trades.empty else []
sel_cusips = st.sidebar.multiselect("CUSIP / Bond", all_cusips, default=all_cusips)

# Date range
min_date = _all_trades["trade_date"].min().date() if not _all_trades.empty else date.today() - timedelta(days=365)
as_of = st.sidebar.date_input("As of date", value=date.today())

# ── apply filters ─────────────────────────────────────────────────────────────

trades_df = _all_trades.copy()
if sel_portfolios:
    trades_df = trades_df[trades_df["portfolio"].isin(sel_portfolios)]
if sel_traders:
    trades_df = trades_df[trades_df["trader"].isin(sel_traders)]
if sel_cusips:
    trades_df = trades_df[trades_df["cusip"].isin(sel_cusips)]

# Country filter: via bonds_static lookup
if sel_countries and bonds_static:
    cusips_in_countries = {c for c, b in bonds_static.items() if b.country in sel_countries}
    trades_df = trades_df[trades_df["cusip"].isin(cusips_in_countries)]

# Positions computed from filtered trades up to as_of
positions = compute_positions(trades_df, as_of=as_of)

# ── sidebar: pricing controls ─────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("Pricing")

prices = load_latest_prices()

latest_snapshots = sorted((DATA_DIR / "prices").glob("prices_*.csv"))
if latest_snapshots:
    ts = latest_snapshots[-1].stem.replace("prices_", "").replace("_", " ")
    st.sidebar.caption(f"Last updated: {ts}")
else:
    st.sidebar.caption("No price snapshot yet")

if st.sidebar.button("Refresh Bloomberg Prices"):
    cusip_list = list(positions.keys())
    if cusip_list:
        with st.spinner("Fetching Bloomberg prices..."):
            prices = get_prices(cusip_list, TEMPLATE_PATH)
        st.sidebar.success(f"Updated {len(prices)} CUSIPs")
        st.rerun()
    else:
        st.sidebar.warning("No positions to price")

# ── sidebar: history controls ─────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("History")

hist_start = st.sidebar.date_input("History from", value=min_date)
hist_portfolio = sel_portfolios[0] if len(sel_portfolios) == 1 else None

if st.sidebar.button("Recompute P&L History"):
    with st.spinner("Computing daily P&L history..."):
        compute_daily_pnl(hist_start, as_of, portfolio=hist_portfolio)
    st.sidebar.success("History updated")
    st.rerun()

# ── compute P&L components (as of date, filtered) ────────────────────────────

accruals_df = total_portfolio_accruals(positions, bonds_static, as_of)
realized_df = total_realized_pnl(trades_df[trades_df["trade_date"] <= pd.Timestamp(as_of)])
mtm_df = mark_to_market(positions, prices, DATA_DIR / "bonds_static.csv", as_of)

total_accruals = accruals_df["accrued"].sum() if not accruals_df.empty and "accrued" in accruals_df else 0.0
total_realized = realized_df["realized_gain"].sum() if not realized_df.empty else 0.0
total_unrealized = (
    mtm_df["mtm_gain"].sum()
    if not mtm_df.empty and "mtm_gain" in mtm_df and mtm_df["mtm_gain"].notna().any()
    else 0.0
)
total_pnl = total_realized + total_unrealized + total_accruals

# ── KPI row ───────────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total P&L", _fmt(total_pnl))
col2.metric("Unrealized MTM", _fmt(total_unrealized))
col3.metric("Realized Gains", _fmt(total_realized))
col4.metric("Accrued Interest", _fmt(total_accruals))

st.markdown("---")

# ── positions table + attribution ─────────────────────────────────────────────

left, right = st.columns([3, 2])

with left:
    st.subheader("Positions")
    if mtm_df.empty:
        if trades_df.empty:
            st.info("No trades match the current filters. Check `data/trades.csv`.")
        else:
            st.info("No prices loaded. Click **Refresh Bloomberg Prices** in the sidebar.")
    else:
        accruals_lookup = (
            accruals_df.set_index("cusip")["accrued"].to_dict() if not accruals_df.empty else {}
        )
        realized_lookup = (
            realized_df.set_index("cusip")["realized_gain"].to_dict() if not realized_df.empty else {}
        )
        bond_names = {c: b.name for c, b in bonds_static.items()}

        rows = []
        for _, row in mtm_df.iterrows():
            cusip = row["cusip"]
            rows.append({
                "CUSIP": cusip,
                "Name": bond_names.get(cusip, ""),
                "Nominal": f"{row['net_nominal']:,.0f}",
                "Clean Px": f"{row['clean_px']:.3f}" if row["clean_px"] else "—",
                "Dirty Px": f"{row['dirty_px']:.3f}" if row["dirty_px"] else "—",
                "MTM Gain": _fmt(row["mtm_gain"]),
                "Accrued": _fmt(accruals_lookup.get(cusip)),
                "Realized": _fmt(realized_lookup.get(cusip, 0.0)),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with right:
    st.subheader("P&L Attribution")
    attr_df = pd.DataFrame({
        "Component": ["Realized", "Unrealized MTM", "Accrued Interest"],
        "Value": [total_realized, total_unrealized, total_accruals],
    })
    if attr_df["Value"].abs().sum() > 0:
        st.bar_chart(attr_df.set_index("Component"), use_container_width=True)
    else:
        st.info("No P&L data to display")

# ── historical P&L chart ──────────────────────────────────────────────────────

st.markdown("---")
st.subheader("P&L History")

pnl_hist = load_pnl_history(
    portfolio=hist_portfolio,
    start_date=hist_start,
    end_date=as_of,
)

if pnl_hist.empty:
    st.info(
        "No history computed yet. Set a start date and click **Recompute P&L History** in the sidebar. "
        "Make sure `price_history.csv` is populated first (run `backfill_prices.py`)."
    )
else:
    # Aggregate across CUSIPs per day
    daily = (
        pnl_hist.groupby("date")[["mtm_gain", "accrued", "realized_gain", "total_pnl"]]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    daily["date"] = pd.to_datetime(daily["date"])

    chart_tab, table_tab = st.tabs(["Chart", "Data"])

    with chart_tab:
        chart_df = daily.set_index("date")[["mtm_gain", "accrued", "realized_gain"]]
        chart_df.columns = ["MTM Gain", "Accrued Interest", "Realized Gain"]
        st.line_chart(chart_df, use_container_width=True)

        st.caption("Total P&L")
        st.line_chart(daily.set_index("date")[["total_pnl"]], use_container_width=True)

    with table_tab:
        st.dataframe(
            daily.rename(columns={
                "date": "Date", "mtm_gain": "MTM Gain",
                "accrued": "Accrued", "realized_gain": "Realized", "total_pnl": "Total P&L",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ── footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
portfolio_label = ", ".join(sel_portfolios) if sel_portfolios else "all portfolios"
st.caption(
    f"As of {as_of}  |  {portfolio_label}  |  "
    f"{len(positions)} CUSIPs  |  {len(trades_df)} trade confirmations"
)
