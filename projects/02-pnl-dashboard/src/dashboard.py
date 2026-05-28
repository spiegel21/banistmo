"""
P&L Dashboard — Streamlit app.

Run with:  streamlit run src/dashboard.py

Total P&L = Realized + Price MTM + Accrued Interest. These three are mutually
exclusive (Price MTM and Accrued together make up total unrealized), so they
sum to the headline figure with no double counting.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

import config
from position_manager import load_all_trades, compute_positions
from accruals import load_bonds_static, total_portfolio_accruals
from trading_gains import total_realized_pnl
from mtm import mark_to_market
from bloomberg import get_prices, load_latest_prices
from history import compute_daily_pnl, load_pnl_history

TEMPLATE_PATH = config.BLOOMBERG_TEMPLATE_PATH
PRICES_DIR = config.PRICES_DIR


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.0f}"


# ── page + data ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="P&L Dashboard", page_icon="📊", layout="wide")
st.title("Fixed-Income P&L Dashboard")

all_trades = load_all_trades()
bonds_static = load_bonds_static()

# ── sidebar: filters ──────────────────────────────────────────────────────────

st.sidebar.header("Filters")


def _options(col):
    return sorted(all_trades[col].dropna().unique().tolist()) if not all_trades.empty else []


sel_portfolios = st.sidebar.multiselect("Portfolio", _options("portfolio"), default=_options("portfolio"))
all_countries = sorted({b.country for b in bonds_static.values() if b.country})
sel_countries = st.sidebar.multiselect("Country", all_countries, default=all_countries)
sel_traders = st.sidebar.multiselect("Trader", _options("trader"), default=_options("trader"))
sel_cusips = st.sidebar.multiselect("CUSIP / Bond", _options("cusip"), default=_options("cusip"))

min_date = all_trades["trade_date"].min().date() if not all_trades.empty else date.today() - timedelta(days=365)
as_of = st.sidebar.date_input("As of date", value=date.today())

# ── apply filters ─────────────────────────────────────────────────────────────

trades_df = all_trades.copy()
if not trades_df.empty:
    if sel_portfolios:
        trades_df = trades_df[trades_df["portfolio"].isin(sel_portfolios)]
    if sel_traders:
        trades_df = trades_df[trades_df["trader"].isin(sel_traders)]
    if sel_cusips:
        trades_df = trades_df[trades_df["cusip"].isin(sel_cusips)]
    if sel_countries and bonds_static:
        in_countries = {c for c, b in bonds_static.items() if b.country in sel_countries}
        trades_df = trades_df[trades_df["cusip"].isin(in_countries)]

positions = compute_positions(trades_df, as_of=as_of)

# ── sidebar: pricing ──────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("Pricing")

prices = load_latest_prices()
snapshots = sorted(PRICES_DIR.glob("prices_*.csv"))
if snapshots:
    ts = snapshots[-1].stem.replace("prices_", "").replace("_", " ")
    st.sidebar.caption(f"Last priced: {ts}")
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

# ── sidebar: history ──────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("History")

hist_start = st.sidebar.date_input("History from", value=min_date)
hist_portfolio = sel_portfolios[0] if len(sel_portfolios) == 1 else None

if st.sidebar.button("Recompute P&L History"):
    with st.spinner("Computing daily P&L history..."):
        compute_daily_pnl(hist_start, as_of, portfolio=hist_portfolio)
    st.sidebar.success("History updated")
    st.rerun()

# ── compute P&L components ────────────────────────────────────────────────────

trades_to_date = trades_df[trades_df["trade_date"] <= pd.Timestamp(as_of)] if not trades_df.empty else trades_df
realized_df = total_realized_pnl(trades_to_date)
mtm_df = mark_to_market(positions, prices, bonds_static, as_of)

total_realized = realized_df["realized_gain"].sum() if not realized_df.empty else 0.0
total_price = mtm_df["price_pnl"].sum() if not mtm_df.empty and mtm_df["price_pnl"].notna().any() else 0.0
total_accrued = mtm_df["accrued_pnl"].sum() if not mtm_df.empty and mtm_df["accrued_pnl"].notna().any() else 0.0
total_pnl = total_realized + total_price + total_accrued

# ── KPI row ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total P&L", _fmt(total_pnl))
c2.metric("Price MTM", _fmt(total_price))
c3.metric("Accrued Interest", _fmt(total_accrued))
c4.metric("Realized Gains", _fmt(total_realized))

st.markdown("---")

# ── positions + attribution ───────────────────────────────────────────────────

left, right = st.columns([3, 2])

with left:
    st.subheader("Positions")
    if mtm_df.empty:
        if trades_df.empty:
            st.info("No trades match the current filters. Check `data/trades.csv`.")
        else:
            st.info("No prices loaded. Click **Refresh Bloomberg Prices** in the sidebar.")
    else:
        realized_lookup = (
            realized_df.set_index("cusip")["realized_gain"].to_dict() if not realized_df.empty else {}
        )
        names = {c: b.name for c, b in bonds_static.items()}
        rows = []
        for _, r in mtm_df.iterrows():
            cusip = r["cusip"]
            rows.append({
                "CUSIP": cusip,
                "Name": names.get(cusip, ""),
                "Nominal": f"{r['net_nominal']:,.0f}",
                "Clean Px": f"{r['clean_px']:.3f}" if pd.notna(r["clean_px"]) else "—",
                "Price MTM": _fmt(r["price_pnl"]),
                "Accrued": _fmt(r["accrued_pnl"]),
                "Realized": _fmt(realized_lookup.get(cusip, 0.0)),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with right:
    st.subheader("P&L Attribution")
    attr = pd.DataFrame({
        "Component": ["Realized", "Price MTM", "Accrued"],
        "Value": [total_realized, total_price, total_accrued],
    })
    if attr["Value"].abs().sum() > 0:
        st.bar_chart(attr.set_index("Component"), use_container_width=True)
    else:
        st.info("No P&L data to display")

# ── history chart ─────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("P&L History")

pnl_hist = load_pnl_history(portfolio=hist_portfolio, start_date=hist_start, end_date=as_of)
if pnl_hist.empty:
    st.info(
        "No history computed yet. Pick a start date and click **Recompute P&L History**. "
        "Populate `price_history.csv` first via `backfill_prices.py`."
    )
else:
    daily = (
        pnl_hist.groupby("date")[["price_pnl", "accrued", "realized_gain", "total_pnl"]]
        .sum().reset_index().sort_values("date")
    )
    daily["date"] = pd.to_datetime(daily["date"])
    chart_tab, data_tab = st.tabs(["Chart", "Data"])
    with chart_tab:
        comp = daily.set_index("date")[["price_pnl", "accrued", "realized_gain"]]
        comp.columns = ["Price MTM", "Accrued", "Realized"]
        st.line_chart(comp, use_container_width=True)
        st.caption("Total P&L")
        st.line_chart(daily.set_index("date")[["total_pnl"]], use_container_width=True)
    with data_tab:
        st.dataframe(daily, use_container_width=True, hide_index=True)

# ── footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
label = ", ".join(sel_portfolios) if sel_portfolios else "all portfolios"
st.caption(f"As of {as_of}  |  {label}  |  {len(positions)} CUSIPs  |  {len(trades_df)} trades")
