"""
P&L Dashboard — Streamlit app.

Run with:  streamlit run src/dashboard.py

Total P&L = Realized + Price MTM + Accrued Interest. These three are mutually
exclusive (Price MTM and Accrued together make up total unrealized), so they
sum to the headline figure with no double counting.

Tabs:
  Overview        — headline KPIs, attribution, cumulative P&L curve
  Positions & MTM — full per-position mark-to-market (clean AND dirty price)
  Daily Ledger    — single-day drill-down: positions, accruals, trades, realized
  Time Series     — every position, every business day (downloadable)
  Data Editor     — add / modify / remove trades, bonds, initial positions, prices
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

import config
import data_io
from position_manager import load_all_trades, compute_positions, get_positions_as_of
from accruals import load_bonds_static, total_portfolio_accruals
from trading_gains import total_realized_pnl, realized_pnl
from mtm import mark_to_market
from bloomberg import get_prices, load_latest_prices
from history import (
    compute_daily_pnl, load_pnl_history,
    daily_snapshot, position_timeseries, accrual_breakdown,
)

TEMPLATE_PATH = config.BLOOMBERG_TEMPLATE_PATH
PRICES_DIR = config.PRICES_DIR


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.0f}"


def _nansum(series) -> float:
    return float(series.sum(min_count=1)) if series.notna().any() else 0.0


def _raw_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a CSV verbatim (cusip as string) or return an empty editable frame."""
    if Path(path).exists() and Path(path).stat().st_size > 0:
        return pd.read_csv(path, dtype={"cusip": str})
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


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

# single-portfolio scope used by per-portfolio views/history
hist_portfolio = sel_portfolios[0] if len(sel_portfolios) == 1 else None

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
total_price = _nansum(mtm_df["price_pnl"]) if not mtm_df.empty else 0.0
total_accrued = _nansum(mtm_df["accrued_pnl"]) if not mtm_df.empty else 0.0
total_pnl = total_realized + total_price + total_accrued

# ── tabs ────────────────────────────────────────────────────────────────────

tab_overview, tab_mtm, tab_ledger, tab_ts, tab_edit = st.tabs(
    ["Overview", "Positions & MTM", "Daily Ledger", "Time Series", "Data Editor"]
)

# ── Tab 1: Overview ───────────────────────────────────────────────────────────

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total P&L", _fmt(total_pnl))
    c2.metric("Price MTM", _fmt(total_price))
    c3.metric("Accrued Interest", _fmt(total_accrued))
    c4.metric("Realized Gains", _fmt(total_realized))

    st.markdown("---")
    left, right = st.columns([3, 2])

    with left:
        st.subheader("P&L Attribution")
        attr = pd.DataFrame({
            "Component": ["Realized", "Price MTM", "Accrued"],
            "Value": [total_realized, total_price, total_accrued],
        })
        if attr["Value"].abs().sum() > 0:
            st.bar_chart(attr.set_index("Component"), width='stretch')
        else:
            st.info("No P&L data to display")

    with right:
        st.subheader("Position summary")
        st.metric("Open positions", len([p for p in positions.values() if p.net_nominal != 0]))
        st.metric("Trades (filtered)", len(trades_df))

    st.markdown("---")
    st.subheader("Cumulative P&L")
    pnl_hist = load_pnl_history(portfolio=hist_portfolio, start_date=hist_start, end_date=as_of)
    if pnl_hist.empty:
        st.info(
            "No history computed yet. Pick a start date and click **Recompute P&L History**. "
            "Populate `price_history.csv` first (Data Editor → Manual Prices or `backfill_prices.py`)."
        )
    else:
        daily = (
            pnl_hist.groupby("date")[["price_pnl", "accrued", "realized_gain", "total_pnl"]]
            .sum().reset_index().sort_values("date")
        )
        daily["date"] = pd.to_datetime(daily["date"])
        comp = daily.set_index("date")[["price_pnl", "accrued", "realized_gain"]].fillna(0).cumsum()
        comp.columns = ["Price MTM", "Accrued", "Realized"]
        st.line_chart(comp, width='stretch')
        st.caption("Total P&L (cumulative)")
        st.line_chart(daily.set_index("date")[["total_pnl"]].fillna(0).cumsum(), width='stretch')

# ── Tab 2: Positions & MTM ────────────────────────────────────────────────────

with tab_mtm:
    st.subheader("Mark-to-Market — clean & dirty price")
    if mtm_df.empty:
        if trades_df.empty:
            st.info("No trades match the current filters. Check `data/trades.csv`.")
        else:
            st.info("No prices loaded. Add prices in **Data Editor → Manual Prices** or click **Refresh Bloomberg Prices**.")
    else:
        names = {c: b.name for c, b in bonds_static.items()}
        realized_lookup = (
            realized_df.set_index("cusip")["realized_gain"].to_dict() if not realized_df.empty else {}
        )
        view = mtm_df.copy()
        view.insert(1, "name", view["cusip"].map(names).fillna(""))
        view["realized"] = view["cusip"].map(realized_lookup).fillna(0.0)
        view = view.rename(columns={
            "cusip": "CUSIP", "name": "Name", "net_nominal": "Nominal",
            "clean_px": "Clean Px", "accrued_today_pct": "Accrued %", "dirty_px": "Dirty Px",
            "mtm_value": "MTM Value", "book_value": "Book Value", "mtm_gain": "MTM Gain",
            "accrued_pnl": "Accrued", "price_pnl": "Price MTM", "realized": "Realized",
            "note": "Note",
        })
        ordered = ["CUSIP", "Name", "Nominal", "Clean Px", "Accrued %", "Dirty Px",
                   "MTM Value", "Book Value", "Price MTM", "Accrued", "MTM Gain", "Realized", "Note"]
        st.dataframe(view[ordered], width='stretch', hide_index=True)

        st.markdown("**Portfolio totals**")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("MTM Value", _fmt(_nansum(mtm_df["mtm_value"])))
        t2.metric("Book Value", _fmt(_nansum(mtm_df["book_value"])))
        t3.metric("Price MTM", _fmt(total_price))
        t4.metric("Accrued", _fmt(total_accrued))
        t5.metric("MTM Gain", _fmt(_nansum(mtm_df["mtm_gain"])))
        st.caption("Dirty Px = Clean Px + Accrued %.  MTM Value = Nominal × Dirty Px / 100.  "
                   "MTM Gain = MTM Value + Book Value = Price MTM + Accrued.")

# ── Tab 3: Daily Ledger ───────────────────────────────────────────────────────

with tab_ledger:
    ledger_day = st.date_input("Ledger day", value=as_of, key="ledger_day")
    st.caption(f"Everything below is marked as of {ledger_day}"
               + (f" · portfolio: {hist_portfolio}" if hist_portfolio else " · all portfolios"))

    st.markdown("##### Positions held — clean & dirty MTM")
    snap = daily_snapshot(ledger_day, portfolio=hist_portfolio)
    if snap.empty:
        st.info("No positions held on this day (or no price available).")
    else:
        st.dataframe(snap, width='stretch', hide_index=True)

    st.markdown("##### Accrual detail")
    pos_day = get_positions_as_of(ledger_day, hist_portfolio)
    acc = accrual_breakdown(pos_day, bonds_static, ledger_day)
    if acc.empty:
        st.info("No accruing positions on this day.")
    else:
        st.dataframe(acc, width='stretch', hide_index=True)

    cL, cR = st.columns(2)
    with cL:
        st.markdown("##### Trades booked this day")
        if trades_df.empty:
            booked = trades_df
        else:
            booked = trades_df[trades_df["trade_date"].dt.date == ledger_day]
        if booked.empty:
            st.info("No trades booked.")
        else:
            cols = ["cusip", "side", "nominal", "price", "net", "accrued", "trader", "portfolio"]
            st.dataframe(booked[[c for c in cols if c in booked.columns]],
                         width='stretch', hide_index=True)

    with cR:
        st.markdown("##### Realized closes this day")
        rl = realized_pnl(trades_df) if not trades_df.empty else pd.DataFrame()
        if not rl.empty:
            rl = rl[rl["close_date"].dt.date == ledger_day]
        if rl is None or rl.empty:
            st.info("No positions closed.")
        else:
            st.dataframe(rl, width='stretch', hide_index=True)

# ── Tab 4: Time Series ────────────────────────────────────────────────────────

with tab_ts:
    st.subheader("Position time series (every business day)")
    st.caption(f"{hist_start} → {as_of}"
               + (f" · portfolio: {hist_portfolio}" if hist_portfolio else " · all portfolios"))
    with st.spinner("Building time series..."):
        ts = position_timeseries(hist_start, as_of, portfolio=hist_portfolio)
    if ts.empty:
        st.info("No data. Ensure positions exist and `price_history.csv` is populated for the range.")
    else:
        st.dataframe(ts, width='stretch', hide_index=True, height=500)
        st.download_button(
            "Download CSV", ts.to_csv(index=False).encode(),
            file_name=f"position_timeseries_{hist_start}_{as_of}.csv", mime="text/csv",
        )

# ── Tab 5: Data Editor ────────────────────────────────────────────────────────

with tab_edit:
    st.subheader("Edit underlying data")
    st.warning(
        "Saving **Trades** overwrites `data/trades.csv` — the same file the email "
        "parser appends to. A timestamped backup is written to `data/backups/` before "
        "every save, so changes are always recoverable."
    )

    def _editor_block(title, raw_df, save_fn, key):
        st.markdown(f"#### {title}")
        edited = st.data_editor(raw_df, num_rows="dynamic", width='stretch', key=key)
        if st.button(f"Save {title}", key=f"save_{key}"):
            try:
                backup = save_fn(edited)
                msg = "Saved."
                if backup:
                    msg += f"  Backup: {Path(backup).name}"
                st.success(msg)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    _editor_block(
        "Trades",
        _raw_csv(config.TRADES_PATH, data_io.TRADES_COLUMNS),
        data_io.save_trades, "ed_trades",
    )
    st.markdown("---")
    _editor_block(
        "Bond Static",
        _raw_csv(config.BONDS_STATIC_PATH, data_io.BONDS_COLUMNS),
        data_io.save_bonds_static, "ed_bonds",
    )
    st.markdown("---")
    _editor_block(
        "Initial Positions",
        _raw_csv(config.INITIAL_POSITIONS_PATH, data_io.INITIAL_COLUMNS),
        data_io.save_initial_positions, "ed_initial",
    )
    st.markdown("---")
    _editor_block(
        "Manual Prices (current)",
        _raw_csv(config.MANUAL_PRICES_PATH, data_io.MANUAL_PRICES_COLUMNS),
        data_io.save_manual_prices, "ed_manual_px",
    )
    st.markdown("---")
    _editor_block(
        "Manual Price History",
        _raw_csv(config.MANUAL_HISTORY_PATH, data_io.MANUAL_HISTORY_COLUMNS),
        data_io.save_manual_price_history, "ed_manual_hist",
    )

# ── footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
label = ", ".join(sel_portfolios) if sel_portfolios else "all portfolios"
st.caption(f"As of {as_of}  |  {label}  |  {len(positions)} CUSIPs  |  {len(trades_df)} trades")
