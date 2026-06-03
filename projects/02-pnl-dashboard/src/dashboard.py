"""
P&L Dashboard — Streamlit app.

Run with:  streamlit run src/dashboard.py

Total P&L = Realized + Price MTM + Accrued Interest. These three are mutually
exclusive (Price MTM and Accrued together make up total unrealized), so they
sum to the headline figure with no double counting.

Tabs:
  Overview          — headline KPIs, attribution, cumulative P&L curve
  Positions & MTM   — full per-position mark-to-market (clean AND dirty price)
  Positions by Date — editable positions snapshot; saves as adjustment trades
  Trades by Date    — editable single-day trade blotter with save + recalculate
  MTM Attribution   — per-bond / rollup transparency views, color-coded
  Daily Ledger      — single-day drill-down: positions, accruals, trades, realized
  Time Series       — every position, every business day (downloadable)
  Data Editor       — add / modify / remove trades, bonds, initial positions, prices
"""
import sys
from datetime import date, datetime, timedelta
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
from bloomberg import (
    load_latest_prices,
    prepare_template, read_prices_from_template,
    prepare_history_template, read_history_from_template,
    last_priced_date, find_price_gaps,
    read_static_from_template, merge_bonds_static,
    open_in_excel,
)
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


def _bonds_static_df(bonds_static: dict) -> pd.DataFrame:
    """Flatten BondStatic dict to a join-ready DataFrame."""
    if not bonds_static:
        return pd.DataFrame(columns=["cusip", "name", "country", "currency",
                                     "coupon_rate", "day_count_convention", "maturity_date"])
    rows = [{
        "cusip": b.cusip,
        "name": b.name,
        "country": b.country,
        "currency": b.currency,
        "coupon_rate": b.coupon_rate,
        "day_count_convention": b.day_count_convention,
        "maturity_date": b.maturity_date,
    } for b in bonds_static.values()]
    return pd.DataFrame(rows)


def _enrich(df: pd.DataFrame, bs_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Left-join df with bs_df on cusip; insert enrichment cols right after cusip."""
    if df.empty or bs_df.empty:
        return df
    available = [c for c in cols if c in bs_df.columns]
    merged = df.merge(bs_df[["cusip"] + available], on="cusip", how="left")
    # reorder: cusip, enrichment cols, then remaining original cols (preserving order)
    others = [c for c in df.columns if c != "cusip"]
    ordered = ["cusip"] + available + others
    return merged[[c for c in ordered if c in merged.columns]]


def _color_pnl_df(df: pd.DataFrame, pnl_cols: list[str]) -> pd.DataFrame:
    """Return df unchanged (color formatting disabled for pandas compatibility)."""
    return df


def _make_adjustment_trades(
    baseline_positions: dict,
    edited_df: pd.DataFrame,
    selected_date: date,
    portfolio: str | None,
) -> pd.DataFrame:
    """
    Diff edited positions against baseline; emit synthetic adjustment trade rows.
    Returns a DataFrame shaped to data_io.TRADES_COLUMNS (empty if no changes).
    """
    rows = []
    for _, row in edited_df.iterrows():
        cusip = str(row.get("cusip", ""))
        baseline = baseline_positions.get(cusip)
        base_nominal = baseline.net_nominal if baseline else 0.0
        base_price = baseline.wavg_price if baseline else 0.0

        edited_nominal = float(row.get("net_nominal", base_nominal))
        edited_price = float(row.get("wavg_price", base_price))

        delta = edited_nominal - base_nominal
        if abs(delta) < 1e-6 and abs(edited_price - base_price) < 1e-6:
            continue

        px = edited_price if edited_price > 0 else base_price
        side = "buy" if delta > 0 else "sell"
        abs_delta = abs(delta)
        principal = round(abs_delta * px / 100, 2)
        net = round(-principal if side == "buy" else principal, 2)

        rows.append({
            "Timestamp": datetime.now().isoformat(),
            "cusip": cusip,
            "side": side,
            "nominal": abs_delta,
            "principal": principal,
            "net": net,
            "accrued": 0.0,
            "price": round(px, 6),
            "yield_closed": float("nan"),
            "trade_date": selected_date.strftime("%m/%d/%y"),
            "settle_date": selected_date.strftime("%m/%d/%y"),
            "trader": "adjustment",
            "portfolio": portfolio or config.DEFAULT_PORTFOLIO,
        })
    return pd.DataFrame(rows, columns=data_io.TRADES_COLUMNS) if rows else pd.DataFrame(columns=data_io.TRADES_COLUMNS)


# ── page + data ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="P&L Dashboard", page_icon="📊", layout="wide")
st.title("Fixed-Income P&L Dashboard")

all_trades = load_all_trades()
bonds_static = load_bonds_static()
bs_df = _bonds_static_df(bonds_static)

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

# ── sidebar: today's prices ───────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("Today's Prices")

prices = load_latest_prices()
snapshots = sorted(PRICES_DIR.glob("prices_*.csv"))
if snapshots:
    _ts = snapshots[-1].stem.replace("prices_", "").replace("_", " ")
    st.sidebar.caption(f"Last priced: {_ts}")
else:
    st.sidebar.caption("No price snapshot yet")

cusip_list = list(positions.keys())

if st.sidebar.button("① Prepare & open template", key="btn_prepare_prices"):
    if cusip_list:
        _path = prepare_template(cusip_list, TEMPLATE_PATH)
        if open_in_excel(_path):
            st.sidebar.success(
                f"Opened **{_path.name}** in Excel — wait for Bloomberg to "
                "finish populating, then save and close."
            )
        else:
            st.sidebar.success(
                f"Template ready ({len(cusip_list)} CUSIP(s)) — open "
                f"**{_path}** in Excel, wait for Bloomberg, then save and close."
            )
    else:
        st.sidebar.warning("No positions to price.")

st.sidebar.caption("② Bloomberg populates → Save → Close Excel")

if st.sidebar.button("③ Import prices & static", key="btn_import_prices"):
    _imported = read_prices_from_template(TEMPLATE_PATH)
    _fetched_static = read_static_from_template(TEMPLATE_PATH)
    _msg_parts = []

    if _imported:
        prices = _imported
        _msg_parts.append(f"{len(_imported)} price(s) imported")
    else:
        _msg_parts.append("no prices found (open template in Excel first)")

    if not _fetched_static.empty and not _fetched_static["cusip"].isna().all():
        try:
            _n_new, _n_filled = merge_bonds_static(_fetched_static)
            if _n_new or _n_filled:
                _msg_parts.append(f"bond static: {_n_new} new CUSIP(s), {_n_filled} field(s) filled")
        except ValueError as _exc:
            st.sidebar.error(
                f"Bond static validation failed — fix rows in Data Editor → Bond Static.\n\n{_exc}"
            )

    if _imported:
        st.sidebar.success(" · ".join(_msg_parts))
        st.rerun()
    else:
        st.sidebar.warning(" · ".join(_msg_parts))

# ── sidebar: price history backfill ──────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("Price History")

# Gap detection runs on every render: picks up trade edits and new CUSIPs.
_price_gaps = find_price_gaps(all_trades) if not all_trades.empty else []

if _price_gaps:
    _n_cusips_gap = len({c for c, _, _ in _price_gaps})
    st.sidebar.warning(
        f"Missing prices: {len(_price_gaps)} range(s) across {_n_cusips_gap} CUSIP(s)."
    )
    if st.sidebar.button("① Prepare & open history template", key="btn_prepare_hist"):
        _path = prepare_history_template(_price_gaps, TEMPLATE_PATH)
        if open_in_excel(_path):
            st.sidebar.success(
                f"Opened **{_path.name}** in Excel — wait for all {len(_price_gaps)} "
                "BDH block(s) to populate, then save and close."
            )
        else:
            st.sidebar.success(
                f"Template ready ({len(_price_gaps)} block(s)) — open "
                f"**{_path}** in Excel, wait for Bloomberg, then save and close."
            )
    st.sidebar.caption("② Bloomberg populates all blocks → Save → Close Excel")
else:
    _last_px_date = last_priced_date()
    if _last_px_date:
        st.sidebar.caption(f"Price history complete through {_last_px_date}.")
    else:
        st.sidebar.caption("No price history yet.")

if st.sidebar.button("③ Import history", key="btn_import_hist"):
    _hist_df = read_history_from_template(TEMPLATE_PATH)
    if not _hist_df.empty:
        st.sidebar.success(
            f"Imported {len(_hist_df)} price records "
            f"through {_hist_df['date'].max()}."
        )
        st.rerun()
    else:
        st.sidebar.warning(
            "No prices found — open the template in Excel, wait for Bloomberg "
            "to populate all blocks, save, then import."
        )

# ── sidebar: P&L history ──────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("P&L History")

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

(tab_overview, tab_mtm, tab_positions_date, tab_trades_date,
 tab_attribution, tab_ledger, tab_ts, tab_edit) = st.tabs([
    "Overview",
    "Positions & MTM",
    "Positions by Date",
    "Trades by Date",
    "MTM Attribution",
    "Daily Ledger",
    "Time Series",
    "Data Editor",
])

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
        realized_lookup = (
            realized_df.set_index("cusip")["realized_gain"].to_dict() if not realized_df.empty else {}
        )
        view = mtm_df.copy()
        view = _enrich(view, bs_df, ["name", "country", "currency"])
        view["realized"] = view["cusip"].map(realized_lookup).fillna(0.0)
        view = view.rename(columns={
            "cusip": "CUSIP", "name": "Name", "country": "Country", "currency": "CCY",
            "net_nominal": "Nominal", "clean_px": "Clean Px",
            "accrued_today_pct": "Accrued %", "dirty_px": "Dirty Px",
            "mtm_value": "MTM Value", "book_value": "Book Value", "mtm_gain": "MTM Gain",
            "accrued_pnl": "Accrued", "price_pnl": "Price MTM", "realized": "Realized",
            "note": "Note",
        })
        ordered = ["CUSIP", "Name", "Country", "CCY", "Nominal",
                   "Clean Px", "Accrued %", "Dirty Px",
                   "MTM Value", "Book Value", "Price MTM", "Accrued", "MTM Gain",
                   "Realized", "Note"]
        styled_mtm = _color_pnl_df(
            view[[c for c in ordered if c in view.columns]],
            ["Price MTM", "Accrued", "MTM Gain", "Realized"],
        )
        st.dataframe(styled_mtm, width="stretch", hide_index=True)

        with st.expander("Accrual detail"):
            acc_detail_mtm = accrual_breakdown(positions, bonds_static, as_of)
            if not acc_detail_mtm.empty:
                acc_view_mtm = _enrich(acc_detail_mtm, bs_df, ["name", "country", "currency"])
                st.dataframe(acc_view_mtm, width="stretch", hide_index=True)
            else:
                st.info("No accruing positions.")

        st.markdown("**Portfolio totals**")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("MTM Value", _fmt(_nansum(mtm_df["mtm_value"])))
        t2.metric("Book Value", _fmt(_nansum(mtm_df["book_value"])))
        t3.metric("Price MTM", _fmt(total_price))
        t4.metric("Accrued", _fmt(total_accrued))
        t5.metric("MTM Gain", _fmt(_nansum(mtm_df["mtm_gain"])))
        st.caption("Dirty Px = Clean Px + Accrued %.  MTM Value = Nominal × Dirty Px / 100.  "
                   "MTM Gain = MTM Value + Book Value = Price MTM + Accrued.")

# ── Tab 3: Positions by Date ──────────────────────────────────────────────────

with tab_positions_date:
    st.subheader("Positions by Date — editable")
    st.caption(
        "Edit **Net Nominal** or **WAVG Price** then click **Save as Adjustment Trade**. "
        "A synthetic trade is appended to `trades.csv` so the full history stays reconcilable."
    )
    pos_date = st.date_input("Positions as of", value=as_of, key="pos_date")

    baseline_pos = compute_positions(all_trades, as_of=pos_date, portfolio=hist_portfolio)

    if not baseline_pos:
        st.info("No positions on this date with the current filters.")
    else:
        pos_rows = [{
            "cusip": p.cusip,
            "net_nominal": p.net_nominal,
            "wavg_price": round(p.wavg_price, 6),
            "book_value": round(p.book_value, 2),
            "last_settle": p.last_settle,
        } for p in baseline_pos.values()]
        pos_base_df = pd.DataFrame(pos_rows)
        pos_display = _enrich(pos_base_df, bs_df, ["name", "country", "currency"])
        display_cols = ["cusip", "name", "country", "currency",
                        "net_nominal", "wavg_price", "book_value", "last_settle"]
        pos_display = pos_display[[c for c in display_cols if c in pos_display.columns]]

        edited_pos = st.data_editor(
            pos_display,
            num_rows="fixed",
            width="stretch",
            key="ed_pos_by_date",
            column_config={
                "cusip":       st.column_config.TextColumn("CUSIP", disabled=True),
                "name":        st.column_config.TextColumn("Name", disabled=True),
                "country":     st.column_config.TextColumn("Country", disabled=True),
                "currency":    st.column_config.TextColumn("CCY", disabled=True),
                "net_nominal": st.column_config.NumberColumn("Net Nominal", format="%.0f"),
                "wavg_price":  st.column_config.NumberColumn("WAVG Price", format="%.4f"),
                "book_value":  st.column_config.NumberColumn("Book Value", disabled=True, format="%.2f"),
                "last_settle": st.column_config.DateColumn("Last Settle", disabled=True),
            },
        )

        col_save_pos, col_recalc_pos = st.columns(2)
        with col_save_pos:
            if st.button("Save as Adjustment Trade", key="save_pos_adj"):
                adj_trades = _make_adjustment_trades(
                    baseline_pos, edited_pos, pos_date, hist_portfolio
                )
                if adj_trades.empty:
                    st.info("No changes detected — nothing to save.")
                else:
                    full_raw = _raw_csv(config.TRADES_PATH, data_io.TRADES_COLUMNS)
                    combined = pd.concat([full_raw, adj_trades], ignore_index=True)
                    try:
                        backup = data_io.save_trades(combined)
                        msg = f"Saved {len(adj_trades)} adjustment trade(s)."
                        if backup:
                            msg += f"  Backup: {Path(backup).name}"
                        st.success(msg)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

        with col_recalc_pos:
            if st.button("Recalculate P&L", key="recalc_pos_date"):
                with st.spinner("Recomputing P&L history..."):
                    compute_daily_pnl(hist_start, as_of, portfolio=hist_portfolio)
                st.success("P&L history updated.")
                st.rerun()

# ── Tab 4: Trades by Date ─────────────────────────────────────────────────────

with tab_trades_date:
    st.subheader("Trades by Date — editable")
    st.caption(
        "Add, modify, or delete trades for a single day. "
        "A timestamped backup is always created before writing. "
        "**Nominal must be unsigned** (sign is stored in the Side column)."
    )
    trades_date = st.date_input("Trade date", value=as_of, key="trades_date_sel")

    full_raw_df = _raw_csv(config.TRADES_PATH, data_io.TRADES_COLUMNS)

    if full_raw_df.empty:
        day_trades = pd.DataFrame(columns=data_io.TRADES_COLUMNS)
        day_mask = pd.Series(dtype=bool)
    else:
        raw_trade_dates = pd.to_datetime(full_raw_df["trade_date"], errors="coerce").dt.date
        day_mask = raw_trade_dates == trades_date
        day_trades = full_raw_df[day_mask].copy()

    st.caption(f"{len(day_trades)} trade(s) on {trades_date}.")

    edited_day = st.data_editor(
        day_trades,
        num_rows="dynamic",
        width="stretch",
        key="ed_trades_by_date",
        column_config={
            "trade_date":   st.column_config.TextColumn("Trade Date"),
            "settle_date":  st.column_config.TextColumn("Settle Date"),
            "side":         st.column_config.SelectboxColumn("Side", options=["buy", "sell"]),
            "nominal":      st.column_config.NumberColumn("Nominal (unsigned)", format="%.0f"),
            "price":        st.column_config.NumberColumn("Price", format="%.4f"),
            "principal":    st.column_config.NumberColumn("Principal", format="%.2f"),
            "net":          st.column_config.NumberColumn("Net", format="%.2f"),
            "accrued":      st.column_config.NumberColumn("Accrued", format="%.2f"),
            "yield_closed": st.column_config.NumberColumn("Yield", format="%.4f"),
        },
    )

    # Backfill dates for newly added rows
    fmt = "%m/%d/%y"
    date_str = trades_date.strftime(fmt)
    if not edited_day.empty:
        edited_day["trade_date"] = edited_day["trade_date"].fillna(date_str)
        edited_day["settle_date"] = edited_day["settle_date"].fillna(date_str)

    col_save_td, col_recalc_td = st.columns(2)
    with col_save_td:
        if st.button("Save Trades", key="save_trades_by_date"):
            if not full_raw_df.empty and day_mask.any():
                other_days = full_raw_df[~day_mask].copy()
            else:
                other_days = full_raw_df.copy()
            reconstructed = pd.concat([other_days, edited_day], ignore_index=True)
            try:
                backup = data_io.save_trades(reconstructed)
                msg = "Trades saved."
                if backup:
                    msg += f"  Backup: {Path(backup).name}"
                st.success(msg)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    with col_recalc_td:
        if st.button("Recalculate P&L", key="recalc_trades_date"):
            with st.spinner("Recomputing P&L history..."):
                compute_daily_pnl(hist_start, as_of, portfolio=hist_portfolio)
            st.success("P&L history updated.")
            st.rerun()

# ── Tab 5: MTM Attribution ────────────────────────────────────────────────────

with tab_attribution:
    st.subheader("MTM Attribution & Transparency")
    attr_view = st.selectbox(
        "View",
        ["Bond Detail", "Rollup", "Accrual Detail", "Price History Table"],
        key="attr_view_sel",
    )

    # ── Bond Detail ──────────────────────────────────────────────────────────
    if attr_view == "Bond Detail":
        snap_tab, range_tab = st.tabs(["Current Snapshot", "Date Range"])

        with snap_tab:
            if mtm_df.empty:
                st.info("No MTM data. Ensure prices are loaded.")
            else:
                snap_view = _enrich(
                    mtm_df.copy(), bs_df,
                    ["name", "country", "currency", "coupon_rate",
                     "day_count_convention", "maturity_date"],
                )
                acc_bk = accrual_breakdown(positions, bonds_static, as_of)
                if not acc_bk.empty:
                    snap_view = snap_view.merge(
                        acc_bk[["cusip", "last_coupon_date", "days_accrued"]],
                        on="cusip", how="left",
                    )
                realized_lookup2 = (
                    realized_df.set_index("cusip")["realized_gain"].to_dict()
                    if not realized_df.empty else {}
                )
                snap_view["realized"] = snap_view["cusip"].map(realized_lookup2).fillna(0.0)

                snap_cols = [
                    "name", "cusip", "country", "currency",
                    "net_nominal", "clean_px", "accrued_today_pct", "dirty_px",
                    "mtm_value", "book_value", "mtm_gain",
                    "price_pnl", "accrued_pnl", "realized",
                    "last_coupon_date", "days_accrued",
                    "coupon_rate", "day_count_convention", "maturity_date",
                ]
                snap_view = snap_view[[c for c in snap_cols if c in snap_view.columns]]
                snap_view = snap_view.rename(columns={
                    "name": "Name", "cusip": "CUSIP", "country": "Country", "currency": "CCY",
                    "net_nominal": "Nominal", "clean_px": "Clean Px",
                    "accrued_today_pct": "Accrued %", "dirty_px": "Dirty Px",
                    "mtm_value": "MTM Value", "book_value": "Book Value",
                    "mtm_gain": "MTM Gain", "price_pnl": "Price P&L",
                    "accrued_pnl": "Accrued P&L", "realized": "Realized",
                    "last_coupon_date": "Last Coupon", "days_accrued": "Days Accrued",
                    "coupon_rate": "Coupon Rate", "day_count_convention": "Day Count",
                    "maturity_date": "Maturity",
                })
                styled_snap = _color_pnl_df(
                    snap_view, ["Price P&L", "Accrued P&L", "MTM Gain", "Realized"]
                )
                st.dataframe(styled_snap, width="stretch", hide_index=True)
                st.caption(
                    "Price P&L = clean price change × nominal.  "
                    "Accrued P&L = daily accrual.  "
                    "MTM Gain = Price P&L + Accrued P&L."
                )

        with range_tab:
            pnl_hist_attr = load_pnl_history(
                portfolio=hist_portfolio, start_date=hist_start, end_date=as_of
            )
            if pnl_hist_attr.empty:
                st.info("No history. Click **Recompute P&L History** in the sidebar first.")
            else:
                enriched_hist = pnl_hist_attr.merge(
                    bs_df[["cusip", "name"]].drop_duplicates("cusip"),
                    on="cusip", how="left",
                )
                enriched_hist["label"] = (
                    enriched_hist["name"].fillna("").str.strip()
                    + " (" + enriched_hist["cusip"] + ")"
                )
                pivot = enriched_hist.pivot_table(
                    index="date", columns="label", values="total_pnl", aggfunc="sum"
                ).fillna(0.0)
                pivot.index = pd.to_datetime(pivot.index).date
                pivot_reset = pivot.reset_index().rename(columns={"date": "Date"})
                styled_pivot = _color_pnl_df(pivot_reset, list(pivot.columns))
                st.dataframe(styled_pivot, width="stretch", hide_index=True, height=420)
                st.caption("Daily total P&L per bond. Each column = bond name (CUSIP).")

    # ── Rollup ───────────────────────────────────────────────────────────────
    elif attr_view == "Rollup":
        group_by = st.selectbox(
            "Group by",
            ["Issuer (6-digit CUSIP)", "Country", "Currency", "Portfolio"],
            key="rollup_group",
        )
        pnl_hist_roll = load_pnl_history(
            portfolio=hist_portfolio, start_date=hist_start, end_date=as_of
        )
        if pnl_hist_roll.empty:
            st.info("No history. Click **Recompute P&L History** in the sidebar first.")
        else:
            rollup = pnl_hist_roll.merge(
                bs_df[["cusip", "country", "currency"]].drop_duplicates("cusip"),
                on="cusip", how="left",
            )
            if group_by == "Issuer (6-digit CUSIP)":
                rollup["group_key"] = rollup["cusip"].str[:6]
                label = "Issuer"
            elif group_by == "Country":
                rollup["group_key"] = rollup["country"].fillna("Unknown")
                label = "Country"
            elif group_by == "Currency":
                rollup["group_key"] = rollup["currency"].fillna("Unknown")
                label = "Currency"
            else:
                rollup["group_key"] = rollup["portfolio"]
                label = "Portfolio"

            agg = (
                rollup.groupby("group_key")
                .agg(
                    price_pnl=("price_pnl", "sum"),
                    accrued=("accrued", "sum"),
                    realized_gain=("realized_gain", "sum"),
                    total_pnl=("total_pnl", "sum"),
                    net_nominal=("net_nominal", "sum"),
                )
                .reset_index()
                .rename(columns={
                    "group_key": label,
                    "price_pnl": "Price P&L",
                    "accrued": "Accrued P&L",
                    "realized_gain": "Realized",
                    "total_pnl": "Total P&L",
                    "net_nominal": "Net Nominal",
                })
                .sort_values("Total P&L", ascending=False)
            )
            styled_rollup = _color_pnl_df(
                agg, ["Price P&L", "Accrued P&L", "Realized", "Total P&L"]
            )
            st.dataframe(styled_rollup, width="stretch", hide_index=True)
            st.caption(
                f"Aggregated P&L grouped by {label.lower()} over "
                f"{hist_start} → {as_of}. Values are cumulative sums."
            )

    # ── Accrual Detail ───────────────────────────────────────────────────────
    elif attr_view == "Accrual Detail":
        acc_detail_attr = accrual_breakdown(positions, bonds_static, as_of)
        if acc_detail_attr.empty:
            st.info("No accruing positions.")
        else:
            acc_full = _enrich(
                acc_detail_attr, bs_df,
                ["name", "country", "currency", "coupon_rate", "day_count_convention"],
            )
            acc_display_cols = [
                "name", "cusip", "country", "currency",
                "coupon_rate", "day_count_convention",
                "last_coupon_date", "days_accrued",
                "accrued_per_100", "accrued_total", "note",
            ]
            acc_full = acc_full[[c for c in acc_display_cols if c in acc_full.columns]]
            acc_full = acc_full.rename(columns={
                "name": "Name", "cusip": "CUSIP", "country": "Country", "currency": "CCY",
                "coupon_rate": "Coupon Rate", "day_count_convention": "Day Count",
                "last_coupon_date": "Last Coupon", "days_accrued": "Days Accrued",
                "accrued_per_100": "Accrued/100", "accrued_total": "Accrued Total",
                "note": "Note",
            })
            styled_acc = _color_pnl_df(acc_full, ["Accrued Total"])
            st.dataframe(styled_acc, width="stretch", hide_index=True)
            total_accrued_attr = (
                _nansum(acc_detail_attr["accrued_total"])
                if "accrued_total" in acc_detail_attr.columns else 0.0
            )
            st.metric("Total Portfolio Accrued", _fmt(total_accrued_attr))
            st.caption(
                "Accrued/100 = accrued interest per 100 par (using bond's day-count convention).  "
                "Accrued Total = Accrued/100 × Net Nominal / 100."
            )

    # ── Price History Table ──────────────────────────────────────────────────
    else:
        st.caption(
            f"Position time series {hist_start} → {as_of}"
            + (f"  ·  portfolio: {hist_portfolio}" if hist_portfolio else "  ·  all portfolios")
        )
        with st.spinner("Building price history table..."):
            ts_attr = position_timeseries(hist_start, as_of, portfolio=hist_portfolio)
        if ts_attr.empty:
            st.info("No data. Populate `price_history.csv` for this date range.")
        else:
            ts_enriched = _enrich(ts_attr, bs_df, ["name", "country", "currency"])
            px_cols = [
                "date", "cusip", "name", "country", "currency",
                "net_nominal", "clean_px", "accrued_pct", "dirty_px",
                "mtm_value", "price_pnl", "accrued_pnl",
            ]
            ts_display = ts_enriched[[c for c in px_cols if c in ts_enriched.columns]]
            ts_display = ts_display.rename(columns={
                "date": "Date", "cusip": "CUSIP", "name": "Name",
                "country": "Country", "currency": "CCY",
                "net_nominal": "Nominal", "clean_px": "Clean Px",
                "accrued_pct": "Accrued %", "dirty_px": "Dirty Px",
                "mtm_value": "MTM Value", "price_pnl": "Price P&L",
                "accrued_pnl": "Accrued P&L",
            })
            styled_ts = _color_pnl_df(ts_display, ["Price P&L", "Accrued P&L"])
            st.dataframe(styled_ts, width="stretch", hide_index=True, height=500)
            st.download_button(
                "Download CSV",
                ts_display.to_csv(index=False).encode(),
                file_name=f"price_history_{hist_start}_{as_of}.csv",
                mime="text/csv",
            )

# ── Tab 6: Daily Ledger ───────────────────────────────────────────────────────

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

# ── Tab 7: Time Series ────────────────────────────────────────────────────────

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

# ── Tab 8: Data Editor ────────────────────────────────────────────────────────

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
