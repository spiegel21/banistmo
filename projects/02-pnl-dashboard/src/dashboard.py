"""
P&L Dashboard — Streamlit app.

Run with:  streamlit run src/dashboard.py

Total P&L = Realized + Price P&L + Accrued P&L. These three are mutually
exclusive (Price P&L and Accrued P&L together make up total unrealized), so they
sum to the headline figure with no double counting.

Tabs:
  Overview          — today's daily KPIs, total cumulative P&L, attribution, P&L by country
  Positions & MTM   — date-range P&L with full per-position mark-to-market + price transparency
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
from accruals import load_bonds_static, upcoming_coupons
from trading_gains import total_realized_pnl, realized_pnl
from mtm import mark_to_market
from bloomberg import (
    load_latest_prices, load_price_history,
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
import reconciliation
from movements import position_movements
import exposure
import analytics
from classification import classify

TEMPLATE_PATH = config.BLOOMBERG_TEMPLATE_PATH
PRICES_DIR = config.PRICES_DIR


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.0f}"


def _nansum(series) -> float:
    return float(series.sum(min_count=1)) if series.notna().any() else 0.0


def _apply_cusip_filter(df: pd.DataFrame, cusips: set[str]) -> pd.DataFrame:
    """Filter a pnl_history DataFrame to rows whose CUSIP is in `cusips`.
    No-op when cusips is empty (no filter active) or df is empty."""
    if cusips and not df.empty and "cusip" in df.columns:
        return df[df["cusip"].isin(cusips)]
    return df


def _prev_bday(d: date) -> date:
    """Return the most recent business day before d."""
    d = d - timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d


def _raw_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a CSV verbatim (cusip as string) or return an empty editable frame."""
    if Path(path).exists() and Path(path).stat().st_size > 0:
        return pd.read_csv(path, dtype={"cusip": str})
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def _bonds_static_df(bonds_static: dict) -> pd.DataFrame:
    """Flatten BondStatic dict to a join-ready DataFrame."""
    if not bonds_static:
        return pd.DataFrame(columns=["cusip", "name", "country", "currency",
                                     "coupon_rate", "day_count_convention",
                                     "maturity_date", "instrument_type"])
    rows = [{
        "cusip": b.cusip,
        "name": b.name,
        "country": b.country,
        "currency": b.currency,
        "coupon_rate": b.coupon_rate,
        "day_count_convention": b.day_count_convention,
        "maturity_date": b.maturity_date,
        "instrument_type": b.instrument_type,
    } for b in bonds_static.values()]
    return pd.DataFrame(rows)


def _classification_df(bonds_static: dict) -> pd.DataFrame:
    """cusip → derived classification dimensions (sovereign/corp, country of risk,
    local/global, sector, seniority, issuer). Centralises classify() so filters,
    rollups, and exposure all agree."""
    cols = ["cusip", "instrument_type", "country_of_risk", "market",
            "sector", "seniority", "issuer"]
    if not bonds_static:
        return pd.DataFrame(columns=cols)
    rows = []
    for cusip, b in bonds_static.items():
        d = classify(b)
        rows.append({"cusip": cusip, **{k: d.get(k) for k in cols[1:]}})
    return pd.DataFrame(rows, columns=cols)


def _enrich(df: pd.DataFrame, bs_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Left-join df with bs_df on cusip; insert enrichment cols right after cusip.

    Skips any column already present in df to avoid _x/_y merge conflicts.
    """
    if df.empty or bs_df.empty:
        return df
    need = [c for c in cols if c in bs_df.columns and c not in df.columns]
    if not need:
        return df
    merged = df.merge(bs_df[["cusip"] + need], on="cusip", how="left")
    others = [c for c in df.columns if c != "cusip"]
    ordered = ["cusip"] + need + others
    return merged[[c for c in ordered if c in merged.columns]]


def _color_pnl_df(df: pd.DataFrame, pnl_cols: list[str]) -> pd.DataFrame:
    """Return df unchanged (color formatting disabled for pandas compatibility)."""
    return df


def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame safe for Streamlit's Arrow serialization.

    Streamlit converts each column to a pyarrow array; an ``object`` column that
    holds numbers plus a stray empty string (e.g. a totals row, or a numeric
    field that came in blank) raises
    ``Could not convert '' with type str: tried to convert to double``.
    Replacing blank/whitespace-only strings with None lets Arrow infer a numeric
    column. Genuine text columns are unaffected (blanks just render empty).
    """
    obj_cols = df.select_dtypes(include="object").columns
    if df.empty or obj_cols.empty:
        return df
    df = df.copy()
    df[obj_cols] = df[obj_cols].replace(r"^\s*$", None, regex=True)
    return df


def _filtered_dataframe(df: pd.DataFrame, key: str, **kwargs) -> None:
    """Render a text filter then display the (filtered) dataframe."""
    filt = st.text_input(
        "Filter", key=f"filt_{key}",
        placeholder="search any column…",
        label_visibility="collapsed",
    )
    if filt.strip():
        mask = df.apply(
            lambda col: col.astype(str).str.contains(filt.strip(), case=False, na=False)
        ).any(axis=1)
        df = df[mask]
    st.dataframe(_arrow_safe(df), **kwargs)


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
clf_df = _classification_df(bonds_static)

# ── sidebar: filters ──────────────────────────────────────────────────────────

st.sidebar.header("Filters")


def _options(col):
    return sorted(all_trades[col].dropna().unique().tolist()) if not all_trades.empty else []


def _clf_options(col):
    return sorted(clf_df[col].dropna().unique().tolist()) if not clf_df.empty else []


sel_portfolios = st.sidebar.multiselect("Portfolio", _options("portfolio"), default=[])
all_countries = sorted({b.country for b in bonds_static.values() if b.country})
sel_countries = st.sidebar.multiselect("Country", all_countries, default=[])
sel_country_risk = st.sidebar.multiselect("Country of Risk", _clf_options("country_of_risk"), default=[])
sel_inst_type = st.sidebar.multiselect("Sovereign / Corp", _clf_options("instrument_type"), default=[])
sel_market = st.sidebar.multiselect("Local / Global", _clf_options("market"), default=[])
sel_sector = st.sidebar.multiselect("Sector", _clf_options("sector"), default=[])
sel_traders = st.sidebar.multiselect("Trader", _options("trader"), default=[])
sel_cusips = st.sidebar.multiselect("CUSIP / Bond", _options("cusip"), default=[])

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
    # classification-based filters (country of risk, sovereign/corp, local/global, sector)
    for _sel, _col in [
        (sel_country_risk, "country_of_risk"),
        (sel_inst_type, "instrument_type"),
        (sel_market, "market"),
        (sel_sector, "sector"),
    ]:
        if _sel and not clf_df.empty:
            _keep = set(clf_df[clf_df[_col].isin(_sel)]["cusip"])
            trades_df = trades_df[trades_df["cusip"].isin(_keep)]

positions = compute_positions(trades_df, as_of=as_of)

# CUSIPs surviving all sidebar filters — used to restrict pnl_history to selected bonds
_filtered_cusips: set[str] = set(trades_df["cusip"].unique()) if not trades_df.empty else set()

# Load price history once here — used by both the sidebar coverage widget and the
# MTM transparency columns computed later in the page.
_ph = load_price_history()

# single-portfolio scope used by per-portfolio views/history
hist_portfolio = sel_portfolios[0] if len(sel_portfolios) == 1 else None

# ── sidebar: bloomberg (prices + history, single workflow) ───────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("Bloomberg")

prices = load_latest_prices()
snapshots = sorted(PRICES_DIR.glob("prices_*.csv"))
if snapshots:
    _ts = snapshots[-1].stem.replace("prices_", "").replace("_", " ")
    st.sidebar.caption(f"Last priced: {_ts}")
else:
    st.sidebar.caption("No price snapshot yet")

_price_gaps = find_price_gaps(all_trades) if not all_trades.empty else []
_gap_cusips = {c for c, _, _ in _price_gaps}
if _price_gaps:
    _n_cusips_gap = len(_gap_cusips)
    st.sidebar.warning(
        f"Missing history: {len(_price_gaps)} range(s) across {_n_cusips_gap} CUSIP(s)."
    )
else:
    _last_px_date = last_priced_date()
    if _last_px_date:
        st.sidebar.caption(f"Price history complete through {_last_px_date}.")
    else:
        st.sidebar.caption("No price history yet.")

with st.sidebar.expander("Price History Coverage"):
    _all_held_cusips = set(positions.keys())
    _ph_counts: dict[str, int] = {}
    _ph_last: dict[str, str] = {}
    if not _ph.empty:
        for _c, _grp in _ph.groupby("cusip"):
            _ph_counts[str(_c)] = len(_grp)
            _ph_last[str(_c)] = str(_grp["date"].max())[:10]

    _cov_rows = []
    for _cusip in sorted(_all_held_cusips):
        _b = bonds_static.get(_cusip)
        _name = (_b.name if _b and _b.name else "") or ""
        _days = _ph_counts.get(_cusip, 0)
        _thru = _ph_last.get(_cusip, "—")
        if _cusip in _gap_cusips:
            _status = "⚠ gaps — in BDH sheet"
        elif _days > 0:
            _status = f"✓ complete ({_thru})"
        else:
            _status = "✗ no history yet"
        _cov_rows.append({"CUSIP": _cusip, "Name": _name, "Days": _days, "Status": _status})

    if _cov_rows:
        st.dataframe(pd.DataFrame(_cov_rows), hide_index=True, use_container_width=True)
    else:
        st.write("No positions found.")
    st.caption(
        "✓ = history already in price_history.csv (excluded from BDH sheet). "
        "⚠ = gaps detected → bond appears in History BDH sheet. "
        "✗ = bought today or BDH not yet run."
    )

cusip_list = list(positions.keys())

if st.sidebar.button("① Prepare & open template", key="btn_prepare_all"):
    if cusip_list:
        prepare_template(cusip_list, TEMPLATE_PATH, bonds_static=bonds_static)
        if _price_gaps:
            prepare_history_template(_price_gaps, TEMPLATE_PATH, bonds_static=bonds_static)
        _n_hist = len({c for c, _, _ in _price_gaps})
        _sheets = "Live MTM + Static" + (f" + {_n_hist} History block(s)" if _n_hist else "")
        if open_in_excel(TEMPLATE_PATH):
            st.sidebar.success(
                f"Opened **{TEMPLATE_PATH.name}** in Excel — wait for Bloomberg to "
                f"populate {_sheets}, then save and close."
            )
        else:
            st.sidebar.success(
                f"Template ready — open **{TEMPLATE_PATH}** in Excel, "
                f"wait for Bloomberg to populate {_sheets}, then save and close."
            )
    else:
        st.sidebar.warning("No positions to price.")

st.sidebar.caption(
    "② Wait until **all** cells show real values "
    "(no '#N/A Requesting data') → **Save** → **Close** Excel"
)

if st.sidebar.button("③ Import all data", key="btn_import_all"):
    try:
        _imported       = read_prices_from_template(TEMPLATE_PATH)
        _fetched_static = read_static_from_template(TEMPLATE_PATH)
        _hist_df        = read_history_from_template(TEMPLATE_PATH)
    except ValueError as _refresh_exc:
        st.sidebar.error(
            f"⏳ **Bloomberg still refreshing** — {_refresh_exc}\n\n"
            "**What to do:** switch to Excel, wait until every cell shows a real value "
            "(prices, dates, names — no '#N/A Requesting data'), "
            "then **File → Save** and **close** the file before clicking ③ again."
        )
        st.stop()
    except Exception as _read_exc:
        st.sidebar.error(f"❌ Error reading template: {_read_exc}")
        st.exception(_read_exc)
        st.stop()

    _msg_parts      = []

    if _imported:
        prices = _imported
        _msg_parts.append(f"{len(_imported)} price(s) imported")
    else:
        _msg_parts.append("no current prices found")

    _static_saved = False
    if not _fetched_static.empty and not _fetched_static["cusip"].isna().all():
        try:
            _n_new, _n_filled, _incomplete = merge_bonds_static(_fetched_static)
            _static_saved = True
            if _n_new or _n_filled:
                _msg_parts.append(f"static: {_n_new} new CUSIP(s), {_n_filled} field(s) filled")
            if _incomplete:
                st.sidebar.warning(
                    f"⚠ {len(_incomplete)} bond(s) returned no maturity date from Bloomberg "
                    f"(likely govt/sovereign — the default \"CUSIP Corp\" ticker doesn't resolve): "
                    f"{', '.join(_incomplete[:8])}{'…' if len(_incomplete) > 8 else ''}.\n\n"
                    "Set each one's **bbg_ticker** (e.g. `<id> Govt`) in "
                    "**Data Editor → Bond Static**, then run ① Prepare and ③ Import again. "
                    "Other bonds were saved normally."
                )
        except Exception as _exc:
            st.sidebar.error(
                f"❌ Bond static save failed: {_exc}\n\n"
                f"Type: `{type(_exc).__name__}`"
            )
            st.exception(_exc)

    if not _hist_df.empty:
        try:
            with st.spinner("Recomputing P&L history..."):
                compute_daily_pnl(min_date, as_of, portfolio=None)
            _msg_parts.append(
                f"{len(_hist_df)} history record(s) through {_hist_df['date'].max()} · P&L recomputed"
            )
        except Exception as _pnl_exc:
            st.sidebar.error(f"❌ P&L recompute failed: {_pnl_exc}")
            st.exception(_pnl_exc)

    if _imported or not _hist_df.empty or _static_saved:
        st.sidebar.success(" · ".join(_msg_parts))
        st.rerun()
    else:
        st.sidebar.warning(
            " · ".join(_msg_parts) or
            "No data found — open the template in Excel, wait for Bloomberg, save, then import."
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

# ── MTM transparency: prev_px, cost_px, px_change, px_vs_cost ────────────────
_prev_day = _prev_bday(as_of)
if not _ph.empty:
    _prev_ph = _ph[_ph["date"].dt.date == _prev_day]
    _prev_prices_map: dict[str, float] = dict(zip(_prev_ph["cusip"], _prev_ph["px_last"]))
else:
    _prev_prices_map = {}

if not mtm_df.empty:
    _wavg_map = {c: p.wavg_price for c, p in positions.items()}
    mtm_df["cost_px"]    = mtm_df["cusip"].map(_wavg_map)
    mtm_df["prev_px"]    = mtm_df["cusip"].map(_prev_prices_map)
    mtm_df["px_change"]  = (mtm_df["clean_px"] - mtm_df["prev_px"]).round(4)
    mtm_df["px_vs_cost"] = (mtm_df["clean_px"] - mtm_df["cost_px"]).round(4)

# Today's daily move — sourced from pnl_history for the Overview KPIs
_today = date.today()
_today_hist = load_pnl_history(start_date=_today, end_date=_today)
_today_hist = _apply_cusip_filter(_today_hist, _filtered_cusips)
if not _today_hist.empty:
    today_price    = float(_today_hist["price_pnl"].sum())
    today_accrued  = float(_today_hist["accrued"].sum())
    today_realized = float(_today_hist["realized_gain"].sum())
    today_total    = float(_today_hist["total_pnl"].sum())
    _today_data_available = True
else:
    today_price = today_accrued = today_realized = today_total = 0.0
    _today_data_available = False

# ── tabs ────────────────────────────────────────────────────────────────────

(tab_overview, tab_debug, tab_risk, tab_mtm, tab_positions_date, tab_trades_date,
 tab_attribution, tab_ledger, tab_ts, tab_edit) = st.tabs([
    "Overview",
    "🔧 Debug",
    "📐 Risk",
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
    if not _today_data_available:
        st.info(
            "Today's P&L not yet computed. Click **Recompute P&L History** in the sidebar "
            "to generate today's daily move data."
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total P&L (Today)",   _fmt(today_total))
    c2.metric("Price P&L (Today)",   _fmt(today_price))
    c3.metric("Accrued P&L (Today)", _fmt(today_accrued))
    c4.metric("Realized (Today)",    _fmt(today_realized))
    st.caption("Daily move as of today. Cumulative P&L is shown in the charts below.")

    # ── Cumulative Total P&L (first chart) ────────────────────────────────────
    st.markdown("---")
    st.subheader("Cumulative Total P&L")
    pnl_hist = load_pnl_history(portfolio=hist_portfolio, start_date=hist_start, end_date=as_of)
    pnl_hist = _apply_cusip_filter(pnl_hist, _filtered_cusips)
    if pnl_hist.empty:
        st.info(
            "No history computed yet. Pick a start date and click **Recompute P&L History**. "
            "Populate `price_history.csv` first (Data Editor → Manual Prices or `backfill_prices.py`)."
        )
    else:
        _daily = (
            pnl_hist.groupby("date")[["price_pnl", "accrued", "realized_gain", "total_pnl"]]
            .sum().reset_index().sort_values("date")
        )
        _daily["date"] = pd.to_datetime(_daily["date"])
        st.line_chart(_daily.set_index("date")[["total_pnl"]].fillna(0).cumsum(), width="stretch")

    # ── P&L Attribution bar chart + position summary ──────────────────────────
    st.markdown("---")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("P&L Attribution (cumulative)")
        attr = pd.DataFrame({
            "Component": ["Realized", "Price P&L", "Accrued P&L"],
            "Value": [total_realized, total_price, total_accrued],
        })
        if attr["Value"].abs().sum() > 0:
            st.bar_chart(attr.set_index("Component"), width="stretch")
        else:
            st.info("No P&L data to display")
    with right:
        st.subheader("Position summary")
        st.metric("Open positions", len([p for p in positions.values() if p.net_nominal != 0]))
        st.metric("Trades (filtered)", len(trades_df))

    # ── Cumulative P&L breakdown by component ────────────────────────────────
    if not pnl_hist.empty:
        st.markdown("---")
        st.subheader("Cumulative P&L by component")
        comp = _daily.set_index("date")[["price_pnl", "accrued", "realized_gain"]].fillna(0).cumsum()
        comp.columns = ["Price P&L", "Accrued P&L", "Realized"]
        st.line_chart(comp, width="stretch")

    # ── P&L by Country ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("P&L by Country")
    if pnl_hist.empty:
        st.info("No history computed yet.")
    elif bs_df.empty or "country" not in bs_df.columns:
        st.info("Bond static data not loaded — country breakdown unavailable.")
    else:
        _bond_type_filter = st.radio(
            "Bond Type",
            ["All", "Corp", "Sovereign", "Agency"],
            horizontal=True,
            key="country_bond_type",
        )
        _bs_country_df = bs_df[["cusip", "country", "instrument_type"]].drop_duplicates("cusip")
        if _bond_type_filter != "All":
            _bs_country_df = _bs_country_df[_bs_country_df["instrument_type"] == _bond_type_filter]

        country_hist = pnl_hist.merge(_bs_country_df, on="cusip", how="inner")
        country_hist["country"] = country_hist["country"].fillna("Unknown")
        country_hist["date"] = pd.to_datetime(country_hist["date"])

        if country_hist.empty:
            st.info(f"No bonds of type '{_bond_type_filter}' in the current filtered portfolio.")
        else:
            # Cumulative by country line chart
            _cum_raw = (
                country_hist.groupby(["date", "country"])["total_pnl"]
                .sum().reset_index().sort_values("date")
            )
            _cum_pivot = (
                _cum_raw.pivot(index="date", columns="country", values="total_pnl")
                .fillna(0).cumsum()
            )
            st.caption("Cumulative P&L by country")
            st.line_chart(_cum_pivot, width="stretch")

            # Daily by country bar chart
            _day_raw = (
                country_hist.groupby(["date", "country"])["total_pnl"]
                .sum().reset_index().sort_values("date")
            )
            _day_pivot = (
                _day_raw.pivot(index="date", columns="country", values="total_pnl")
                .fillna(0)
            )
            st.caption("Daily P&L by country")
            st.bar_chart(_day_pivot, width="stretch")

    # ── Exposure & Concentration ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Exposure & Concentration")
    _exp_base = exposure.exposure_base(positions, bonds_static, prices, as_of)
    if _exp_base.empty:
        st.info("No exposure to show — load prices and bond static data first.")
    else:
        _dim_label = st.selectbox(
            "Break down by", list(exposure.DIMENSIONS.keys()), key="exp_dim",
        )
        _dim_col = exposure.DIMENSIONS[_dim_label]
        if _dim_col == "portfolio":
            # portfolio isn't on the bond; attach the (single) scope if filtered
            _exp_base = _exp_base.assign(portfolio=hist_portfolio or "all")
        _agg = exposure.aggregate_exposure(_exp_base, _dim_col)
        if _agg.empty:
            st.info("No data for this dimension.")
        else:
            _ec1, _ec2 = st.columns([3, 2])
            with _ec1:
                _agg_view = _agg.rename(columns={
                    _dim_col: _dim_label, "net_nominal": "Nominal", "mtm_value": "MTM Value",
                    "pct_nominal": "% Nominal", "pct_mtm": "% MTM", "n_bonds": "# Bonds",
                })
                st.dataframe(
                    _arrow_safe(_agg_view), hide_index=True, width="stretch",
                    column_config={
                        "Nominal":   st.column_config.NumberColumn(format="%,.0f"),
                        "MTM Value": st.column_config.NumberColumn(format="%,.0f"),
                        "% Nominal": st.column_config.NumberColumn(format="%.1f"),
                        "% MTM":     st.column_config.NumberColumn(format="%.1f"),
                    },
                )
            with _ec2:
                _chart = _agg.set_index(_dim_col)[["mtm_value"]]
                st.caption("MTM value by " + _dim_label.lower())
                st.bar_chart(_chart, width="stretch")
                _conc = exposure.concentration(_exp_base, _dim_col, top_n=5)
                st.metric(
                    f"Top-5 {_dim_label} concentration (% MTM)",
                    f"{_conc['top_n_pct']:.1f}%",
                    help=f"Largest: {_conc['largest']} ({_conc['largest_pct']:.1f}%)",
                )

        # ── Maturity ladder ───────────────────────────────────────────────────
        st.markdown("**Maturity ladder**")
        _ladder = exposure.maturity_ladder(_exp_base, as_of)
        if _ladder.empty:
            st.info("No maturity data.")
        else:
            _lc1, _lc2 = st.columns([3, 2])
            with _lc1:
                _ladder_view = _ladder.rename(columns={
                    "bucket": "Tenor", "net_nominal": "Nominal",
                    "mtm_value": "MTM Value", "n_bonds": "# Bonds",
                })
                st.dataframe(
                    _arrow_safe(_ladder_view), hide_index=True, width="stretch",
                    column_config={
                        "Nominal":   st.column_config.NumberColumn(format="%,.0f"),
                        "MTM Value": st.column_config.NumberColumn(format="%,.0f"),
                    },
                )
            with _lc2:
                st.caption("MTM value by remaining tenor")
                st.bar_chart(_ladder.set_index("bucket")[["mtm_value"]], width="stretch")


# ── Tab 2: Debug / Needs Attention ────────────────────────────────────────────

with tab_debug:
    st.subheader("Debug — data quality & items needing attention")
    st.caption(
        "Every trade, bond, and price that is broken or incomplete, in one place. "
        "**Errors** block correct P&L, **warnings** look suspicious (e.g. weird prices), "
        "**needs-input** marks missing manual fields (name, country of risk, rating…)."
    )

    _held_cusips = {c for c, p in positions.items() if p.net_nominal != 0}
    with st.spinner("Running data-quality checks..."):
        findings_df, summary = reconciliation.run_all_checks(
            raw_trades=reconciliation._read_raw_trades(),
            bonds_static=bonds_static,
            held_cusips=_held_cusips,
            current_prices=prices,
            price_history=_ph,
            as_of=as_of,
        )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("🔴 Errors", summary.get("error", 0))
    d2.metric("🟠 Warnings", summary.get("warning", 0))
    d3.metric("🟡 Needs input", summary.get("needs_input", 0))
    d4.metric("Total findings", summary.get("total", 0))

    if findings_df.empty:
        st.success("✅ No data-quality issues found. Trades, bonds, and prices all look clean.")
    else:
        fc1, fc2 = st.columns(2)
        with fc1:
            sev_filter = st.multiselect(
                "Severity", ["error", "warning", "needs_input"],
                default=["error", "warning", "needs_input"], key="dbg_sev",
            )
        with fc2:
            cats = sorted(findings_df["category"].unique().tolist())
            cat_filter = st.multiselect("Category", cats, default=cats, key="dbg_cat")

        view = findings_df.copy()
        if sev_filter:
            view = view[view["severity"].isin(sev_filter)]
        if cat_filter:
            view = view[view["category"].isin(cat_filter)]

        _sev_icon = {"error": "🔴 error", "warning": "🟠 warning", "needs_input": "🟡 needs input"}
        view["severity"] = view["severity"].map(_sev_icon).fillna(view["severity"])
        view = view.rename(columns={
            "severity": "Severity", "category": "Category", "source": "Source",
            "key": "CUSIP / Row", "field": "Field", "issue": "Issue",
            "suggested_fix": "Suggested fix",
        })
        _filtered_dataframe(view, "debug", width="stretch", hide_index=True, height=460)
        st.download_button(
            "Download findings CSV",
            findings_df.to_csv(index=False).encode(),
            file_name=f"debug_findings_{date.today()}.csv",
            mime="text/csv",
        )
        st.caption(
            "Fix items in **Data Editor** (trades / bond static / manual prices) or via the "
            "**Bloomberg** import in the sidebar, then revisit this tab — it re-runs every load."
        )


# ── Tab 3: Risk Analytics ─────────────────────────────────────────────────────

with tab_risk:
    st.subheader("Risk Analytics — yield, duration, DV01, convexity")
    st.caption(
        "In-house bond math: YTM solved from the clean price, then modified/Macaulay "
        "duration, DV01 (per 1bp), and convexity by finite differences. Portfolio "
        "duration/convexity/YTM are market-value weighted; DV01 sums in cash terms. "
        f"Marked as of {as_of}. Spreads (G/Z) need a yield curve and are out of scope."
    )

    risk_df, risk_summary = analytics.portfolio_risk(positions, bonds_static, prices, as_of)

    rs1, rs2, rs3, rs4, rs5 = st.columns(5)
    rs1.metric("MTM Value", _fmt(risk_summary["mtm_value"]))
    rs2.metric("Portfolio DV01", _fmt(risk_summary["dv01_dollar"]),
               help="Cash P&L for a 1bp parallel yield rise (sum across bonds).")
    _md = risk_summary["mod_duration"]
    rs3.metric("Mod. Duration", "—" if _md != _md else f"{_md:.2f}",
               help="MV-weighted modified duration (years).")
    _yt = risk_summary["ytm"]
    rs4.metric("Portfolio YTM", "—" if _yt != _yt else f"{_yt*100:.2f}%",
               help="MV-weighted yield to maturity.")
    _cx = risk_summary["convexity"]
    rs5.metric("Convexity", "—" if _cx != _cx else f"{_cx:.1f}",
               help="MV-weighted convexity.")

    if risk_df.empty:
        st.info("No positions to analyse. Load positions and prices first.")
    else:
        rv = _enrich(risk_df, bs_df, ["country", "currency"])
        rv = rv.rename(columns={
            "cusip": "CUSIP", "name": "Name", "country": "Country", "currency": "CCY",
            "net_nominal": "Nominal", "clean_px": "Clean Px", "dirty_px": "Dirty Px",
            "mtm_value": "MTM Value", "ytm": "YTM", "mac_duration": "Mac Dur",
            "mod_duration": "Mod Dur", "dv01_dollar": "DV01 ($)", "convexity": "Convexity",
            "note": "Note",
        })
        if "YTM" in rv.columns:
            rv["YTM"] = pd.to_numeric(rv["YTM"], errors="coerce") * 100  # show as %
        _filtered_dataframe(
            rv, "risk_tbl", width="stretch", hide_index=True, height=420,
            column_config={
                "Nominal":   st.column_config.NumberColumn(format="%,.0f"),
                "Clean Px":  st.column_config.NumberColumn(format="%.4f"),
                "Dirty Px":  st.column_config.NumberColumn(format="%.4f"),
                "MTM Value": st.column_config.NumberColumn(format="%,.0f"),
                "YTM":       st.column_config.NumberColumn(format="%.3f"),
                "Mac Dur":   st.column_config.NumberColumn(format="%.3f"),
                "Mod Dur":   st.column_config.NumberColumn(format="%.3f"),
                "DV01 ($)":  st.column_config.NumberColumn(format="%,.2f"),
                "Convexity": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption("YTM shown in %. DV01 ($) = cash P&L per 1bp; positive = loses on a yield rise.")
        st.download_button(
            "Download risk table CSV", risk_df.to_csv(index=False).encode(),
            file_name=f"risk_analytics_{as_of}.csv", mime="text/csv",
        )

    # ── Risk by classification dimension ──────────────────────────────────────
    if not risk_df.empty:
        st.markdown("---")
        st.subheader("Rate risk by dimension")
        st.caption("DV01 and market-value-weighted modified duration grouped by classification.")
        _rbg_dims = {
            "Country of Risk": "country_of_risk", "Sector": "sector",
            "Sovereign / Corp": "instrument_type", "Local / Global": "market",
            "Currency": "currency",
        }
        _rbg_label = st.selectbox("Group risk by", list(_rbg_dims.keys()), key="risk_dim")
        _rbg_col = _rbg_dims[_rbg_label]
        if _rbg_col == "currency":
            _gmap = {b.cusip: (b.currency or "Unknown") for b in bonds_static.values()}
        elif not clf_df.empty:
            _gmap = dict(zip(clf_df["cusip"], clf_df[_rbg_col]))
        else:
            _gmap = {}
        _rbg = analytics.risk_by_group(risk_df, _gmap)
        if _rbg.empty:
            st.info("No solved risk to group yet.")
        else:
            _rc1, _rc2 = st.columns([3, 2])
            with _rc1:
                _rbg_view = _rbg.rename(columns={
                    "group": _rbg_label, "mtm_value": "MTM Value",
                    "dv01_dollar": "DV01 ($)", "mod_duration": "Mod Dur", "n_bonds": "# Bonds",
                })
                st.dataframe(
                    _arrow_safe(_rbg_view), hide_index=True, width="stretch",
                    column_config={
                        "MTM Value": st.column_config.NumberColumn(format="%,.0f"),
                        "DV01 ($)":  st.column_config.NumberColumn(format="%,.2f"),
                        "Mod Dur":   st.column_config.NumberColumn(format="%.2f"),
                    },
                )
            with _rc2:
                st.caption("DV01 ($) by " + _rbg_label.lower())
                st.bar_chart(_rbg.set_index("group")[["dv01_dollar"]], width="stretch")

    # ── Historical VaR / Expected Shortfall ───────────────────────────────────
    st.markdown("---")
    st.subheader("Historical VaR / Expected Shortfall")
    st.caption(
        "Empirical tail of realised daily total P&L over the selected history window "
        "(sidebar **History from** → **As of**). Needs computed P&L history."
    )
    _conf = st.select_slider("Confidence", options=[0.90, 0.95, 0.99], value=0.99, key="var_conf")
    _var_hist = load_pnl_history(portfolio=hist_portfolio, start_date=hist_start, end_date=as_of)
    _var_hist = _apply_cusip_filter(_var_hist, _filtered_cusips)
    if _var_hist.empty:
        st.info("No P&L history yet — click **Recompute P&L History** in the sidebar.")
    else:
        _daily_pnl = _var_hist.groupby("date")["total_pnl"].sum()
        _var = analytics.var_historical(_daily_pnl, confidence=_conf)
        v1, v2, v3, v4 = st.columns(4)
        v1.metric(f"VaR ({int(_conf*100)}%)", _fmt(-_var["var"]),
                  help="Loss not expected to be exceeded on a normal day at this confidence.")
        v2.metric("Expected Shortfall", _fmt(-_var["es"]),
                  help="Average loss on days worse than VaR.")
        v3.metric("Worst day", _fmt(_var["worst_day"]))
        v4.metric("Observations", _var["n_obs"])


# ── Tab 4: Positions & MTM ────────────────────────────────────────────────────

with tab_mtm:
    st.subheader("Mark-to-Market — clean & dirty price")

    # Date range filter
    _mc1, _mc2, _mc3 = st.columns([2, 2, 1])
    with _mc1:
        mtm_start = st.date_input("P&L from", value=date.today(), key="mtm_start")
    with _mc2:
        mtm_end = st.date_input("P&L to", value=date.today(), key="mtm_end")
    with _mc3:
        st.write("")  # vertical spacer
        if st.button("Today", key="mtm_today"):
            st.session_state["mtm_start"] = date.today()
            st.session_state["mtm_end"] = date.today()
            st.rerun()

    # Positions / prices as of range end
    positions_mtm = compute_positions(trades_df, as_of=mtm_end)
    mtm_df_range = mark_to_market(positions_mtm, prices, bonds_static, mtm_end)

    # Transparency columns for the range-end snapshot
    _positions_mtm_wavg = {c: p.wavg_price for c, p in positions_mtm.items()}
    if not mtm_df_range.empty:
        mtm_df_range["cost_px"]    = mtm_df_range["cusip"].map(_positions_mtm_wavg)
        mtm_df_range["prev_px"]    = mtm_df_range["cusip"].map(_prev_prices_map)
        mtm_df_range["px_change"]  = (mtm_df_range["clean_px"] - mtm_df_range["prev_px"]).round(4)
        mtm_df_range["px_vs_cost"] = (mtm_df_range["clean_px"] - mtm_df_range["cost_px"]).round(4)

    # P&L summed over range from pnl_history
    pnl_range = load_pnl_history(portfolio=hist_portfolio, start_date=mtm_start, end_date=mtm_end)
    pnl_range = _apply_cusip_filter(pnl_range, _filtered_cusips)
    if not pnl_range.empty:
        pnl_sum = (
            pnl_range.groupby("cusip")
            .agg(
                price_pnl=("price_pnl", "sum"),
                accrued_pnl=("accrued", "sum"),
                realized_gain=("realized_gain", "sum"),
            )
            .reset_index()
        )
    else:
        pnl_sum = pd.DataFrame(columns=["cusip", "price_pnl", "accrued_pnl", "realized_gain"])

    if mtm_df_range.empty:
        if trades_df.empty:
            st.info("No trades match the current filters. Check `data/trades.csv`.")
        else:
            st.info("No prices loaded. Add prices in **Data Editor → Manual Prices** or click **Refresh Bloomberg Prices**.")
    else:
        view = mtm_df_range.copy()
        view = _enrich(view, bs_df, ["name", "country", "currency"])

        # Replace P&L columns with range-summed values from pnl_history
        view = view.drop(columns=["price_pnl", "accrued_pnl", "mtm_gain"], errors="ignore")
        if not pnl_sum.empty:
            view = view.merge(pnl_sum, on="cusip", how="left")
        else:
            view["price_pnl"] = float("nan")
            view["accrued_pnl"] = float("nan")
            view["realized_gain"] = 0.0
        view["unrealized_pnl"] = view["price_pnl"].fillna(0) + view["accrued_pnl"].fillna(0)
        view["realized"] = view.get("realized_gain", pd.Series(0.0, index=view.index)).fillna(0)

        view = view.rename(columns={
            "cusip": "CUSIP", "name": "Name", "country": "Country", "currency": "CCY",
            "net_nominal": "Nominal",
            "cost_px": "Cost Px", "prev_px": "Prev Px", "px_change": "Px Change",
            "clean_px": "Clean Px", "accrued_today_pct": "Accrued %", "dirty_px": "Dirty Px",
            "px_vs_cost": "Vs Cost",
            "mtm_value": "MTM Value", "book_value": "Book Value",
            "price_pnl": "Price P&L", "accrued_pnl": "Accrued P&L",
            "unrealized_pnl": "Unrealized P&L", "realized": "Realized",
            "note": "Note",
        })
        ordered = [
            "CUSIP", "Name", "Country", "CCY", "Nominal",
            "Cost Px", "Prev Px", "Px Change", "Clean Px", "Accrued %", "Dirty Px",
            "Vs Cost", "MTM Value", "Book Value",
            "Price P&L", "Accrued P&L", "Unrealized P&L", "Realized", "Note",
        ]
        styled_mtm = _color_pnl_df(
            view[[c for c in ordered if c in view.columns]],
            ["Price P&L", "Accrued P&L", "Unrealized P&L", "Realized"],
        )
        _filtered_dataframe(
            styled_mtm, "mtm", width="stretch", hide_index=True,
            column_config={
                "Nominal":        st.column_config.NumberColumn(format="%,.0f"),
                "Cost Px":        st.column_config.NumberColumn(format="%.4f"),
                "Prev Px":        st.column_config.NumberColumn(format="%.4f"),
                "Px Change":      st.column_config.NumberColumn(format="%.4f"),
                "Clean Px":       st.column_config.NumberColumn(format="%.4f"),
                "Accrued %":      st.column_config.NumberColumn(format="%.6f"),
                "Dirty Px":       st.column_config.NumberColumn(format="%.4f"),
                "Vs Cost":        st.column_config.NumberColumn(format="%.4f"),
                "MTM Value":      st.column_config.NumberColumn(format="%,.0f"),
                "Book Value":     st.column_config.NumberColumn(format="%,.0f"),
                "Price P&L":      st.column_config.NumberColumn(format="%,.0f"),
                "Accrued P&L":    st.column_config.NumberColumn(format="%,.0f"),
                "Unrealized P&L": st.column_config.NumberColumn(format="%,.0f"),
                "Realized":       st.column_config.NumberColumn(format="%,.0f"),
            },
        )

        with st.expander("Accrual detail"):
            acc_detail_mtm = accrual_breakdown(positions_mtm, bonds_static, mtm_end)
            if not acc_detail_mtm.empty:
                acc_view_mtm = _enrich(acc_detail_mtm, bs_df, ["name", "country", "currency"])
                _filtered_dataframe(acc_view_mtm, "mtm_acc", width="stretch", hide_index=True)
            else:
                st.info("No accruing positions.")

        range_price = float(pnl_sum["price_pnl"].sum()) if not pnl_sum.empty else 0.0
        range_accrued = float(pnl_sum["accrued_pnl"].sum()) if not pnl_sum.empty else 0.0
        range_realized = float(pnl_sum["realized_gain"].sum()) if not pnl_sum.empty else 0.0
        range_unrealized = range_price + range_accrued

        st.markdown("**Portfolio totals**")
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        t1.metric("MTM Value",      _fmt(_nansum(mtm_df_range["mtm_value"])))
        t2.metric("Book Value",     _fmt(_nansum(mtm_df_range["book_value"])))
        t3.metric("Price P&L",      _fmt(range_price))
        t4.metric("Accrued P&L",    _fmt(range_accrued))
        t5.metric("Unrealized P&L", _fmt(range_unrealized))
        t6.metric("Realized",       _fmt(range_realized))
        st.caption(
            f"Cost Px = WAVG purchase price.  Prev Px = {_prev_day} close.  "
            f"Px Change = today vs prev.  Vs Cost = today vs cost basis.  "
            f"Dirty Px = Clean Px + Accrued %.  MTM Value = Nominal × Dirty Px / 100.  "
            f"P&L columns = sum from {mtm_start} to {mtm_end}. Positions as of {mtm_end}."
        )

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
                "net_nominal": st.column_config.NumberColumn("Net Nominal", format="%,.0f"),
                "wavg_price":  st.column_config.NumberColumn("WAVG Price", format="%.2f"),
                "book_value":  st.column_config.NumberColumn("Book Value", disabled=True, format="%,.2f"),
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
    _date_range = st.date_input(
        "Trade date range",
        value=(min_date, as_of),
        key="trades_date_range",
    )
    if isinstance(_date_range, (list, tuple)) and len(_date_range) == 2:
        trades_start, trades_end = _date_range
    else:
        trades_start = trades_end = _date_range[0] if _date_range else as_of

    full_raw_df = _raw_csv(config.TRADES_PATH, data_io.TRADES_COLUMNS)

    if full_raw_df.empty:
        day_trades = pd.DataFrame(columns=data_io.TRADES_COLUMNS)
        day_mask = pd.Series(dtype=bool)
    else:
        _td = pd.to_datetime(full_raw_df["trade_date"], format="%m/%d/%y", errors="coerce")
        _unresolved = _td.isna() & full_raw_df["trade_date"].notna()
        if _unresolved.any():
            _td[_unresolved] = pd.to_datetime(full_raw_df.loc[_unresolved, "trade_date"], errors="coerce")
        raw_trade_dates = _td.dt.date
        day_mask = (raw_trade_dates >= trades_start) & (raw_trade_dates <= trades_end)
        day_trades = full_raw_df[day_mask].copy()

    st.caption(f"{len(day_trades)} trade(s) from {trades_start} to {trades_end}.")

    # Show the bond name (display-only) right after cusip. Dropped on save —
    # save_trades() filters to TRADES_COLUMNS, so "name" never reaches disk.
    if not day_trades.empty and "cusip" in day_trades.columns:
        day_trades = _enrich(day_trades, bs_df, ["name"])

    edited_day = st.data_editor(
        day_trades,
        num_rows="dynamic",
        width="stretch",
        key="ed_trades_by_date",
        column_config={
            "name":         st.column_config.TextColumn("Name", disabled=True),
            "trade_date":   st.column_config.TextColumn("Trade Date"),
            "settle_date":  st.column_config.TextColumn("Settle Date"),
            "side":         st.column_config.SelectboxColumn("Side", options=["buy", "sell"]),
            "nominal":      st.column_config.NumberColumn("Nominal (unsigned)", format="%,.0f"),
            "price":        st.column_config.NumberColumn("Price", format="%.2f"),
            "principal":    st.column_config.NumberColumn("Principal", format="%,.2f"),
            "net":          st.column_config.NumberColumn("Net", format="%,.2f"),
            "accrued":      st.column_config.NumberColumn("Accrued", format="%,.2f"),
            "yield_closed": st.column_config.NumberColumn("Yield", format="%.4f"),
        },
    )

    fmt = "%m/%d/%y"
    date_str = trades_end.strftime(fmt)
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
        ["Bond Detail", "Bond Movements", "Rollup", "Accrual Detail",
         "Coupon Calendar", "Price History Table"],
        key="attr_view_sel",
    )

    # ── Bond Detail ──────────────────────────────────────────────────────────
    if attr_view == "Bond Detail":
        snap_tab, range_tab = st.tabs(["Current Snapshot", "Date Range"])

        with snap_tab:
            today_bond_hist = load_pnl_history(start_date=_today, end_date=_today)
            today_bond_hist = _apply_cusip_filter(today_bond_hist, _filtered_cusips)

            if mtm_df.empty and today_bond_hist.empty:
                st.info("No data. Ensure prices are loaded and P&L history has been computed for today.")
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

                # Merge today's daily P&L per bond
                if not today_bond_hist.empty:
                    today_pnl_by_cusip = (
                        today_bond_hist.groupby("cusip")
                        .agg(
                            today_price_pnl=("price_pnl", "sum"),
                            today_accrued=("accrued", "sum"),
                            today_realized=("realized_gain", "sum"),
                            today_total=("total_pnl", "sum"),
                        )
                        .reset_index()
                    )
                    snap_view = snap_view.merge(today_pnl_by_cusip, on="cusip", how="left")
                else:
                    for col in ["today_price_pnl", "today_accrued", "today_realized", "today_total"]:
                        snap_view[col] = 0.0

                snap_view["today_unrealized"] = (
                    snap_view["today_price_pnl"].fillna(0)
                    + snap_view["today_accrued"].fillna(0)
                )

                snap_cols = [
                    "name", "cusip", "country", "currency",
                    "net_nominal", "cost_px", "prev_px", "px_change",
                    "clean_px", "dirty_px", "px_vs_cost",
                    "today_price_pnl", "today_accrued", "today_unrealized",
                    "today_realized", "today_total",
                    "last_coupon_date", "days_accrued",
                    "coupon_rate", "day_count_convention", "maturity_date",
                ]
                snap_view = snap_view[[c for c in snap_cols if c in snap_view.columns]]
                snap_view = snap_view.rename(columns={
                    "name": "Name", "cusip": "CUSIP", "country": "Country", "currency": "CCY",
                    "net_nominal": "Nominal",
                    "cost_px": "Cost Px", "prev_px": "Prev Px", "px_change": "Px Change",
                    "clean_px": "Clean Px", "dirty_px": "Dirty Px", "px_vs_cost": "Vs Cost",
                    "today_price_pnl": "Price P&L", "today_accrued": "Accrued P&L",
                    "today_unrealized": "Unrealized P&L",
                    "today_realized": "Realized", "today_total": "Total P&L",
                    "last_coupon_date": "Last Coupon", "days_accrued": "Days Accrued",
                    "coupon_rate": "Coupon Rate", "day_count_convention": "Day Count",
                    "maturity_date": "Maturity",
                })

                # Totals row
                _pnl_total_cols = ["Nominal", "Price P&L", "Accrued P&L", "Unrealized P&L",
                                   "Realized", "Total P&L"]
                # Use None (not "") for non-total columns: an empty string in an
                # otherwise-numeric column breaks Streamlit's Arrow serialization
                # ("Could not convert '' with type str: tried to convert to double").
                _totals = {c: None for c in snap_view.columns}
                _totals["Name"] = "TOTAL"
                for _col in _pnl_total_cols:
                    if _col in snap_view.columns:
                        _vals = pd.to_numeric(snap_view[_col], errors="coerce")
                        _totals[_col] = _vals.sum()
                snap_with_totals = pd.concat(
                    [snap_view, pd.DataFrame([_totals])], ignore_index=True
                )

                styled_snap = _color_pnl_df(
                    snap_with_totals,
                    ["Price P&L", "Accrued P&L", "Unrealized P&L", "Realized", "Total P&L"],
                )
                _filtered_dataframe(
                    styled_snap, "attr_snap", width="stretch", hide_index=True,
                    column_config={
                        "Nominal":        st.column_config.NumberColumn(format="%,.0f"),
                        "Cost Px":        st.column_config.NumberColumn(format="%.4f"),
                        "Prev Px":        st.column_config.NumberColumn(format="%.4f"),
                        "Px Change":      st.column_config.NumberColumn(format="%.4f"),
                        "Clean Px":       st.column_config.NumberColumn(format="%.4f"),
                        "Dirty Px":       st.column_config.NumberColumn(format="%.4f"),
                        "Vs Cost":        st.column_config.NumberColumn(format="%.4f"),
                        "Price P&L":      st.column_config.NumberColumn(format="%,.0f"),
                        "Accrued P&L":    st.column_config.NumberColumn(format="%,.0f"),
                        "Unrealized P&L": st.column_config.NumberColumn(format="%,.0f"),
                        "Realized":       st.column_config.NumberColumn(format="%,.0f"),
                        "Total P&L":      st.column_config.NumberColumn(format="%,.0f"),
                        "Coupon Rate":    st.column_config.NumberColumn(format="%.4f"),
                    },
                )
                if not today_bond_hist.empty:
                    st.caption(
                        "Cost Px = WAVG purchase price.  Prev Px = prior business day close.  "
                        "Px Change = today vs prev.  Vs Cost = today vs cost basis.  "
                        "Today's P&L contribution per bond. Unrealized P&L = Price P&L + Accrued P&L."
                    )
                else:
                    st.caption(
                        "No P&L history for today — run **Recompute P&L History** to populate. "
                        "Position/price data shown from as-of date."
                    )

        with range_tab:
            pnl_hist_attr = load_pnl_history(
                portfolio=hist_portfolio, start_date=hist_start, end_date=as_of
            )
            pnl_hist_attr = _apply_cusip_filter(pnl_hist_attr, _filtered_cusips)
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
                _filtered_dataframe(styled_pivot, "attr_pivot", width="stretch", hide_index=True, height=420)
                st.caption("Daily total P&L per bond. Each column = bond name (CUSIP).")

    # ── Bond Movements ───────────────────────────────────────────────────────
    elif attr_view == "Bond Movements":
        st.caption(
            "Full audit trail per bond: every trade and its effect on the running "
            "position, WAVG cost basis, cash, and realized gains. The realized column "
            "reconciles exactly with the rest of the app."
        )
        _mv_cusips = sorted(trades_df["cusip"].dropna().unique().tolist()) if not trades_df.empty else []
        if not _mv_cusips:
            st.info("No trades match the current filters.")
        else:
            _label_map = {
                c: (f"{bonds_static[c].name} ({c})" if c in bonds_static and bonds_static[c].name else c)
                for c in _mv_cusips
            }
            _sel_mv = st.selectbox(
                "Bond", _mv_cusips, format_func=lambda c: _label_map.get(c, c),
                key="mv_cusip_sel",
            )
            mv = position_movements(trades_df, cusip=_sel_mv, portfolio=hist_portfolio)
            if mv.empty:
                st.info("No movements for this bond with the current filters.")
            else:
                mv_view = mv.rename(columns={
                    "trade_date": "Trade Date", "portfolio": "Portfolio", "cusip": "CUSIP",
                    "trader": "Trader", "side": "Side", "nominal": "Nominal", "price": "Price",
                    "cash_flow": "Cash Flow", "running_nominal": "Running Nominal",
                    "running_wavg_cost": "Running WAVG Cost", "realized_gain": "Realized Gain",
                    "cumulative_cash": "Cumulative Cash", "cumulative_realized": "Cumulative Realized",
                })
                if "Trade Date" in mv_view.columns:
                    mv_view["Trade Date"] = pd.to_datetime(mv_view["Trade Date"]).dt.date
                _filtered_dataframe(
                    mv_view, "bond_mv", width="stretch", hide_index=True, height=420,
                    column_config={
                        "Nominal":             st.column_config.NumberColumn(format="%,.0f"),
                        "Price":               st.column_config.NumberColumn(format="%.4f"),
                        "Cash Flow":           st.column_config.NumberColumn(format="%,.2f"),
                        "Running Nominal":     st.column_config.NumberColumn(format="%,.0f"),
                        "Running WAVG Cost":   st.column_config.NumberColumn(format="%.4f"),
                        "Realized Gain":       st.column_config.NumberColumn(format="%,.2f"),
                        "Cumulative Cash":     st.column_config.NumberColumn(format="%,.2f"),
                        "Cumulative Realized": st.column_config.NumberColumn(format="%,.2f"),
                    },
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Current Nominal", _fmt(mv["running_nominal"].iloc[-1]))
                m2.metric("Realized (lifetime)", _fmt(mv["cumulative_realized"].iloc[-1]))
                m3.metric("Net Cash", _fmt(mv["cumulative_cash"].iloc[-1]))
                st.download_button(
                    "Download movements CSV",
                    mv.to_csv(index=False).encode(),
                    file_name=f"movements_{_sel_mv}.csv", mime="text/csv",
                )

    # ── Rollup ───────────────────────────────────────────────────────────────
    elif attr_view == "Rollup":
        group_by = st.selectbox(
            "Group by",
            ["Issuer (6-digit CUSIP)", "Country", "Country of Risk", "Currency",
             "Sovereign / Corp", "Local / Global", "Sector", "Portfolio"],
            key="rollup_group",
        )
        pnl_hist_roll = load_pnl_history(
            portfolio=hist_portfolio, start_date=hist_start, end_date=as_of
        )
        pnl_hist_roll = _apply_cusip_filter(pnl_hist_roll, _filtered_cusips)
        if pnl_hist_roll.empty:
            st.info("No history. Click **Recompute P&L History** in the sidebar first.")
        else:
            rollup = pnl_hist_roll.merge(
                bs_df[["cusip", "country", "currency"]].drop_duplicates("cusip"),
                on="cusip", how="left",
            )
            if not clf_df.empty:
                rollup = rollup.merge(
                    clf_df[["cusip", "country_of_risk", "instrument_type", "market", "sector"]]
                    .drop_duplicates("cusip"),
                    on="cusip", how="left",
                )
            if group_by == "Issuer (6-digit CUSIP)":
                rollup["group_key"] = rollup["cusip"].str[:6]
                label = "Issuer"
            elif group_by == "Country":
                rollup["group_key"] = rollup["country"].fillna("Unknown")
                label = "Country"
            elif group_by == "Country of Risk":
                rollup["group_key"] = rollup.get("country_of_risk", pd.Series(dtype=str)).fillna("Unknown")
                label = "Country of Risk"
            elif group_by == "Currency":
                rollup["group_key"] = rollup["currency"].fillna("Unknown")
                label = "Currency"
            elif group_by == "Sovereign / Corp":
                rollup["group_key"] = rollup.get("instrument_type", pd.Series(dtype=str)).fillna("Unknown")
                label = "Sovereign / Corp"
            elif group_by == "Local / Global":
                rollup["group_key"] = rollup.get("market", pd.Series(dtype=str)).fillna("Unknown")
                label = "Local / Global"
            elif group_by == "Sector":
                rollup["group_key"] = rollup.get("sector", pd.Series(dtype=str)).fillna("Unknown")
                label = "Sector"
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
                    "net_nominal": "Nominal",
                })
                .sort_values("Total P&L", ascending=False)
            )
            styled_rollup = _color_pnl_df(
                agg, ["Price P&L", "Accrued P&L", "Realized", "Total P&L"]
            )
            _filtered_dataframe(
                styled_rollup, "attr_rollup", width="stretch", hide_index=True,
                column_config={
                    "Nominal":     st.column_config.NumberColumn(format="%,.0f"),
                    "Price P&L":   st.column_config.NumberColumn(format="%,.0f"),
                    "Accrued P&L": st.column_config.NumberColumn(format="%,.0f"),
                    "Realized":    st.column_config.NumberColumn(format="%,.0f"),
                    "Total P&L":   st.column_config.NumberColumn(format="%,.0f"),
                },
            )
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

            _safe_days = acc_full["days_accrued"].replace(0, 1)
            acc_full["accrued_bps_day"] = (
                acc_full["accrued_per_100"] / _safe_days * 100
            ).round(4)
            acc_full["accrued_cash_day"] = (
                acc_full["accrued_total"] / _safe_days
            ).round(2)

            acc_display_cols = [
                "name", "cusip", "country", "currency",
                "coupon_rate", "day_count_convention",
                "last_coupon_date", "days_accrued",
                "accrued_bps_day", "accrued_cash_day", "accrued_total", "note",
            ]
            acc_full = acc_full[[c for c in acc_display_cols if c in acc_full.columns]]
            acc_full = acc_full.rename(columns={
                "name": "Name", "cusip": "CUSIP", "country": "Country", "currency": "CCY",
                "coupon_rate": "Coupon Rate", "day_count_convention": "Day Count",
                "last_coupon_date": "Last Coupon", "days_accrued": "Days Accrued",
                "accrued_bps_day": "Accrued bps/day",
                "accrued_cash_day": "Accrued cash/day",
                "accrued_total": "Accrued Total",
                "note": "Note",
            })
            styled_acc = _color_pnl_df(acc_full, ["Accrued Total"])
            _filtered_dataframe(
                styled_acc, "attr_acc", width="stretch", hide_index=True,
                column_config={
                    "Coupon Rate":      st.column_config.NumberColumn(format="%.4f"),
                    "Accrued bps/day":  st.column_config.NumberColumn(format="%.4f"),
                    "Accrued cash/day": st.column_config.NumberColumn(format="%,.2f"),
                    "Accrued Total":    st.column_config.NumberColumn(format="%,.2f"),
                },
            )
            total_accrued_attr = (
                _nansum(acc_detail_attr["accrued_total"])
                if "accrued_total" in acc_detail_attr.columns else 0.0
            )
            st.metric("Total Portfolio Accrued", _fmt(total_accrued_attr))
            st.caption(
                "Accrued bps/day = daily carry rate in basis points per 100 par.  "
                "Accrued cash/day = daily carry in currency units at current position.  "
                "Accrued Total = cumulative accrued interest since last coupon."
            )

    # ── Coupon Calendar ──────────────────────────────────────────────────────
    elif attr_view == "Coupon Calendar":
        st.caption(
            "Forecast of coupon cash flows for currently-held positions. "
            "Short positions show negative (paid-away) coupons."
        )
        _cc1, _cc2 = st.columns(2)
        with _cc1:
            cal_start = st.date_input("From", value=as_of, key="cal_start")
        with _cc2:
            cal_end = st.date_input("To", value=as_of + timedelta(days=365), key="cal_end")

        cal = upcoming_coupons(positions, bonds_static, cal_start, cal_end)
        if cal.empty:
            st.info("No coupons scheduled in this window for held positions.")
        else:
            cal = _enrich(cal, bs_df, ["name", "country", "currency"])
            cal_view = cal.rename(columns={
                "coupon_date": "Coupon Date", "cusip": "CUSIP", "name": "Name",
                "country": "Country", "currency": "CCY", "net_nominal": "Nominal",
                "coupon_rate": "Coupon Rate", "coupon_amount": "Coupon Amount",
            })
            _filtered_dataframe(
                cal_view, "coupon_cal", width="stretch", hide_index=True, height=420,
                column_config={
                    "Nominal":       st.column_config.NumberColumn(format="%,.0f"),
                    "Coupon Rate":   st.column_config.NumberColumn(format="%.4f"),
                    "Coupon Amount": st.column_config.NumberColumn(format="%,.2f"),
                },
            )
            cc_m1, cc_m2 = st.columns(2)
            cc_m1.metric("Total coupons (window)", _fmt(cal["coupon_amount"].sum()))
            cc_m2.metric("Coupon events", len(cal))
            # Monthly bar chart of coupon cash
            _cal_m = cal.copy()
            _cal_m["month"] = pd.to_datetime(_cal_m["coupon_date"]).dt.to_period("M").astype(str)
            _by_month = _cal_m.groupby("month")["coupon_amount"].sum()
            st.caption("Coupon cash by month")
            st.bar_chart(_by_month, width="stretch")
            st.download_button(
                "Download coupon calendar CSV",
                cal.to_csv(index=False).encode(),
                file_name=f"coupon_calendar_{cal_start}_{cal_end}.csv", mime="text/csv",
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
            # Apply CUSIP filter to the timeseries too
            if _filtered_cusips and not ts_enriched.empty and "cusip" in ts_enriched.columns:
                ts_enriched = ts_enriched[ts_enriched["cusip"].isin(_filtered_cusips)]
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
            _filtered_dataframe(
                styled_ts, "attr_ts", width="stretch", hide_index=True, height=500,
                column_config={
                    "Nominal":     st.column_config.NumberColumn(format="%,.0f"),
                    "Clean Px":    st.column_config.NumberColumn(format="%.4f"),
                    "Accrued %":   st.column_config.NumberColumn(format="%.6f"),
                    "Dirty Px":    st.column_config.NumberColumn(format="%.4f"),
                    "MTM Value":   st.column_config.NumberColumn(format="%,.0f"),
                    "Price P&L":   st.column_config.NumberColumn(format="%,.0f"),
                    "Accrued P&L": st.column_config.NumberColumn(format="%,.0f"),
                },
            )
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
        _filtered_dataframe(snap, "ledger_snap", width="stretch", hide_index=True)

    st.markdown("##### Accrual detail")
    pos_day = get_positions_as_of(ledger_day, hist_portfolio)
    acc = accrual_breakdown(pos_day, bonds_static, ledger_day)
    if acc.empty:
        st.info("No accruing positions on this day.")
    else:
        acc = _enrich(acc, bs_df, ["name"])
        _filtered_dataframe(acc, "ledger_acc", width="stretch", hide_index=True)

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
            cols = ["cusip", "name", "side", "nominal", "price", "net", "accrued", "trader", "portfolio"]
            booked = _enrich(booked, bs_df, ["name"])
            _filtered_dataframe(
                booked[[c for c in cols if c in booked.columns]],
                "ledger_trades", width="stretch", hide_index=True,
            )

    with cR:
        st.markdown("##### Realized closes this day")
        rl = realized_pnl(trades_df) if not trades_df.empty else pd.DataFrame()
        if not rl.empty:
            rl = rl[rl["close_date"].dt.date == ledger_day]
        if rl is None or rl.empty:
            st.info("No positions closed.")
        else:
            rl = _enrich(rl, bs_df, ["name"])
            _filtered_dataframe(rl, "ledger_rl", width="stretch", hide_index=True)

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
        _filtered_dataframe(ts, "ts_main", width="stretch", hide_index=True, height=500)
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
        edited = st.data_editor(raw_df, num_rows="dynamic", width="stretch", key=key)
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
