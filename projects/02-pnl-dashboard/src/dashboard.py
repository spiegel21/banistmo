"""
P&L Dashboard — Streamlit app.

Run with:  streamlit run src/dashboard.py
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))

from position_manager import get_positions, load_trades
from accruals import load_bonds_static, total_portfolio_accruals
from trading_gains import total_realized_pnl, unrealized_pnl
from mtm import mark_to_market
from bloomberg import get_prices, load_latest_prices

DATA_DIR = Path(__file__).parent.parent / "data"
TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "bloomberg_prices.xlsx"


# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt(value: float | None, prefix: str = "") -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    return f"{prefix}{sign}{value:,.0f}"


def _color(value: float | None) -> str:
    if value is None:
        return "gray"
    return "green" if value >= 0 else "red"


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="P&L Dashboard",
    page_icon="📊",
    layout="wide",
)
st.title("Fixed-Income P&L Dashboard")

# ── load data ─────────────────────────────────────────────────────────────────

as_of = st.sidebar.date_input("As of date", value=date.today())
trades_df = load_trades(DATA_DIR / "trades")
positions = get_positions()
bonds_static = load_bonds_static(DATA_DIR / "bonds_static.csv")
prices = load_latest_prices()

# ── sidebar: refresh Bloomberg prices ────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("Pricing")

if prices:
    latest_snapshot = sorted((DATA_DIR / "prices").glob("prices_*.csv"))
    if latest_snapshot:
        ts = latest_snapshot[-1].stem.replace("prices_", "").replace("_", " ")
        st.sidebar.caption(f"Last updated: {ts}")
    else:
        st.sidebar.caption("Using manual prices")
else:
    st.sidebar.caption("No prices loaded")

if st.sidebar.button("Refresh Bloomberg Prices"):
    isins = list(positions.keys())
    if isins:
        with st.spinner("Fetching Bloomberg prices..."):
            prices = get_prices(isins, TEMPLATE_PATH)
        st.sidebar.success(f"Updated {len(prices)} ISINs")
        st.rerun()
    else:
        st.sidebar.warning("No positions to price")

# ── compute P&L components ───────────────────────────────────────────────────

accruals_df = total_portfolio_accruals(positions, bonds_static, as_of)
realized_df = total_realized_pnl(trades_df)
unrealized_df = unrealized_pnl(positions, prices, DATA_DIR / "bonds_static.csv", as_of)
mtm_df = mark_to_market(positions, prices, DATA_DIR / "bonds_static.csv", as_of)

total_accruals = accruals_df["accrued"].sum() if not accruals_df.empty and "accrued" in accruals_df else 0.0
total_realized = realized_df["realized_gain"].sum() if not realized_df.empty else 0.0
total_unrealized = unrealized_df["unrealized_gain"].sum() if not unrealized_df.empty and "unrealized_gain" in unrealized_df else 0.0
total_pnl = total_realized + total_unrealized + total_accruals

# ── KPI row ───────────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total P&L", _fmt(total_pnl),
              delta=None, delta_color="normal")

with col2:
    st.metric("Unrealized MTM", _fmt(total_unrealized))

with col3:
    st.metric("Realized Gains", _fmt(total_realized))

with col4:
    st.metric("Accrued Interest", _fmt(total_accruals))

st.markdown("---")

# ── positions table ───────────────────────────────────────────────────────────

left, right = st.columns([1, 1])

with left:
    st.subheader("Positions")

    if mtm_df.empty:
        if trades_df.empty:
            st.info("No trades found. Drop CSVs into `data/trades/`.")
        else:
            st.info("Positions loaded but no prices available. Refresh Bloomberg prices.")
    else:
        accruals_lookup = (
            accruals_df.set_index("isin")["accrued"].to_dict()
            if not accruals_df.empty else {}
        )
        realized_lookup = (
            realized_df.set_index("isin")["realized_gain"].to_dict()
            if not realized_df.empty else {}
        )

        display_rows = []
        for _, row in mtm_df.iterrows():
            isin = row["isin"]
            display_rows.append({
                "ISIN": isin,
                "Nominal": f"{row['net_nominal']:,.0f}",
                "Clean Px": f"{row['clean_px']:.3f}" if row["clean_px"] else "—",
                "Dirty Px": f"{row['dirty_px']:.3f}" if row["dirty_px"] else "—",
                "MTM Gain": _fmt(row["mtm_gain"]),
                "Accrued": _fmt(accruals_lookup.get(isin)),
                "Realized": _fmt(realized_lookup.get(isin, 0.0)),
            })

        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

with right:
    st.subheader("P&L Attribution")

    attribution = {
        "Realized": total_realized,
        "Unrealized MTM": total_unrealized,
        "Accrued Interest": total_accruals,
    }
    attr_df = pd.DataFrame(
        {"Component": list(attribution.keys()), "Value": list(attribution.values())}
    )

    if attr_df["Value"].abs().sum() > 0:
        st.bar_chart(attr_df.set_index("Component"), use_container_width=True)
    else:
        st.info("No P&L data to display")

# ── P&L over time ─────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Cumulative Realized P&L over Time")

if not trades_df.empty:
    from trading_gains import realized_pnl as _realized_detail
    detail = _realized_detail(trades_df)
    if not detail.empty:
        detail["sell_date"] = pd.to_datetime(detail["sell_date"])
        timeline = (
            detail.groupby("sell_date")["realized_gain"]
            .sum()
            .sort_index()
            .cumsum()
            .reset_index()
        )
        st.line_chart(timeline.set_index("sell_date"), use_container_width=True)
    else:
        st.info("No realized trades yet.")
else:
    st.info("No trade data loaded.")

# ── footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(f"As of {as_of}  |  {len(positions)} ISINs  |  "
           f"{len(trades_df)} trade confirmations")
