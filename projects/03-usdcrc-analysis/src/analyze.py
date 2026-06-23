"""Quantitative analysis of the BCCR/MONEX USD/CRC market.

Sections:
  1. Feature engineering (returns, gaps, ranges, trade size, VWAP skew)
  2. Reverse-engineering the price formation / "pricing model"
  3. Volume seasonality (day-of-week, month, quincena)
  4. Volume profile (volume at each price level)
  5. Volume vs intraday movement (range & |return| regressions)
  6. Trading signals / backtests

Outputs: prints a structured report to stdout and writes PNG charts to out/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "monex_clean.csv"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
pd.set_option("display.width", 120)


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def load() -> pd.DataFrame:
    df = pd.read_csv(CLEAN, parse_dates=["date"])
    df = df[df["is_trading_day"]].copy().sort_values("date").reset_index(drop=True)

    # --- feature engineering -------------------------------------------------
    df["dow"] = df["date"].dt.dayofweek
    df["dow_name"] = df["dow"].map(dict(enumerate(DOW)))
    df["month"] = df["date"].dt.month
    df["dom"] = df["date"].dt.day
    df["volume_musd"] = df["volume_usd"] / 1e6
    df["avg_ticket_usd"] = df["volume_usd"] / df["n_trades"]

    # price-formation features
    df["ret_vwap"] = df["vwap"] - df["prev_vwap"]          # session-over-session move (VWAP basis)
    df["ret_vwap_bps"] = df["ret_vwap"] / df["prev_vwap"] * 1e4
    df["open_gap"] = df["open"] - df["prev_vwap"]          # how far open jumps from last VWAP
    df["open_gap_bps"] = df["open_gap"] / df["prev_vwap"] * 1e4
    df["intraday"] = df["close"] - df["open"]              # open->close drift
    df["intraday_bps"] = df["intraday"] / df["open"] * 1e4
    df["range"] = df["high"] - df["low"]                   # absolute intraday range
    df["range_bps"] = df["range"] / df["vwap"] * 1e4
    df["vwap_skew"] = df["vwap"] - df["avg_simple"]        # >0: big trades done above simple avg
    # where did VWAP sit within the day's range? 0=at low, 1=at high
    df["vwap_pos"] = (df["vwap"] - df["low"]) / df["range"].replace(0, np.nan)
    df["close_pos"] = (df["close"] - df["low"]) / df["range"].replace(0, np.nan)
    return df


def section_overview(df):
    hdr("1. DATA OVERVIEW")
    print(f"Trading days        : {len(df)}")
    print(f"Date range          : {df.date.min().date()} -> {df.date.max().date()}")
    print(f"VWAP start / end    : {df.vwap.iloc[0]:.2f} -> {df.vwap.iloc[-1]:.2f} "
          f"({(df.vwap.iloc[-1]/df.vwap.iloc[0]-1)*100:+.2f}% over period)")
    print(f"VWAP min / max      : {df.vwap.min():.2f} / {df.vwap.max():.2f}")
    print(f"Daily volume (US$M) : mean {df.volume_musd.mean():.1f}  median {df.volume_musd.median():.1f}"
          f"  min {df.volume_musd.min():.1f}  max {df.volume_musd.max():.1f}")
    print(f"Trades / day        : mean {df.n_trades.mean():.0f}  median {df.n_trades.median():.0f}"
          f"  max {df.n_trades.max():.0f}")
    print(f"Avg ticket (US$)    : mean {df.avg_ticket_usd.mean():,.0f}  median {df.avg_ticket_usd.median():,.0f}")
    print(f"Daily |VWAP move|   : mean {df.ret_vwap_bps.abs().mean():.1f} bps  "
          f"std {df.ret_vwap_bps.std():.1f} bps  (annualized vol ~{df.ret_vwap_bps.std()/1e4*np.sqrt(252)*100:.1f}%)")


def section_price_model(df):
    hdr("2. REVERSE-ENGINEERING THE PRICE FORMATION MODEL")

    # (a) Is the open anchored to the previous session VWAP?
    g = df.dropna(subset=["open_gap_bps"])
    print("\n(a) OPEN vs PREVIOUS-SESSION VWAP")
    print(f"    open - prev_vwap: mean {g.open_gap_bps.mean():+.2f} bps  "
          f"median {g.open_gap_bps.median():+.2f}  std {g.open_gap_bps.std():.2f}")
    print(f"    % of days open within +/-10 bps of prev VWAP: "
          f"{(g.open_gap_bps.abs() <= 10).mean()*100:.1f}%")
    print("    => the market opens essentially AT the prior session's weighted avg "
          "(anchoring / crawling reference).")

    # (b) Mean reversion vs momentum of session VWAP moves
    r = df["ret_vwap_bps"].dropna()
    ac1 = r.autocorr(1)
    ac2 = r.autocorr(2)
    ac5 = r.autocorr(5)
    print("\n(b) SESSION-OVER-SESSION VWAP MOVE — autocorrelation")
    print(f"    lag1 {ac1:+.3f}   lag2 {ac2:+.3f}   lag5 {ac5:+.3f}")
    print("    (negative lag1 => mean reversion;  positive => momentum/trend)")

    # (c) Does the OPEN GAP predict the rest-of-day reversal? (intraday mean reversion)
    sub = df.dropna(subset=["open_gap_bps", "intraday_bps"])
    sl, ic, rr, pv, se = stats.linregress(sub.open_gap_bps, sub.intraday_bps)
    print("\n(c) DOES THE OPENING GAP GET FADED INTRADAY?")
    print(f"    intraday_bps = {ic:+.2f} + {sl:+.3f} * open_gap_bps   (R^2={rr**2:.3f}, p={pv:.1e})")
    print(f"    slope {sl:+.3f}: a negative slope means an up-gap tends to drift back down.")
    print("    CAVEAT: open_gap=open-prevVWAP and intraday=close-open SHARE the noisy `open`")
    print("    print with opposite signs, which mechanically biases this slope toward -1.")
    # Clean version free of that shared-noise artifact: regress (close-prevVWAP) on
    # (open-prevVWAP). `open` noise now sits only in the regressor -> errors-in-variables
    # ATTENUATES the slope toward 0, so a slope clearly < 1 is a genuine partial fade.
    sub2 = df.dropna(subset=["open_gap_bps", "ret_vwap_bps"]).copy()
    sub2["close_gap_bps"] = (sub2["close"] - sub2["prev_vwap"]) / sub2["prev_vwap"] * 1e4
    cl_sl, cl_ic, cl_rr, cl_pv, _ = stats.linregress(sub2.open_gap_bps, sub2.close_gap_bps)
    print(f"    CLEAN: close_gap_bps = {cl_ic:+.2f} + {cl_sl:.3f} * open_gap_bps "
          f"(R^2={cl_rr**2:.3f}) -> {1-cl_sl:.0%} of the opening gap is retraced by the close.")

    # (d) Reference-rate reconstruction: is the published VWAP ~ volume-weighted mid?
    #     We can at least confirm VWAP sits inside [low, high] and its position.
    print("\n(d) WHERE DOES VWAP SIT IN THE DAILY RANGE?")
    print(f"    vwap position in [low,high]: mean {df.vwap_pos.mean():.2f} (0=low,1=high), "
          f"std {df.vwap_pos.std():.2f}")
    print(f"    close position in [low,high]: mean {df.close_pos.mean():.2f}")
    print(f"    VWAP-vs-simple-avg skew: mean {df.vwap_skew.mean():+.3f} colones "
          f"({(df.vwap_skew>0).mean()*100:.0f}% of days VWAP>simple avg => larger tickets lifted offers)")

    # (e) Trend / drift regime
    df["t"] = np.arange(len(df))
    sl, ic, rr, pv, se = stats.linregress(df["t"], df["vwap"])
    print("\n(e) SECULAR DRIFT (crawl)")
    print(f"    VWAP ~ {ic:.2f} {sl:+.4f}*day   => {sl*21:+.3f} colones/month, "
          f"{sl*252:+.2f} colones/yr  (R^2={rr**2:.3f})")

    # chart: VWAP path + open-gap anchoring
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[2, 1])
    ax[0].plot(df.date, df.vwap, lw=1, color="navy", label="Session VWAP")
    ax[0].fill_between(df.date, df.low, df.high, color="navy", alpha=0.12, label="Daily range")
    ax[0].set_title("USD/CRC MONEX — session VWAP and daily range")
    ax[0].legend(loc="upper left")
    ax[0].set_ylabel("colones / US$")
    ax[1].hist(g.open_gap_bps.clip(-40, 40), bins=60, color="teal")
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_title("Open gap vs previous-session VWAP (bps) — tight anchoring")
    ax[1].set_xlabel("bps")
    fig.tight_layout()
    fig.savefig(OUT / "01_price_formation.png", dpi=110)
    plt.close(fig)


def section_seasonality(df):
    hdr("3. VOLUME SEASONALITY")

    print("\n(a) DAY-OF-WEEK")
    dow = df.groupby("dow_name").agg(
        days=("volume_musd", "size"),
        vol_musd=("volume_musd", "mean"),
        n_trades=("n_trades", "mean"),
        ticket=("avg_ticket_usd", "mean"),
        ret_bps=("ret_vwap_bps", "mean"),
        absret_bps=("ret_vwap_bps", lambda s: s.abs().mean()),
    ).reindex([d for d in DOW if d in df.dow_name.values])
    print(dow.round(1).to_string())

    print("\n(b) MONTH-OF-YEAR")
    mon = df.groupby("month").agg(
        days=("volume_musd", "size"),
        vol_musd=("volume_musd", "mean"),
        n_trades=("n_trades", "mean"),
        ret_bps=("ret_vwap_bps", "mean"),
        absret_bps=("ret_vwap_bps", lambda s: s.abs().mean()),
    )
    mon.index = [pd.Timestamp(2025, m, 1).strftime("%b") for m in mon.index]
    print(mon.round(1).to_string())

    print("\n(c) INTRA-MONTH (quincena effect: payroll / tax dates)")
    df["fortnight"] = np.where(df.dom <= 15, "1st half (1-15)", "2nd half (16-31)")
    fn = df.groupby("fortnight").agg(
        days=("volume_musd", "size"),
        vol_musd=("volume_musd", "mean"),
        ret_bps=("ret_vwap_bps", "mean"),
    )
    print(fn.round(2).to_string())
    # month-end window
    df["near_monthend"] = df.dom >= 25
    me = df.groupby("near_monthend").agg(vol_musd=("volume_musd", "mean"),
                                         ret_bps=("ret_vwap_bps", "mean"))
    print(f"\n    Month-end (day>=25) vol: {me.loc[True,'vol_musd']:.1f}M vs rest "
          f"{me.loc[False,'vol_musd']:.1f}M  "
          f"({me.loc[True,'vol_musd']/me.loc[False,'vol_musd']-1:+.1%})")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    dow.vol_musd.plot.bar(ax=ax[0], color="steelblue")
    ax[0].set_title("Avg daily volume by day-of-week (US$M)")
    ax[0].set_ylabel("US$M")
    mon.vol_musd.plot.bar(ax=ax[1], color="indianred")
    ax[1].set_title("Avg daily volume by month (US$M)")
    fig.tight_layout()
    fig.savefig(OUT / "02_volume_seasonality.png", dpi=110)
    plt.close(fig)
    return dow, mon


def section_volume_profile(df):
    hdr("4. VOLUME PROFILE — volume at each price level")
    # Bucket by 1-colon price level using session VWAP as the price proxy.
    df["px_bin"] = df["vwap"].round(0)
    prof = df.groupby("px_bin").agg(
        days=("volume_musd", "size"),
        vol_musd=("volume_musd", "sum"),
    ).sort_index()
    prof["vol_pct"] = prof.vol_musd / prof.vol_musd.sum() * 100
    poc = prof.vol_musd.idxmax()
    # value area: smallest set of bins holding 70% of volume
    s = prof.sort_values("vol_musd", ascending=False)
    cum = s.vol_musd.cumsum() / s.vol_musd.sum()
    va = s[cum <= 0.70]
    print(f"Point of Control (highest-volume level): {poc:.0f} colones "
          f"({prof.loc[poc,'vol_pct']:.1f}% of all volume, {int(prof.loc[poc,'days'])} days)")
    print(f"70% Value Area: {va.index.min():.0f} - {va.index.max():.0f} colones")
    print("\nTop 10 highest-volume price levels:")
    print(prof.sort_values("vol_musd", ascending=False).head(10).round(1).to_string())

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(prof.index, prof.vol_musd, height=0.8, color="darkgreen", alpha=0.7)
    ax.axhline(poc, color="red", lw=1.2, label=f"POC {poc:.0f}")
    ax.set_title("USD/CRC volume profile (US$M traded at each ~1-colon level)")
    ax.set_xlabel("US$M total")
    ax.set_ylabel("colones / US$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "03_volume_profile.png", dpi=110)
    plt.close(fig)
    return prof, poc


def section_vol_vs_move(df):
    hdr("5. VOLUME vs INTRADAY MOVEMENT")
    sub = df.dropna(subset=["range_bps", "volume_musd", "ret_vwap_bps"])

    # range vs volume
    sl, ic, rr, pv, se = stats.linregress(sub.volume_musd, sub.range_bps)
    print("\n(a) DAILY RANGE vs VOLUME")
    print(f"    range_bps = {ic:.1f} {sl:+.4f} * volume_musd   (R^2={rr**2:.3f}, p={pv:.1e})")
    print(f"    Pearson r = {np.corrcoef(sub.volume_musd, sub.range_bps)[0,1]:+.3f} "
          f"(volume up => range {'wider' if sl>0 else 'tighter'})")

    # |return| vs volume
    sl2, ic2, rr2, pv2, se2 = stats.linregress(sub.volume_musd, sub.ret_vwap_bps.abs())
    print("\n(b) |SESSION MOVE| vs VOLUME")
    print(f"    |ret_bps| = {ic2:.1f} {sl2:+.4f} * volume_musd   (R^2={rr2**2:.3f}, p={pv2:.1e})")

    # signed move vs volume (does volume have a directional sign? e.g. bank/CB selling caps price)
    sl3, ic3, rr3, pv3, se3 = stats.linregress(sub.volume_musd, sub.ret_vwap_bps)
    print("\n(c) SIGNED MOVE vs VOLUME")
    print(f"    ret_bps = {ic3:+.2f} {sl3:+.5f} * volume_musd   (R^2={rr3**2:.3f}, p={pv3:.1e})")
    print("    (a significant NEGATIVE slope is a fingerprint of supply/CB USD selling on heavy days)")

    # high-volume vs low-volume terciles
    print("\n(d) BEHAVIOUR BY VOLUME TERCILE")
    sub = sub.copy()
    sub["vol_t"] = pd.qcut(sub.volume_musd, 3, labels=["low", "mid", "high"])
    ter = sub.groupby("vol_t", observed=True).agg(
        vol_musd=("volume_musd", "mean"),
        range_bps=("range_bps", "mean"),
        absret_bps=("ret_vwap_bps", lambda s: s.abs().mean()),
        mean_ret_bps=("ret_vwap_bps", "mean"),
    )
    print(ter.round(2).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(sub.volume_musd, sub.range_bps, s=10, alpha=0.4, color="purple")
    xx = np.linspace(sub.volume_musd.min(), sub.volume_musd.max(), 50)
    ax[0].plot(xx, ic + sl * xx, color="k")
    ax[0].set_title(f"Daily range vs volume (r={np.corrcoef(sub.volume_musd, sub.range_bps)[0,1]:.2f})")
    ax[0].set_xlabel("volume US$M"); ax[0].set_ylabel("range bps")
    ax[1].scatter(sub.volume_musd, sub.ret_vwap_bps, s=10, alpha=0.4, color="darkorange")
    ax[1].plot(xx, ic3 + sl3 * xx, color="k")
    ax[1].axhline(0, color="gray", lw=0.6)
    ax[1].set_title("Signed session move vs volume")
    ax[1].set_xlabel("volume US$M"); ax[1].set_ylabel("ret bps")
    fig.tight_layout()
    fig.savefig(OUT / "04_volume_vs_move.png", dpi=110)
    plt.close(fig)


def _sharpe(daily_bps):
    daily_bps = pd.Series(daily_bps).dropna()
    if daily_bps.std() == 0 or len(daily_bps) < 5:
        return np.nan
    return daily_bps.mean() / daily_bps.std() * np.sqrt(252)


def section_signals(df):
    hdr("6. TRADING OPPORTUNITIES (signal backtests)")
    df = df.copy()

    # --- Signal 1: Open-gap fade (intraday mean reversion) -------------------
    # If open gaps ABOVE prev VWAP, short to close; if BELOW, long to close.
    s = df.dropna(subset=["open_gap_bps", "intraday_bps"]).copy()
    thr = 5  # bps
    s1 = s[s.open_gap_bps.abs() >= thr].copy()
    # pnl in bps = -sign(gap) * intraday move
    s1["pnl_bps"] = -np.sign(s1.open_gap_bps) * s1.intraday_bps
    print("\nSIGNAL 1 — Fade the opening gap to the close (intraday mean reversion)")
    print(f"    Trades: {len(s1)} (|gap|>={thr}bps)   Hit rate: {(s1.pnl_bps>0).mean()*100:.1f}%")
    print(f"    Avg P&L: {s1.pnl_bps.mean():+.2f} bps/trade   "
          f"Total: {s1.pnl_bps.sum():+.0f} bps   Sharpe~{_sharpe(s1.pnl_bps):.2f}")

    # --- Signal 2: Session-VWAP momentum (overnight) -------------------------
    # lag-1 autocorr is strongly POSITIVE (+0.37) => the market TRENDS. So follow,
    # don't fade: take the sign of today's move into the next session.
    df["ret_next"] = df.ret_vwap_bps.shift(-1)
    z = (df.ret_vwap_bps - df.ret_vwap_bps.mean()) / df.ret_vwap_bps.std()
    s2 = df[z.abs() >= 1.0].dropna(subset=["ret_next"]).copy()
    s2["pnl_bps"] = np.sign(s2.ret_vwap_bps) * s2.ret_next   # momentum (follow the move)
    print("\nSIGNAL 2 — FOLLOW a >1sigma session move into the NEXT session (overnight momentum)")
    print(f"    (lag-1 autocorr is +{df.ret_vwap_bps.autocorr(1):.2f}, i.e. the market trends)")
    print(f"    Trades: {len(s2)}   Hit rate: {(s2.pnl_bps>0).mean()*100:.1f}%")
    print(f"    Avg P&L: {s2.pnl_bps.mean():+.2f} bps/trade   Sharpe~{_sharpe(s2.pnl_bps):.2f}")

    # --- Signal 2b: simple VWAP trend filter (5d vs 20d) ---------------------
    df["ma5"] = df.vwap.rolling(5).mean()
    df["ma20"] = df.vwap.rolling(20).mean()
    df["trend_pos"] = np.sign(df.ma5 - df.ma20)          # +1 above (depreciating), -1 below
    df["fwd_ret"] = -df.ret_vwap_bps.shift(-1)            # P&L of being SHORT USD = -USD move
    tf = df.dropna(subset=["trend_pos", "fwd_ret"])
    pnl_tf = -tf.trend_pos * tf.fwd_ret                   # follow trend: long USD when ma5>ma20
    print("\nSIGNAL 2b — 5d/20d VWAP trend-follow (trade the persistent appreciation crawl)")
    print(f"    Hit rate: {(pnl_tf>0).mean()*100:.1f}%   Avg {pnl_tf.mean():+.2f} bps/day   "
          f"Sharpe~{_sharpe(pnl_tf):.2f}")

    # --- Signal 3: Day-of-week directional bias -----------------------------
    print("\nSIGNAL 3 — Day-of-week directional bias (session VWAP move)")
    dwret = df.groupby("dow_name").ret_vwap_bps.agg(["mean", "std", "count"])
    dwret["t_stat"] = dwret["mean"] / (dwret["std"] / np.sqrt(dwret["count"]))
    dwret = dwret.reindex([d for d in DOW if d in df.dow_name.values])
    print(dwret.round(2).to_string())

    # --- Signal 4: Volume as an order-flow / continuation signal -------------
    # Contemporaneously, heavy volume => colon appreciation (rate falls). Test
    # whether that pressure PERSISTS: does a heavy-volume day predict the next day?
    df["vol_z"] = (df.volume_musd - df.volume_musd.rolling(20).mean()) / \
                  df.volume_musd.rolling(20).std()
    heavy = df[df.vol_z >= 1.0].dropna(subset=["ret_next"])
    light = df[df.vol_z <= -1.0].dropna(subset=["ret_next"])
    print("\nSIGNAL 4 — Volume as order-flow: next-session move after volume spikes")
    print(f"    After HEAVY day (vol_z>=+1, n={len(heavy)}): next-session move "
          f"{heavy.ret_next.mean():+.2f} bps (same-day was {heavy.ret_vwap_bps.mean():+.2f})")
    print(f"    After LIGHT day (vol_z<=-1, n={len(light)}): next-session move "
          f"{light.ret_next.mean():+.2f} bps")
    print("    => heavy USD-supply days keep pushing the rate down the next session "
          "(appreciation pressure persists, consistent with the +0.37 autocorrelation).")

    # --- equity curves for the two clean, tradeable signals -----------------
    # Order-flow strategy: position = -sign(vol_z) held into next session
    of = df.dropna(subset=["vol_z", "ret_next"]).copy()
    # Position into the next session: SHORT USD after a heavy day (vol_z>0), LONG USD
    # after a light day. P&L of a short-USD position = -ret_next (rate falling => profit).
    # => pnl = -sign(vol_z) * ret_next.
    of["pnl_bps"] = -np.sign(of.vol_z) * of.ret_next

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(of.date, of.pnl_bps.cumsum(), label="S4 order-flow (next-session)", color="green")
    mom = s2.sort_values("date")
    ax.plot(mom.date, mom.pnl_bps.cumsum(), label="S2 overnight momentum (>1sigma days)",
            color="navy")
    ax.set_title("Cumulative P&L (bps) — clean tradeable signals")
    ax.set_ylabel("cumulative bps"); ax.axhline(0, color="gray", lw=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "05_signal_equity.png", dpi=110)
    plt.close(fig)
    print(f"\n    [S4 full backtest: {len(of)} days, "
          f"{of.pnl_bps.sum():+.0f} bps total, {of.pnl_bps.mean():+.2f}/day, "
          f"hit {(of.pnl_bps>0).mean()*100:.0f}%, Sharpe~{_sharpe(of.pnl_bps):.2f}]")
    return s1, s2


def main():
    df = load()
    section_overview(df)
    section_price_model(df)
    section_seasonality(df)
    section_volume_profile(df)
    section_vol_vs_move(df)
    section_signals(df)
    hdr("CHARTS WRITTEN")
    for p in sorted(OUT.glob("*.png")):
        print("   ", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
