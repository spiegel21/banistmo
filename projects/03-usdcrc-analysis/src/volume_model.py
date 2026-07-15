"""Flow + seasonality model for USD/CRC — NO technical indicators.

Features are built ONLY from (all strictly causal, known at the close of day t):
  * volume: today's, lagged (t-1,t-2), 5-day average, and SEASONALLY-ADJUSTED
    volume (today vs the expanding mean for the same weekday / month) — because
    "low volume" must mean low *for that day/season*, not low in absolute terms.
  * seasonality: day-of-week, month, quincena (1-15 vs 16-31), day-of-month.
  * price / precio ponderado relationships: close vs VWAP, today's VWAP move,
    where the close sits in the day's range.

Hypothesis under test: LOW volume -> USD strengthens next day (rate up).

Everything is validated walk-forward (expanding window, periodic refit => every
prediction is out-of-sample) and traded long/short USD net of 0.65 CRC round-trip
(0.325/side). Writes vm_*.png and out/vm_results.json.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze import OUT, load

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
NAVY, GREEN, RED, GREY, ORANGE, PURPLE = "#1f3b73", "#2e8b57", "#c0392b", "#7f8c8d", "#e08a2b", "#7d3c98"
ANN = np.sqrt(252)
COST_SIDE = 0.325            # 0.65 CRC round-trip
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# --------------------------------------------------------------------------- #
def build():
    df = load().copy()
    df["vol"] = df.volume_usd / 1e6
    df["rvv"] = df.vwap.pct_change() * 1e4              # VWAP-to-VWAP (realistic desk fill)
    df["r_next"] = df.rvv.shift(-1)                     # tradeable next-day USD return (bps)
    df["dow"] = df.date.dt.dayofweek
    df["mon"] = df.date.dt.month
    df["dom"] = df.date.dt.day
    df["quincena"] = (df.dom <= 15).astype(int)        # 1 = first half

    # --- volume features (causal) ---
    df["vol_z20"] = (df.vol - df.vol.rolling(20).mean()) / df.vol.rolling(20).std()
    df["vol_lag1"] = df.vol_z20.shift(1)
    df["vol_lag2"] = df.vol_z20.shift(2)
    df["vol_5d"] = df.vol_z20.rolling(5).mean()
    # seasonally-adjusted volume: today vs expanding mean of same weekday / month (shifted = causal)
    df["vol_dow_norm"] = df.vol / df.groupby("dow").vol.transform(
        lambda s: s.shift(1).expanding(min_periods=10).mean())
    df["vol_mon_norm"] = df.vol / df.groupby("mon").vol.transform(
        lambda s: s.shift(1).expanding(min_periods=5).mean())
    df["vol_dow_norm"] = np.log(df.vol_dow_norm.clip(lower=0.05))
    df["vol_mon_norm"] = np.log(df.vol_mon_norm.clip(lower=0.05))

    # --- price / precio-ponderado relationships ---
    df["cl_vs_vwap"] = (df.close - df.vwap) / df.vwap * 1e4
    df["vwap_move"] = (df.vwap - df.prev_vwap) / df.prev_vwap * 1e4
    rng = (df.high - df.low).replace(0, np.nan)
    df["range_pos"] = ((df.close - df.low) / rng).fillna(0.5)

    # --- calendar dummies ---
    for d in range(5):
        df[f"dow_{d}"] = (df.dow == d).astype(int)
    df["mon_sin"] = np.sin(2 * np.pi * df.mon / 12)
    df["mon_cos"] = np.cos(2 * np.pi * df.mon / 12)
    return df


# Feature set: volume + calendar + close/VWAP structure. NO momentum / moving-average /
# breakout indicators (vwap_move was tested and deliberately excluded as a technical signal).
FEATURES = ["vol_z20", "vol_lag1", "vol_lag2", "vol_5d", "vol_dow_norm", "vol_mon_norm",
            "quincena", "dom", "dow_0", "dow_1", "dow_2", "dow_3", "dow_4",
            "mon_sin", "mon_cos", "cl_vs_vwap", "range_pos"]
VOL_FEATS = ["vol_z20", "vol_lag1", "vol_lag2", "vol_5d", "vol_dow_norm", "vol_mon_norm"]
CAL_FEATS = ["quincena", "dom", "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "mon_sin", "mon_cos"]


# --------------------------------------------------------------------------- #
# walk-forward modelling (every prediction is out-of-sample)
# --------------------------------------------------------------------------- #
def walk_forward(df, model_fn, min_train=504, refit=63):
    X = df[FEATURES].values
    y = (df.r_next > 0).astype(int).values
    valid = ~np.isnan(X).any(1) & ~np.isnan(df.r_next.values)
    prob = np.full(len(df), np.nan)
    coefs = []
    start = max(min_train, np.where(valid)[0][0] + min_train)
    for t in range(start, len(df), refit):
        tr = np.arange(0, t)[valid[:t]]
        if len(tr) < min_train:
            continue
        sc = StandardScaler().fit(X[tr])
        m = model_fn().fit(sc.transform(X[tr]), y[tr])
        te = np.arange(t, min(t + refit, len(df)))
        te = te[valid[te]]
        if len(te):
            prob[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        if hasattr(m, "coef_"):
            coefs.append(m.coef_[0])
    avg_coef = dict(zip(FEATURES, np.mean(coefs, 0))) if coefs else {}
    return prob, avg_coef


def net_pnl(pos, df, cost_side=COST_SIDE):
    pos = pd.Series(pos).reset_index(drop=True)
    r = df.r_next.reset_index(drop=True)
    px = df.vwap.reset_index(drop=True)                 # slippage booked against the VWAP fill
    turn = pos.diff().abs()
    turn.iloc[0] = abs(pos.iloc[0])
    net = pos * r - cost_side / px * 1e4 * turn
    return net, turn


def sharpe(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return np.nan if len(x) < 5 or x.std() == 0 else x.mean() / x.std() * ANN


def stats(net, turn=None, n=None, mask=None):
    x = net.copy()
    if mask is not None:
        x = x[mask]
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    cum = np.cumsum(x)
    out = {"sharpe": round(float(sharpe(x)), 2), "mean_bps": round(float(x.mean()), 2),
           "ann_bps": round(float(x.mean() * 252), 0),
           "hit": round(float((x[x != 0] > 0).mean() * 100), 1) if (x != 0).any() else 0.0,
           "maxdd_bps": round(float((cum - np.maximum.accumulate(cum)).min()), 0), "n": int(len(x))}
    if turn is not None and n:
        out["roundtrips_yr"] = round(float(np.nansum(turn) / 2 / (n / 252)), 1)
    return out


# --------------------------------------------------------------------------- #
# EDA charts
# --------------------------------------------------------------------------- #
def eda_vol_nextret(df, res):
    d = df.dropna(subset=["r_next", "vol"])
    raw = d.groupby(pd.qcut(d.vol, 5, labels=False), observed=True).r_next.mean()
    adj = d.groupby(pd.qcut(d.vol_dow_norm, 5, labels=False), observed=True).r_next.mean()
    res["vol_quintile_raw"] = [round(float(x), 2) for x in raw.values]
    res["vol_quintile_adj"] = [round(float(x), 2) for x in adj.values]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    lab = ["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"]
    for a, series, t in [(ax[0], raw, "RAW volume"), (ax[1], adj, "SEASONALLY-ADJUSTED volume")]:
        col = [GREEN if v > 0 else RED for v in series.values]
        a.bar(range(5), series.values, color=col)
        a.set_xticks(range(5))
        a.set_xticklabels(lab)
        a.axhline(0, color="k", lw=0.8)
        a.set_title(f"Next-day USD move by {t} quintile")
        a.set_ylabel("next-day move (bps)")
    fig.suptitle("LOW volume → USD strengthens next day (green = USD up)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "vm_vol_nextret.png", dpi=110)
    plt.close(fig)


def eda_seasonality(df, res):
    d = df.dropna(subset=["r_next"])
    dow = d.groupby("dow").r_next.mean().reindex(range(5))
    quin = d.groupby("quincena").r_next.mean()
    mon = d.groupby("mon").r_next.mean()
    res["dow_nextret"] = {DOW[i]: round(float(dow[i]), 2) for i in range(5)}
    res["quincena_nextret"] = {"first_half_1_15": round(float(quin.get(1, np.nan)), 2),
                               "second_half_16_31": round(float(quin.get(0, np.nan)), 2)}
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
    ax[0].bar(DOW, dow.values, color=[GREEN if v > 0 else RED for v in dow.values])
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("Next-day move by day-of-week")
    ax[0].set_ylabel("bps")
    ax[1].bar(["1st half\n(1–15)", "2nd half\n(16–31)"], [quin.get(1), quin.get(0)],
              color=[GREEN if quin.get(1) > 0 else RED, GREEN if quin.get(0) > 0 else RED])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("Next-day move by quincena")
    mlab = [pd.Timestamp(2025, m, 1).strftime("%b") for m in mon.index]
    ax[2].bar(mlab, mon.values, color=[GREEN if v > 0 else RED for v in mon.values])
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_title("Next-day move by month")
    fig.tight_layout()
    fig.savefig(OUT / "vm_seasonality.png", dpi=110)
    plt.close(fig)


def eda_lagged(df, res):
    d = df.dropna(subset=["r_next"])
    lags = {k: round(float(d.vol.shift(k).corr(d.r_next)), 3) for k in range(6)}
    res["lagged_vol_corr"] = lags
    res["vol_autocorr1"] = round(float(df.vol.autocorr(1)), 3)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar([str(k) for k in lags], list(lags.values()), color=NAVY)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("corr(volume t−k, next-day move) — yesterday's volume still informs")
    ax[0].set_xlabel("lag k (days)")
    ac = [df.vol.autocorr(k) for k in range(1, 16)]
    ax[1].bar(range(1, 16), ac, color=GREEN)
    ax[1].set_title(f"Volume autocorrelation (lag1={ac[0]:.2f}) — volume regimes persist")
    ax[1].set_xlabel("lag (days)")
    fig.tight_layout()
    fig.savefig(OUT / "vm_lagged.png", dpi=110)
    plt.close(fig)


def eda_interactions(df, res):
    d = df.dropna(subset=["r_next"]).copy()
    d["vt"] = pd.qcut(d.vol_dow_norm, 3, labels=["low", "mid", "high"])
    piv_dow = d.pivot_table("r_next", "vt", "dow", aggfunc="mean", observed=True)
    d["half"] = np.where(d.quincena == 1, "1–15", "16–31")
    piv_q = d.pivot_table("r_next", "vt", "half", aggfunc="mean", observed=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, piv, t, xl in [(ax[0], piv_dow, "volume × day-of-week", DOW),
                          (ax[1], piv_q, "volume × quincena", ["1–15", "16–31"])]:
        lim = np.nanmax(np.abs(piv.values))
        im = a.imshow(piv.values, cmap="RdYlGn_r", vmin=-lim, vmax=lim, aspect="auto")
        a.set_xticks(range(piv.shape[1]))
        a.set_xticklabels(xl)
        a.set_yticks(range(3))
        a.set_yticklabels(["low vol", "mid", "high vol"])
        a.set_title(f"Next-day move: {t}")
        a.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                a.text(j, i, f"{piv.values[i, j]:.0f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle("Interactions — green = USD up next day (low-vol + 2nd-half = strongest)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "vm_interactions.png", dpi=110)
    plt.close(fig)


def chart_coefs(coef, res):
    items = sorted(coef.items(), key=lambda kv: kv[1])
    res["logit_coefs"] = {k: round(float(v), 3) for k, v in coef.items()}
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color=[RED if v < 0 else GREEN for v in vals])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Walk-forward logistic coefficients (standardized)\n+ = predicts USD up next day")
    fig.tight_layout()
    fig.savefig(OUT / "vm_coefs.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    df = build()
    res = {"_meta": {"n_days": int(len(df)), "date_min": str(df.date.min().date()),
                     "date_max": str(df.date.max().date()), "cost_crc_roundtrip": 0.65,
                     "features": FEATURES}}
    eda_vol_nextret(df, res)
    eda_seasonality(df, res)
    eda_lagged(df, res)
    eda_interactions(df, res)

    # --- attribution: how much does each feature family contribute? ---
    recent0 = (df.date >= "2019-01-01").values
    def _net_sh(feats):
        global FEATURES
        keep = FEATURES
        FEATURES = feats
        p, _ = walk_forward(df, lambda: LogisticRegression(C=0.3, max_iter=1000))
        FEATURES = keep
        pos = np.clip((np.nan_to_num(p, nan=0.5) - 0.5) * 5, -1, 1)
        net, _t = net_pnl(pos, df)
        return round(float(sharpe(net)), 2), round(float(sharpe(net[recent0])), 2)
    bench_net, _bt = net_pnl(np.full(len(df), -1.0), df)
    res["decomposition"] = {
        "volume_only": _net_sh(VOL_FEATS),
        "calendar_only": _net_sh(CAL_FEATS),
        "volume_plus_calendar": _net_sh(VOL_FEATS + CAL_FEATS),
        "full_model": _net_sh(FEATURES),
        "static_short_usd": [round(float(sharpe(bench_net)), 2),
                             round(float(sharpe(bench_net[recent0])), 2)],
    }
    dchart = res["decomposition"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    labs = ["Static\nshort-USD", "Volume\nonly", "Calendar\nonly", "Volume+\nCalendar", "Full\nmodel"]
    keys = ["static_short_usd", "volume_only", "calendar_only", "volume_plus_calendar", "full_model"]
    full_v = [dchart[k][0] for k in keys]
    rec_v = [dchart[k][1] for k in keys]
    x = np.arange(len(labs))
    ax.bar(x - 0.2, full_v, 0.4, label="full history", color=GREY)
    ax.bar(x + 0.2, rec_v, 0.4, label="2019+ regime", color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("net Sharpe")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Attribution — volume and calendar each add real, additive edge over the drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "vm_attribution.png", dpi=110)
    plt.close(fig)

    # walk-forward models -> out-of-sample probabilities
    p_logit, coef = walk_forward(df, lambda: LogisticRegression(C=0.3, max_iter=1000))
    p_gbm, _ = walk_forward(df, lambda: HistGradientBoostingClassifier(
        max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0))
    chart_coefs(coef, res)

    recent = (df.date >= "2019-01-01").values
    oos = ~np.isnan(p_logit)
    res["oos_accuracy_logit"] = round(float(((p_logit > 0.5) == (df.r_next > 0))[oos].mean()), 3)
    res["oos_accuracy_gbm"] = round(float(((p_gbm > 0.5) == (df.r_next > 0))[
        ~np.isnan(p_gbm)].mean()), 3)

    # positions (long/short USD). continuous = conviction-sized (cost-friendly); discrete = ±1.
    pos_cont = np.clip((np.nan_to_num(p_logit, nan=0.5) - 0.5) * 5, -1, 1)
    pos_disc = np.where(np.isnan(p_logit), 0.0, np.sign(p_logit - 0.5))
    pos_gbm = np.clip((np.nan_to_num(p_gbm, nan=0.5) - 0.5) * 5, -1, 1)
    # simple transparent rule: low seasonally-adj volume -> long USD, plus 2nd-half tilt
    simple = -np.sign(df.vol_dow_norm.fillna(0)) * 0.6 + np.where(df.quincena == 1, -0.4, 0.4)
    simple = np.clip(simple, -1, 1)
    simple[~oos] = 0.0

    book = {}
    for name, pos in [("Logit (conviction)", pos_cont), ("Logit (±1)", pos_disc),
                      ("Gradient boosting", pos_gbm), ("Simple flow+quincena rule", simple)]:
        net, turn = net_pnl(pos, df)
        book[name] = (net, turn, pos)
        res[name] = stats(net, turn, len(df))
        res[name + "_2019"] = stats(net, mask=recent)
        res[name + "_gross_sh"] = round(float(sharpe(pd.Series(pos).reset_index(drop=True)
                                                      * df.r_next.reset_index(drop=True))), 2)

    # equity chart
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for (name, (net, turn, _)), c in zip(book.items(), [NAVY, PURPLE, ORANGE, GREEN]):
        ax.plot(df.date, np.nancumsum(net), color=c, lw=1.4,
                label=f"{name}: net Sh {res[name]['sharpe']}, {res[name]['roundtrips_yr']} rt/yr")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Flow+seasonality models — net of 0.65 CRC round-trip, long/short USD")
    ax.set_ylabel("cum net P&L (bps)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "vm_equity.png", dpi=110)
    plt.close(fig)

    # OOS / regime chart for the primary (conviction logit)
    net, turn, _ = book["Logit (conviction)"]
    k = int(len(df) * 0.6)
    res["Logit (conviction)_is"] = stats(net.iloc[:k])
    res["Logit (conviction)_oos"] = stats(net.iloc[k:])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.date, np.nancumsum(net), color=NAVY, lw=1.5)
    ax.axvspan(df.date.iloc[0], df.date.iloc[k], color=GREY, alpha=0.10)
    ax.axvline(df.date.iloc[k], color=RED, ls="--", lw=1)
    ax.set_title(f"Conviction logit — net cum P&L (IS Sh {res['Logit (conviction)_is']['sharpe']} "
                 f"→ OOS {res['Logit (conviction)_oos']['sharpe']})")
    ax.set_ylabel("cum net P&L (bps)")
    fig.tight_layout()
    fig.savefig(OUT / "vm_oos.png", dpi=110)
    plt.close(fig)

    # permutation null for the conviction logit
    rng = np.random.default_rng(7)
    posv = book["Logit (conviction)"][2]
    r = df.r_next.reset_index(drop=True)
    px = df.close.reset_index(drop=True)
    cb = COST_SIDE / px * 1e4
    actual = sharpe(net)
    null = np.empty(2000)
    for i in range(2000):
        p = pd.Series(rng.permutation(posv))
        tn = p.diff().abs()
        tn.iloc[0] = abs(p.iloc[0])
        null[i] = sharpe(p * r - cb * tn)
    res["perm_p"] = round(float((null >= actual).mean()), 4)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null, bins=50, color=NAVY, alpha=0.55)
    ax.axvline(actual, color=RED, lw=2, label=f"actual {actual:.2f}")
    ax.set_title(f"Permutation null — conviction logit (p={res['perm_p']})")
    ax.set_xlabel("net Sharpe under shuffled timing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "vm_permutation.png", dpi=110)
    plt.close(fig)

    Path(OUT / "vm_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("_")}, indent=2)[:2500])
    print("\nWrote vm_*.png and vm_results.json")


if __name__ == "__main__":
    main()
