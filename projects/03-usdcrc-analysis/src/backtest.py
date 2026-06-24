"""Rigorous, overfitting-aware backtest of the USD/CRC signals.

A quant's checklist, applied to every candidate strategy:
  1. Strictly causal signals (decide at close t, earn t -> t+1).
  2. In-sample vs out-of-sample chronological split.
  3. Expanding walk-forward (the *sign* is re-estimated live, never peeked).
  4. Transaction-cost sensitivity + breakeven cost.
  5. Benchmark decomposition: alpha/beta vs static short-USD (the trend).
  6. Monte-Carlo permutation null (does timing beat random same-exposure?).
  7. Probabilistic & Deflated Sharpe (haircut for multiple trials + fat tails).
  8. Parameter-sensitivity heatmap (plateau vs lucky spike).
  9. Regime robustness (range vs appreciation sub-periods).

Writes bt_*.png charts and out/backtest_results.json (consumed by build_report).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyze import OUT, load

plt.rcParams.update({
    "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
NAVY, GREEN, RED, GREY, ORANGE = "#1f3b73", "#2e8b57", "#c0392b", "#7f8c8d", "#e08a2b"
ANN = np.sqrt(252)
SPLIT_FRAC = 0.65
REGIME_SPLIT = pd.Timestamp("2025-10-15")
N_TRIALS = 6          # how many distinct strategies we eyeballed -> multiple-testing haircut
RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------- #
# core helpers
# --------------------------------------------------------------------------- #
def sharpe(bps):
    bps = np.asarray(bps, float)
    bps = bps[~np.isnan(bps)]
    return np.nan if len(bps) < 5 or bps.std() == 0 else bps.mean() / bps.std() * ANN


def net_pnl(pos, r_next, cost_bps=0.0):
    """Position decided at t, return earned t->t+1; cost on |change in position|."""
    pos = pd.Series(pos).reset_index(drop=True)
    r_next = pd.Series(r_next).reset_index(drop=True)
    turn = pos.diff().abs()
    turn.iloc[0] = pos.iloc[0].__abs__()
    return (pos * r_next - cost_bps * turn).values


def sharpe_lo(bps, max_lag=10):
    """Lo (2002) autocorrelation-adjusted annualized Sharpe.

    Positive serial correlation in the P&L makes the naive sqrt(252) overstate
    the annual Sharpe; this corrects for it.
    """
    bps = np.asarray(bps, float)
    bps = bps[~np.isnan(bps)]
    sr_d = bps.mean() / bps.std()
    x = bps - bps.mean()
    d = (x * x).sum()
    q = 252
    s = 0.0
    for k in range(1, min(max_lag, q - 1) + 1):
        rho = (x[k:] * x[:-k]).sum() / d
        s += (q - k) * rho
    eta = q / np.sqrt(q + 2 * s)
    return float(sr_d * eta)


def stats_block(bps):
    bps = np.asarray(bps, float)
    bps = bps[~np.isnan(bps)]
    active = bps[bps != 0]
    if len(active) == 0:
        active = bps
    return {
        "n": int(len(bps)),
        "n_active": int(len(active)),
        "sharpe": round(float(sharpe(bps)), 2),
        "sharpe_lo": round(float(sharpe_lo(bps)), 2),
        "mean_bps": round(float(bps.mean()), 2),
        "hit": round(float((active > 0).mean() * 100), 1),
        "total_bps": round(float(bps.sum()), 0),
        "maxdd_bps": round(float((np.cumsum(bps) -
                                  np.maximum.accumulate(np.cumsum(bps))).min()), 0),
    }


def probabilistic_sharpe(bps, sr_benchmark_ann=0.0):
    """PSR: prob the true Sharpe exceeds a benchmark, adjusting for skew/kurtosis."""
    bps = np.asarray(bps, float)
    bps = bps[~np.isnan(bps)]
    n = len(bps)
    sr = bps.mean() / bps.std()                       # daily
    sk = stats.skew(bps)
    ku = stats.kurtosis(bps, fisher=False)
    sr_b = sr_benchmark_ann / ANN
    denom = np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_b) * np.sqrt(n - 1) / denom))


def deflated_sharpe(bps, n_trials):
    """DSR: PSR against the Sharpe you'd expect as the best of n_trials lucky tries."""
    bps = np.asarray(bps, float)
    bps = bps[~np.isnan(bps)]
    sr = bps.mean() / bps.std() * ANN
    e = np.e
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * e))
    sr_var = (1 - np.euler_gamma) * z1 + np.euler_gamma * z2   # expected-max in SR-std units
    # Sharpe std across trials ~ sqrt((1 + 0.5 sr^2)/n); approximate with observed.
    n = len(bps)
    sr_d = sr / ANN
    sr_std = np.sqrt((1 + 0.5 * sr_d**2) / (n - 1)) * ANN
    sr_benchmark_ann = sr_std * sr_var
    return probabilistic_sharpe(bps, sr_benchmark_ann), round(float(sr_benchmark_ann), 2)


# --------------------------------------------------------------------------- #
# signal construction (all causal)
# --------------------------------------------------------------------------- #
def make_frame():
    df = load().copy()
    df["r_next"] = df.ret_vwap_bps.shift(-1)          # USD return earned next session (bps)
    df["vol_z"] = (df.volume_musd - df.volume_musd.rolling(20).mean()) / \
                  df.volume_musd.rolling(20).std()
    df["ret_z"] = (df.ret_vwap_bps - df.ret_vwap_bps.rolling(60).mean()) / \
                  df.ret_vwap_bps.rolling(60).std()
    df = df.dropna(subset=["r_next", "vol_z", "ret_z"]).reset_index(drop=True)
    df["pos_of"] = -np.sign(df.vol_z)                          # order-flow: short USD after heavy
    df["pos_mo"] = np.where(df.ret_z.abs() >= 1, np.sign(df.ret_vwap_bps), 0.0)  # momentum
    df["pos_bench"] = -1.0                                     # static short USD (the trend)
    return df


# --------------------------------------------------------------------------- #
# tests / charts
# --------------------------------------------------------------------------- #
def test_oos(df, res):
    k = int(len(df) * SPLIT_FRAC)
    tr, te = df.iloc[:k], df.iloc[k:]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for axi, (name, col, color) in zip(
            ax, [("Order-flow", "pos_of", GREEN), ("Momentum", "pos_mo", NAVY)]):
        res.setdefault(name, {})
        res[name]["in_sample"] = stats_block(net_pnl(tr[col], tr.r_next))
        res[name]["out_sample"] = stats_block(net_pnl(te[col], te.r_next))
        full = net_pnl(df[col], df.r_next)
        cum = np.cumsum(full)
        axi.plot(df.date, cum, color=color, lw=1.5)
        axi.axvspan(df.date.iloc[0], df.date.iloc[k], color=GREY, alpha=0.10)
        axi.axvline(df.date.iloc[k], color=RED, ls="--", lw=1)
        axi.set_title(f"{name}: IS Sharpe {res[name]['in_sample']['sharpe']} "
                      f"-> OOS {res[name]['out_sample']['sharpe']}")
        axi.set_ylabel("cum P&L (bps)")
        axi.text(df.date.iloc[2], cum.max() * 0.92, "train", color=GREY)
        axi.text(df.date.iloc[k + 2], cum.max() * 0.92, "test (held out)", color=RED)
    fig.suptitle("Out-of-sample test — fit nothing on the grey region, judge on the red",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "bt_oos.png", dpi=110)
    plt.close(fig)


def test_walkforward(df, res):
    """Re-estimate the order-flow SIGN on an expanding window; trade strictly forward."""
    start = 60
    pnl = np.full(len(df), np.nan)
    for t in range(start, len(df)):
        past = df.iloc[:t]
        # Live-estimated direction of the volume->next-move relationship (no peeking).
        sign = np.sign(np.corrcoef(past.vol_z, past.r_next)[0, 1])
        # Expected sign is negative, so position = sign * sign(vol_z): when corr<0 and the
        # day was heavy (vol_z>0) we go short USD (pos<0).
        pos_t = sign * np.sign(df.vol_z.iloc[t])
        pnl[t] = pos_t * df.r_next.iloc[t]
    wf = pnl[start:]
    res["Order-flow"]["walk_forward"] = stats_block(wf)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.date.iloc[start:], np.cumsum(wf), color=GREEN, lw=1.6)
    ax.set_title(f"Order-flow expanding walk-forward (sign re-estimated live) — "
                 f"Sharpe {res['Order-flow']['walk_forward']['sharpe']}, "
                 f"hit {res['Order-flow']['walk_forward']['hit']}%")
    ax.set_ylabel("cum P&L (bps)")
    fig.tight_layout()
    fig.savefig(OUT / "bt_walkforward.png", dpi=110)
    plt.close(fig)


def test_costs(df, res):
    costs = np.linspace(0, 12, 25)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    breakeven = {}
    for name, col, color in [("Order-flow", "pos_of", GREEN), ("Momentum", "pos_mo", NAVY)]:
        srs = [sharpe(net_pnl(df[col], df.r_next, c)) for c in costs]
        ax.plot(costs, srs, color=color, lw=1.8, label=name)
        be = next((c for c, s in zip(costs, srs) if s <= 0), None)
        breakeven[name] = round(float(be), 1) if be is not None else None
        res[name]["breakeven_cost_bps"] = breakeven[name]
        res[name]["net_sharpe_at_2bps"] = round(float(sharpe(net_pnl(df[col], df.r_next, 2.0))), 2)
    ax.axhline(0, color=RED, lw=0.8)
    ax.axvline(2, color=GREY, ls="--", lw=1)
    ax.text(2.1, ax.get_ylim()[1] * 0.9, "~MONEX spread", color=GREY, fontsize=9)
    ax.set_title("Sharpe vs per-side transaction cost (bps) — breakeven where it crosses 0")
    ax.set_xlabel("transaction cost (bps per side)")
    ax.set_ylabel("net Sharpe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "bt_costs.png", dpi=110)
    plt.close(fig)


def test_benchmark(df, res):
    """Alpha/beta of order-flow vs static short-USD (is there edge beyond the trend?)."""
    of = net_pnl(df.pos_of, df.r_next)
    bench = net_pnl(df.pos_bench, df.r_next)
    beta, alpha, r, p, se = stats.linregress(bench, of)
    t_alpha = alpha / se if se else np.nan  # se is for slope; approximate alpha t via resid
    resid = of - (alpha + beta * bench)
    t_alpha = alpha / (resid.std() / np.sqrt(len(resid)))
    res["Order-flow"]["benchmark"] = {
        "bench_sharpe": round(float(sharpe(bench)), 2),
        "alpha_bps": round(float(alpha), 2),
        "beta": round(float(beta), 2),
        "t_alpha": round(float(t_alpha), 2),
        "corr_with_bench": round(float(r), 2),
    }
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(df.date, np.cumsum(of), color=GREEN, lw=1.6, label="Order-flow")
    ax[0].plot(df.date, np.cumsum(bench), color=GREY, lw=1.6, label="Static short-USD (trend)")
    ax[0].set_title("Order-flow vs the trend benchmark")
    ax[0].set_ylabel("cum P&L (bps)")
    ax[0].legend()
    ax[1].scatter(bench, of, s=10, alpha=0.4, color=NAVY)
    xx = np.linspace(bench.min(), bench.max(), 40)
    ax[1].plot(xx, alpha + beta * xx, color=RED, lw=1.6)
    ax[1].axhline(0, color=GREY, lw=0.5)
    ax[1].axvline(0, color=GREY, lw=0.5)
    ax[1].set_title(f"alpha {alpha:+.1f} bps/day (t={t_alpha:.1f}), beta {beta:+.2f}")
    ax[1].set_xlabel("benchmark daily P&L (bps)")
    ax[1].set_ylabel("order-flow daily P&L (bps)")
    fig.tight_layout()
    fig.savefig(OUT / "bt_benchmark.png", dpi=110)
    plt.close(fig)


def test_permutation(df, res, n=3000):
    """Shuffle positions (preserve exposure mix) -> null Sharpe distribution."""
    r = df.r_next.values
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for axi, (name, col, color) in zip(
            ax, [("Order-flow", "pos_of", GREEN), ("Momentum", "pos_mo", NAVY)]):
        pos = df[col].values
        actual = sharpe(pos * r)
        null = np.empty(n)
        for i in range(n):
            null[i] = sharpe(RNG.permutation(pos) * r)
        pval = float((null >= actual).mean())
        res[name]["permutation_p"] = round(pval, 4)
        axi.hist(null, bins=50, color=color, alpha=0.55)
        axi.axvline(actual, color=RED, lw=2, label=f"actual {actual:.2f}")
        axi.set_title(f"{name}: permutation null (p={pval:.3f})")
        axi.set_xlabel("Sharpe under shuffled timing")
        axi.legend()
    fig.suptitle("Does the signal's TIMING beat random timing with the same exposure?",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "bt_permutation.png", dpi=110)
    plt.close(fig)


def test_sensitivity(df, res):
    """Order-flow Sharpe across (lookback, z-threshold) -> plateau check."""
    lookbacks = [10, 15, 20, 30, 40, 60]
    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    grid = np.full((len(lookbacks), len(thresholds)), np.nan)
    base = load().copy()
    base["r_next"] = base.ret_vwap_bps.shift(-1)
    for i, lb in enumerate(lookbacks):
        vz = (base.volume_musd - base.volume_musd.rolling(lb).mean()) / \
             base.volume_musd.rolling(lb).std()
        for j, thr in enumerate(thresholds):
            pos = np.where(vz.abs() >= thr, -np.sign(vz), 0.0)
            d = pd.DataFrame({"pos": pos, "rn": base.r_next}).dropna()
            grid[i, j] = sharpe(net_pnl(d.pos, d.rn))
    res["Order-flow"]["sensitivity"] = {
        "min": round(float(np.nanmin(grid)), 2),
        "max": round(float(np.nanmax(grid)), 2),
        "median": round(float(np.nanmedian(grid)), 2),
    }
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=0, vmax=np.nanmax(grid))
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels(thresholds)
    ax.set_yticks(range(len(lookbacks)))
    ax.set_yticklabels(lookbacks)
    ax.set_xlabel("volume z-score entry threshold")
    ax.set_ylabel("volume lookback (days)")
    ax.set_title("Order-flow Sharpe across parameters — broad green = robust, not a spike")
    ax.grid(False)
    for i in range(len(lookbacks)):
        for j in range(len(thresholds)):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "bt_sensitivity.png", dpi=110)
    plt.close(fig)


def test_regime(df, res):
    a = df[df.date < REGIME_SPLIT]
    b = df[df.date >= REGIME_SPLIT]
    res["Order-flow"]["regime"] = {
        "range": stats_block(net_pnl(a.pos_of, a.r_next)),
        "appreciation": stats_block(net_pnl(b.pos_of, b.r_next)),
    }
    res["Momentum"]["regime"] = {
        "range": stats_block(net_pnl(a.pos_mo, a.r_next)),
        "appreciation": stats_block(net_pnl(b.pos_mo, b.r_next)),
    }


def test_execution(df, res):
    """You cannot trade *at* VWAP. Re-price the order-flow edge on a realistic
    close-to-close fill: enter at close(t), exit at close(t+1)."""
    base = load().copy()
    base["cc"] = base.close.pct_change() * 1e4          # close-to-close USD return (bps)
    base["cc_next"] = base.cc.shift(-1)
    base["vol_z"] = (base.volume_musd - base.volume_musd.rolling(20).mean()) / \
                    base.volume_musd.rolling(20).std()
    base = base.dropna(subset=["cc_next", "vol_z"])
    pos = -np.sign(base.vol_z)
    res["Order-flow"]["exec_close_to_close"] = stats_block(net_pnl(pos, base.cc_next))


def finalize(df, res):
    for name, col in [("Order-flow", "pos_of"), ("Momentum", "pos_mo")]:
        full = net_pnl(df[col], df.r_next)
        res[name]["full"] = stats_block(full)
        res[name]["PSR_vs0"] = round(probabilistic_sharpe(full, 0.0), 3)
        dsr, sr_haircut = deflated_sharpe(full, N_TRIALS)
        res[name]["DSR"] = round(dsr, 3)
        res[name]["DSR_haircut_sharpe"] = sr_haircut
    res["_meta"] = {
        "n_days": int(len(df)),
        "split_frac": SPLIT_FRAC,
        "n_trials_haircut": N_TRIALS,
        "regime_split": str(REGIME_SPLIT.date()),
        "date_min": str(df.date.min().date()),
        "date_max": str(df.date.max().date()),
    }


def main():
    df = make_frame()
    res = {}
    test_oos(df, res)
    test_walkforward(df, res)
    test_costs(df, res)
    test_benchmark(df, res)
    test_permutation(df, res)
    test_sensitivity(df, res)
    test_regime(df, res)
    test_execution(df, res)
    finalize(df, res)
    Path(OUT / "backtest_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("\nWrote bt_*.png and out/backtest_results.json")


if __name__ == "__main__":
    main()
