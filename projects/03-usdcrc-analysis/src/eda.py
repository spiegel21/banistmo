"""Exploratory data analysis charts for the USD/CRC MONEX data.

Purely informational visualizations to *see* structure before modelling:
  eda_price_volume.png  : price path with regime shading + signed volume bars
  eda_returns.png       : return distribution, QQ, ACF(returns), ACF(|returns|)
  eda_seasonality.png   : return boxplots by month and by day-of-week
  eda_volume_return.png : volume vs same-day & next-day move + rolling correlation
  eda_calendar.png      : day-of-week x month return heatmap + volume heatmap
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyze import DOW, OUT, load

plt.rcParams.update({
    "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
NAVY, GREEN, RED, GREY = "#1f3b73", "#2e8b57", "#c0392b", "#7f8c8d"

# Data-driven regime split: the colon was broadly range-bound, then began a
# sustained appreciation. We split for *descriptive* shading only.
REGIME_SPLIT = pd.Timestamp("2025-10-15")


def acf(x, n):
    x = np.asarray(x, float)
    x = x - x.mean()
    d = (x * x).sum()
    return [1.0] + [float((x[k:] * x[:-k]).sum() / d) for k in range(1, n + 1)]


def price_volume(df):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[2, 1], sharex=True)
    ax[0].plot(df.date, df.vwap, color=NAVY, lw=1.1)
    ax[0].set_title("USD/CRC session VWAP, 2014–2026 — pegged/quiet (2015–17) then two-way regimes")
    ax[0].set_ylabel("colones / US$")

    col = np.where(df.ret_vwap_bps >= 0, RED, GREEN)  # red = colon weaker, green = stronger
    ax[1].bar(df.date, df.volume_musd, color=col, width=1.0)
    ax[1].set_title("Daily volume (US$M) — green = colón appreciated that day, red = depreciated")
    ax[1].set_ylabel("US$M")
    fig.tight_layout()
    fig.savefig(OUT / "eda_price_volume.png", dpi=110)
    plt.close(fig)


def returns_diagnostics(df):
    r = df.ret_vwap_bps.dropna()
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    ax[0, 0].hist(r, bins=45, color=NAVY, alpha=0.85, density=True)
    xs = np.linspace(r.min(), r.max(), 200)
    ax[0, 0].plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), color=RED, lw=1.5, label="normal")
    ax[0, 0].set_title(f"Daily move dist (bps)  skew {r.skew():+.2f}  kurt {r.kurtosis():+.2f}")
    ax[0, 0].legend(fontsize=8)

    stats.probplot(r, dist="norm", plot=ax[0, 1])
    ax[0, 1].set_title("QQ plot vs normal (fat tails)")
    ax[0, 1].get_lines()[0].set(marker="o", ms=3, color=NAVY, alpha=0.5)
    ax[0, 1].get_lines()[1].set(color=RED)

    n = 20
    a = acf(r, n)
    ci = 1.96 / np.sqrt(len(r))
    ax[1, 0].bar(range(n + 1), a, color=[RED if i == 1 else NAVY for i in range(n + 1)])
    ax[1, 0].axhline(ci, color=GREY, ls="--", lw=0.8)
    ax[1, 0].axhline(-ci, color=GREY, ls="--", lw=0.8)
    ax[1, 0].set_title(f"ACF of daily moves — lag1 = {a[1]:+.2f} (real momentum)")
    ax[1, 0].set_xlabel("lag (days)")

    aa = acf(r.abs(), n)
    ax[1, 1].bar(range(n + 1), aa, color=GREEN)
    ax[1, 1].axhline(ci, color=GREY, ls="--", lw=0.8)
    ax[1, 1].set_title("ACF of |moves| — volatility clustering")
    ax[1, 1].set_xlabel("lag (days)")
    fig.tight_layout()
    fig.savefig(OUT / "eda_returns.png", dpi=110)
    plt.close(fig)


def seasonality_box(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    months = sorted(df.month.unique())
    ax[0].boxplot([df[df.month == m].ret_vwap_bps.dropna() for m in months],
                  tick_labels=[pd.Timestamp(2025, m, 1).strftime("%b") for m in months],
                  showfliers=False)
    ax[0].axhline(0, color=RED, lw=0.8)
    ax[0].set_title("Daily move by month (bps) — dispersion, not just mean")
    ax[0].set_ylabel("bps")

    dows = [d for d in DOW if d in df.dow_name.values]
    ax[1].boxplot([df[df.dow_name == d].ret_vwap_bps.dropna() for d in dows],
                  tick_labels=dows, showfliers=False)
    ax[1].axhline(0, color=RED, lw=0.8)
    ax[1].set_title("Daily move by day-of-week (bps)")
    fig.tight_layout()
    fig.savefig(OUT / "eda_seasonality.png", dpi=110)
    plt.close(fig)


def volume_return(df):
    df = df.copy()
    df["ret_next"] = df.ret_vwap_bps.shift(-1)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))

    hb = ax[0].hexbin(df.volume_musd, df.ret_vwap_bps, gridsize=22, cmap="Blues", mincnt=1)
    b, a = np.polyfit(df.volume_musd, df.ret_vwap_bps, 1)
    xx = np.linspace(df.volume_musd.min(), df.volume_musd.max(), 50)
    ax[0].plot(xx, a + b * xx, color=RED, lw=1.6)
    ax[0].axhline(0, color=GREY, lw=0.6)
    ax[0].set_title(f"SAME-day: move vs volume\nslope {b:+.2f} bps/$M (contemporaneous)")
    ax[0].set_xlabel("volume US$M")
    ax[0].set_ylabel("move bps")
    fig.colorbar(hb, ax=ax[0], shrink=0.8)

    d2 = df.dropna(subset=["ret_next"])
    ax[1].scatter(d2.volume_musd, d2.ret_next, s=10, alpha=0.4, color=NAVY)
    b2, a2 = np.polyfit(d2.volume_musd, d2.ret_next, 1)
    ax[1].plot(xx, a2 + b2 * xx, color=RED, lw=1.6)
    ax[1].axhline(0, color=GREY, lw=0.6)
    ax[1].set_title(f"NEXT-day: move vs today volume\nslope {b2:+.2f} bps/$M (tradeable)")
    ax[1].set_xlabel("today volume US$M")
    ax[1].set_ylabel("next-day move bps")

    rc = df.set_index("date")[["volume_musd", "ret_vwap_bps"]].rolling(60).corr().unstack()
    rc = rc[("volume_musd", "ret_vwap_bps")].dropna()
    ax[2].plot(rc.index, rc.values, color=NAVY, lw=1.3)
    ax[2].axhline(0, color=GREY, lw=0.6)
    ax[2].fill_between(rc.index, rc.values, 0, where=rc.values < 0, color=GREEN, alpha=0.25)
    ax[2].set_title("Rolling 60d corr(volume, move)\npersistently negative = stable signal")
    ax[2].set_ylabel("correlation")
    fig.tight_layout()
    fig.savefig(OUT / "eda_volume_return.png", dpi=110)
    plt.close(fig)


def calendar_heat(df):
    dows = [d for d in DOW if d in df.dow_name.values]
    months = sorted(df.month.unique())
    mlab = [pd.Timestamp(2025, m, 1).strftime("%b") for m in months]
    ret = np.full((len(dows), len(months)), np.nan)
    vol = np.full((len(dows), len(months)), np.nan)
    for i, d in enumerate(dows):
        for j, m in enumerate(months):
            sl = df[(df.dow_name == d) & (df.month == m)]
            if len(sl):
                ret[i, j] = sl.ret_vwap_bps.mean()
                vol[i, j] = sl.volume_musd.mean()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    lim = np.nanmax(np.abs(ret))
    im0 = ax[0].imshow(ret, cmap="RdYlGn_r", vmin=-lim, vmax=lim, aspect="auto")
    ax[0].set_title("Avg daily move (bps) — green = colón stronger")
    im1 = ax[1].imshow(vol, cmap="viridis", aspect="auto")
    ax[1].set_title("Avg daily volume (US$M)")
    for a in ax:
        a.set_xticks(range(len(mlab)))
        a.set_xticklabels(mlab)
        a.set_yticks(range(len(dows)))
        a.set_yticklabels(dows)
        a.grid(False)
    fig.colorbar(im0, ax=ax[0], shrink=0.8)
    fig.colorbar(im1, ax=ax[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "eda_calendar.png", dpi=110)
    plt.close(fig)


def main():
    df = load()
    price_volume(df)
    returns_diagnostics(df)
    seasonality_box(df)
    volume_return(df)
    calendar_heat(df)
    print("Wrote EDA charts:")
    for f in ["eda_price_volume", "eda_returns", "eda_seasonality",
              "eda_volume_return", "eda_calendar"]:
        print("   out/" + f + ".png")


if __name__ == "__main__":
    main()
