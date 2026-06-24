# USD/CRC (MONEX) — Long/Short findings (full history, net of real slippage)

**Data:** BCCR MONEX export, **2,663 trading days, 2014-12-08 → 2026-06-19** (11.5 yrs).
**Execution cost:** 0.65 CRC slippage per side (~13.7 bps at a ~475 spot), adverse.
Reproduce: `parse_monex → analyze → eda → strategies → backtest → build_report`.

## The headline revision

An earlier 18-month study showed a ~Sharpe-5 daily order-flow strategy. With 11.5
years of data and the real 0.65 CRC/side slippage, that result does **not** hold:

1. **The 18-month "Sharpe ~5" was a mirage** — a favourable post-2018 window, marked
   at an untradeable VWAP fill.
2. **Daily order-flow is a real signal but uneconomic for a taker.** Gross next-day
   close-to-close Sharpe is 2.8 (positive every year since 2017), but it flips ~95×/yr
   and the ~5 bps/trade edge is smaller than the ~13.7 bps/side cost → **net Sharpe −2.1**.
   It only pays if you are the liquidity *provider*.
3. **A slow trend-following long/short USD survives.** Donchian-40 breakout: **net
   Sharpe 1.04**, 2.5 round-trips/yr, +452 bps/yr, max drawdown −12.9%. Both long-USD
   (+1,621 bps) and short-USD (+3,158 bps) legs are positive; permutation p≈0.
4. **But it only earns when the colón trends.** In-sample (2015–21, mostly the pegged
   years) net Sharpe was 0.28; out-of-sample (2021–26) 1.83. A BCCR re-peg flattens the
   edge (no bleed, no earn). Managed-float tail risk is the dominant danger.

## The regime lesson (why short samples deceive)

The volume→direction correlation was **~0 in 2015–17** and only emerged from 2018 on.
The original study sat entirely inside the favourable regime — which is exactly how a
real edge and an overfit one look identical on 18 months. Full-history + real-cost
testing is what separated them.

## Is it worth pursuing?

- **Standalone speculative book:** marginal (net Sharpe ~1, ~13% drawdowns, tail risk,
  and you need an instrument to be long *and* short USD/CRC at 0.65 CRC slippage).
- **Treasury / hedging overlay (most likely use):** yes — a slow trend filter telling you
  when to be long vs short USD, trading a few times/yr, is a robust, cheap improvement over
  a static hedge; it survived 11.5 years and real costs.
- **Daily order-flow as a taker:** no — you pay the edge away in slippage. Monetisable only
  as a market-maker capturing the spread you currently pay.

## Note on the slippage convention

Your example described price *improvement* (buy below / sell above spot), which is
impossible for a price-taker. This report models the economically standard *adverse* 0.65
CRC per side. If you meant something else, the cost layer flips and the order-flow
conclusion changes — flag it and it's a one-line change.
