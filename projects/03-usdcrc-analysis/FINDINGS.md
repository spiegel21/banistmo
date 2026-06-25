# USD/CRC (MONEX) — Flow & Seasonality model (no technical indicators)

**Data:** 2,663 trading days, 2014-12-08 → 2026-06-19. **Cost:** 0.65 CRC round-trip.
Features use ONLY previous/lagged volumes, seasonality (day-of-week, month, quincena),
and price/precio-ponderado relationships. No moving averages, breakouts, or momentum.
Everything is walk-forward (every prediction out-of-sample). Reproduce:
`parse_monex → analyze → eda → volume_model → build_report`.

## Your thesis is confirmed — and it's tradeable

1. **Low volume → USD strengthens next day, monotonically.** Seasonally-adjusted volume
   quintiles: lowest **+7.5 bps** next-day USD move → highest **−11.3 bps**. Yesterday's
   volume also informs (corr −0.12) and volume regimes persist (autocorr 0.58).
2. **Seasonality is an independent, equally strong driver.** Quincena: first half of month
   **−5.5 bps** (USD weak), second half **+4.1 bps** (USD strong) — a payment/tax cycle.
   Wednesday is the softest weekday.
3. **Volume and calendar are additive.** Net Sharpe (full / 2019+):
   volume-only **1.40 / 2.03**, calendar-only **1.56 / 2.02**, combined **2.43 / 3.16**.
4. **It is NOT the trend.** Static short-USD nets only **0.33**; the model nets **2.43**.
   The walk-forward model predicts next-day direction at **63% out-of-sample** (52% base
   rate), Sharpe **2.23 full / 3.34 in 2019+** net of cost, out-of-sample ≥ in-sample,
   permutation p≈0, and lagging the features 5 days collapses the accuracy (no look-ahead).

## How to trade it (long/short USD)

Each day score next-day direction from: (i) seasonally-adjusted volume — low ⇒ long USD;
(ii) quincena — first half ⇒ lean short USD, second half ⇒ long; (iii) yesterday's volume
and close-vs-VWAP. Size by conviction to keep turnover ~40 round-trips/yr. Even a hand-built
version of this rule nets Sharpe ~1.8.

## Caveats

- Turnover ~40 rt/yr → cost-sensitive (already netted at 0.65 CRC round-trip).
- Most of the edge is in the volatile 2019+ regime; the 2015–17 pegged years carried less.
- High Sharpe partly reflects the colón's low managed-float volatility. The dominant,
  unpriceable risk is a BCCR re-peg / intervention, which would mute both signals.
- Net of slippage on close-to-close fills; gross of financing/borrow and market impact; the
  colón is not freely shortable for all participants (a directional/treasury edge).
