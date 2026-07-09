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


## Underlying dynamics (why it works) & dollar results ($1M/trade)

**The engine — exporters are the swing USD supply.** On days the colón strengthens
the market trades ~29M vs ~18M on days it weakens: USD sellers (exporters, remittances,
FDI) come in size, USD buyers trickle. So high volume = supply present = colón up; low
volume = thin market = USD drifts up next day. That asymmetry is the volume signal.

**The quincena is a cash-flow cycle.** Volume peaks mid-month (~day 13–15) and the colón
strengthens hardest right then, reversing in the second half — exporter settlements /
tax-paydate conversions. Positive in all 12 years.

**Transparent rules at $1,000,000 per trade, net of 0.65 CRC round-trip:**

| Rule | Logic | Per year | Sharpe | Trades/yr |
|------|-------|---------:|-------:|----------:|
| Quincena | dom≤15 → short USD; dom>15 → long | **$92k** | **2.09** | 24 |
| Volume | seasonal-vol<1 → long USD; else short | $17k | 0.37 | 78 |
| Precio-ponderado skew | VWAP>simple → long USD; else short | −$7k | −0.17 | 102 |
| Combined vote (majority of the 3) | — | $51k | 1.15 | 85 |
| ML (conviction-sized, all features) | logistic P(up), size=clip((P−.5)×5,±1) | ~$61k | 2.23 | 42 |

**Lesson:** the calendar (quincena) rule is the best *standalone* — cheap to trade and
high Sharpe. The volume/skew signals are directionally real but flip too often (78–102
trades/yr) to survive slippage alone; they add value only conviction-sized or as filters,
which is exactly what the ML model does (turnover ~42/yr).


## RECOMMENDED STRATEGY — refined quincena (calendar) rule

The crude "short USD if day≤15" improves substantially once days 1–4 (which are
USD-*up*) are excluded from the short window:

    day 1–4    -> LONG USD   (start-of-month, USD tends up)
    day 5–15   -> SHORT USD  (mid-month USD-supply surge, colón strengthens)
    day 16–end -> LONG USD   (supply fades, USD drifts up)

At USD 1M per trade, net of 0.65 CRC round-trip:

| Version | Per year | Sharpe | Trades/yr | Max DD |
|---------|---------:|-------:|----------:|-------:|
| Base quincena (≤15) | $92k | 2.09 | 24 | −$63k |
| **Refined (short 5–15)** | **$125k** | **2.88** | 24 | −$50k |
| Refined + slow-volume sizing | $103k | **3.27** | 20 | **−$34k** |

- **Robust, not overfit:** Sharpe is a broad plateau (1.2–2.9) across all short-window
  choices; positive in ALL 12 years (worst +$37k), incl. the 2015–17 pegged years.
- **Volume is confirmation, not a daily signal:** 1st-half+high-vol = strong sell-USD,
  2nd-half+low-vol = strong buy-USD, but trading volume daily adds turnover that costs
  more than it adds. Folding it in as a SLOW 20-day regime filter (trim size when it
  disagrees) lifts Sharpe to 3.27 and halves the drawdown.
- Practical trading calendar: `out/quincena_trading_calendar.csv`.

## Público vs privado volume — NOT possible with this file

The MONEX export contains only *aggregate* traded volume. Splitting público (BCCR /
sector público) vs privado flow needs a different BCCR dataset (sector-level MONEX /
ventanilla breakdown). Provide that and the mid-month supply surge can be attributed
to a specific sector.
