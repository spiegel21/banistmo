# USD/CRC (MONEX) — Pricing-Model Reverse-Engineering & Trading Analysis

**Source:** BCCR *"frmVerCatCuadro"* export (MONEX wholesale FX market).
**Sample:** 362 trading days, 2024-12-06 → 2026-06-19.
**Fields per day:** open / close / low / high / simple-avg / **VWAP** / prior-session
VWAP of the traded rate, plus total US$ volume, number of matched trades (`calces`),
and per-trade size stats.

> Reproduce: `python src/parse_monex.py && python src/analyze.py`.
> Full numeric log in `out/analysis_report.txt`; charts in `out/*.png`.

---

## TL;DR — what the data says

1. **The colón is in a strong appreciation crawl that *accelerated*.** VWAP was range-bound
   ~500–511 through ~Oct-2025, then fell sharply to ~454 (**−10.1%** peak-to-date; average
   drift −37 ₡/yr, R² = 0.70 on a linear fit, but the regime is "flat → steep down", not
   constant-speed). Realized vol is low (~4% annualized) — a *managed, trending* market, not
   a random walk.
2. **Price formation is VWAP-anchored.** Each session opens essentially *at* the prior
   session's weighted-average rate (median gap −1.6 bps; 48% of days open within ±10 bps).
   The published session VWAP is the de-facto reference price the next day prices off of.
3. **Volume is a one-directional order-flow signal.** Heavy-volume days are USD-*supply*
   days: `ret_bps = +14.8 − 0.45·volume_$M` (R² = 0.14, p ≈ 1e-13). The top volume
   tercile averages **−14 bps** (colón up); the bottom tercile **+7 bps** (colón down).
4. **The appreciation pressure persists overnight.** Session moves have **+0.37 lag-1
   autocorrelation** (trend, not reversion). After a heavy day the rate keeps falling
   **−14 bps** the next session; after a light day it rises **+8.5 bps**.
5. **Best clean edge:** an *order-flow continuation* strategy (short USD after heavy days,
   long after light days) earns **+7 bps/day, 59% hit, Sharpe ≈ 4.4** gross over 342 days.

---

## 1. The "pricing model" (price formation)

The MONEX rate is **not** a random walk around a fixed level — it is a **crawling,
VWAP-anchored managed float**. The mechanics inferred from the data:

| Mechanism | Evidence |
|---|---|
| **Anchor**: today's session is benchmarked to yesterday's VWAP. | `open − prev_VWAP`: mean −2.9 bps, median −1.6 bps, 48% within ±10 bps. |
| **Crawl**: a persistent appreciation drift is layered on the anchor. | VWAP ≈ 520.25 − 0.148·day → −3.1 ₡/month, R² = 0.70. |
| **Intraday discovery is mild and mean-reverting around the open.** | Clean regression `close−prevVWAP = 0.20·(open−prevVWAP)` → ~**80% of the opening gap is retraced** by the close. VWAP sits mid-range (mean position 0.53 in [low,high]). |
| **Large tickets are roughly symmetric.** | VWAP > simple-avg on only 51% of days; mean skew +0.04 ₡ → no systematic "big players always lift offers" bias. |

> ⚠️ **Caveat on the headline intraday "fade".** A naive regression of `close−open` on
> `open−prevVWAP` gives slope −0.81 (R²=0.24), but `open` appears in both with opposite
> signs, so noise in the single opening print mechanically biases it toward −1. The
> **clean** specification above (which puts the noisy `open` only on the right-hand side)
> shows the *real* retracement is ~80% — still strong, but the raw Sharpe of the
> "fade-the-open" trade (Signal 1, Sharpe ≈ 6) is **inflated** and not reliably tradeable.

## 2. Volume seasonality

- **Day-of-week:** volume falls steadily into the week-end — Tue ~42.6 $M is the peak,
  **Friday is the lightest (33.5 $M, −18%)** with the smallest average ticket
  (145k vs 205k on Tue). Trade *count* is highest Friday but size is smallest → Friday is
  retail/clean-up flow, not block flow.
- **Month-of-year:** **April is the heaviest month (49 $M/day)** — income-tax / repatriation
  season — and also shows the cleanest appreciation. **February prints the strongest
  appreciation (−15.6 bps/day)**; **December is the only depreciation month (+7.2 bps/day)**
  — Christmas (*aguinaldo*) import demand for dollars. (Jul–Oct have fewer sample days.)
- **Intra-month:** the **first half of the month appreciates (−7.7 bps/day)** vs the
  second half (+1.9 bps); month-end (day ≥ 25) volume is **−11%** lighter. Fits a
  payroll/tax-cycle USD-supply rhythm.

## 3. Volume profile (volume at each price level)

- **Point of Control: 507 ₡** (8.5% of all volume); the 1-₡ levels **504–509** together
  hold ~40% of lifetime volume — the congestion zone the market traded through on the way down.
- **70% value area: 453 – 510 ₡.** Current price (~454) sits at the **bottom edge** — i.e.
  in price-discovery territory below the historical congestion, consistent with an ongoing
  trend rather than range-bound mean reversion.

## 4. Volume vs intraday movement

- **Volume does *not* widen the daily range** (r = +0.04, n.s.) and barely affects the
  *magnitude* of the move. The MONEX daily range is fairly constant (~50–55 bps) regardless of size.
- **Volume affects the *direction*.** The signed-move/volume slope is strongly negative
  (−0.45 bps per $M, p≈1e-13): **size = USD supply = colón strength.** This is the single
  most robust structural relationship in the data and the basis of the order-flow signal.

## 5. Trading opportunities (gross of costs)

| # | Idea | Result | Verdict |
|---|------|--------|---------|
| **S4** | **Order-flow continuation** — short USD after a heavy day (vol z>0), long after a light day; hold into next session. | **+7.0 bps/day, 59% hit, Sharpe ≈ 4.4** (342 days, +2412 bps total) | ✅ **Best clean edge.** Robust, mechanism-backed. |
| **S2** | Overnight **momentum** — follow a >1σ session move into the next session. | +17 bps/trade, 73% hit, Sharpe ≈ 8.4 (89 trades) | ✅ Strong but few trades; same trend driver as S4. |
| S1 | Intraday **fade the opening gap** to the close. | 63% hit, Sharpe ≈ 6 | ⚠️ Real (~80% retrace) but Sharpe inflated by shared-`open` noise; needs intraday execution. |
| S2b | 5d/20d VWAP **MA-crossover** trend-follow. | −1.3 bps/day, Sharpe ≈ −0.8 | ❌ Whipsaws — the edge lives at the 1-day horizon, MAs are too slow. |
| S3 | Day-of-week directional bias. | Wed weakest (−5.6 bps, t≈−1.9) | ⚠️ Suggestive only, not significant. |

### How to trade the edge
- **Direction:** the dominant, persistent trade is **long colón / short USD** (the crawl),
  sized up after **heavy-volume sessions** and into the **first half of the month / Feb & Apr**;
  trimmed in **December**.
- **Execution:** trade VWAP-to-VWAP (next-session reference) since the open is anchored to
  the prior VWAP — you can transact near a *known* reference rate.
- **Mechanism (why it should keep working):** structural USD oversupply (exports, tourism,
  FDI, possible BCCR reserve accumulation) shows up as volume, and the managed-float anchor
  makes that pressure bleed out gradually rather than gapping — hence the autocorrelation.

## Caveats
- All P&L is **gross**: no MONEX spread, brokerage, or settlement costs; the colón is not
  freely shortable for most participants (this is a *directional/hedging* edge, not arb).
- Backtests are **in-sample** on 18 months dominated by one appreciation regime; a policy
  shift (BCCR intervention, rate cuts, terms-of-trade reversal) would break the crawl.
- The "best board offers" (`MEJORES OFERTAS EN PIZARRA`) block is empty in this export, so
  no bid/ask microstructure was available — VWAP is the price proxy throughout.
