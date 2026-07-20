# USD/CRC (MONEX) — Flow & Seasonality model (no technical indicators)

> **⚠️ Annualisation caveat — read before quoting any figure below.**
> Every module except `rank_strategies.py` and `exit_lab.py` annualises with a
> hardcoded **252** trading days/year. MONEX actually trades **231.0** sessions/year
> (2,663 sessions over 11.53 calendar years). So every `$/yr` figure on this page
> that came from the legacy modules is **overstated by ~9.1%**, and every Sharpe by
> **~4.5%**. To convert to the correct basis: divide $/yr by 1.091 and Sharpe by 1.045.
> `out/ranking.json` and `out/exit_lab.json` are already on the correct basis.
> See **[Unified ranking](#unified-ranking--every-strategy-on-one-basis)** below, which
> supersedes the per-module tables for cross-strategy comparison.

**Data:** 2,663 trading days, 2014-12-08 → 2026-06-19. **Cost:** 0.65 CRC round-trip.
Features use ONLY previous/lagged volumes, seasonality (day-of-week, month, quincena),
and price/precio-ponderado relationships. No moving averages, breakouts, or momentum.
Everything is walk-forward (every prediction out-of-sample). Reproduce:
`parse_monex → analyze → eda → volume_model → build_report`.

## Your thesis is confirmed — and it's tradeable

1. **Low volume → USD strengthens next day, monotonically.** Seasonally-adjusted volume
   quintiles: lowest **+9.3 bps** next-day USD move → highest **−13.7 bps**. Yesterday's
   volume also informs (corr −0.14) and volume regimes persist (autocorr 0.58).
2. **Seasonality is an independent, equally strong driver.** Quincena: first half of month
   **−5.3 bps** (USD weak), second half **+3.9 bps** (USD strong) — a payment/tax cycle.
   Wednesday is the softest weekday.
3. **Volume and calendar are additive.** Net Sharpe (full / 2019+):
   volume-only **1.40 / 2.03**, calendar-only **1.56 / 2.02**, combined **2.43 / 3.16**.
4. **It is NOT the trend.** Static short-USD nets only **0.35**; the calendar model nets far
   more (see the ranking below). The walk-forward model predicts next-day direction at
   **69.5% out-of-sample** (logit; GBM 69.4%) against a ~52% base rate, permutation p≈0,
   and lagging the features 5 days collapses the accuracy (no look-ahead).

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

Values below are the FRESH `out/dynamics_results.json` figures (full sample, legacy 252
annualisation — apply the divisors in the caveat at the top of this page):

| Rule | Logic | Per year | Sharpe | Trades/yr |
|------|-------|---------:|-------:|----------:|
| Quincena | dom≤15 → short USD; dom>15 → long | **$87k** | **2.05** | 24 |
| Volume | seasonal-vol<1 → long USD; else short | $47k | 1.10 | 78 |
| Precio-ponderado skew | VWAP>simple → long USD; else short | $41k | 0.98 | 102 |
| Combined vote (majority of the 3) | — | $89k | 2.12 | 85 |

**Lesson:** the calendar (quincena) rule is the best *standalone* — cheap to trade and
high Sharpe. The volume/skew signals are directionally real but flip too often (78–102
trades/yr), so on a full-sample view they look respectable while carrying 5–10× the
drawdown of the calendar rule.

> **Correction (this revision).** An earlier version of this table reported the Volume rule
> at $17k/0.37, the skew rule at **−$7k/−0.17**, and the Combined vote at $51k/1.15. Those
> figures did not reproduce: the volume and combined rules were understated ~2.7× and ~1.7×,
> and the **skew rule's sign was inverted** — it is profitable on the full sample, not a
> loser. The prior "ML (conviction-sized)" row ($61k / 2.23 / 42 trades) has no reproducing
> key in any current output and has been removed rather than left unsourced.
>
> **But note the window matters more than the sign.** Split in-sample vs out-of-sample, all
> three flow rules are worthless-to-negative on the first 60% of history (Sharpe −0.26 to
> +0.49) and only shine on the last 40% (2.0–3.7). Their full-sample numbers average a dead
> regime with a live one. See the ranking below.


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

### Anchoring to the REAL payment/tax calendar (recommended implementation)

The fixed day-of-month band is only a proxy for the events that actually move the
colón: companies sell USD to raise colones for the **IVA (D-104) filing, due the 15th
of the month (rolled to the next business day), and the mid-month quincena payroll**.
`payment_calendar.py` snaps each statutory date (IVA/retenciones 15th, CCSS 4th business
day, month-end quincena, quarterly renta) onto MONEX's own trading calendar, so
weekend/holiday rolling is exact. Re-indexing the same price data from raw day-of-month
to **business-days-to-the-deadline** makes the mechanism unmistakable:

| Business days before IVA/quincena deadline | Avg next-day USD move |
|--------------------------------------------|-----------------------|
| 5 → 1 days before | **−9 to −16 bps** (colón strengthens — USD-supply surge) |
| on the deadline | −3 bps |
| after (supply cleared) | **+3 to +10 bps** (USD drifts back up) |

Shorting USD within **6 business days** of the deadline (long otherwise) earns **~$126k/yr
at Sharpe 2.91, max DD −$45k** — it edges the fixed 5–15 window ($125k / 2.88 / −$50k) at
the same ~24 trades/yr, with a broad Sharpe plateau (2.5–2.9 across 5–9-day lookbacks) and
every year positive. Because the anchor moves with the calendar, it both **trades better and
explains the flow** — the edge is specifically the pre-deadline conversion window, not a
fixed date. Chart: `out/q_calendar.png`; the daily sheet (`daily_signal.py`) now keys off
`td_to_iva`. Sources: Hacienda/TRIBU-CR fiscal calendar (IVA D-104 due the 15th) and the
CCSS planilla 4th-business-day rule.

### Enhancing the rule with a dynamic exit (`exits.py`)

The calendar rule is always in the market, but its edge **front-loads** — the colón
strengthens *into* the deadline and reverts after — so a trade tends to peak mid-hold and
give the move back. Overlaying a **trailing stop** on each trade (bank it once the running
P&L gives back a fixed band from its peak, then stay flat until the calendar opens the next
trade) removes that losing tail **without changing any entry**. Both parameters are tuned on
the in-sample 60% only. VWAP-priced, USD 1M/trade, net of 0.65 CRC round-trip:

| Calendar rule | Full /yr | Sharpe | Max DD | OOS /yr | OOS Sharpe |
|---------------|---------:|-------:|-------:|--------:|-----------:|
| No exit (always in market) | $125k | 3.00 | −$48k | $152k | 2.91 |
| **+ trailing-stop exit (80 bps)** | **$128k** | 3.22 | −$45k | **$155k** | 3.08 |
| + trailing (30) & hard floor (60) bps | $126k | **3.91** | **−$26k** | $153k | **4.09** |

- The trailing stop **raises P&L in both windows** (+$3k/yr in-sample and out-of-sample), and
  the improvement is a **broad plateau** — every band from ~30 to ~100 bps beats the no-exit
  rule on Sharpe in both windows, so it is not a curve-fit point.
- Adding a hard floor stop takes the full-sample Sharpe to 3.91 and nearly halves the worst
  drawdown (−$48k → −$26k) while still edging the baseline on P&L.
- It is an **overlay, not a replacement**: the recommendation is still the calendar rule; the
  exit only decides when to bank a trade early. Charts: `out/ex_optimization.png` (the search),
  `out/ex_equity.png` (before/after), `out/ex_episode.png` (the mechanism on one trade).

## Unified ranking — every strategy on one basis

Until now each module carried its own accounting, so the headline numbers were **not
comparable**: `strategies.py` reported GROSS Sharpe (no slippage) and, for the opening-gap
rule, an *intraday* return stream; `backtest.py` priced close-to-close in bps; `quincena.py`
and `dynamics.py` priced VWAP in dollars. Ranking those side by side was meaningless.

`src/rank_strategies.py` re-implements every rule against ONE frame and ONE basis —
VWAP-to-VWAP, $1M/trade, 0.325 CRC per side, annualised on the real 231 sessions/yr — and
ranks by **in-sample** Sharpe (first 60% of history). The out-of-sample column is carried
for honesty and never feeds the ordering. Chart: `out/ranking.png`; data: `out/ranking.json`.

| # | Strategy | Family | IS Sharpe | OOS Sharpe | IS $/yr | IS max DD |
|---|----------|--------|----------:|-----------:|--------:|----------:|
| 1 | Calendar + trail 30 + floor 40 | exit | **3.70** | **3.92** | $100,673 | **−$15,986** |
| 2 | Calendar + trail 30 + floor 60 | exit | 3.65 | 3.92 | $99,468 | −$17,539 |
| 3 | Calendar + trail 80 | exit | 3.43 | 2.95 | $101,041 | −$23,048 |
| 4 | Refined + slow-vol | calendar | 3.30 | 3.31 | $77,223 | −$25,292 |
| 5 | Calendar + slow-vol | calendar | 3.22 | 3.53 | $75,907 | −$23,662 |
| 6 | Refined (5–15) | calendar | 3.19 | 2.67 | $98,865 | −$46,427 |
| 7 | Real calendar (≤6d) | calendar | 3.14 | 2.79 | $97,221 | −$44,687 |
| 8 | Base quincena (≤15) | calendar | 1.96 | 2.08 | $61,996 | −$55,144 |
| 9 | Always long USD | benchmark | 0.83 | −1.44 | $26,052 | −$110,626 |
| 10 | SMA-120 trend | trend | 0.74 | 1.41 | $22,737 | −$65,638 |
| 11 | Combined vote | flow | 0.49 | 3.66 | $16,015 | −$278,033 |
| 12 | Donchian-40 trend | trend | 0.42 | 1.81 | $12,870 | −$127,827 |
| 13 | Volume rule | flow | 0.06 | 2.06 | $2,076 | −$241,581 |
| 14 | Order-flow daily | flow | −0.15 | 2.01 | −$4,798 | −$273,433 |
| 15 | VWAP-skew rule | flow | −0.26 | 2.14 | −$8,295 | −$396,557 |
| 16 | Always short USD | benchmark | −0.84 | 1.43 | −$26,228 | −$194,806 |

**Three things this makes visible that the per-module tables hid:**

1. **The calendar family owns the top of the table and the flow family owns the bottom** —
   in-sample. Every flow rule (order-flow, volume, skew, combined vote) is worthless or
   negative on 2014–2021 and only works on 2021–2026. Their published full-sample numbers
   average a dead regime with a live one, which is why they looked mediocre-but-real rather
   than regime-contingent.
2. **The flow rules carry catastrophic drawdowns** — $240k–$400k in-sample against the
   calendar rule's $16k–$55k, at 80–97 round trips/yr. Even where their OOS Sharpe looks
   excellent (Combined vote 3.66), the path is not something a desk would fund.
3. **Sanity check on the harness:** "Always long USD" (IS +0.83 / OOS −1.44) and "Always
   short USD" (IS −0.84 / OOS +1.43) are near-exact mirrors, as two opposite always-on
   positions must be. That the mirror holds to 0.01 confirms the P&L and cost accounting.

An honest caveat on the ranking metric: because the in-sample window (2014–2021) spans the
pegged/low-vol regime and the out-of-sample window (2021–2026) spans the float, **in-sample
selection structurally penalises any regime-contingent signal.** The calendar rule wins partly
*because* it is the one edge that works in both. That is a real virtue, but it is not the same
claim as "the flow signals don't work."

## Exit methodology — stop-loss helps, take-profit hurts

`src/exit_lab.py` extends `exits.py` into a four-mechanism exit engine layered on the
unchanged calendar entry: **trailing stop**, **hard stop-loss**, **take-profit**, and a
**time stop**, each testable alone or jointly, in absolute bps or in multiples of trailing
realised volatility. All parameters are chosen on the in-sample 60% only.

**Marginal value of each mechanism alone** (baseline no-exit: IS 3.14 → OOS 2.79):

| Mechanism | Best setting | IS Sharpe | OOS Sharpe | Verdict |
|-----------|--------------|----------:|-----------:|---------|
| Trailing stop | 30 bps | 3.57 | 3.06 | **helps** |
| Hard stop-loss | 40 bps | 3.26 | 3.58 | **helps, more OOS than IS** |
| Take-profit | 160 bps | 3.37 | 2.38 | **HURTS out-of-sample** |
| Time stop | 12 sessions | 3.00 | 2.94 | no value |

**Take-profit is the clear negative result and worth stating plainly.** Tightening it
degrades both windows monotonically — OOS Sharpe runs 3.92 (none) → 3.50 (160 bps) →
3.04 (80 bps). The reason is mechanical: the calendar edge is a *drift* that accrues while
the position is held into the deadline, so capping the winner truncates exactly the move the
strategy exists to capture. When the joint grid appears to "select" a 320 bps target it is
switching the mechanism off, not using it.

**Recommended overlay: trailing 30 bps + hard stop 40 bps, no take-profit.**

| | IS Sharpe | OOS Sharpe | OOS max DD | OOS total P&L |
|---|----------:|-----------:|-----------:|--------------:|
| Calendar, no exit | 3.14 | 2.79 | −$47,692 | $644,112 |
| **+ trail 30 / floor 40** | **3.70** | **3.92** | **−$26,192** | $616,768 |

**This is a risk improvement, not an alpha improvement — do not sell it as more money.**
A paired t-test on the daily out-of-sample P&L difference gives t = −0.35, **p = 0.73**: total
P&L is statistically unchanged (indeed nominally $27k lower). What the overlay buys is a
**45% cut in max drawdown** and **+1.13 out-of-sample Sharpe**. The value is that a smoother
path can be sized up, not that the same notional earns more.

Overfitting controls, all passed:
- **Held-out confirmation:** OOS Sharpe (3.92) ≥ IS Sharpe (3.70) — the overlay does not decay.
- **Plateau, not a spike:** the 18 parameter sets adjacent to the winner have a median
  in-sample Sharpe at 96% of the peak.
- **Selection survives:** corr(IS, OOS) across all 391 grid cells is **+0.82** — in-sample
  ranking genuinely predicts out-of-sample performance for this parameter family.
- **Multiple-testing haircut:** deflated Sharpe (Bailey/López de Prado) over 391 trials gives
  P(true Sharpe > 0) ≈ 1.0.
- **Selection rule:** among sets within one standard error of the peak Sharpe (SE = 0.39, 113
  sets tie), take the smallest in-sample drawdown, then the fewest armed mechanisms. This is
  what drops the inert take-profit. Drawdown is used as the second criterion deliberately —
  a stop-loss buys tail control, and Sharpe barely prices tails.

**Volatility-scaled bands did not help.** Expressing the same mechanisms in multiples of
20-session realised vol — motivated by the regime shift, median daily vol 13.3 bps in-sample
vs 23.4 bps out-of-sample — scores *worse* than fixed bps in both windows (IS 3.47 / OOS 3.21
vs 3.70 / 3.92). Reported as a negative result rather than dropped.

## Público vs privado volume — NOT possible with this file

The MONEX export contains only *aggregate* traded volume. Splitting público (BCCR /
sector público) vs privado flow needs a different BCCR dataset (sector-level MONEX /
ventanilla breakdown). Provide that and the mid-month supply surge can be attributed
to a specific sector.
