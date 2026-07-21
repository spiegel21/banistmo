# USD/CRC (MONEX) — Calendar (quincena) strategy: findings & recommendation

A flow-and-seasonality trading model for the USD/CRC wholesale FX market (BCCR MONEX),
built from a single BCCR price/volume export plus two BCCR official-flow exports. No
technical indicators — no moving averages, breakouts, or momentum.

> **One accounting basis, everywhere in this document.** Every figure is priced
> **VWAP-to-VWAP** next session, **USD 1,000,000** per trade, **net of 0.65 CRC
> round-trip** slippage (0.325/side), and annualised on the **231 sessions/yr** MONEX
> actually trades (2,663 sessions over 11.53 calendar years) — *not* the 252
> equity-market convention. All modules read this basis from one file, `src/basis.py`,
> so every table here and in `out/report.html` agrees by construction. The `out/*.json`
> outputs are the source of truth; `tests/` reconciles every headline number below
> against them.

**Data:** 2,663 trading sessions, **2014-12-08 → 2026-06-19**. In-sample = first 60%
(through 2021-11-17); out-of-sample (OOS) = last 40% (from **2021-11-18**). Selection is
always on in-sample; OOS is reported as honest confirmation and never used to choose.

---

## 1. Recommendation (headline)

**Trade Costa Rica's mid-month payment calendar in USD/CRC.**

1. **Entry (the alpha).** **SHORT USD** within **6 business days** of the IVA (D-104) /
   mid-month quincena deadline — the 15th, rolled to the next business day — and **LONG
   USD** the rest of the month. Firms sell USD to raise colones for the 15th tax filing
   and payroll, so the colón strengthens *into* the deadline and reverts after.
2. **Sizing (the default risk control).** Hold **half size** whenever a slow 20-day
   volume regime disagrees with the trade. This never changes the position's sign, only
   its magnitude, and it removes the fattest losing trades.
3. **Exit overlay (recommended, optional).** A **trailing 30 bps + hard floor 40 bps**
   stop on each trade. It tops the unified ranking and roughly halves drawdown. It is a
   **risk** improvement, not more alpha (see §5).
4. **Further trim (optional).** Cut long-USD size when BCCR reserves are rising (§6).

The calendar entry is the only edge in the whole study that works in **both** the pegged
2014–2021 regime and the 2021–2026 float. Everything else — daily order-flow, volume, and
skew rules — is regime-contingent and carries catastrophic drawdowns (§4).

### What each layer earns (VWAP-priced, $1M/trade, 231 sess/yr)

| Configuration | IS Sharpe | OOS Sharpe | IS $/yr | Max DD | Turnover |
|---|--:|--:|--:|--:|--:|
| Calendar entry, flat size (`Real calendar ≤6 bd`) | 3.14 | 2.79 | $97,221 | −$44,687 | ~22 rt/yr |
| **+ slow-volume trim** (`Calendar + slow-vol`) | 3.22 | 3.53 | $75,907 | −$23,662 | ~18 rt/yr |
| **+ trail 30 / floor 40 exit** (`top of the ranking`) | **3.70** | **3.92** | $100,673 | **−$15,986** | ~22 rt/yr |
| + reserve-regime long-trim (full-sample) | 3.55 | **4.09** | — | −$22,226 | ~22 rt/yr |

The always-in-market recommendation is **Calendar + slow-vol** (no timing overlay to
manage); the highest risk-adjusted configuration is the same entry **with the trail
30 / floor 40 exit**. They are complementary layers on one entry, not rival strategies.

---

## 2. The unified ranking — every strategy on one basis

`src/rank_strategies.py` re-prices **every** rule through a single code path (VWAP-to-VWAP,
$1M/trade, net cost, 231 sess/yr) and ranks by **in-sample** Sharpe. This is the master
cross-strategy comparison; the per-module tables elsewhere add mechanism and context but
this table is authoritative. Chart: `out/ranking.png`; data: `out/ranking.json`.

| # | Strategy | Family | IS Sharpe | OOS Sharpe | IS $/yr | IS max DD |
|--:|---|---|--:|--:|--:|--:|
| 1 | Calendar + trail 30 + floor 40 | exit | **3.70** | **3.92** | $100,673 | **−$15,986** |
| 2 | Calendar + trail 30 + floor 60 | exit | 3.65 | 3.92 | $99,468 | −$17,539 |
| 3 | Calendar + trail 80 | exit | 3.43 | 2.95 | $101,041 | −$23,048 |
| 4 | Refined (5–15) + slow-vol | calendar | 3.30 | 3.31 | $77,223 | −$25,292 |
| 5 | Calendar + slow-vol | calendar | 3.22 | 3.53 | $75,907 | −$23,662 |
| 6 | Refined (5–15) | calendar | 3.19 | 2.67 | $98,865 | −$46,427 |
| 7 | Real calendar (≤6 bd) | calendar | 3.14 | 2.79 | $97,221 | −$44,687 |
| 8 | Base quincena (≤15) | calendar | 1.96 | 2.08 | $61,996 | −$55,144 |
| 9 | Always long USD | benchmark | 0.83 | −1.44 | $26,052 | −$110,626 |
| 10 | SMA-120 trend | trend | 0.74 | 1.41 | $22,737 | −$65,638 |
| 11 | Combined vote | flow | 0.49 | 3.66 | $16,015 | −$278,033 |
| 12 | Donchian-40 trend | trend | 0.42 | 1.81 | $12,870 | −$127,827 |
| 13 | Volume rule | flow | 0.06 | 2.06 | $2,076 | −$241,581 |
| 14 | Order-flow daily | flow | −0.15 | 2.01 | −$4,798 | −$273,433 |
| 15 | VWAP-skew rule | flow | −0.26 | 2.14 | −$8,295 | −$396,557 |
| 16 | Always short USD | benchmark | −0.84 | 1.43 | −$26,228 | −$194,806 |

**Three things the ranking makes plain.**
1. **The calendar family owns the top; the flow family owns the bottom** — in-sample.
   Every flow rule (order-flow, volume, skew, combined vote) is worthless-to-negative on
   2014–2021 and only fires on 2021–2026, so its full-sample average blends a dead regime
   with a live one.
2. **The flow rules carry catastrophic drawdowns** — $240k–$400k in-sample vs the calendar
   rule's $16k–$55k, at 80–97 round trips/yr. Even where a flow rule's OOS Sharpe looks
   excellent (Combined vote 3.66), the path is not fundable.
3. **Harness sanity check.** "Always long USD" (IS +0.83 / OOS −1.44) and "Always short
   USD" (IS −0.84 / OOS +1.43) are near-exact mirrors, as two opposite always-on positions
   must be — confirming the P&L and cost accounting.

**Honest caveat on the metric.** Because the in-sample window (2014–2021) is the pegged,
low-vol regime and OOS (2021–2026) is the float, in-sample selection *structurally* favours
whatever works in both. The calendar rule wins partly *because* it is that edge. That is a
real virtue, but it is not the same claim as "the flow signals never work."

---

## 3. Why it works (mechanism)

**The quincena is a cash-flow cycle.** Re-indexing the price series from raw day-of-month
to **business-days-to-the-IVA/quincena-deadline** makes the mechanism unmistakable:

| Business days before the deadline | Avg next-day USD move |
|---|---|
| 5 → 1 days before | **−9 to −16 bps** (colón strengthens — USD-supply surge) |
| on the deadline | −3 bps |
| after (supply cleared) | **+3 to +10 bps** (USD drifts back up) |

The anchor is snapped to MONEX's own trading calendar so weekend/holiday rolling is exact
(`src/payment_calendar.py`). Sources: Hacienda / TRIBU-CR fiscal calendar (IVA D-104 due the
15th) and the CCSS planilla 4th-business-day rule.

**Exporters are the swing USD supply.** On days the colón strengthens the market trades
~28.9M vs ~18.4M on days it weakens: USD sellers (exporters, remittances, FDI) come in
size, USD buyers trickle. High volume = supply present = colón up; low volume = thin market
= USD drifts up next day. That asymmetry is why the **slow-volume trim** works as a size
filter — it cuts trades the supply picture does not confirm.

---

## 4. Robustness & overfitting controls

The rules carry **no fitted parameters** — the ≤6-day anchor, the 20-day volume window, and
the half-size trim are fixed a-priori choices. Even so:

- **Window sensitivity.** Net Sharpe is a **broad plateau** across every choice of the
  short-window start/end, not a lucky point (`out/q_sensitivity.png`).
- **Every year positive.** The refined rule is positive in **all 12 calendar years**,
  including the 2015–17 pegged years where the volume signal was dead.
- **Held-out confirmation.** The slow-vol variants **hold or improve** out of sample
  (Calendar + slow-vol 3.22 → 3.53), while the strong flat-sized entries give the edge
  back (Real calendar 3.14 → 2.79). The slow-vol size trim is the design choice that
  matters most.

---

## 5. Exit methodology — stop-loss helps, take-profit hurts

`src/exit_lab.py` layers a four-mechanism exit engine on the unchanged calendar entry —
**trailing stop, hard stop-loss, take-profit, time stop** — each testable alone or jointly,
in absolute bps or in multiples of trailing realised vol. All parameters are chosen on the
in-sample 60% only.

**Marginal value of each mechanism alone** (baseline no-exit: IS 3.14 → OOS 2.79):

| Mechanism | Best setting | IS Sharpe | OOS Sharpe | Verdict |
|---|---|--:|--:|---|
| Trailing stop | 30 bps | 3.57 | 3.06 | **helps** |
| Hard stop-loss | 40 bps | 3.26 | 3.58 | **helps — more OOS than IS** |
| Take-profit | 160 bps | 3.37 | 2.38 | **HURTS out-of-sample** |
| Time stop | 12 sessions | 3.00 | 2.94 | no value |

**Take-profit is the clear negative result, stated plainly.** Tightening it degrades both
windows monotonically — OOS Sharpe runs 3.92 (none) → 3.50 (160 bps) → 3.04 (80 bps).
Mechanically, the calendar edge is a *drift* that accrues while the position is held into
the deadline, so capping the winner truncates exactly the move the strategy exists to
capture.

**Recommended overlay: trailing 30 bps + hard floor 40 bps, no take-profit.**

| | IS Sharpe | OOS Sharpe | OOS max DD | OOS total P&L |
|---|--:|--:|--:|--:|
| Calendar, no exit | 3.14 | 2.79 | −$47,692 | $644,112 |
| **+ trail 30 / floor 40** | **3.70** | **3.92** | **−$26,192** | $616,768 |

**This is a risk improvement, not an alpha improvement — do not sell it as more money.**
A paired t-test on the daily OOS P&L difference gives **p = 0.73**: total P&L is
statistically unchanged (indeed nominally ~$27k lower). What the overlay buys is a **~45%
cut in max drawdown** and **+1.13 OOS Sharpe** — a smoother path that can be sized up.

Overfitting controls, all passed:
- **Held-out confirmation:** OOS Sharpe (3.92) ≥ IS Sharpe (3.70) — no decay.
- **Selection survives:** corr(IS, OOS) across all grid cells is **+0.82** — in-sample
  ranking predicts OOS for this parameter family.
- **Multiple-testing haircut:** deflated Sharpe (Bailey / López de Prado) over the grid
  gives P(true Sharpe > 0) ≈ 1.0.
- **Selection rule:** among sets within one standard error of the peak Sharpe, take the
  smallest in-sample drawdown, then the fewest armed mechanisms — this is what drops the
  inert take-profit.
- **Volatility-scaled bands did *not* help** — reported as a negative result, not dropped.

---

## 6. BCCR official flow & reserves — mechanism, and one tradeable overlay

Two more BCCR exports let us see the official hand behind MONEX: FX **intervention**
(CodCuadro 1587) and month-end **net reserves** (CodCuadro 8). `src/parse_bccr.py` parses
them; `src/intervention.py` folds them into the calendar strategy.

- **The official footprint is large.** Official flow — the BCCR plus the non-bank public
  sector (RECOPE et al.) routed through MONEX — is a **median ~52%** of daily MONEX volume,
  present on **~89%** of sessions. So MONEX volume is not a clean read of private
  supply/demand; the *differential* (total − official) is the private flow.
- **It explains the calendar reversion.** Official USD demand **peaks at the IVA
  deadline, exactly where the colón turns** (contemporaneous corr **−0.25**; top-quintile
  official-buying days average −13.6 bps next-day vs +6.6 for the bottom). Heavy official
  demand doesn't push USD up — it **marks the turn** where private mid-month supply
  overwhelms. But the signal **decays to −0.08 when lagged one session** and depends on
  same-day disclosure, so it stays a *mechanism*, **not a live entry**.
- **The tradeable win is the reserves.** Trimming long-USD size when reserves are **rising**
  (a colón-supportive regime — strictly causal, a lagged monthly figure) lifts OOS Sharpe
  **3.92 → 4.09** and cuts max drawdown **−$26k → −$22k**. It is **not just "trim longs"**:
  a blind control that halves *every* long reaches a similar OOS Sharpe, but the
  reserve-selective trim earns **+$56k more out-of-sample** (paired **t = 3.31, p = 0.001**)
  at the same risk reduction — the reserve regime picks *which* longs to cut.

Like the exit overlay, these are **risk** improvements, not new alpha. Charts: `out/iv_*.png`.

---

## 7. Risks & limitations

- **Re-peg / heavy intervention is the dominant, unpriceable risk.** The BCCR is already
  ~half of MONEX and could mute both the calendar and the flow edges at will. The reserve
  trim is a partial hedge, not an escape.
- **The float regime is thinly sampled.** The 11.5-year sample is real, but the *float*
  that carries most of the edge is only the last ~4.5 years.
- **Execution assumptions.** Net of slippage on VWAP fills; gross of financing/borrow and
  market impact. The colón is not freely shortable for all participants — this is a
  directional/treasury edge, not a market-neutral one.
- **Turnover is cost-sensitive** (~18–22 round trips/yr), already netted at 0.65 CRC
  round-trip.

---

## 8. Reproduce

```bash
cd projects/03-usdcrc-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/parse_monex.py && python src/parse_bccr.py
for m in analyze eda strategies backtest backtest_vwap volume_model \
        dynamics quincena exits rank_strategies exit_lab intervention daily_signal; do
  python src/$m.py
done
python src/build_report.py
python -c "from weasyprint import HTML; HTML('out/report.html').write_pdf('out/report.pdf')"
python -m pytest tests/ -q          # reconcile the report against the JSON outputs
```

The daily position sheet is `python src/daily_signal.py [YYYY-MM-DD]`; the practical trading
calendar is `out/quincena_trading_calendar.csv`. See **README.md** for the module map.
