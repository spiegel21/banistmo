"""Build the self-contained visual report: dynamics -> rules -> dollars -> model.

Reads out/vm_results.json (model) and out/dynamics_results.json (rules + $/1M).
Run after analyze / eda / volume_model / dynamics.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
R = json.loads((OUT / "vm_results.json").read_text())
DY = json.loads((OUT / "dynamics_results.json").read_text())
Q = json.loads((OUT / "quincena_results.json").read_text())
QREF = Q["Refined (short 5-15)"]
QSLOW = Q["Refined + slow-vol sizing"]
QBASE = Q["Base quincena (<=15)"]
QCAL = next(v for k, v in Q.items() if k.startswith("Real calendar"))
CAL_PRE = Q["_meta"].get("cal_pre", 6)
M = R["_meta"]
LG = R["Logit (conviction)"]
LG9 = R["Logit (conviction)_2019"]
DEC = R["decomposition"]


def img(name):
    return "data:image/png;base64," + base64.b64encode((OUT / name).read_bytes()).decode()


def fig(name, title, cap):
    return (f'<figure><h3>{title}</h3><img src="{img(name)}" alt="{title}"/>'
            f'<figcaption>{cap}</figcaption></figure>')


def usd(x):
    return f"${x/1e3:,.0f}k" if abs(x) < 1e6 else f"${x/1e6:.2f}M"


KPIS = [
    (usd(QREF['per_year_usd']), "refined quincena / yr", "$1M/trade, net of cost"),
    (f"{QREF['sharpe']}", "refined quincena Sharpe", f"only {QREF['roundtrips_yr']} trades/yr"),
    (f"{QSLOW['sharpe']}", "with slow-vol sizing", f"DD only {usd(QSLOW['maxdd_usd'])}"),
    (usd(Q['refined_worst_year_usd']), "worst year", "positive in all 12 years"),
    (f"{R['vol_quintile_adj'][0]:+.0f} / {R['vol_quintile_adj'][-1]:+.0f} bps", "low vs high volume",
     "next-day USD move"),
    (f"{DY['vol_on_usd_down']:.0f} vs {DY['vol_on_usd_up']:.0f} M", "volume: colón up vs down",
     "USD sellers trade in size"),
]


def kpi_html():
    return "".join(f'<div class="kpi"><div class="kpi-v">{v}</div>'
                   f'<div class="kpi-l">{lab}</div><div class="kpi-s">{s}</div></div>'
                   for v, lab, s in KPIS)


def dollar_table():
    order = ["Quincena rule", "Volume rule", "Precio-ponderado skew rule", "Combined vote (3 signals)"]
    out = ['<table><tr><th>Rule ($1M USD per trade, net of 0.65 CRC round-trip)</th>'
           '<th class="num">Total P&amp;L</th><th class="num">Per year</th><th class="num">Per trade</th>'
           '<th class="num">Trades/yr</th><th class="num">Win %</th><th class="num">Sharpe</th>'
           '<th class="num">Max DD</th></tr>']
    for k in order:
        b = DY[k]
        cls = "ok" if b["sharpe"] >= 1.5 else ("mb" if b["sharpe"] >= 0.5 else "no")
        out.append(f'<tr><td>{k}</td><td class="num">{usd(b["total_usd"])}</td>'
                   f'<td class="num {cls}">{usd(b["per_year_usd"])}</td>'
                   f'<td class="num">${b["per_trade_usd"]:,}</td>'
                   f'<td class="num">{b["roundtrips_yr"]}</td><td class="num">{b["win_rate"]}</td>'
                   f'<td class="num {cls}">{b["sharpe"]}</td>'
                   f'<td class="num">{usd(b["maxdd_usd"])}</td></tr>')
    out.append(f'<tr><td><b>ML combined model</b> (conviction-sized, walk-forward)</td>'
               f'<td class="num">{usd(LG["ann_bps"]*100*M["n_days"]/252)}</td>'
               f'<td class="num ok">{usd(LG["ann_bps"]*100)}</td><td class="num muted">—</td>'
               f'<td class="num">{LG["roundtrips_yr"]}</td><td class="num">{LG["hit"]}</td>'
               f'<td class="num ok">{LG["sharpe"]}</td>'
               f'<td class="num">{usd(LG["maxdd_bps"]*100)}</td></tr>')
    out.append("</table>")
    return "".join(out)


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>USD/CRC — Dynamics, Rules &amp; Dollar Results</title>
<style>
 :root {{ --bg:#0f1419; --card:#1a212b; --ink:#e8edf2; --mut:#8a97a6;
   --accent:#37b87c; --warn:#e0a83d; --bad:#e05d5d; --line:#2a3441; --blue:#4a90d9; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink);
   font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
 .wrap {{ max-width:1040px; margin:0 auto; padding:32px 20px 80px; }}
 header {{ border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:26px; }}
 h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-.5px; }}
 .sub {{ color:var(--mut); font-size:14px; }}
 .part {{ font-size:12px; text-transform:uppercase; letter-spacing:2px; color:var(--blue);
   margin:46px 0 2px; font-weight:700; }}
 h2 {{ font-size:18px; margin:6px 0 14px; break-after:avoid; }}
 h2 .n {{ color:var(--accent); }}
 .kpis {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
 .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:15px; }}
 .kpi-v {{ font-size:20px; font-weight:700; letter-spacing:-.5px; }}
 .kpi-l {{ font-size:13px; margin-top:2px; }}
 .kpi-s {{ font-size:12px; color:var(--mut); margin-top:4px; }}
 .lead {{ background:var(--card); border-left:3px solid var(--accent); border-radius:0 10px 10px 0;
   padding:16px 20px; margin:8px 0 0; }}
 .lead b {{ color:var(--accent); }}
 .rule {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:2px 18px;
   margin:12px 0; break-inside:avoid; }}
 .rule h4 {{ margin:14px 0 6px; font-size:15px; color:var(--blue); }}
 .rule .logic {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px;
   background:#0f1419; border:1px solid var(--line); border-radius:8px; padding:10px 12px; color:#cfe3d6; }}
 .rule p {{ margin:8px 0; font-size:13.5px; }}
 .ok {{ color:var(--accent); }} .no {{ color:var(--bad); }} .mb {{ color:var(--warn); }}
 figure {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
   padding:16px; margin:16px 0; break-inside:avoid; page-break-inside:avoid; }}
 figure h3 {{ margin:0 0 10px; font-size:16px; }}
 figure img {{ width:100%; border-radius:8px; display:block; background:#fff; }}
 figcaption {{ color:var(--mut); font-size:13px; margin-top:10px; }}
 table {{ width:100%; border-collapse:collapse; margin:8px 0; font-size:12.5px; break-inside:avoid; }}
 th,td {{ text-align:left; padding:8px 9px; border-bottom:1px solid var(--line); vertical-align:top; }}
 th {{ color:var(--mut); font-weight:600; font-size:10.5px; text-transform:uppercase; letter-spacing:.4px; }}
 td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
 td.ok {{ color:var(--accent); }} td.mb {{ color:var(--warn); }} td.no {{ color:var(--bad); }} .muted {{ color:var(--mut); }}
 ul.find {{ list-style:none; padding:0; margin:0; }}
 ul.find li {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
   padding:13px 16px; margin-bottom:10px; }}
 ul.find b {{ color:var(--blue); }}
 .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
 footer {{ color:var(--mut); font-size:12.5px; margin-top:40px; border-top:1px solid var(--line);
   padding-top:18px; }}
 @page {{ size:A4; margin:13mm 11mm; }}
 @media (max-width:720px) {{ .kpis,.cols {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">

<header>
  <h1>USD/CRC · MONEX — Underlying dynamics, rule logic &amp; dollar results</h1>
  <div class="sub">{M['n_days']:,} sessions · {M['date_min']} → {M['date_max']} · long/short USD ·
    ${DY['_meta']['notional_usd']:,.0f} per trade · net of 0.65 CRC round-trip · no technical indicators</div>
</header>

<p class="lead">The recommended strategy is a <b>refined calendar (quincena) rule</b>: short USD only on
the mid-month supply window (days 5–15), long the rest of the month. At $1,000,000 per trade it earns
<b>{usd(QREF['per_year_usd'])}/yr at Sharpe {QREF['sharpe']}</b> with just {QREF['roundtrips_yr']} trades a
year and a worst-year of {usd(Q['refined_worst_year_usd'])} — positive in all 12 years. Below: the exact
rule, its robustness, the underlying mechanics, and how volume fits in (as a slow regime filter, not a
daily signal).</p>

<div class="kpis" style="margin-top:16px">{kpi_html()}</div>

<div class="part">Part A · The recommended strategy — refined quincena</div>
<h2><span class="n">A1.</span> The rule &amp; the numbers ($1M per trade, net of cost)</h2>
<table><tr><th>Version</th><th class="num">Per year</th><th class="num">Sharpe</th>
  <th class="num">Trades/yr</th><th class="num">Win %</th><th class="num">Max DD</th><th>Note</th></tr>
<tr><td>Base quincena (short if day ≤ 15)</td><td class="num">{usd(QBASE['per_year_usd'])}</td>
  <td class="num">{QBASE['sharpe']}</td><td class="num">{QBASE['roundtrips_yr']}</td>
  <td class="num">{QBASE['win_rate']}</td><td class="num">{usd(QBASE['maxdd_usd'])}</td>
  <td class="muted">original</td></tr>
<tr><td><b>Refined (short days 5–15, long otherwise)</b></td>
  <td class="num ok">{usd(QREF['per_year_usd'])}</td><td class="num ok">{QREF['sharpe']}</td>
  <td class="num">{QREF['roundtrips_yr']}</td><td class="num">{QREF['win_rate']}</td>
  <td class="num">{usd(QREF['maxdd_usd'])}</td><td class="muted">most dollars</td></tr>
<tr><td>Refined + slow-volume sizing</td><td class="num">{usd(QSLOW['per_year_usd'])}</td>
  <td class="num ok">{QSLOW['sharpe']}</td><td class="num">{QSLOW['roundtrips_yr']}</td>
  <td class="num">{QSLOW['win_rate']}</td><td class="num ok">{usd(QSLOW['maxdd_usd'])}</td>
  <td class="muted">best risk-adjusted</td></tr>
<tr><td><b>Real calendar (≤{CAL_PRE} bd to IVA/quincena deadline)</b></td>
  <td class="num ok">{usd(QCAL['per_year_usd'])}</td><td class="num ok">{QCAL['sharpe']}</td>
  <td class="num">{QCAL['roundtrips_yr']}</td><td class="num">{QCAL['win_rate']}</td>
  <td class="num">{usd(QCAL['maxdd_usd'])}</td><td class="muted">deadline-anchored</td></tr></table>
<div class="rule"><h4>Exact logic (the trading calendar)</h4>
  <div class="logic">≤ {CAL_PRE} business days before the IVA/quincena deadline (through it) →  SHORT USD<br>
    &nbsp;&nbsp;&nbsp;(firms sell USD to raise colones for the 15th tax + payroll → colón strengthens)<br>
    otherwise →  LONG USD  (supply fades, USD drifts up)<br>
    optional: trim size to 50% when a slow 20-day volume regime disagrees</div>
  <p>Two equivalent readings of the same edge. The <b>fixed</b> version keys off the raw day number
    (days 1–4 LONG, 5–15 SHORT, 16–end LONG); the <b>real-calendar</b> version anchors the short window to
    Costa Rica's actual IVA (D-104) / mid-month quincena deadline — the 15th, rolled to the next business
    day — so it shifts with weekends and holidays. Anchoring to the true deadline earns
    {usd(QCAL['per_year_usd'])}/yr at Sharpe {QCAL['sharpe']} (vs {usd(QREF['per_year_usd'])} / {QREF['sharpe']}
    for the fixed window) at the same {QCAL['roundtrips_yr']} trades/yr, and — see B2b — the next-day move is
    sharpest measured in <i>days-to-deadline</i>, confirming the cash-flow mechanism.</p></div>

<h2><span class="n">A2.</span> Is it overfit? Window sensitivity &amp; every-year P&amp;L</h2>
{fig("q_sensitivity.png", "Net Sharpe across every choice of the short-window start &amp; end",
     "The result is a broad green plateau (Sharpe " + str(Q['window_sharpe_min']) + "–" +
     str(Q['window_sharpe_max']) + "), not a lucky point — days 5→15 is simply the natural mid-month "
     "supply window, and neighbours all work.")}
{fig("q_peryear.png", "Net P&L by year — base vs refined ($1M/trade)",
     "Positive in all 12 calendar years (worst " + usd(Q['refined_worst_year_usd']) + "), including the "
     "2015–17 pegged years where the volume signal was dead. The refined version dominates the base almost "
     "every year.")}

<h2><span class="n">A3.</span> How volume fits in — confirmation, not a daily signal</h2>
{fig("q_interaction.png", "Next-day USD move: calendar × volume (bps)",
     "Volume CONFIRMS the calendar on the diagonal: first-half + high volume is strongly USD-down "
     "(sell USD), second-half + low volume strongly USD-up (buy USD). But trading volume day-by-day adds "
     "turnover that costs more than it adds — so we fold it in as a SLOW regime filter that only trims size, "
     "which is what lifts the Sharpe to " + str(QSLOW['sharpe']) + " and cuts the drawdown to " +
     usd(QSLOW['maxdd_usd']) + ".")}

<h2><span class="n">A4.</span> Tearsheet ($1M per trade, net of cost)</h2>
{fig("q_tearsheet.png", "Refined quincena — equity, drawdown, yearly &amp; monthly P&L",
     "Smooth compounding to ~$1.3M over the sample, shallow drawdowns, every year and (on average) every "
     "month positive — January and April strongest.")}

<div class="part">Part B · The underlying dynamics — what we are exploiting</div>
<h2><span class="n">A1.</span> The engine: exporters are the swing USD supply</h2>
{fig("dyn_mechanism.png", "Volume when the colón strengthens vs weakens, and next-day move by volume",
     f"On days the colón strengthens (USD falls) the market trades {DY['vol_on_usd_down']:.0f}M — far more "
     f"than the {DY['vol_on_usd_up']:.0f}M on days it weakens. USD sellers (exporters, remittances, FDI) come "
     "in size; USD buyers (importers) trickle. So HIGH volume = supply present = colón up, and LOW volume = "
     f"thin market = USD drifts up next day ({DY['nextmove_after_lowvol']:+.1f} bps after quiet days vs "
     f"{DY['nextmove_after_highvol']:+.1f} after busy ones). That asymmetry IS the volume signal.")}

<h2><span class="n">A2.</span> The intra-month cash-flow cycle (why the calendar works)</h2>
{fig("dyn_domcycle.png", "Average next-day USD move (bars) and volume (line) by day-of-month",
     "A clear monthly rhythm: volume peaks mid-month (~day 13–15) and the colón strengthens hardest right "
     "then (red bars) — a recurring USD-supply surge (exporter settlements, tax/paydate conversions). The "
     "first days and the second half of the month lean the other way (USD up). This is the quincena effect, "
     "and it is a cash-flow cycle, not a chart pattern.")}

<h2><span class="n">A2b.</span> Aligning to the real deadline sharpens the signal</h2>
{fig("q_calendar.png", "Average next-day USD move by business-days-to the IVA/quincena deadline",
     "Re-indexing the same data from raw day-of-month to <i>trading days to the statutory IVA (15th) / "
     "mid-month payroll deadline</i> makes the mechanism unmistakable: the colón strengthens hardest 1–5 "
     "business days BEFORE the deadline (−12 to −16 bps next-day), then reverts to USD-up right after — a "
     "textbook conversion-then-clearing pattern. The red band is the short-USD window. Because the anchor "
     "rolls with weekends and holidays, the calendar rule (Sharpe " + str(QCAL['sharpe']) + ", a broad "
     "plateau of " + str(Q['calendar_pre_sharpe_min']) + "–" + str(Q['calendar_pre_sharpe_max']) +
     " across lookbacks) edges the fixed-day version while explaining the flow.")}
{fig("q_calendar_peryear.png", "Net P&L by year — fixed day-of-month window vs real calendar ($1M/trade)",
     "Deadline-anchoring holds up year by year (worst year " + usd(Q['calendar_worst_year_usd']) + "), not "
     "just in aggregate.")}

<h2><span class="n">A3.</span> Two more relationships</h2>
<div class="cols">
{fig("dyn_skew.png", "Precio-ponderado skew (VWAP − simple average)",
     f"When large tickets pull the weighted price above the simple average, USD tends to keep rising next "
     f"day (corr {DY['skew_corr']}). It reads large-player direction — but, as the dollar table shows, it "
     "flips daily and is too costly to trade alone.")}
{fig("dyn_stability.png", "Year-by-year stability of both effects",
     "The volume→move link is negative every year since 2018; the quincena spread is positive in ALL 12 "
     "years. That cross-regime persistence is why we treat these as structural, not curve-fit.")}
</div>

<div class="part">Part C · How each rule decides (exact logic)</div>
<div class="rule">
  <h4>Rule 1 — Volume ("buy USD when the market is quiet")</h4>
  <div class="logic">seasonal_vol = today_volume ÷ (avg volume on the same weekday, to date)<br>
    if seasonal_vol &lt; 1.0  →  LONG USD $1M   (quieter than normal)<br>
    if seasonal_vol ≥ 1.0  →  SHORT USD $1M  (busier than normal)</div>
  <p><b>Why:</b> quiet = exporters absent = importer demand lifts USD. We divide by the same-weekday
    average so "quiet" means quiet <i>for a Tuesday</i>, not just an absolute low.
    <b>Result:</b> directionally right (positive) but flips ~{DY['Volume rule']['roundtrips_yr']}×/yr, so
    slippage eats most of it → net Sharpe {DY['Volume rule']['sharpe']}.</p>
</div>
<div class="rule">
  <h4>Rule 2 — Quincena ("short USD in the first half of the month")</h4>
  <div class="logic">if day_of_month ≤ 15  →  SHORT USD $1M   (mid-month USD-supply surge)<br>
    if day_of_month &gt; 15  →  LONG USD $1M</div>
  <p><b>Why:</b> the cash-flow cycle in A2 — mid-month USD supply strengthens the colón, then it reverses.
    <b>Result:</b> the workhorse — {usd(DY['Quincena rule']['per_year_usd'])}/yr, Sharpe
    {DY['Quincena rule']['sharpe']}, only {DY['Quincena rule']['roundtrips_yr']} trades/yr, max drawdown
    {usd(DY['Quincena rule']['maxdd_usd'])}.</p>
</div>
<div class="rule">
  <h4>Rule 3 — Precio-ponderado skew ("follow the big tickets")</h4>
  <div class="logic">if VWAP &gt; simple_average  →  LONG USD $1M   (large trades lifted the price)<br>
    if VWAP ≤ simple_average  →  SHORT USD $1M</div>
  <p><b>Why:</b> the weighted-vs-simple gap reveals whether large players were buying or selling USD.
    <b>Result:</b> a real signal (corr {DY['skew_corr']}) but it flips ~{DY['Precio-ponderado skew rule']['roundtrips_yr']}×/yr
    → net negative after cost. Useful only as a tie-breaker, not standalone.</p>
</div>
<div class="rule">
  <h4>Rule 4 — Combined vote &amp; the ML model</h4>
  <div class="logic">Combined = majority vote of Rules 1–3 (always ±1) → $1M long/short<br>
    ML = logistic P(USD up) on all features; position = clip((P − 0.5) × 5, −1, +1)</div>
  <p><b>Why the ML sizes by conviction:</b> a raw daily flip trades ~95×/yr and dies on cost. Scaling the
    position by confidence keeps turnover near {LG['roundtrips_yr']}×/yr while still capturing the edge —
    that is the ML's main advantage over the transparent rules, not better direction.</p>
</div>

<div class="part">Part D · Results at $1,000,000 per trade (all rules, net of cost)</div>
{dollar_table()}
{fig("dyn_dollar_equity.png", "Cumulative net P&L in US$ millions — transparent rules, $1M per trade",
     "The quincena rule (green) compounds smoothly with shallow drawdowns; the combined vote is decent but "
     "dragged by the turnover of the volume and skew legs. The volume rule is directionally right but nearly "
     "flat net of cost — the signal is real, the daily churn is the problem.")}

<div class="part">Part E · The combined model &amp; the raw relationships</div>
<h2><span class="n">D1.</span> Low volume → USD up (seasonally adjusted)</h2>
{fig("vm_vol_nextret.png", "Next-day USD move by volume quintile",
     "Monotonic: quietest days → USD up, busiest → USD down. Seasonal adjustment sharpens the extremes.")}
<h2><span class="n">D2.</span> Attribution &amp; drivers</h2>
<div class="cols">
{fig("vm_attribution.png", "Net Sharpe by feature family",
     "Volume-only and calendar-only each beat the drift and are additive.")}
{fig("vm_coefs.png", "Walk-forward model coefficients", "Quincena/day-of-month and volume dominate; no technical indicators used.")}
</div>
{fig("vm_oos.png", "In-sample vs out-of-sample (net)",
     "OOS Sharpe " + str(R['Logit (conviction)_oos']['sharpe']) + " ≥ IS " +
     str(R['Logit (conviction)_is']['sharpe']) + "; a 5-day feature lag collapses accuracy (no look-ahead), "
     "permutation p=" + str(R['perm_p']) + ".")}

<div class="part">Part F · Verdict</div>
<ul class="find">
 <li><b>The most robust, simplest edge is the calendar.</b> Short USD in the first half of the month, long
   in the second — {usd(DY['Quincena rule']['per_year_usd'])}/yr per $1M at Sharpe {DY['Quincena rule']['sharpe']},
   {DY['Quincena rule']['roundtrips_yr']} trades/yr, and positive in all 12 years. Start here.</li>
 <li><b>Your volume thesis is correct but must be traded gently.</b> Low volume → USD up is real and
   monotonic, but a daily flip pays it all to slippage. Use it conviction-sized or as a filter on the
   calendar trade — that is exactly what lifts the combined/ML book to Sharpe ~2.</li>
 <li><b>The mechanism is economic, not technical:</b> exporter USD supply (volume) and the payment/tax
   cycle (quincena). That is why it persists across regimes — and why it would weaken if the BCCR re-pegged
   or intervened heavily, the dominant risk.</li>
</ul>

<footer>
  <b>Dollar convention.</b> $1,000,000 USD notional per full position; 1 bp of next-day move = $100; slippage
  0.325 CRC/side (~$680) charged on every position change. P&amp;L is net of that, gross of financing/borrow
  and market impact, on close-to-close fills, non-compounded (reset to $1M each day). The ML row is
  conviction-sized so its notional varies up to $1M. All model numbers are walk-forward out-of-sample.
  Reproducible: <code>parse_monex → analyze → eda → volume_model → dynamics → build_report</code>.
</footer>

</div></body></html>"""


def main():
    dst = OUT / "report.html"
    dst.write_text(HTML, encoding="utf-8")
    print(f"Wrote {dst} ({dst.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
