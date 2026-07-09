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
    (usd(DY['Quincena rule']['per_year_usd']), "quincena rule / yr", "best transparent rule, $1M/trade"),
    (f"{DY['Quincena rule']['sharpe']}", "quincena Sharpe", f"only {DY['Quincena rule']['roundtrips_yr']} trades/yr"),
    (f"{R['vol_quintile_adj'][0]:+.0f} / {R['vol_quintile_adj'][-1]:+.0f} bps", "low vs high volume",
     "next-day USD move"),
    (f"{DY['vol_on_usd_down']:.0f} vs {DY['vol_on_usd_up']:.0f} M", "volume: colón up vs down day",
     "USD sellers trade in size"),
    (f"{R['oos_accuracy_logit']:.0%}", "model OOS accuracy", "combined model, next-day direction"),
    (f"{usd(LG['ann_bps']*100)}/yr", "ML model / yr", f"net Sharpe {LG['sharpe']}, $1M scaling"),
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

<p class="lead">Three things here: <b>why</b> the signals exist (the flow &amp; cash-flow mechanics we are
exploiting), <b>exactly how</b> each non-ML rule decides to be long or short USD, and <b>what it makes in
dollars</b> at $1,000,000 per trade. Headline: the simplest rule — a <b>calendar (quincena) rule</b> —
earns about <b>{usd(DY['Quincena rule']['per_year_usd'])} per year</b> at Sharpe {DY['Quincena rule']['sharpe']}
with only {DY['Quincena rule']['roundtrips_yr']} trades a year; the volume signal is real but flips too often
to survive slippage on its own.</p>

<div class="kpis" style="margin-top:16px">{kpi_html()}</div>

<div class="part">Part A · The underlying dynamics — what we are exploiting</div>
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

<div class="part">Part B · How each rule decides (exact logic)</div>
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

<div class="part">Part C · Results at $1,000,000 per trade (net of cost)</div>
{dollar_table()}
{fig("dyn_dollar_equity.png", "Cumulative net P&L in US$ millions — transparent rules, $1M per trade",
     "The quincena rule (green) compounds smoothly with shallow drawdowns; the combined vote is decent but "
     "dragged by the turnover of the volume and skew legs. The volume rule is directionally right but nearly "
     "flat net of cost — the signal is real, the daily churn is the problem.")}

<div class="part">Part D · The combined model &amp; the raw relationships</div>
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

<div class="part">Part E · Verdict</div>
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
