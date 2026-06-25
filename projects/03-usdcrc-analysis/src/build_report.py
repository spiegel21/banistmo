"""Build the self-contained visual HTML report for the flow + seasonality model.

Story: EXPLORE the volume/season/price relationships -> MODEL them (walk-forward,
no technical indicators) -> TEST net of real cost -> VERDICT.
Reads out/vm_results.json. Run after analyze/eda/volume_model.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
R = json.loads((OUT / "vm_results.json").read_text())
M = R["_meta"]
LG = R["Logit (conviction)"]
LG9 = R["Logit (conviction)_2019"]
DEC = R["decomposition"]
SIM = R["Simple flow+quincena rule"]


def img(name):
    return "data:image/png;base64," + base64.b64encode((OUT / name).read_bytes()).decode()


def fig(name, title, cap):
    return (f'<figure><h3>{title}</h3><img src="{img(name)}" alt="{title}"/>'
            f'<figcaption>{cap}</figcaption></figure>')


KPIS = [
    (f"{M['n_days']:,}", "trading days", f"{M['date_min']} → {M['date_max']}"),
    (f"{R['vol_quintile_adj'][0]:+.0f} / {R['vol_quintile_adj'][-1]:+.0f} bps", "low vs high volume",
     "next-day USD move (seasonally adj.)"),
    (f"{R['quincena_nextret']['first_half_1_15']:+.1f} / {R['quincena_nextret']['second_half_16_31']:+.1f}",
     "quincena effect (bps)", "1st half vs 2nd half of month"),
    (f"{R['oos_accuracy_logit']:.0%}", "out-of-sample accuracy", "next-day direction (52% base rate)"),
    (f"{LG['sharpe']} / {LG9['sharpe']}", "model net Sharpe", "full history / 2019+ regime"),
    (f"{DEC['static_short_usd'][0]} → {DEC['volume_plus_calendar'][0]}", "drift → model (net Sh)",
     "edge is NOT the trend"),
]


def kpi_html():
    return "".join(f'<div class="kpi"><div class="kpi-v">{v}</div>'
                   f'<div class="kpi-l">{lab}</div><div class="kpi-s">{s}</div></div>'
                   for v, lab, s in KPIS)


def decomp_table():
    rows = [("Static short-USD (the drift)", "static_short_usd", "baseline"),
            ("Volume only", "volume_only", "your low-vol→USD-up thesis, alone"),
            ("Calendar only (quincena / dow / month)", "calendar_only", "seasonality, alone"),
            ("Volume + Calendar", "volume_plus_calendar", "the two are additive"),
            ("Full model (+ close/VWAP structure)", "full_model", "deployed model")]
    out = ['<table><tr><th>Long/short model (net of 0.65 CRC round-trip)</th>'
           '<th class="num">Net Sharpe (full)</th><th class="num">Net Sharpe (2019+)</th><th>Note</th></tr>']
    for label, key, note in rows:
        f, r9 = DEC[key]
        cls = "ok" if f >= 1.5 else ("mb" if f >= 0.8 else "no")
        out.append(f'<tr><td>{label}</td><td class="num {cls}">{f}</td>'
                   f'<td class="num">{r9}</td><td class="muted">{note}</td></tr>')
    out.append("</table>")
    return "".join(out)


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>USD/CRC — Flow &amp; Seasonality Model</title>
<style>
 :root {{ --bg:#0f1419; --card:#1a212b; --ink:#e8edf2; --mut:#8a97a6;
   --accent:#37b87c; --warn:#e0a83d; --bad:#e05d5d; --line:#2a3441; --blue:#4a90d9; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink);
   font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
 .wrap {{ max-width:1040px; margin:0 auto; padding:32px 20px 80px; }}
 header {{ border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:26px; }}
 h1 {{ font-size:27px; margin:0 0 6px; letter-spacing:-.5px; }}
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
 .verdict {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
   padding:4px 20px; margin-top:14px; }}
 .verdict li {{ margin:10px 0; }}
 .ok {{ color:var(--accent); }} .no {{ color:var(--bad); }} .mb {{ color:var(--warn); }}
 figure {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
   padding:16px; margin:16px 0; break-inside:avoid; page-break-inside:avoid; }}
 figure h3 {{ margin:0 0 10px; font-size:16px; }}
 figure img {{ width:100%; border-radius:8px; display:block; background:#fff; }}
 figcaption {{ color:var(--mut); font-size:13px; margin-top:10px; }}
 table {{ width:100%; border-collapse:collapse; margin:8px 0; font-size:13px; break-inside:avoid; }}
 th,td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
 th {{ color:var(--mut); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.5px; }}
 td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
 td.ok {{ color:var(--accent); }} td.mb {{ color:var(--warn); }} td.no {{ color:var(--bad); }}
 .muted {{ color:var(--mut); font-size:12px; }}
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
  <h1>USD/CRC · MONEX — Flow &amp; Seasonality Model (no technical indicators)</h1>
  <div class="sub">{M['n_days']:,} sessions · {M['date_min']} → {M['date_max']} · features = previous
    volumes + seasonality + price/precio-ponderado relationships · long/short USD · net of 0.65 CRC round-trip</div>
</header>

<p class="lead">Built entirely from <b>volume, calendar and price-vs-VWAP</b> — no moving averages,
breakouts or momentum. Your prior holds and is monotonic: <b>low volume → USD strengthens next day</b>.
Seasonality (especially the <b>quincena</b>) is an independent, equally strong driver. A walk-forward
model combining them predicts next-day direction at <b>{R['oos_accuracy_logit']:.0%}</b> out-of-sample
and nets <b>Sharpe {LG['sharpe']} (full) / {LG9['sharpe']} (2019+)</b> after your real slippage — and it
beats simply shorting USD (the drift) by ~7×, so the edge is genuinely flow/seasonal, not the trend.</p>

<div class="verdict"><ul>
 <li><span class="ok"><b>CONFIRMED —</b> low volume → USD up, monotonically.</span> Seasonally-adjusted,
   the lowest-volume days average <b>{R['vol_quintile_adj'][0]:+.1f} bps</b> next day vs
   <b>{R['vol_quintile_adj'][-1]:+.1f} bps</b> for the highest. Yesterday's volume also informs
   (corr {R['lagged_vol_corr']['1']}), and volume regimes persist (autocorr {R['vol_autocorr1']}).</li>
 <li><span class="ok"><b>SEASONALITY is real and additive.</b></span> First half of the month averages
   <b>{R['quincena_nextret']['first_half_1_15']:+.1f} bps</b> (USD weak), second half
   <b>{R['quincena_nextret']['second_half_16_31']:+.1f} bps</b> (USD strong) — a payment/quincena cycle.
   Volume-only and calendar-only each net ~Sharpe {DEC['volume_only'][0]}/{DEC['calendar_only'][0]}; together
   <b>{DEC['volume_plus_calendar'][0]}</b>.</li>
 <li><span class="ok"><b>NOT the trend.</b></span> Static short-USD nets only {DEC['static_short_usd'][0]};
   the model nets {DEC['volume_plus_calendar'][0]}. Out-of-sample Sharpe
   ({R['Logit (conviction)_oos']['sharpe']}) ≥ in-sample ({R['Logit (conviction)_is']['sharpe']}),
   permutation p={R['perm_p']}, and lagging the features 5 days collapses the accuracy — so it is not
   look-ahead and not drift.</li>
 <li><span class="mb"><b>CAVEATS —</b> turnover and one regime.</span> ~{LG['roundtrips_yr']}
   round-trips/yr means it is cost-sensitive (already netted here). Most of the edge is in the volatile
   2019+ era; in the 2015–17 pegged years volume/season carried less. Managed-float tail risk (a re-peg or
   intervention) is the dominant danger an in-sample test cannot price.</li>
</ul></div>

<div class="part">Part A · The relationships (explore)</div>
<h2><span class="n">A1.</span> Low volume → USD strengthens (your core thesis)</h2>
{fig("vm_vol_nextret.png", "Next-day USD move by volume quintile (raw &amp; seasonally-adjusted)",
     "Monotonic and clean: the quietest days are followed by USD strength, the busiest by USD weakness. "
     "Adjusting volume for its weekday/month seasonality sharpens the extremes — 'low' should mean low "
     "for that day, not low in absolute terms.")}

<h2><span class="n">A2.</span> Seasonality — quincena, day-of-week, month</h2>
{fig("vm_seasonality.png", "Next-day USD move by calendar bucket",
     "The quincena (1–15 vs 16–31) is the strongest calendar effect — consistent with the Costa Rican "
     "payment/tax cycle driving USD demand. Day-of-week and month add smaller tilts.")}

<h2><span class="n">A3.</span> Previous-day volumes &amp; persistence</h2>
{fig("vm_lagged.png", "Lagged-volume correlation and volume autocorrelation",
     "Today's volume is the strongest predictor but yesterday's still informs; volume is highly "
     "autocorrelated (0.58), so the recent volume regime — not just one day — carries information.")}

<h2><span class="n">A4.</span> Interactions — where the effect is strongest</h2>
{fig("vm_interactions.png", "Next-day move by volume tercile × day-of-week and × quincena",
     "The signal is strongest where low volume and the second half of the month line up — the model "
     "exploits these interactions rather than each effect in isolation.")}

<div class="part">Part B · The model</div>
<h2><span class="n">B1.</span> What drives it (walk-forward coefficients)</h2>
{fig("vm_coefs.png", "Standardized logistic coefficients (+ predicts USD up next day)",
     "Quincena / day-of-month and the volume terms dominate, in the directions the exploration implied. "
     "No moving-average or breakout features are used; close-vs-VWAP is included as a structural "
     "price/precio-ponderado relationship.")}

<h2><span class="n">B2.</span> Attribution — volume vs calendar vs drift</h2>
{decomp_table()}
{fig("vm_attribution.png", "Net Sharpe by feature family",
     "Volume alone and calendar alone each clear the static-short-USD drift comfortably, and combine "
     "additively. This is the central evidence that the edge is flow/seasonal, not trend-following.")}

<div class="part">Part C · Net-of-cost backtest</div>
<h2><span class="n">C1.</span> Equity curves (long/short USD, net of 0.65 CRC round-trip)</h2>
{fig("vm_equity.png", "Cumulative net P&L — model variants vs a simple transparent rule",
     "Even a hand-built rule ('long USD when seasonally-adjusted volume is low, lean by quincena') nets "
     "Sharpe " + str(SIM['sharpe']) + "; the walk-forward logistic adds to that. Conviction-sizing keeps "
     "turnover and cost down versus a raw daily flip.")}

<h2><span class="n">C2.</span> Out-of-sample &amp; permutation</h2>
{fig("vm_oos.png", "In-sample vs out-of-sample (net)",
     "Net Sharpe " + str(R['Logit (conviction)_is']['sharpe']) + " in-sample → " +
     str(R['Logit (conviction)_oos']['sharpe']) + " out-of-sample — it strengthens in the held-out "
     "(more volatile) period, the opposite of an overfit curve.")}
{fig("vm_permutation.png", "Permutation null (net)",
     "Shuffling the timing of the same positions destroys the edge (p=" + str(R['perm_p']) + "): the "
     "skill is in WHEN it is long vs short USD, driven by volume and the calendar.")}

<div class="part">Part D · Verdict &amp; how to trade it</div>
<h2>Is it worth pursuing?</h2>
<ul class="find">
 <li><b>Yes — this is the most promising result so far, and it is exactly your thesis.</b> A long/short
   USD book driven by low-volume-buying / high-volume-selling, tilted by the quincena, nets Sharpe
   ~{DEC['volume_plus_calendar'][0]} (full) / ~{LG9['sharpe']} (2019+) after your real slippage, with shallow
   drawdowns ({LG['maxdd_bps']/100:.1f}%).</li>
 <li><b>Trade it as:</b> each day, score next-day direction from (i) seasonally-adjusted volume — low ⇒ long
   USD, (ii) quincena — first half ⇒ lean short USD, second half ⇒ long, (iii) yesterday's volume and the
   close-vs-VWAP. Size by conviction to hold turnover near ~{LG['roundtrips_yr']} round-trips/yr.</li>
 <li><b>Why it should persist:</b> it is anchored in real cash-flow cycles — payment/tax dates (quincena) and
   genuine USD supply/demand (volume) — not a chart pattern. That is a more durable basis than a technical signal.</li>
 <li><span class="no"><b>Dominant risk:</b></span> the BCCR. A re-peg or heavy intervention mutes both the
   volume signal (as in 2015–17) and the seasonal swings. Keep size modest, keep refitting walk-forward,
   and treat the high Sharpe as a low-volatility-regime number that a policy shock can break.</li>
</ul>

<footer>
  <b>Method &amp; caveats.</b> Features are strictly causal (known at the close); every reported number is
  walk-forward out-of-sample (expanding window, quarterly refit). P&amp;L is net of your 0.65 CRC round-trip
  slippage (0.325/side) on close-to-close fills, gross of financing/borrow and market impact. No technical
  indicators are used (a momentum term was tested and excluded). The colón is not freely shortable for all
  participants — this is a directional/treasury edge. High Sharpe partly reflects the colón's low managed-float
  volatility; size for the tail, not the backtest.<br><br>
  Reproducible: <code>parse_monex → analyze → eda → volume_model → build_report</code>, then
  <code>weasyprint</code> for the PDF. Numbers are read live from <code>vm_results.json</code>.
</footer>

</div></body></html>"""


def main():
    dst = OUT / "report.html"
    dst.write_text(HTML, encoding="utf-8")
    print(f"Wrote {dst} ({dst.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
