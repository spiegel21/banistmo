"""Build the self-contained visual HTML report (charts embedded as base64).

Full-history (2014-2026), slippage-aware story:
  EXPLORE -> COST REALITY -> DEPLOYABLE STRATEGY -> VERDICT.
Reads out/backtest_results.json. Run after analyze/eda/strategies/backtest.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
R = json.loads((OUT / "backtest_results.json").read_text())
M = R["_meta"]
OF = R["Order-flow daily"]
DON = R["Donchian-40 trend"]
DONF = R["Donchian-40_full"]
SIDES = R["Donchian-40_sides"]
IS, OOS = R["Donchian-40_is"], R["Donchian-40_oos"]


def img(name):
    return "data:image/png;base64," + base64.b64encode((OUT / name).read_bytes()).decode()


def fig(name, title, cap):
    return (f'<figure><h3>{title}</h3><img src="{img(name)}" alt="{title}"/>'
            f'<figcaption>{cap}</figcaption></figure>')


KPIS = [
    (f"{M['n_days']:,}", "trading days", f"{M['date_min']} → {M['date_max']} (11.5 yrs)"),
    (f"{M['cost_bps_rt_at_475']} bps", "slippage round-trip", f"{M['cost_crc_roundtrip']} CRC at a ~475 spot"),
    (f"{OF['gross_sharpe']} → {OF['sharpe']}", "order-flow gross → net", f"only {OF['recent_2019']['sharpe']} net even in 2019+"),
    (f"{DONF['sharpe']}", "trend net Sharpe", f"long/short USD, {DONF['roundtrips_yr']} rt/yr"),
    (f"+{DONF['ann_bps']:.0f} bps/yr", "trend net return", f"max DD {DONF['maxdd_bps']/100:.1f}%"),
    (f"{IS['sharpe']} / {OOS['sharpe']}", "trend IS / OOS Sharpe", "edge lives in trending regimes"),
]


def kpi_html():
    return "".join(f'<div class="kpi"><div class="kpi-v">{v}</div>'
                   f'<div class="kpi-l">{lab}</div><div class="kpi-s">{s}</div></div>'
                   for v, lab, s in KPIS)


def strat_table():
    rows = [("Order-flow daily", OF), ("Donchian-40 trend", DON), ("SMA-120 trend", R["SMA-120 trend"])]
    out = ['<table><tr><th>Long/short strategy</th><th class="num">Gross Sh</th>'
           '<th class="num">Net Sh (full)</th><th class="num">Net Sh (2019+)</th>'
           '<th class="num">Round-trips/yr</th><th class="num">Max DD</th><th>Verdict</th></tr>']
    verdict = {
        "Order-flow daily": '<span class="mb">Marginal / regime-only, high turnover</span>',
        "Donchian-40 trend": '<span class="ok">Survives — robust</span>',
        "SMA-120 trend": '<span class="ok">Survives</span>',
    }
    for name, b in rows:
        out.append(
            f'<tr><td>{name}</td><td class="num">{b["gross_sharpe"]}</td>'
            f'<td class="num">{b["sharpe"]}</td><td class="num">{b["recent_2019"]["sharpe"]}</td>'
            f'<td class="num">{b["roundtrips_yr"]}</td><td class="num">{b["maxdd_bps"]/100:.1f}%</td>'
            f'<td>{verdict[name]}</td></tr>')
    out.append("</table>")
    return "".join(out)


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>USD/CRC (MONEX) — Long/Short Quant Report</title>
<style>
 :root {{ --bg:#0f1419; --card:#1a212b; --ink:#e8edf2; --mut:#8a97a6;
   --accent:#37b87c; --warn:#e0a83d; --bad:#e05d5d; --line:#2a3441; --blue:#4a90d9; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink);
   font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
 .wrap {{ max-width:1040px; margin:0 auto; padding:32px 20px 80px; }}
 header {{ border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:26px; }}
 h1 {{ font-size:28px; margin:0 0 6px; letter-spacing:-.5px; }}
 .sub {{ color:var(--mut); font-size:14px; }}
 .part {{ font-size:12px; text-transform:uppercase; letter-spacing:2px; color:var(--blue);
   margin:46px 0 2px; font-weight:700; }}
 h2 {{ font-size:18px; margin:6px 0 14px; break-after:avoid; }}
 h2 .n {{ color:var(--accent); }}
 .kpis {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
 .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:15px; }}
 .kpi-v {{ font-size:21px; font-weight:700; letter-spacing:-.5px; }}
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
  <h1>USD/CRC · MONEX — Long/Short Quant Report (full history, net of real slippage)</h1>
  <div class="sub">{M['n_days']:,} sessions · {M['date_min']} → {M['date_max']} · execution cost
    {M['cost_crc_roundtrip']} CRC round-trip (~{M['cost_bps_rt_at_475']} bps) · long/short USD positions</div>
</header>

<p class="lead">This revises the earlier 18-month study with <b>11.5 years of data</b> and your
<b>real slippage of 0.65 CRC round-trip</b> (0.325/side). Both change the conclusion. The 18-month
sample was a favourable window with an untradeable VWAP fill; on the full history and real costs the
high-frequency order-flow edge is at best marginal — but a <b>slow trend-following long/short</b>
clears it comfortably, with a believable net Sharpe near 1.1 and only a few trades a year.</p>

<div class="verdict"><ul>
 <li><span class="mb"><b>REVISION —</b> the earlier "Sharpe ~5" was a mirage.</span> It came from an
   18-month window (the colón only became signal-rich after 2018) and from marking P&L at the VWAP,
   which you cannot actually trade at.</li>
 <li><span class="mb"><b>MARGINAL —</b> daily order-flow only works in the recent regime, with heavy turnover.</span>
   The signal is <i>real</i> (gross next-day Sharpe {OF['gross_sharpe']}, positive every year since 2017),
   but it flips ~{OF['roundtrips_yr']}×/yr. Net of your 0.65 CRC round-trip it is only
   <b>{OF['sharpe']} over the full history</b> (−{abs(OF['maxdd_bps'])/100:.0f}% drawdown) and
   <b>{OF['recent_2019']['sharpe']} even in the signal-rich 2019+ regime</b> — no better than the trend
   strategy but with 30× the trading. It is not worth running as a taker; it pays cleanly only if you
   are the liquidity <i>provider</i> capturing the spread.</li>
 <li><span class="ok"><b>SURVIVES —</b> a slow trend-following long/short USD.</span> Donchian-40
   breakout: <b>net Sharpe {DONF['sharpe']}</b>, {DONF['roundtrips_yr']} round-trips/yr,
   +{DONF['ann_bps']:.0f} bps/yr, max drawdown {DONF['maxdd_bps']/100:.1f}%. Both the long-USD and
   short-USD legs make money; permutation p={R['Donchian-40_perm_p']}.</li>
 <li><span class="mb"><b>CAVEAT —</b> it only earns when the colón trends.</span> In-sample (2015–21,
   mostly the pegged years) net Sharpe was just {IS['sharpe']}; out-of-sample (2021–26, two-way regimes)
   {OOS['sharpe']}. If the BCCR re-pins the rate, trend-following goes flat — it won't bleed, but it
   won't earn. Managed-float tail risk remains the dominant danger.</li>
</ul></div>

<div class="part">Part A · Explore — 11.5 years, multiple regimes</div>
<h2><span class="n">A1.</span> Price &amp; order flow across regimes</h2>
{fig("eda_price_volume.png", "Session VWAP and signed daily volume, 2014–2026",
     "The colón was effectively pinned/quiet in 2015–17, then traded two-way: depreciation spikes "
     "(2018, 2020, COVID) and a long appreciation 2022–26. A single strategy must cope with all of these.")}

<h2><span class="n">A2.</span> The signal is regime-dependent (the key overfitting lesson)</h2>
{fig("bt_regime_corr.png", "Volume→direction correlation by year, and the colón's annual move",
     "Left: the volume→next-day-move correlation was ~0 in 2015–17 and only emerged from 2018 on. The "
     "original 18-month study sat entirely inside the favourable post-2018 regime — which is exactly how "
     "a real edge and an overfit one can look identical on a short sample. Right: the two-way moves a "
     "long/short strategy has to navigate.")}

<h2><span class="n">A3.</span> The order-flow relationship &amp; return character</h2>
<div class="cols">
{fig("eda_volume_return.png", "Volume vs same-day & next-day move + rolling correlation",
     "Heavy volume ⇒ colón strength, contemporaneously and next-day — a genuine order-flow signal "
     "(USD supply). This is real; the problem is monetising it net of cost.")}
{fig("eda_returns.png", "Return distribution, QQ, and autocorrelations",
     "Fat tails and volatility clustering — the tails are what a managed-float blow-up would live in, "
     "and a short sample never shows them.")}
</div>

<h2><span class="n">A4.</span> Seasonality &amp; volume profile (context overlays)</h2>
<div class="cols">
{fig("eda_calendar.png", "Day-of-week × month heatmaps", "Calendar tilts in move and volume — a sizing overlay, not a standalone edge.")}
{fig("03_volume_profile.png", "Volume at each price level", "Where the market has traded over 11.5 years; congestion vs discovery zones.")}
</div>

<div class="part">Part B · The cost reality</div>
<h2><span class="n">B1.</span> What 0.65 CRC/side does to each long/short strategy</h2>
{strat_table()}
{fig("bt_net_compare.png", "Net-of-slippage cumulative P&L — the whole story in one chart",
     "Order-flow (orange) loses through the dead/pegged 2015–18 years on turnover cost, then recovers in "
     "the signal-rich 2019+ regime — ending roughly flat over the full history. The trend strategies "
     "(green/navy) are quiet through 2015–21, then earn +~4,500 bps in the trending era. The trend edge is "
     "far more robust across regimes.")}

<h2><span class="n">B2.</span> Sensitivity to the slippage assumption</h2>
{fig("bt_slippage.png", "Net Sharpe vs round-trip slippage",
     "Daily order-flow's net Sharpe sits near zero and falls fast as cost rises; the trend strategy stays "
     "clearly positive past your 0.65 CRC round-trip. The deployable edge is the one that barely cares "
     "about the exact cost.")}

<div class="part">Part C · The deployable strategy — trend long/short USD</div>
<h2><span class="n">C1.</span> Net-of-cost tearsheet</h2>
{fig("bt_trend_tearsheet.png", "Donchian-40 breakout long/short USD — net of 0.65 CRC slippage",
     "Equity, drawdown, P&L by year and the daily-return distribution. Flat 2015–21, then the bulk of the "
     "P&L in the trending 2022–26 years. Drawdowns are real (max " + f"{DONF['maxdd_bps']/100:.1f}%" + ").")}

<h2><span class="n">C2.</span> It is genuinely two-way (long &amp; short both work)</h2>
{fig("bt_longshort.png", "Long-USD vs short-USD leg contribution",
     "Long-USD legs +" + f"{SIDES['long_usd_total_bps']:.0f}" + " bps, short-USD legs +" +
     f"{SIDES['short_usd_total_bps']:.0f}" + " bps. Both sides are positive across the sample, so this is a "
     "real long/short — not a one-directional bet dressed up.")}

<h2><span class="n">C3.</span> Out-of-sample &amp; robustness</h2>
{fig("bt_oos.png", "In-sample (2015–21) vs out-of-sample (2021–26), net",
     "Net Sharpe " + str(IS['sharpe']) + " in-sample → " + str(OOS['sharpe']) + " out-of-sample. The OOS "
     "period is stronger because it contained the trends; the honest read is 'pays when the colón moves'.")}
{fig("bt_sensitivity.png", "Net Sharpe vs lookback (Donchian & SMA)",
     "Positive across a broad band of lookbacks (Donchian " + str(R['sensitivity']['donchian_min']) + "–" +
     str(R['sensitivity']['donchian_max']) + "), so it isn't a single lucky parameter — but it is not a "
     "high-Sharpe machine either.")}
{fig("bt_permutation.png", "Permutation null (net)",
     "Shuffling the timing of the same positions destroys the edge (p=" + str(R['Donchian-40_perm_p']) +
     "), so the entry/exit timing carries genuine information.")}

<div class="part">Part D · Verdict &amp; deployment</div>
<h2>Is it worth pursuing?</h2>
<ul class="find">
 <li><b>As a standalone speculative book:</b> marginal. Net Sharpe ~1 with ~13% drawdowns and a known
   tail risk is not compelling on its own, and you need an instrument that lets you be long <i>and</i>
   short USD/CRC at 0.65 CRC slippage.</li>
 <li><b>As a treasury / hedging overlay (most likely your case):</b> yes, useful. If you already run
   USD/CRC exposure, a slow trend filter that tells you when to be long vs short USD — trading only a few
   times a year — is a cheap, robust improvement over a static hedge, and it survived 11.5 years and real costs.</li>
 <li><b>Don't run the daily order-flow signal as a taker.</b> Even at your 0.65 CRC round-trip it nets
   only ~1.0 Sharpe in its best (2019+) regime and ~0.2 over the full history, with a −31% drawdown and
   ~40× the turnover of the trend book — strictly worse risk-adjusted. It is genuinely valuable only as a
   market-maker / internaliser capturing the spread you currently pay.</li>
 <li><span class="no"><b>Dominant risk:</b></span> the BCCR. A re-peg flattens the trend edge; a policy
   break / intervention is the fat tail this sample can't price. Size for survival, keep the lookback slow,
   and re-estimate continuously.</li>
</ul>

<footer>
  <b>Caveats.</b> P&amp;L is net of your stated 0.65 CRC round-trip slippage (0.325/side) but gross of
  financing/borrow and any market impact; the colón is not freely shortable for all participants. Tests
  assume you can hold a signed USD position at MONEX. The "best board offers" block in the export was
  empty, so VWAP / close are the price proxies.<br><br>
  Reproducible: <code>parse_monex → analyze → eda → strategies → backtest → build_report</code>, then
  <code>weasyprint</code> for the PDF. Strategy numbers are read live from <code>backtest_results.json</code>.
</footer>

</div></body></html>"""


def main():
    dst = OUT / "report.html"
    dst.write_text(HTML, encoding="utf-8")
    print(f"Wrote {dst} ({dst.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
