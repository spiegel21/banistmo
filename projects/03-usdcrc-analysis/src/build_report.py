"""Build a single self-contained HTML visual report (charts embedded as base64).

Run after analyze.py has written out/*.png. Produces out/report.html.
"""
from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def img(name: str) -> str:
    b = base64.b64encode((OUT / name).read_bytes()).decode()
    return f"data:image/png;base64,{b}"


KPIS = [
    ("−10.1%", "colón appreciation", "504.99 → 454.10 ₡ over the sample"),
    ("39 $M", "avg daily volume", "median 35 $M · 216 trades/day"),
    ("−0.45", "bps move per $M", "signed move vs volume · p≈1e-13"),
    ("+0.37", "lag-1 autocorr", "session moves trend, don't revert"),
    ("Sharpe 4.4", "order-flow signal", "+7 bps/day · 59% hit (gross)"),
    ("507 ₡", "point of control", "value area 453–510 ₡"),
]

SIGNALS = [
    ("Order-flow continuation", "short USD after heavy days, long after light",
     "+7.0 bps/day · 59% hit · Sharpe ≈ 4.4", "good", "Best clean edge — works even in the flat regime"),
    ("Overnight momentum", "follow a >1σ session move into the next session",
     "+17 bps/trade · 73% hit · Sharpe ≈ 8.4", "good", "Strong but fewer trades · same trend driver"),
    ("Fade the opening gap", "buy/sell the gap vs prior VWAP back to the close",
     "63% hit · Sharpe ≈ 6", "warn", "Real (~80% retrace) but Sharpe inflated by shared-open noise"),
    ("5d/20d MA crossover", "classic trend-follow on the VWAP",
     "−1.3 bps/day · Sharpe ≈ −0.8", "bad", "Whipsaws — the edge lives at the 1-day horizon"),
    ("Day-of-week bias", "directional bias by weekday",
     "Wed weakest (t ≈ −1.9)", "warn", "Suggestive only, not significant"),
]


def kpi_html():
    return "".join(
        f'<div class="kpi"><div class="kpi-v">{v}</div>'
        f'<div class="kpi-l">{lab}</div><div class="kpi-s">{s}</div></div>'
        for v, lab, s in KPIS
    )


def signal_rows():
    badge = {"good": "✅", "warn": "⚠️", "bad": "❌"}
    return "".join(
        f'<tr class="{cls}"><td><b>{name}</b><div class="muted">{desc}</div></td>'
        f'<td class="num">{res}</td><td>{badge[cls]} {note}</td></tr>'
        for name, desc, res, cls, note in SIGNALS
    )


def fig(name, title, caption):
    return f"""
    <figure>
      <h3>{title}</h3>
      <img src="{img(name)}" alt="{title}"/>
      <figcaption>{caption}</figcaption>
    </figure>"""


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>USD/CRC (MONEX) — Pricing & Trading Report</title>
<style>
 :root {{ --bg:#0f1419; --card:#1a212b; --ink:#e8edf2; --mut:#8a97a6;
   --accent:#37b87c; --warn:#e0a83d; --bad:#e05d5d; --line:#2a3441; --blue:#4a90d9; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink);
   font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
 .wrap {{ max-width:1040px; margin:0 auto; padding:32px 20px 80px; }}
 header {{ border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:28px; }}
 h1 {{ font-size:30px; margin:0 0 6px; letter-spacing:-.5px; }}
 .sub {{ color:var(--mut); font-size:14px; }}
 h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:1.5px; color:var(--accent);
   margin:42px 0 14px; }}
 .kpis {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
 .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }}
 .kpi-v {{ font-size:26px; font-weight:700; letter-spacing:-.5px; }}
 .kpi-l {{ font-size:13px; color:var(--ink); margin-top:2px; }}
 .kpi-s {{ font-size:12px; color:var(--mut); margin-top:4px; }}
 .lead {{ background:var(--card); border-left:3px solid var(--accent); border-radius:0 10px 10px 0;
   padding:16px 20px; margin:8px 0 0; }}
 .lead b {{ color:var(--accent); }}
 figure {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
   padding:16px; margin:16px 0; break-inside:avoid; page-break-inside:avoid; }}
 h2 {{ break-after:avoid; }}
 @page {{ size:A4; margin:14mm 12mm; background:var(--bg); }}
 figure h3 {{ margin:0 0 10px; font-size:16px; }}
 figure img {{ width:100%; border-radius:8px; display:block; background:#fff; }}
 figcaption {{ color:var(--mut); font-size:13px; margin-top:10px; }}
 table {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:14px; }}
 th,td {{ text-align:left; padding:11px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
 th {{ color:var(--mut); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.5px; }}
 td.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; color:var(--ink); }}
 .muted {{ color:var(--mut); font-size:12.5px; margin-top:2px; }}
 tr.good td:first-child {{ border-left:3px solid var(--accent); }}
 tr.warn td:first-child {{ border-left:3px solid var(--warn); }}
 tr.bad  td:first-child {{ border-left:3px solid var(--bad); }}
 ul.find {{ list-style:none; padding:0; margin:0; }}
 ul.find li {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
   padding:14px 16px; margin-bottom:10px; }}
 ul.find b {{ color:var(--blue); }}
 .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
 footer {{ color:var(--mut); font-size:12.5px; margin-top:40px; border-top:1px solid var(--line);
   padding-top:18px; }}
 @media (max-width:720px) {{ .kpis,.cols {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">

<header>
  <h1>USD/CRC · MONEX — Pricing-Model Reverse-Engineering &amp; Trading Report</h1>
  <div class="sub">BCCR MONEX daily data · 362 trading days · 6 Dec 2024 → 19 Jun 2026 ·
    open/close/low/high · VWAP · volume · trade counts</div>
</header>

<p class="lead">The USD/CRC is <b>not a random walk</b> — it is a <b>VWAP-anchored crawling
managed float</b> in a strong colón-appreciation trend. <b>Volume barely affects how big a
move is, but strongly predicts its direction</b>: heavy days are USD-supply days that push the
colón stronger, and that pressure <b>persists into the next session</b> — the basis of the
best tradeable edge.</p>

<h2>Headline numbers</h2>
<div class="kpis">{kpi_html()}</div>

<h2>1 · Price formation</h2>
{fig("01_price_formation.png", "Session VWAP, daily range & opening-gap distribution",
     "Top: the colón was range-bound ~500–511 through Oct-2025, then appreciated sharply to ~454. "
     "Bottom: the open clusters tightly around the prior session's VWAP (median gap −1.6 bps) — "
     "the market re-anchors to yesterday's weighted average each day.")}
<ul class="find">
  <li><b>Anchor.</b> 48% of sessions open within ±10 bps of the prior VWAP; ~80% of any opening
      gap is retraced by the close.</li>
  <li><b>Crawl.</b> Average drift ≈ −37 ₡/yr (R²=0.70), but the regime is "flat → steep down",
      not constant speed.</li>
</ul>

<h2>2 · Volume seasonality</h2>
{fig("02_volume_seasonality.png", "Average daily volume by day-of-week and month",
     "Friday is the lightest day with the smallest tickets (clean-up flow). April is the heaviest "
     "month (tax/repatriation); the first half of each month and Feb show the strongest appreciation; "
     "December is the only depreciation month (aguinaldo import demand).")}

<h2>3 · Volume profile — where the market traded</h2>
<div class="cols">
{fig("03_volume_profile.png", "US$ volume at each ~1-colón level",
     "Point of Control 507 ₡; the 504–509 band holds ~40% of lifetime volume. 70% value area "
     "453–510 ₡ — current price sits at the bottom edge, i.e. price-discovery territory, not a range to fade.")}
{fig("04_volume_vs_move.png", "Volume vs daily range (left) and signed move (right)",
     "Volume does NOT widen the range (r≈0.04). It drives DIRECTION: the right panel's clear "
     "negative slope (−0.45 bps/$M, p≈1e-13) is the order-flow fingerprint — size = USD supply = colón strength.")}
</div>

<h2>4 · Trading signals (gross of costs)</h2>
<table>
 <tr><th>Signal</th><th>Result</th><th>Verdict</th></tr>
 {signal_rows()}
</table>
{fig("05_signal_equity.png", "Cumulative P&L of the two clean tradeable signals",
     "Both grind upward through the flat regime (Mar–Sep 2025) and accelerate into the trend — "
     "the order-flow signal (green) has standalone value beyond simply riding the appreciation.")}

<h2>5 · Strategy deep-dive</h2>
{fig("s_comparison.png", "Signals side-by-side — Sharpe, hit rate, average P&L",
     "The order-flow and momentum signals clear 50% hit comfortably; the opening-gap fade is "
     "shown for completeness but its headline stats are inflated by the shared-open artifact.")}
{fig("s_mechanics.png", "Why the order-flow edge exists",
     "Left: today's volume z-score vs the NEXT session's move — a clean negative slope, so heavy "
     "days keep pushing the colón stronger tomorrow. Right: bucketed, the gradient is monotonic — "
     "very-heavy days → next-session appreciation, very-light days → depreciation.")}
{fig("s1_orderflow_tearsheet.png", "Order-flow continuation — full tearsheet",
     "Equity, drawdown, monthly P&L, and the daily-P&L distribution. Smooth equity, only one "
     "losing month (Feb-25), shallow drawdowns, and a right-skewed daily distribution.")}
{fig("s2_momentum_tearsheet.png", "Overnight momentum — full tearsheet",
     "Fewer trades (only >1σ session moves qualify) but a higher per-trade edge; the same "
     "trend/order-flow driver underlies both strategies.")}

<h2>How to trade it</h2>
<ul class="find">
  <li>The dominant trade is <b>long colón / short USD</b> (the crawl), sized <b>up after
      heavy-volume sessions</b> and into the <b>first half of the month / Feb &amp; Apr</b>,
      trimmed in <b>December</b>.</li>
  <li>Execute <b>VWAP-to-VWAP</b>: the open anchors to a known prior reference, so you can
      transact near a predictable rate.</li>
  <li><b>Why it should persist:</b> structural USD oversupply (exports, tourism, FDI, possible
      BCCR reserve accumulation) shows up as volume, and the managed-float anchor bleeds that
      pressure out gradually rather than gapping — hence the autocorrelation.</li>
</ul>

<footer>
  <b>Caveats.</b> All P&amp;L is gross (no MONEX spread / brokerage / settlement); the colón is
  not freely shortable for most participants → a directional/hedging edge, not arbitrage.
  Backtests are in-sample over one appreciation regime; a policy shift would break the crawl.
  The "best board offers" block in the export was empty, so VWAP is the price proxy throughout.
  <br><br>Generated from <code>projects/03-usdcrc-analysis</code> · parser + analysis reproducible via
  <code>python src/parse_monex.py &amp;&amp; python src/analyze.py &amp;&amp; python src/build_report.py</code>.
</footer>

</div></body></html>"""


def main():
    dst = OUT / "report.html"
    dst.write_text(HTML, encoding="utf-8")
    print(f"Wrote {dst} ({dst.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
