"""Build the self-contained visual report: recommendation -> dynamics -> rules -> dollars.

Everything is priced VWAP-to-VWAP (the realistic desk fill) net of the 0.65 CRC
round-trip slippage. The recommended calendar/quincena rule leads every section;
secondary signals (volume, skew, combined vote, ML, trend books) are kept but
tucked into tabs that default to the recommended strategy.

Reads out/{quincena,dynamics,vm,exits,backtest_vwap}_results.json. Run last, after
analyze / eda / volume_model / dynamics / quincena / exits / backtest_vwap.
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
XT = json.loads((OUT / "exits_results.json").read_text())
XB, XTR, XCB, XI, XM = (XT["baseline"], XT["trailing"], XT["trailing_floor"],
                        XT["improve"], XT["_meta"])
BV = json.loads((OUT / "backtest_vwap_results.json").read_text())
BVM = BV["_meta"]
QREF = Q["Refined (short 5-15)"]
QSLOW = Q["Refined + slow-vol sizing"]
QBASE = Q["Base quincena (<=15)"]
QCAL = next(v for k, v in Q.items() if k.startswith("Real calendar"))
OC = Q["oos_calendar"]
ORF = Q["oos_refined"]
EX = Q["execution"]
CAL_PRE = Q["_meta"].get("cal_pre", 6)
M = R["_meta"]
LG = R["Logit (conviction)"]


def img(name):
    return "data:image/png;base64," + base64.b64encode((OUT / name).read_bytes()).decode()


def fig(name, title, cap):
    return (f'<figure><h3>{title}</h3><img src="{img(name)}" alt="{title}"/>'
            f'<figcaption>{cap}</figcaption></figure>')


def usd(x):
    return f"${x/1e3:,.0f}k" if abs(x) < 1e6 else f"${x/1e6:.2f}M"


def scls(s):
    """Colour class from a Sharpe value."""
    return "ok" if s >= 1.5 else ("mb" if s >= 0.5 else "no")


def tabs(items):
    """Render a tab group. items = [(label, is_best, html), ...]; first is default.

    On screen a tiny bit of JS toggles panels; in print (PDF) the tab bar is
    hidden and every panel is shown with its own heading, so no result is lost.
    """
    bar = ['<div class="tabs"><div class="tabbar">']
    pans = []
    for i, (label, best, html) in enumerate(items):
        act = " active" if i == 0 else ""
        star = '<span class="best">best</span>' if best else ""
        bar.append(f'<button class="tabbtn{act}">{label}{star}</button>')
        pans.append(f'<div class="tabpanel{act}"><div class="tab-h">{label}</div>{html}</div>')
    bar.append("</div>")
    return "".join(bar) + "".join(pans) + "</div>"


# --------------------------------------------------------------------------- #
# KPI strip — headline the recommended calendar rule (VWAP-priced) + its OOS
# --------------------------------------------------------------------------- #
KPIS = [
    (usd(QCAL["per_year_usd"]), "calendar rule / yr", "$1M/trade · VWAP-priced"),
    (f"{QCAL['sharpe']}", "Sharpe (full sample)", f"only {QCAL['roundtrips_yr']} trades/yr"),
    (f"{OC['oos']['sharpe']}", "out-of-sample Sharpe", f"OOS {usd(OC['oos']['per_year_usd'])}/yr · 40% held out"),
    (f"{XCB['full']['sharpe']}", "Sharpe with exit overlay",
     f"trailing stop cuts DD to {usd(XCB['full']['maxdd_usd'])} (A5)"),
    (usd(Q["refined_worst_year_usd"]), "worst calendar year", "positive in all 12 years"),
    (f"{DY['vol_on_usd_down']:.0f} vs {DY['vol_on_usd_up']:.0f} M", "volume: colón up vs down",
     "USD sellers trade in size"),
]


def kpi_html():
    return "".join(f'<div class="kpi"><div class="kpi-v">{v}</div>'
                   f'<div class="kpi-l">{lab}</div><div class="kpi-s">{s}</div></div>'
                   for v, lab, s in KPIS)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def money_table():
    """Unified $1M/trade table — recommended calendar family first, then the raw
    building-block signals and the ML model. All VWAP-priced, net of cost."""
    reco = [(f"Real calendar (≤{CAL_PRE} bd to deadline)", QCAL, True),
            ("Refined quincena (short days 5–15)", QREF, False),
            ("Refined + slow-vol sizing", QSLOW, False),
            ("Base quincena (short ≤15)", QBASE, False)]
    block = [("Volume rule", DY["Volume rule"]),
             ("Precio-ponderado skew rule", DY["Precio-ponderado skew rule"]),
             ("Combined vote (3 signals)", DY["Combined vote (3 signals)"])]
    out = ['<table><tr><th>Strategy · $1M/trade · VWAP-priced, net 0.65 CRC RT</th>'
           '<th class="num">Per year</th><th class="num">Sharpe</th><th class="num">Trades/yr</th>'
           '<th class="num">Win %</th><th class="num">Max DD</th></tr>']
    out.append('<tr><td colspan="6" class="grp">Recommended — calendar / quincena family</td></tr>')
    for name, b, best in reco:
        rc = ' class="rowbest"' if best else ""
        star = ' <span class="best">best</span>' if best else ""
        out.append(f'<tr{rc}><td>{name}{star}</td>'
                   f'<td class="num {scls(b["sharpe"])}">{usd(b["per_year_usd"])}</td>'
                   f'<td class="num {scls(b["sharpe"])}">{b["sharpe"]}</td>'
                   f'<td class="num">{b["roundtrips_yr"]}</td><td class="num">{b["win_rate"]}</td>'
                   f'<td class="num">{usd(b["maxdd_usd"])}</td></tr>')
    out.append('<tr><td colspan="6" class="grp">Building-block signals — trade gently / as filters</td></tr>')
    for name, b in block:
        out.append(f'<tr><td>{name}</td>'
                   f'<td class="num {scls(b["sharpe"])}">{usd(b["per_year_usd"])}</td>'
                   f'<td class="num {scls(b["sharpe"])}">{b["sharpe"]}</td>'
                   f'<td class="num">{b["roundtrips_yr"]}</td><td class="num">{b["win_rate"]}</td>'
                   f'<td class="num">{usd(b["maxdd_usd"])}</td></tr>')
    out.append(f'<tr><td><b>ML combined model</b> (conviction-sized, walk-forward)</td>'
               f'<td class="num ok">{usd(LG["ann_bps"]*100)}</td>'
               f'<td class="num ok">{LG["sharpe"]}</td><td class="num">{LG["roundtrips_yr"]}</td>'
               f'<td class="num">{LG["hit"]}</td><td class="num">{usd(LG["maxdd_bps"]*100)}</td></tr>')
    out.append("</table>")
    return "".join(out)


def oos_table():
    def row(name, o):
        return (f'<tr><td>{name}</td>'
                f'<td class="num">{usd(o["is"]["per_year_usd"])}</td><td class="num">{o["is"]["sharpe"]}</td>'
                f'<td class="num ok">{usd(o["oos"]["per_year_usd"])}</td>'
                f'<td class="num ok">{o["oos"]["sharpe"]}</td></tr>')
    return ('<table><tr><th>Rule (chronological 60/40 split, no re-fitting)</th>'
            '<th class="num">IS / yr</th><th class="num">IS Sharpe</th>'
            '<th class="num">OOS / yr</th><th class="num">OOS Sharpe</th></tr>'
            + row(f"Real calendar (≤{CAL_PRE} bd)", OC)
            + row("Refined (short 5–15)", ORF) + "</table>")


def exits_table():
    """Baseline calendar rule vs the two optimised exit overlays (full + OOS)."""
    def row(name, b, best=False, note=""):
        rc = ' class="rowbest"' if best else ""
        star = ' <span class="best">recommended</span>' if best else ""
        f, o = b["full"], b["oos"]
        return (f'<tr{rc}><td>{name}{star}<div class="muted" style="font-size:11px">{note}</div></td>'
                f'<td class="num {scls(f["sharpe"])}">{usd(f["per_year_usd"])}</td>'
                f'<td class="num {scls(f["sharpe"])}">{f["sharpe"]}</td>'
                f'<td class="num">{usd(f["maxdd_usd"])}</td>'
                f'<td class="num ok">{usd(o["per_year_usd"])}</td>'
                f'<td class="num ok">{o["sharpe"]}</td></tr>')
    return ('<table><tr><th>Calendar rule · $1M/trade · VWAP-priced, net cost</th>'
            '<th class="num">Full / yr</th><th class="num">Sharpe</th><th class="num">Max DD</th>'
            '<th class="num">OOS / yr</th><th class="num">OOS Sharpe</th></tr>'
            + row("No exit — always in the market", XB, note="the rule from A1–A4")
            + row(f"+ trailing-stop exit ({XM['trail_bps']} bps)", XTR, best=True,
                  note="tuned on in-sample P&amp;L; biggest raw P&amp;L")
            + row(f"+ trailing ({XM['combo_trail_bps']}) &amp; hard floor ({XM['combo_floor_bps']}) bps", XCB,
                  note="tuned on in-sample Sharpe; best risk-adjusted")
            + "</table>")


def exec_table():
    return ('<table><tr><th>Recommended calendar rule · $1M/trade · net 0.65 CRC RT</th>'
            '<th class="num">Per year</th><th class="num">Sharpe</th><th class="num">Max DD</th></tr>'
            f'<tr><td>Closing-print fill <span class="muted">(optimistic mark)</span></td>'
            f'<td class="num">{usd(EX["close"]["per_year_usd"])}</td>'
            f'<td class="num">{EX["close"]["sharpe"]}</td>'
            f'<td class="num">{usd(EX["close"]["maxdd_usd"])}</td></tr>'
            f'<tr class="rowbest"><td>Session VWAP fill <span class="best">used everywhere in this report</span></td>'
            f'<td class="num ok">{usd(EX["vwap"]["per_year_usd"])}</td>'
            f'<td class="num ok">{EX["vwap"]["sharpe"]}</td>'
            f'<td class="num">{usd(EX["vwap"]["maxdd_usd"])}</td></tr></table>')


def vwap_table():
    """Secondary trend / order-flow books — close-print vs VWAP execution."""
    order = ["Order-flow daily", "Donchian-40 trend", "SMA-120 trend"]
    cv = BV["close_vs_vwap"]
    out = ['<table><tr><th>Secondary strategy (net 0.65 CRC RT, VWAP-to-VWAP)</th>'
           '<th class="num">Close Sharpe</th><th class="num">VWAP Sharpe</th>'
           '<th class="num">VWAP bps/yr</th><th class="num">Trades/yr</th></tr>']
    for k in order:
        c = cv[k]
        b = BV[k]
        cls = "ok" if c["vwap_sharpe"] >= c["close_sharpe"] else "no"
        out.append(f'<tr><td>{k}</td><td class="num">{c["close_sharpe"]}</td>'
                   f'<td class="num {cls}">{c["vwap_sharpe"]}</td>'
                   f'<td class="num {cls}">{c["vwap_ann_bps"]:.0f}</td>'
                   f'<td class="num">{b["roundtrips_yr"]}</td></tr>')
    out.append("</table>")
    return "".join(out)


def dist_table():
    """Distribution shape of each calendar-family variant's daily net return (VWAP-priced)."""
    rd = Q["return_dist"]
    order = ["Real calendar (recommended)", "Refined (5–15)", "Refined + slow-vol",
             "Base quincena (≤15)"]
    out = ['<table><tr><th>Variant · daily net bps, $1M/trade, VWAP-priced</th><th class="num">Mean</th>'
           '<th class="num">Std</th><th class="num">Sharpe</th><th class="num">Pos days</th>'
           '<th class="num">Skew</th><th class="num">Ex-kurt</th><th class="num">5% VaR</th>'
           '<th class="num">Worst day</th></tr>']
    for k in order:
        d = rd[k]
        best = k.startswith("Real calendar")
        rc = ' class="rowbest"' if best else ""
        star = ' <span class="best">recommended</span>' if best else ""
        out.append(f'<tr{rc}><td>{k}{star}</td><td class="num">{d["mean"]:+.2f}</td>'
                   f'<td class="num">{d["std"]:.1f}</td><td class="num {scls(d["sharpe"])}">{d["sharpe"]:.2f}</td>'
                   f'<td class="num">{d["pos"]:.0f}%</td><td class="num">{d["skew"]:+.2f}</td>'
                   f'<td class="num">{d["kurt"]:+.1f}</td><td class="num">{d["var95"]:.1f}</td>'
                   f'<td class="num">{d["min"]:.0f}</td></tr>')
    out.append("</table>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Part C — per-rule logic, as tabs (recommended rule is the default)
# --------------------------------------------------------------------------- #
RULE_CAL = f"""<div class="rule"><h4>Calendar / quincena rule — the recommendation</h4>
  <div class="logic">≤ {CAL_PRE} business days before the IVA/quincena deadline (through it) →  SHORT USD $1M<br>
    &nbsp;&nbsp;&nbsp;(firms sell USD to raise colones for the 15th tax + payroll → colón strengthens)<br>
    otherwise →  LONG USD $1M   (supply fades, USD drifts up)<br>
    optional: trim size to 50% when a slow 20-day volume regime disagrees</div>
  <p><b>Why:</b> the intra-month cash-flow cycle (Part B) — companies convert USD→CRC for the D-104 IVA
    filing (due the 15th, rolled to the next business day) and the mid-month quincena payroll, a recurring
    supply surge that strengthens the colón into the deadline and reverts after. <b>Two readings:</b> a
    <i>fixed</i> day-of-month window (days 1–4 LONG, 5–15 SHORT, 16–end LONG) and the <i>real-calendar</i>
    anchor that rolls with weekends/holidays. <b>Result (VWAP-priced):</b> {usd(QCAL['per_year_usd'])}/yr,
    Sharpe {QCAL['sharpe']}, {QCAL['roundtrips_yr']} trades/yr, max drawdown {usd(QCAL['maxdd_usd'])} — and
    an out-of-sample Sharpe of {OC['oos']['sharpe']} (A4). The base "short the whole first half" rule
    ({usd(QBASE['per_year_usd'])}/yr, Sharpe {QBASE['sharpe']}) is the crude ancestor this refines.</p></div>"""

RULE_VOL = f"""<div class="rule"><h4>Volume — "buy USD when the market is quiet"</h4>
  <div class="logic">seasonal_vol = today_volume ÷ (avg volume on the same weekday, to date)<br>
    if seasonal_vol &lt; 1.0  →  LONG USD $1M   (quieter than normal)<br>
    if seasonal_vol ≥ 1.0  →  SHORT USD $1M  (busier than normal)</div>
  <p><b>Why:</b> quiet = exporters absent = importer demand lifts USD. Dividing by the same-weekday average
    makes "quiet" mean quiet <i>for a Tuesday</i>. <b>Result:</b> directionally right but flips
    ~{DY['Volume rule']['roundtrips_yr']}×/yr — even VWAP-priced it only nets Sharpe
    {DY['Volume rule']['sharpe']} standalone. Best used as the slow regime filter on the calendar trade.</p></div>"""

RULE_SKEW = f"""<div class="rule"><h4>Precio-ponderado skew — "follow the big tickets"</h4>
  <div class="logic">if VWAP &gt; simple_average  →  LONG USD $1M   (large trades lifted the price)<br>
    if VWAP ≤ simple_average  →  SHORT USD $1M</div>
  <p><b>Why:</b> the weighted-vs-simple gap reveals whether large players were buying or selling USD
    (corr {DY['skew_corr']} with the next-day move). <b>Result:</b> flips
    ~{DY['Precio-ponderado skew rule']['roundtrips_yr']}×/yr → Sharpe {DY['Precio-ponderado skew rule']['sharpe']}
    net; a tie-breaker, not a standalone book.</p></div>"""

RULE_ML = f"""<div class="rule"><h4>Combined vote &amp; the ML model</h4>
  <div class="logic">Combined = majority vote of the three signals (always ±1) → $1M long/short<br>
    ML = logistic P(USD up) on all features; position = clip((P − 0.5) × 5, −1, +1)</div>
  <p><b>Why the ML sizes by conviction:</b> a raw daily flip trades ~95×/yr and dies on cost. Scaling the
    position by confidence keeps turnover near {LG['roundtrips_yr']}×/yr while still capturing the edge. On
    the realistic VWAP basis the combined vote nets Sharpe {DY['Combined vote (3 signals)']['sharpe']} and the
    ML {LG['sharpe']} — but both lean on the same calendar + volume economics as the simple rule above.</p></div>"""


JS = """<script>
document.querySelectorAll('.tabs').forEach(function(t){
  var btns=t.querySelectorAll('.tabbtn'), pans=t.querySelectorAll('.tabpanel');
  btns.forEach(function(b,i){ b.addEventListener('click', function(){
    btns.forEach(function(x){x.classList.remove('active');});
    pans.forEach(function(x){x.classList.remove('active');});
    b.classList.add('active'); pans[i].classList.add('active');
  });});
});
</script>"""


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>USD/CRC — Calendar Strategy, Dynamics &amp; VWAP-Priced Results</title>
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
 td.grp {{ color:var(--blue); font-weight:700; font-size:10.5px; text-transform:uppercase; letter-spacing:.5px;
   background:#141b24; border-bottom:1px solid var(--line); }}
 tr.rowbest td {{ background:rgba(55,184,124,0.09); }}
 .best {{ color:var(--accent); font-size:10px; text-transform:uppercase; letter-spacing:.5px;
   font-weight:700; margin-left:6px; border:1px solid var(--accent); border-radius:5px; padding:1px 5px; }}
 ul.find {{ list-style:none; padding:0; margin:0; }}
 ul.find li {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
   padding:13px 16px; margin-bottom:10px; }}
 ul.find b {{ color:var(--blue); }}
 .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
 /* tabs */
 .tabs {{ margin:14px 0; }}
 .tabbar {{ display:flex; flex-wrap:wrap; gap:6px; border-bottom:1px solid var(--line); }}
 .tabbtn {{ background:transparent; color:var(--mut); border:1px solid var(--line); border-bottom:none;
   border-radius:9px 9px 0 0; padding:8px 14px; font:inherit; font-size:13px; cursor:pointer; }}
 .tabbtn:hover {{ color:var(--ink); }}
 .tabbtn.active {{ color:var(--ink); background:var(--card); box-shadow:inset 0 -2px 0 var(--accent); }}
 .tabpanel {{ display:none; background:var(--card); border:1px solid var(--line); border-top:none;
   border-radius:0 0 12px 12px; padding:6px 18px 14px; }}
 .tabpanel.active {{ display:block; }}
 .tabpanel figure, .tabpanel .rule {{ background:#0f1419; }}
 .tab-h {{ display:none; font-weight:700; color:var(--blue); font-size:13px; margin:12px 0 2px; }}
 footer {{ color:var(--mut); font-size:12.5px; margin-top:40px; border-top:1px solid var(--line);
   padding-top:18px; }}
 @page {{ size:A4; margin:13mm 11mm; }}
 @media (max-width:720px) {{ .kpis,.cols {{ grid-template-columns:1fr; }} }}
 @media print {{
   .tabbar {{ display:none; }}
   .tabpanel {{ display:block !important; border-radius:12px; border-top:1px solid var(--line); margin-top:10px; }}
   .tab-h {{ display:block; }}
 }}
</style></head>
<body><div class="wrap">

<header>
  <h1>USD/CRC · MONEX — the calendar strategy: dynamics, rule logic &amp; VWAP-priced results</h1>
  <div class="sub">{M['n_days']:,} sessions · {M['date_min']} → {M['date_max']} · long/short USD ·
    ${DY['_meta']['notional_usd']:,.0f} per trade · <b>priced VWAP-to-VWAP</b>, net of 0.65 CRC round-trip ·
    no technical indicators</div>
</header>

<p class="lead">The recommended strategy is a <b>calendar (quincena) rule</b>: short USD only into Costa
Rica's mid-month IVA / payroll deadline (within {CAL_PRE} business days of it, or equivalently day-of-month
5–15), long the rest of the month. Priced at the session VWAP net of cost it earns
<b>{usd(QCAL['per_year_usd'])}/yr at Sharpe {QCAL['sharpe']}</b> on just {QCAL['roundtrips_yr']} trades a year,
with an <b>out-of-sample Sharpe of {OC['oos']['sharpe']}</b> ({usd(OC['oos']['per_year_usd'])}/yr on the 40%
held out after {OC['split_date']}) and a worst year of {usd(Q['refined_worst_year_usd'])} — positive in all
12 years. A <b>dynamic trailing-stop exit</b> (A5) enhances it further — to {usd(XTR['full']['per_year_usd'])}/yr
at Sharpe {XTR['full']['sharpe']}, or Sharpe {XCB['full']['sharpe']} with a hard floor that nearly halves the
drawdown — <b>lifting P&amp;L both in- and out-of-sample</b>. Every number below is VWAP-priced; secondary
signals are kept in tabs but default to this rule.</p>

<div class="kpis" style="margin-top:16px">{kpi_html()}</div>

<div class="part">Part A · The recommended strategy — calendar / quincena rule</div>
<h2><span class="n">A1.</span> The rule &amp; the numbers ($1M per trade, VWAP-priced, net of cost)</h2>
<table><tr><th>Version</th><th class="num">Per year</th><th class="num">Sharpe</th>
  <th class="num">Trades/yr</th><th class="num">Win %</th><th class="num">Max DD</th><th>Note</th></tr>
<tr><td>Base quincena (short if day ≤ 15)</td><td class="num">{usd(QBASE['per_year_usd'])}</td>
  <td class="num">{QBASE['sharpe']}</td><td class="num">{QBASE['roundtrips_yr']}</td>
  <td class="num">{QBASE['win_rate']}</td><td class="num">{usd(QBASE['maxdd_usd'])}</td>
  <td class="muted">crude ancestor</td></tr>
<tr><td>Refined (short days 5–15, long otherwise)</td>
  <td class="num ok">{usd(QREF['per_year_usd'])}</td><td class="num ok">{QREF['sharpe']}</td>
  <td class="num">{QREF['roundtrips_yr']}</td><td class="num">{QREF['win_rate']}</td>
  <td class="num">{usd(QREF['maxdd_usd'])}</td><td class="muted">fixed window</td></tr>
<tr><td>Refined + slow-volume sizing</td><td class="num">{usd(QSLOW['per_year_usd'])}</td>
  <td class="num ok">{QSLOW['sharpe']}</td><td class="num">{QSLOW['roundtrips_yr']}</td>
  <td class="num">{QSLOW['win_rate']}</td><td class="num ok">{usd(QSLOW['maxdd_usd'])}</td>
  <td class="muted">best risk-adjusted</td></tr>
<tr class="rowbest"><td><b>Real calendar (≤{CAL_PRE} bd to IVA/quincena deadline)</b>
  <span class="best">recommended</span></td>
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
{fig("q_peryear.png", "Net P&L by year — base vs refined ($1M/trade, VWAP-priced)",
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

<h2><span class="n">A4.</span> Sample vs out-of-sample — the edge is not just the early years</h2>
<p class="lead">The rule is a fixed heuristic with <b>no fitted parameters</b>, but we still hold out the last
40% of history (everything after <b>{OC['split_date']}</b>) to show the edge is not a property of the calmer,
more-pegged early years. Trained-window intuition only; nothing is re-fit on the test window.</p>
{oos_table()}
{fig("q_oos.png", "Real-calendar rule (VWAP-priced): in-sample vs out-of-sample equity",
     "In-sample " + OC['is_start'] + " → " + OC['is_end'] + " (Sharpe " + str(OC['is']['sharpe']) +
     ") and out-of-sample → " + OC['oos_end'] + " (Sharpe " + str(OC['oos']['sharpe']) + ", "
     + usd(OC['oos']['per_year_usd']) + "/yr). The out-of-sample slope is if anything steeper — the "
     "post-2021 float is where the cash-flow cycle is most tradeable.")}

<h2><span class="n">A5.</span> Enhancing the rule with a dynamic exit (stop-loss / take-profit)</h2>
<p class="lead">The calendar rule is <b>always in the market</b> (short into the deadline, long otherwise). Because
the edge <b>front-loads</b> — the colón strengthens <i>into</i> the deadline and reverts after — a trade often
peaks mid-hold and then gives the move back. So we overlay a <b>trailing stop</b>: within each trade, once the
running P&amp;L gives back {XM['trail_bps']} bps from its best level, we bank it and stay flat until the calendar
opens the next trade. This <b>never changes the entry logic</b> — it can only remove the losing tail of a trade.
Both parameters are optimised on the <b>in-sample 60% only</b>; the out-of-sample 40% below is untouched.</p>
{exits_table()}
{fig("ex_optimization.png", "Exit optimiser — net P&L and Sharpe across the trailing band",
     "This is the search itself. Every trailing band from ~30 to ~100 bps beats the no-exit rule (dashed) in "
     "<b>both</b> the in-sample and out-of-sample windows on Sharpe, and lifts in-sample P&L across the whole "
     "range — a broad plateau, not a single lucky point (the same robustness test we applied to the calendar "
     "window in A2). The chosen " + str(XM['trail_bps']) + " bps maximises in-sample P&L/yr.")}
{fig("ex_equity.png", "Calendar rule vs calendar rule + trailing exit — equity &amp; drawdown",
     "Adding the exit lifts full-sample P&L from " + usd(XB['full']['per_year_usd']) + "/yr to " +
     usd(XTR['full']['per_year_usd']) + "/yr (Sharpe " + str(XB['full']['sharpe']) + " → " +
     str(XTR['full']['sharpe']) + ") and — with the hard floor variant — cuts the worst drawdown from " +
     usd(XB['full']['maxdd_usd']) + " to " + usd(XCB['full']['maxdd_usd']) + ". The green curve sits above the "
     "grey throughout, most visibly through the deep 2017–18 drawdown.")}
{fig("ex_episode.png", "How the exit enhances one trade — bank the move at the peak, skip the reversion",
     "A concrete " + XT['episode']['side'] + "-USD trade: the position banks " +
     f"{XT['episode']['banked_bps']:+d} bps at the trailing exit near the peak, instead of holding to the "
     "calendar's episode end and watching the same trade turn into a " + f"{XT['episode']['held_bps']:+d} bps" +
     " result on the post-deadline reversion — that one exit is worth ~" + str(XT['episode']['saved_bps']) +
     " bps. The overlay does this systematically across trades; it is the mechanism, on a single trade.")}

<h2><span class="n">A6.</span> The position against the price</h2>
{fig("q_position_shading.png", "USD/CRC session VWAP shaded by the recommended calendar position",
     f"Green = long USD, red = short USD (short ≤{CAL_PRE} business days into the IVA/quincena deadline). "
     "The rule is always in a position — there is no flat/no-trade state — so every session carries one shade "
     "or the other. The short (red) bands cluster mid-month, exactly where the cash-flow supply surge lands.")}

<h2><span class="n">A7.</span> Return distributions — how the variants compare</h2>
<p class="lead">Daily net P&amp;L per $1M traded (1 bp = $100), <b>VWAP-to-VWAP</b>. The tabs isolate one
variant; the overlay shows all four on a shared x-range, so the shapes are directly comparable. The story is
in the <i>shape</i>, not just the mean: slow-vol sizing trades a slightly lower mean for a much tighter body
(smaller std, positive skew), which is why its Sharpe tops the table even though the recommended calendar rule
earns more dollars.</p>
{tabs([
    ("Overlay (all four)", False, fig("q_dist_overlay.png", "All four calendar-family variants overlaid",
        "The slow-vol variant is visibly the most peaked (smallest std); the recommended calendar and the "
        "fixed refined window sit almost on top of each other, both shifted right of the crude base rule.")),
    ("Real calendar", True, fig("q_dist_calendar.png", "Real calendar (recommended) — daily net return distribution",
        f"Mean {Q['return_dist']['Real calendar (recommended)']['mean']:+.2f} bps/day at Sharpe "
        f"{Q['return_dist']['Real calendar (recommended)']['sharpe']:.2f}; a fat left tail "
        f"(skew {Q['return_dist']['Real calendar (recommended)']['skew']:+.2f}) is the post-deadline reversion "
        "that the A5 trailing exit is designed to cut.")),
    ("Refined (5–15)", False, fig("q_dist_refined.png", "Refined quincena — daily net return distribution",
        f"The fixed day-of-month twin of the calendar rule — nearly identical shape "
        f"(Sharpe {Q['return_dist']['Refined (5–15)']['sharpe']:.2f}, "
        f"std {Q['return_dist']['Refined (5–15)']['std']:.1f}), confirming the two readings capture the same edge.")),
    ("Refined + slow-vol", False, fig("q_dist_slowvol.png", "Refined + slow-vol sizing — daily net return distribution",
        f"Trimming size when a slow 20-day volume regime disagrees collapses the std to "
        f"{Q['return_dist']['Refined + slow-vol']['std']:.1f} bps and flips the skew positive "
        f"({Q['return_dist']['Refined + slow-vol']['skew']:+.2f}) — the highest Sharpe "
        f"({Q['return_dist']['Refined + slow-vol']['sharpe']:.2f}) of the four.")),
    ("Base quincena", False, fig("q_dist_base.png", "Base quincena (≤15) — daily net return distribution",
        f"The crude ancestor: same volatility but a lower mean "
        f"({Q['return_dist']['Base quincena (≤15)']['mean']:+.2f} bps) and the heaviest left tail, dragging "
        f"its Sharpe to {Q['return_dist']['Base quincena (≤15)']['sharpe']:.2f}.")),
])}
{dist_table()}

<div class="part">Part B · The underlying dynamics — what we are exploiting</div>
<h2><span class="n">B1.</span> The engine: exporters are the swing USD supply</h2>
{fig("dyn_mechanism.png", "Volume when the colón strengthens vs weakens, and next-day move by volume",
     f"On days the colón strengthens (USD falls) the market trades {DY['vol_on_usd_down']:.0f}M — far more "
     f"than the {DY['vol_on_usd_up']:.0f}M on days it weakens. USD sellers (exporters, remittances, FDI) come "
     "in size; USD buyers (importers) trickle. So HIGH volume = supply present = colón up, and LOW volume = "
     f"thin market = USD drifts up next day ({DY['nextmove_after_lowvol']:+.1f} bps after quiet days vs "
     f"{DY['nextmove_after_highvol']:+.1f} after busy ones). That asymmetry IS the volume signal.")}

<h2><span class="n">B2.</span> The intra-month cash-flow cycle (why the calendar works)</h2>
{fig("dyn_domcycle.png", "Average next-day USD move (bars) and volume (line) by day-of-month",
     "A clear monthly rhythm: volume peaks mid-month (~day 13–15) and the colón strengthens hardest right "
     "then (red bars) — a recurring USD-supply surge (exporter settlements, tax/paydate conversions). The "
     "first days and the second half of the month lean the other way (USD up). This is the quincena effect, "
     "and it is a cash-flow cycle, not a chart pattern.")}

<h2><span class="n">B2b.</span> Aligning to the real deadline sharpens the signal</h2>
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

<h2><span class="n">B3.</span> Two more relationships</h2>
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
<p class="sub" style="margin:0 0 4px">The recommended calendar rule is the default tab; the secondary
signals are one click away.</p>
{tabs([("Calendar rule", True, RULE_CAL),
       ("Volume", False, RULE_VOL),
       ("Precio-ponderado skew", False, RULE_SKEW),
       ("Combined &amp; ML", False, RULE_ML)])}

<div class="part">Part D · Results at $1,000,000 per trade (VWAP-priced, net of cost)</div>
<p class="sub" style="margin:0 0 4px">The recommended calendar family sits at the top; the raw building-block
signals and the ML model follow. Equity curves are tabbed — recommended first.</p>
{money_table()}
{tabs([
    ("Recommended calendar", True,
     fig("q_tearsheet.png", "Recommended calendar rule — equity, drawdown, yearly &amp; monthly P&L",
         "Smooth compounding at ~" + usd(QCAL['per_year_usd']) + "/yr with shallow drawdowns; every year "
         "and (on average) every month positive — January and April strongest. VWAP-priced, net of cost.")),
    ("Building-block rules", False,
     fig("dyn_dollar_equity.png", "Cumulative net P&L in US$ millions — building-block rules, $1M per trade",
         "The base quincena (green) compounds smoothly; the combined vote is decent but dragged by the "
         "turnover of the volume and skew legs. The volume rule is directionally right but churns — the "
         "signal is real, the daily flipping is the cost problem.")),
    ("ML model", False,
     fig("vm_equity.png", "Conviction-sized ML model — cumulative net P&L (walk-forward, VWAP-priced)",
         "Every prediction is out-of-sample (walk-forward). Conviction sizing keeps turnover near "
         + str(LG['roundtrips_yr']) + "×/yr; net Sharpe " + str(LG['sharpe']) + ". It mostly repackages the "
         "same calendar + volume economics as the simple rule, at higher complexity.") +
     fig("vm_oos.png", "ML: in-sample vs out-of-sample (net)",
         "OOS Sharpe " + str(R['Logit (conviction)_oos']['sharpe']) + " vs IS " +
         str(R['Logit (conviction)_is']['sharpe']) + "; a 5-day feature lag collapses accuracy (no "
         "look-ahead), permutation p=" + str(R['perm_p']) + ".")),
])}

<div class="part">Part E · Raw relationships behind the signals</div>
<h2><span class="n">E1.</span> Low volume → USD up (seasonally adjusted)</h2>
{fig("vm_vol_nextret.png", "Next-day USD move by volume quintile",
     "Monotonic: quietest days → USD up, busiest → USD down. Seasonal adjustment sharpens the extremes. This "
     "is the raw relationship the volume filter and the ML model lean on.")}
<h2><span class="n">E2.</span> Attribution &amp; drivers</h2>
<div class="cols">
{fig("vm_attribution.png", "Net Sharpe by feature family",
     "Volume-only and calendar-only each beat the drift and are additive — the calendar carries the most, "
     "which is why the standalone calendar rule is the recommendation.")}
{fig("vm_coefs.png", "Walk-forward model coefficients",
     "Quincena/day-of-month and volume dominate; no technical indicators used.")}
</div>

<div class="part">Part F · Execution realism — VWAP vs the closing print</div>
<h2><span class="n">F1.</span> The recommended rule does not depend on catching the close</h2>
<p class="lead">Every result in this report is already priced <b>VWAP-to-VWAP</b> — the fill a desk can
actually work by spreading an order across the session, rather than the optimistic assumption of trading the
last print. Here is the proof for the strategy we care about: the same calendar rule, same 0.65 CRC
round-trip slippage, marked at the closing print vs the session VWAP. The VWAP fill <b>does not haircut the
edge — it slightly improves the Sharpe</b> ({EX['close']['sharpe']} → {EX['vwap']['sharpe']}), because the
VWAP is a steadier reference than a single closing tick.</p>
{exec_table()}
{fig("q_execution.png", "Recommended calendar rule — session VWAP vs closing-print execution",
     "Both curves use the identical signal and the identical 0.65 CRC round-trip slippage; only the fill "
     "reference differs. They track each other closely and the VWAP (realistic) curve carries the higher "
     "Sharpe — the calendar edge is an economic flow, not an artifact of marking to the close.")}
<h2><span class="n">F2.</span> Secondary strategies at VWAP (kept for completeness)</h2>
{tabs([("Why they're here", True,
        "<p style='font-size:13.5px;margin:10px 2px'>These trend / order-flow books are <b>not</b> the "
        "recommendation — they are shown only to demonstrate that the VWAP-execution finding generalises "
        "beyond the calendar rule. On the realistic VWAP basis the noisy daily order-flow book improves most "
        "(its turnover is where a single closing tick hurts), while the slow trend books barely move. If you "
        "only care about the recommended strategy, F1 above is the whole story.</p>" + vwap_table()),
       ("Close vs VWAP chart", False,
        fig("bt_vwap_close_compare.png", "Secondary books &amp; slippage — close-print vs VWAP execution",
            "Left priced at the close, right at the VWAP. The trend books (Donchian-40 " +
            str(BV['close_vs_vwap']['Donchian-40 trend']['close_sharpe']) + "→" +
            str(BV['close_vs_vwap']['Donchian-40 trend']['vwap_sharpe']) + ", SMA-120 " +
            str(BV['close_vs_vwap']['SMA-120 trend']['close_sharpe']) + "→" +
            str(BV['close_vs_vwap']['SMA-120 trend']['vwap_sharpe']) + ") barely move; the daily order-flow "
            "book improves most (" + str(BV['close_vs_vwap']['Order-flow daily']['close_sharpe']) + "→" +
            str(BV['close_vs_vwap']['Order-flow daily']['vwap_sharpe']) + ").")),
       ("Out-of-sample", False,
        fig("bt_vwap_oos.png", "Donchian-40 (VWAP-priced): in-sample vs out-of-sample net Sharpe",
            "Held out " + str(int(BVM['oos_frac']*100)) + "/" + str(int((1-BVM['oos_frac'])*100)) +
            ", the VWAP-priced Donchian-40 keeps working — OOS net Sharpe " +
            str(BV['Donchian-40_oos']['sharpe']) + " (≥ IS " + str(BV['Donchian-40_is']['sharpe']) + ").")),
      ])}

<div class="part">Part G · Verdict</div>
<ul class="find">
 <li><b>The recommendation is the calendar rule.</b> Short USD into Costa Rica's mid-month IVA / payroll
   deadline (within {CAL_PRE} business days of it, or day-of-month 5–15), long the rest of the month —
   <b>{usd(QCAL['per_year_usd'])}/yr per $1M at Sharpe {QCAL['sharpe']}</b>, VWAP-priced, {QCAL['roundtrips_yr']}
   trades/yr, worst year {usd(Q['refined_worst_year_usd'])}, positive in all 12. A slow 20-day volume filter
   lifts the risk-adjusted return to Sharpe {QSLOW['sharpe']} (drawdown just {usd(QSLOW['maxdd_usd'])}).
   Start here.</li>
 <li><b>A dynamic exit enhances it (A5).</b> Because the calendar edge front-loads and reverts, a
   {XM['trail_bps']}-bps <b>trailing stop</b> that banks each trade once the move stalls raises P&amp;L to
   {usd(XTR['full']['per_year_usd'])}/yr (Sharpe {XTR['full']['sharpe']}) — <b>up in both the in-sample and
   out-of-sample windows</b>. Adding a hard {XM['combo_floor_bps']}-bps floor takes the Sharpe to
   {XCB['full']['sharpe']} and nearly halves the worst drawdown ({usd(XB['full']['maxdd_usd'])} →
   {usd(XCB['full']['maxdd_usd'])}). Both were tuned on in-sample data only, and the whole 30–100-bps band
   works — it is an overlay on the calendar rule, not a replacement.</li>
 <li><b>It holds out of sample.</b> On a chronological 60/40 split (test window after {OC['split_date']},
   nothing re-fit) the rule earns {usd(OC['oos']['per_year_usd'])}/yr at Sharpe {OC['oos']['sharpe']}
   out-of-sample — if anything stronger than the {OC['is']['sharpe']} in-sample, because the post-2021 float
   is where the cash-flow cycle is most tradeable.</li>
 <li><b>Your volume thesis is correct but must be traded gently.</b> Low volume → USD up is real and
   monotonic, but a daily flip pays much of it to slippage even VWAP-priced. Use it conviction-sized or as
   the slow filter on the calendar trade — that is what lifts the combined/ML book to Sharpe
   {DY['Combined vote (3 signals)']['sharpe']}–{LG['sharpe']}.</li>
 <li><b>The mechanism is economic, not technical:</b> exporter USD supply (volume) and the payment/tax
   cycle (quincena). That is why it persists across regimes — and why it would weaken if the BCCR re-pegged
   or intervened heavily, the dominant risk.</li>
 <li><b>Priced realistically, the edge survives.</b> Everything here is marked at the session VWAP, not the
   closing print. For the recommended rule that is a slight <i>improvement</i>
   ({EX['close']['sharpe']} → {EX['vwap']['sharpe']} Sharpe), so the result is not an artifact of marking
   to the close.</li>
</ul>

<footer>
  <b>Dollar &amp; pricing convention.</b> $1,000,000 USD notional per full position; 1 bp of next-session move
  = $100; slippage 0.325 CRC/side (~$680) charged on every position change. <b>All P&amp;L is priced
  VWAP-to-VWAP</b> — a position decided from information through session <i>t</i> earns the next session's
  VWAP move, and the slippage is booked against the VWAP (the realistic desk fill), net, gross of
  financing/borrow and market impact, non-compounded (reset to $1M each day). The ML row is conviction-sized
  so its notional varies up to $1M; all model numbers are walk-forward out-of-sample. In-sample /
  out-of-sample = chronological 60/40 split at {OC['split_date']}. Reproducible:
  <code>parse_monex → analyze → eda → volume_model → dynamics → quincena → exits → backtest_vwap → build_report</code>.
</footer>

</div>{JS}</body></html>"""


def main():
    dst = OUT / "report.html"
    dst.write_text(HTML, encoding="utf-8")
    print(f"Wrote {dst} ({dst.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
