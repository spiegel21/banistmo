"""Build the self-contained visual report — condensed to the result, not the project history.

Five sections, one idea each: why the edge exists (the quincena cash-flow cycle and the
BCCR's footprint that marks the turn), the rule and what each layer earns, every strategy
on one honest basis, an overfitting check, and the verdict. The BCCR official-flow/reserves
data is treated as a first-class input to the mechanism, not a late addendum.

Everything is priced VWAP-to-VWAP (the realistic desk fill), $1,000,000 per trade, net of
the 0.65 CRC round-trip slippage, annualised on the venue's real ~231 sessions/yr — one
basis, sourced from src/basis.py, so every table agrees by construction. The out/*.json
files are the source of truth; tests/ reconciles this report against them.

Reads out/{quincena,dynamics,exit_lab,ranking,intervention}_results.json.
Run last, after the analysis modules and rank_strategies / exit_lab / intervention.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DY = json.loads((OUT / "dynamics_results.json").read_text())
Q = json.loads((OUT / "quincena_results.json").read_text())
RK = json.loads((OUT / "ranking.json").read_text())          # unified ranking, one 231-session basis
RKM = RK["_meta"]
XL = json.loads((OUT / "exit_lab.json").read_text())          # full exit engine, same basis
XLB = XL["baseline"]
XLR = XL["variants"][XL["recommended"]]                        # recommended overlay (trail 30 / floor 40)
XLP = XL["stage2_joint"]["bps"]                               # bps joint-grid selection block
IV = json.loads((OUT / "intervention_results.json").read_text())  # BCCR official-flow & reserves
IVF, IVR, IVS = IV["flow"], IV["reserves"], IV["strategy"]
IVRB = IV["reserve_vs_blind"]

QREC = Q["Calendar + slow-vol (recommended)"]                 # the recommended rule
OREC = Q["oos_recommended"]                                    # IS/OOS split for it
RD = Q["return_dist"]
M = DY["_meta"]
NOTIONAL = M["notional_usd"]

# ranking rows keyed by label, so every headline number ties to ranking.json --------------- #
RKBY = {r["label"]: r for r in RK["ranking"]}
R_ENTRY = RKBY["Real calendar (<=6d)"]
R_TRIM = RKBY["Calendar + slow-vol"]
R_EXIT = RKBY["Calendar + trail 30 + floor 40 (exit_lab)"]
R_LONG = RKBY["Always long USD"]
IV_RES = IVS["+ reserve-regime long-trim"]


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


# --------------------------------------------------------------------------- #
# KPI strip — the recommended rule, out-of-sample first
# --------------------------------------------------------------------------- #
KPIS = [
    (f"{OREC['oos']['sharpe']}", "out-of-sample Sharpe",
     f"recommended rule · {usd(OREC['oos']['per_year_usd'])}/yr · 40% held out"),
    (f"{OREC['is']['sharpe']} → {OREC['oos']['sharpe']}", "in-sample → out-of-sample",
     "the edge strengthens on held-out data, it does not decay"),
    (f"{QREC['sharpe']}", "Sharpe (full sample)",
     f"~{QREC['roundtrips_yr']:.0f} trades/yr · {usd(NOTIONAL)}/trade · VWAP-priced"),
    (f"{R_EXIT['is']['sharpe']} / {R_EXIT['oos']['sharpe']}", "with the exit overlay (IS/OOS)",
     f"top of the ranking · max DD {usd(XLR['is']['maxdd_usd'])} (§2)"),
    (f"{IVF['official_share_median']*100:.0f}%", "BCCR + public sector of MONEX volume",
     "the official hand marks the mid-month turn (§1)"),
    (f"{DY['vol_on_usd_down']:.0f} vs {DY['vol_on_usd_up']:.0f} M", "volume: colón up vs down",
     "exporters are the swing USD supply"),
]


def kpi_html():
    return "".join(f'<div class="kpi"><div class="kpi-v">{v}</div>'
                   f'<div class="kpi-l">{lab}</div><div class="kpi-s">{s}</div></div>'
                   for v, lab, s in KPIS)


# --------------------------------------------------------------------------- #
# §2 — one layered table: benchmark, then the rule built up one layer at a time
# --------------------------------------------------------------------------- #
def layer_table():
    """Each row adds ONE design choice on top of the row above. IS Sharpe is the selection
    metric (first 60%); OOS is the honest confirmation (last 40%, never used to choose).
    Every calendar/exit row ties to ranking.json; the reserve row to intervention.json."""
    def rk_row(name, r, note, best=False, grp=False):
        rc = ' class="rowbest"' if best else ""
        star = ' <span class="best">recommended</span>' if best else ""
        i, o = r["is"], r["oos"]
        return (f'<tr{rc}><td>{name}{star}<div class="muted" style="font-size:11px">{note}</div></td>'
                f'<td class="num {scls(i["sharpe"])}">{i["sharpe"]}</td>'
                f'<td class="num {scls(o["sharpe"])}">{o["sharpe"]}</td>'
                f'<td class="num">{usd(i["per_year_usd"])}</td>'
                f'<td class="num">{usd(i["maxdd_usd"])}</td></tr>')
    out = ['<table><tr><th>Layer · $1M/trade · VWAP-priced, net 0.65 CRC RT, 231 sess/yr</th>'
           '<th class="num">IS Sharpe</th><th class="num">OOS Sharpe</th>'
           '<th class="num">IS $/yr</th><th class="num">IS max DD</th></tr>']
    out.append('<tr><td colspan="5" class="grp">Benchmark</td></tr>')
    out.append(rk_row("Always long USD (buy-and-hold)", R_LONG,
                      "the naive alternative — dies out of sample"))
    out.append('<tr><td colspan="5" class="grp">The rule, one design choice at a time</td></tr>')
    out.append(rk_row("① Calendar entry — short USD into the deadline", R_ENTRY,
                      "≤6 business days before the IVA/quincena deadline, flat size"))
    out.append(rk_row("② + slow-volume size trim", R_TRIM,
                      "half size when the slow 20-day volume regime disagrees — cuts the fat losers",
                      best=True))
    out.append(rk_row("③ + trailing 30 / hard floor 40 exit", R_EXIT,
                      "optional timing overlay — best risk-adjusted, roughly halves drawdown"))
    # reserve row comes from intervention.json (full-sample rule, IS/OOS split)
    rt = IV_RES
    out.append(f'<tr><td>④ + BCCR reserve-regime long-trim'
               f'<div class="muted" style="font-size:11px">trim long-USD size when reserves rise — '
               f'strictly-causal monthly figure</div></td>'
               f'<td class="num {scls(rt["is"]["sharpe"])}">{rt["is"]["sharpe"]}</td>'
               f'<td class="num {scls(rt["oos"]["sharpe"])}">{rt["oos"]["sharpe"]}</td>'
               f'<td class="num muted">—</td>'
               f'<td class="num">{usd(rt["full"]["maxdd_usd"])}</td></tr>')
    out.append("</table>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# §3 — the unified ranking: every strategy re-priced through one code path
# --------------------------------------------------------------------------- #
def ranking_table():
    fam_col = {"exit": "blue", "calendar": "ok", "flow": "no", "trend": "mb", "benchmark": "muted"}
    out = ['<table><tr><th>#</th><th>Strategy · VWAP-priced, net 0.65 CRC RT, 231 sess/yr</th>'
           '<th>Family</th><th class="num">IS Sharpe</th><th class="num">OOS Sharpe</th>'
           '<th class="num">IS $/yr</th><th class="num">IS max DD</th></tr>']
    for r in RK["ranking"]:
        best = r["rank_is"] == 1
        rc = ' class="rowbest"' if best else ""
        star = ' <span class="best">top of book</span>' if best else ""
        fc = fam_col.get(r["family"], "muted")
        i, o = r["is"], r["oos"]
        out.append(f'<tr{rc}><td class="num">{r["rank_is"]}</td><td>{r["label"]}{star}</td>'
                   f'<td class="{fc}">{r["family"]}</td>'
                   f'<td class="num {scls(i["sharpe"])}">{i["sharpe"]}</td>'
                   f'<td class="num {scls(o["sharpe"])}">{o["sharpe"]}</td>'
                   f'<td class="num">{usd(i["per_year_usd"])}</td>'
                   f'<td class="num">{usd(i["maxdd_usd"])}</td></tr>')
    out.append("</table>")
    return "".join(out)


RULE_LOGIC = """<div class="rule"><h4>Exact logic</h4>
  <div class="logic">≤ 6 business days before the IVA (D-104) / quincena deadline (through it) →  SHORT USD $1M<br>
    &nbsp;&nbsp;&nbsp;(firms sell USD to raise colones for the 15th tax filing + payroll → colón strengthens)<br>
    otherwise →  LONG USD $1M   (supply fades, USD drifts back up)<br>
    size trim: hold HALF size whenever a slow 20-day volume regime disagrees with the trade<br>
    optional exit: bank the trade once it gives back 30 bps from its best (hard floor 40 bps)</div></div>"""


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>USD/CRC — The Calendar Strategy (condensed)</title>
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
 h2 {{ font-size:19px; margin:6px 0 14px; break-after:avoid; }}
 h2 .n {{ color:var(--accent); }}
 .kpis {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
 .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:15px; }}
 .kpi-v {{ font-size:20px; font-weight:700; letter-spacing:-.5px; }}
 .kpi-l {{ font-size:13px; margin-top:2px; }}
 .kpi-s {{ font-size:12px; color:var(--mut); margin-top:4px; }}
 .lead {{ background:var(--card); border-left:3px solid var(--accent); border-radius:0 10px 10px 0;
   padding:16px 20px; margin:8px 0 0; }}
 .lead b {{ color:var(--accent); }}
 .note {{ background:rgba(224,168,61,0.08); border:1px solid var(--warn); border-radius:10px;
   padding:12px 16px; margin:16px 0 0; font-size:13px; color:var(--ink); }}
 .note b {{ color:var(--warn); }}
 .rule {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:2px 18px;
   margin:12px 0; break-inside:avoid; }}
 .rule h4 {{ margin:14px 0 6px; font-size:15px; color:var(--blue); }}
 .rule .logic {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px;
   background:#0f1419; border:1px solid var(--line); border-radius:8px; padding:10px 12px; color:#cfe3d6; }}
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
 ul.find {{ list-style:none; padding:0; margin:14px 0 0; }}
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
  <h1>USD/CRC · MONEX — the calendar strategy</h1>
  <div class="sub">{RKM['n_days']:,} sessions · {RKM['date_min']} → {RKM['date_max']} · long/short USD ·
    {usd(NOTIONAL)} per trade · <b>priced VWAP-to-VWAP</b>, net of 0.65 CRC round-trip ·
    no technical indicators</div>
</header>

<p class="lead">One tradeable edge: <b>Costa Rica's mid-month payment calendar</b>. Firms sell USD to raise
colones for the 15th IVA (D-104) tax filing and the quincena payroll, so the colón strengthens <i>into</i> the
deadline and reverts after. The rule is <b>short USD within 6 business days of the deadline, long the rest of
the month</b>, at <b>half size whenever a slow 20-day volume regime disagrees</b>. Selection is disciplined —
variants are ranked <b>in-sample</b> (first 60% of history) and every headline here is the <b>out-of-sample</b>
realisation on the 40% held out after <b>{OREC['split_date']}</b>. On that held-out window the rule earns
<b>{usd(OREC['oos']['per_year_usd'])}/yr at Sharpe {OREC['oos']['sharpe']}</b> (full-sample Sharpe
{QREC['sharpe']}, ~{QREC['roundtrips_yr']:.0f} trades/yr), and it <b>strengthens out of sample</b>
({OREC['is']['sharpe']} → {OREC['oos']['sharpe']}) where the flat-sized entries decay. It is the only edge in
the study that works in <b>both</b> the pegged 2014–2021 regime and the 2021–2026 float.</p>

<div class="kpis" style="margin-top:16px">{kpi_html()}</div>

<p class="note"><b>One accounting basis throughout.</b> Every figure — every table and chart — is priced
<b>VWAP-to-VWAP</b> next session, <b>{usd(NOTIONAL)}</b> per trade, net of the <b>0.65 CRC</b> round-trip
slippage (0.325/side), annualised on the <b>{RKM['sessions_per_year_actual']:.0f}</b> sessions/yr MONEX
actually trades — not the 252 equity-market convention. All modules read this from one file
(<code>src/basis.py</code>), so no two tables can silently disagree; <code>tests/</code> reconciles this
report against the <code>out/*.json</code> source of truth.</p>

<div class="part">§1 · Why it works — the cash-flow cycle and the BCCR's hand</div>
<h2><span class="n">1.</span> The mechanism, in one picture</h2>
<p class="lead">Re-index the price series from raw day-of-month to <b>business-days-to-the-deadline</b> and the
mechanism is unmistakable. Private firms convert USD→CRC for the 15th, a recurring supply surge that
strengthens the colón into the deadline (−9 to −16 bps next-day) and reverts after. The BCCR is not a
bystander: official flow — the central bank plus the non-bank public sector (RECOPE et al.) routed through
MONEX — is a <b>median {IVF['official_share_median']*100:.0f}%</b> of daily volume, and official USD demand
<b>peaks exactly at the turn</b>. Heavy official demand does not push USD up; it marks the point where private
mid-month supply overwhelms (contemporaneous corr {IVF['corr_contemp']:+.2f}). So the calendar reversion and
the official footprint are two views of one cash-flow cycle.</p>
{fig("iv_intramonth.png", "The turn: next-day USD move and official USD demand, by trading-days-to the deadline",
     f"Bars: BCCR + public-sector net USD demand ($M). Line: average next-day USD move (bps). Both are indexed "
     f"to the statutory IVA (15th) / mid-month payroll deadline, snapped to MONEX's own trading calendar so "
     f"weekend/holiday rolling is exact. Official demand crests in the shaded window right as the colón "
     f"strengthens hardest — the reversion the exit overlay (§2) harvests. The signal decays to "
     f"{IVF['corr_lag1']:+.2f} when lagged one session and depends on same-day disclosure, so it is the "
     f"mechanism, not a live entry.")}
{fig("dyn_mechanism.png", "Why volume is the size filter: exporters are the swing USD supply",
     f"On days the colón strengthens the market trades {DY['vol_on_usd_down']:.0f}M vs {DY['vol_on_usd_up']:.0f}M "
     f"when it weakens — USD sellers (exporters, remittances, FDI) come in size, buyers trickle. So a thin "
     f"market drifts USD up next day ({DY['nextmove_after_lowvol']:+.1f} bps after quiet days vs "
     f"{DY['nextmove_after_highvol']:+.1f} after busy ones). That asymmetry is why a <b>slow</b> volume regime "
     f"trims size on trades the supply picture does not confirm — without ever flipping the direction.")}

<div class="part">§2 · The rule &amp; what each layer earns</div>
<h2><span class="n">2.</span> Built one design choice at a time</h2>
{RULE_LOGIC}
<p class="sub" style="margin:0 0 8px">Each row adds ONE choice on top of the row above. <b>IS Sharpe</b> is the
selection metric (first 60%); <b>OOS Sharpe</b> is the honest confirmation on the 40% held out after
{OREC['split_date']} and never used to choose. The recommended always-in-market rule is <b>② Calendar +
slow-vol</b>; the exit and reserve layers (③–④) are optional <b>risk</b> improvements, not more alpha.</p>
{layer_table()}
<p class="lead">Read the layers as risk, not money. The <b>slow-vol trim</b> (②) is what makes it trade-ready —
it holds or improves out of sample while flat-sized entries decay, turning out-of-sample per-trade skew
positive (<b>{RD['Calendar + slow-vol']['skew']:+.2f}</b>) and halving the worst trade
({usd(RD['Base quincena (≤15)']['min'])} → {usd(RD['Calendar + slow-vol']['min'])}). The <b>exit overlay</b>
(③) is a second, independent tail control: a paired t-test on daily OOS P&amp;L gives
p = {XL['alpha_vs_risk']['paired_p']} — total P&amp;L is statistically unchanged, but max drawdown drops
~{abs(XL['alpha_vs_risk']['oos_maxdd_delta'])/abs(XLB['oos']['maxdd_usd'])*100:.0f}% and OOS Sharpe rises
+{XL['alpha_vs_risk']['oos_sharpe_delta']}. A take-profit actively <i>hurts</i> (the edge is a drift into the
deadline) and is excluded. The <b>reserve trim</b> (④) beats a blind "halve every long" control by
{usd(IVRB['oos_total_delta_usd'])} out of sample (paired t = {IVRB['paired_t']}, p = {IVRB['paired_p']}) — the
reserve regime picks <i>which</i> longs to cut.</p>
{fig("q_position_shading.png", "What the rule does: session VWAP shaded by the recommended position",
     "Green = long USD, red = short USD (short ≤6 business days into the deadline). The rule is always in a "
     "position — no flat state — and holds these at half size when the slow volume regime disagrees (same "
     "colour, thinner book). The short bands cluster mid-month, exactly where the cash-flow supply surge lands.")}

<div class="part">§3 · Every strategy on one honest basis</div>
<h2><span class="n">3.</span> One code path, one ranking — the master comparison</h2>
<p class="lead">Every rule in the study — the calendar family, the daily flow signals (order-flow, volume,
skew), trend books, the ML model, and the two always-on benchmarks — is re-priced through a <b>single</b> code
path (VWAP-to-VWAP, {usd(NOTIONAL)}/trade, {RKM['sessions_per_year_actual']:.0f} sessions/yr) and ranked by
<b>in-sample</b> Sharpe. This one table and chart replace every per-strategy tearsheet: the calendar family
owns the top, the flow family owns the bottom, and the out-of-sample column (carried for honesty, never used to
order) tells you which survive.</p>
{ranking_table()}
{fig("ranking.png", "Every strategy on one accounting basis, ranked by in-sample net Sharpe",
     "Faded bar = held-out confirmation. The calendar family (green) plus its exit overlays (red) take the top "
     "seven seats; the flow rules (orange) sit at the bottom with " +
     usd(RK['ranking'][14]['is']['maxdd_usd']) + "-class drawdowns despite occasionally strong OOS Sharpe — the "
     "path is not fundable. The mirror-image benchmarks (Always long IS " +
     f"{R_LONG['is']['sharpe']} / OOS {R_LONG['oos']['sharpe']}, Always short " +
     f"{RKBY['Always short USD']['is']['sharpe']} / {RKBY['Always short USD']['oos']['sharpe']}) confirm the "
     "P&L and cost accounting.")}
<ul class="find">
 <li><b>The flow signals are regime-contingent.</b> Order-flow, volume, and skew are worthless-to-negative on
   the pegged 2014–2021 window and only fire on the 2021–2026 float, so their full-sample numbers average a
   dead regime with a live one — and they carry {usd(RK['ranking'][14]['is']['maxdd_usd'])}-class in-sample
   drawdowns at 80–97 round trips/yr. Their value here is only as the <i>slow filter</i> that sizes the
   calendar trade.</li>
 <li class="muted"><b>Honest caveat.</b> Because in-sample is the pegged/low-vol regime and out-of-sample is
   the float, in-sample selection structurally favours whatever works in both. The calendar rule wins partly
   <i>because</i> it is that edge — a real virtue, but not the same claim as "the flow signals never work."</li>
</ul>

<div class="part">§4 · Is it overfit? One look</div>
<h2><span class="n">4.</span> A plateau, a clean held-out split, and passed controls</h2>
<div class="cols">
{fig("q_sensitivity.png", "Net Sharpe across every choice of the short-window start &amp; end",
     "A broad plateau (Sharpe " + str(Q['window_sharpe_min']) + "–" + str(Q['window_sharpe_max']) + "), not a "
     "lucky point — days 5→15 is simply the natural mid-month supply window and every neighbour works. The "
     "≤6-day anchor, 20-day volume window and half-size trim are fixed a-priori, not fitted.")}
{fig("iv_strategy.png", "Held-out equity: the exit overlay + reserve trim vs the plain rule",
     "Red line = the in-sample / out-of-sample boundary; nothing right of it is fitted. Both curves compound "
     "through the boundary and the overlaid rule's drawdowns (lower panel) stay visibly shallower, most "
     "clearly through the deep late-sample selloff.")}
</div>
<ul class="find">
 <li><b>Held out, the edge grows.</b> The recommended rule goes IS {OREC['is']['sharpe']} → OOS
   {OREC['oos']['sharpe']}; the exit overlay IS {R_EXIT['is']['sharpe']} → OOS {R_EXIT['oos']['sharpe']} — no
   decay. Across the exit parameter grid corr(IS, OOS) = {XLP['is_oos_corr']:+.2f}, so in-sample ranking
   genuinely predicts out-of-sample, and the deflated Sharpe (Bailey / López de Prado) over
   {XLP['n_trials']} trials gives P(true&gt;0) ≈ {XLP['deflated_sharpe_prob']}.</li>
 <li><b>Positive every year.</b> The rule is positive in all 12 calendar years (worst
   {usd(Q['refined_worst_year_usd'])}), including the 2015–17 pegged years where the volume signal was dead.
   Priced at the closing print instead of the VWAP the edge slightly <i>improves</i> — it is an economic flow,
   not an artifact of marking to the close.</li>
</ul>

<div class="part">§5 · Verdict &amp; risks</div>
<ul class="find">
 <li><b>Trade the calendar + slow-vol rule.</b> Short USD into Costa Rica's mid-month IVA / payroll deadline
   (within 6 business days), long the rest of the month, at half size when a slow 20-day volume regime
   disagrees. Full-sample {usd(QREC['per_year_usd'])}/yr at Sharpe {QREC['sharpe']},
   ~{QREC['roundtrips_yr']:.0f} trades/yr — and, the number that matters, <b>out-of-sample Sharpe
   {OREC['oos']['sharpe']}</b> ({usd(OREC['oos']['per_year_usd'])}/yr on the 40% held out).</li>
 <li><b>The optional overlays are risk, not alpha.</b> A trailing 30 / hard-floor 40 exit tops the ranking
   (IS {R_EXIT['is']['sharpe']} → OOS {R_EXIT['oos']['sharpe']}) and roughly halves drawdown, and a BCCR
   reserve-regime long-trim lifts OOS Sharpe to {IV_RES['oos']['sharpe']} — both smoother paths to size up,
   with total P&amp;L statistically unchanged. Add them if you want the drawdown, skip them for simplicity.</li>
 <li><b>The mechanism is economic, not technical:</b> exporter USD supply and the tax/payroll cycle. That is
   why it persists across the peg and the float — and why the <b>dominant, unpriceable risk is a re-peg or
   heavy BCCR intervention</b>: the bank is already ~half of MONEX and could mute the edge at will. The reserve
   trim is a partial hedge, not an escape.</li>
 <li><b>Limits.</b> The float that carries most of the edge is only the last ~4.5 years; results are net of
   VWAP slippage but gross of financing/borrow and market impact; the colón is not freely shortable for all
   participants — this is a directional/treasury edge, not a market-neutral one.</li>
</ul>

<footer>
  <b>Dollar &amp; pricing convention.</b> {usd(NOTIONAL)} USD notional per full position; 1 bp of next-session
  move = $100; slippage 0.325 CRC/side charged on every position change. <b>All P&amp;L is priced
  VWAP-to-VWAP</b> — a position decided from information through session <i>t</i> earns the next session's VWAP
  move — net, gross of financing/borrow and market impact, non-compounded (reset to {usd(NOTIONAL)} each day).
  <b>Selection discipline:</b> variants are ranked on the in-sample Sharpe (first 60%, {OREC['is_start']} →
  {OREC['is_end']}); every headline result is the out-of-sample realisation on the held-out 40%
  ({OREC['split_date']} → {OREC['oos_end']}). Nothing is re-fit on the test window. Every part is annualised on
  the venue's real {RKM['sessions_per_year_actual']:.0f} sessions/yr from a single basis
  (<code>src/basis.py</code>). Inputs: one BCCR MONEX price/volume export plus two BCCR official-flow exports
  (FX intervention CodCuadro 1587, net reserves CodCuadro 8). Reproducible:
  <code>parse_monex → parse_bccr → analyze → … → rank_strategies → exit_lab → intervention → build_report</code>.
  See <b>FINDINGS.md</b> for the full written record.
</footer>

</div></body></html>"""


def main():
    dst = OUT / "report.html"
    dst.write_text(HTML, encoding="utf-8")
    print(f"Wrote {dst} ({dst.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
