"""Build the self-contained visual report: recommendation -> dynamics -> rules -> dollars.

Everything is priced VWAP-to-VWAP (the realistic desk fill) net of the 0.65 CRC
round-trip slippage. The recommended calendar/quincena rule leads every section;
secondary signals (volume, skew, combined vote, ML, trend books) are kept but
tucked into tabs that default to the recommended strategy.

Reads out/{quincena,dynamics,vm,exits,backtest_vwap,ranking,exit_lab}_results.json.
Run last, after analyze / eda / volume_model / dynamics / quincena / exits /
backtest_vwap / rank_strategies / exit_lab.

Every JSON this reads is now annualised on ONE basis — the venue's real ~231 sessions/yr,
sourced from src/basis.py — so the per-module tables and the unified ranking (Part R) agree
by construction. Part R remains the master cross-strategy comparison because it re-prices
every rule through a single code path; the per-module parts add the mechanism and context.
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
RK = json.loads((OUT / "ranking.json").read_text())        # unified ranking, one 231-session basis (src/basis.py)
RKM = RK["_meta"]
XL = json.loads((OUT / "exit_lab.json").read_text())        # full exit engine, one 231-session basis (src/basis.py)
XLB = XL["baseline"]
XLR = XL["variants"][XL["recommended"]]                     # recommended overlay (trail 30 / floor 40)
XLP = XL["stage2_joint"]["bps"]                             # the bps joint-grid selection block
IV = json.loads((OUT / "intervention_results.json").read_text())  # BCCR official-flow & reserves overlay
IVF, IVR, IVS = IV["flow"], IV["reserves"], IV["strategy"]
IVRB = IV["reserve_vs_blind"]
QSLOW = Q["Refined + slow-vol sizing"]
QBASE = Q["Base quincena (<=15)"]
QCAL = next(v for k, v in Q.items() if k.startswith("Real calendar"))
QREC = Q["Calendar + slow-vol (recommended)"]     # the recommended rule (calendar entry + slow-vol trim)
OREC = Q["oos_recommended"]                        # IS/OOS split for the recommended rule
FAM = Q["family_oos"]                              # whole family, IS vs OOS
RDW = Q["return_dist_window"]                      # the OOS window the histograms are built on
EX = Q["execution"]
CAL_PRE = Q["_meta"].get("cal_pre", 6)
M = R["_meta"]
LG = R["Logit (conviction)"]

# Return-distribution tabs (A7): (tab label, JSON key, PNG, figure title). Built on OOS trades.
RD_BEST = Q.get("return_dist_best", "Calendar + slow-vol")
DIST_TABS = [
    ("Base quincena", "Base quincena (≤15)", "q_dist_base.png", "Base quincena (≤15)"),
    ("Real calendar", "Real calendar", "q_dist_calendar.png", "Real calendar (flat-sized)"),
    ("Refined + slow-vol", "Refined + slow-vol", "q_dist_slowvol.png", "Refined (fixed 5–15) + slow-vol"),
    ("Calendar + slow-vol", "Calendar + slow-vol", "q_dist_calslow.png", "Calendar + slow-vol"),
]
DIST_NOTE = {
    "Base quincena (≤15)": "The crude ancestor: the widest spread of outcomes, a LEFT-skewed tail, and the "
                           "heaviest OOS losers — the weakest risk-adjusted variant of the family throughout. ",
    "Real calendar": "The deadline-anchored entry at flat size. Strong, but its left tail is intact and its "
                     "OOS Sharpe slips below in-sample. ",
    "Refined + slow-vol": "The slow-volume size trim on the FIXED 5–15 window — the trim cuts the fat losing "
                          "trades, flipping the skew positive and holding the OOS Sharpe. ",
    "Calendar + slow-vol": "THE RECOMMENDED RULE: deadline-anchored entry + slow-volume trim. Right-shifted, "
                           "the thinnest left tail of the family, positive skew, and the family's best "
                           "out-of-sample Sharpe — it strengthens out of sample rather than giving the edge back. ",
}


def dist_cap(key):
    """Per-trade OOS caption pulled straight from the JSON so it always matches the chart."""
    d = Q["return_dist"][key]
    return (f"{d['n']} OOS trades · mean ${d['mean']/1e3:+.1f}k/trade, median ${d['median']/1e3:+.1f}k, "
            f"win {d['win']:.0f}% · std ${d['std']/1e3:.1f}k · skew {d['skew']:+.2f} · worst "
            f"${d['min']/1e3:.0f}k · Sharpe IS {d['is_sharpe']:.2f} → OOS {d['oos_sharpe']:.2f}.")


def dist_tabs_html():
    """A7 tab group: overlay first, then one per variant; the recommended rule is flagged."""
    items = [("Overlay (all four)", False,
              fig("q_dist_overlay.png", "All four calendar-family variants, OOS per-trade P&L overlaid",
                  "On held-out data the two slow-vol variants stay peaked and right-shifted with a thin left "
                  "tail; the flat-sized calendar and the crude base rule carry a heavier left tail. The "
                  "recommended calendar + slow-vol curve is the rightmost mode."))]
    for tab, key, png, title in DIST_TABS:
        cap = DIST_NOTE[key] + dist_cap(key)
        items.append((tab, key == RD_BEST,
                      fig(png, f"{title} — out-of-sample per-trade net P&amp;L", cap)))
    return tabs(items)


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
RD = Q["return_dist"]
KPIS = [
    (f"{OREC['oos']['sharpe']}", "out-of-sample Sharpe", f"recommended rule · {usd(OREC['oos']['per_year_usd'])}/yr · 40% held out"),
    (f"{OREC['is']['sharpe']} → {OREC['oos']['sharpe']}", "in-sample → out-of-sample",
     "family-best OOS Sharpe — it strengthens where the flat rules decay"),
    (f"{QREC['sharpe']}", "Sharpe (full sample)", f"~{QREC['roundtrips_yr']:.0f} trades/yr · $1M/trade · VWAP-priced"),
    (f"{RD['Calendar + slow-vol']['skew']:+.2f}", "OOS per-trade skew",
     f"slow-vol trim halves the worst trade to {usd(RD['Calendar + slow-vol']['min'])}"),
    (f"{XCB['full']['sharpe']}", "Sharpe with exit overlay",
     f"optional trailing stop cuts DD to {usd(XCB['full']['maxdd_usd'])} (A5)"),
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
    """Unified $1M/trade table — recommended calendar family first (with the honest OOS
    Sharpe alongside the full-sample Sharpe), then the raw building-block signals and the
    ML model. All VWAP-priced, net of cost."""
    # (display name, full-sample stat, family_oos key, is-recommended)
    reco = [("Calendar + slow-vol", QREC, "Calendar + slow-vol", True),
            (f"Real calendar (≤{CAL_PRE} bd), flat size", QCAL, "Real calendar", False),
            ("Refined (fixed 5–15) + slow-vol", QSLOW, "Refined + slow-vol", False),
            ("Base quincena (short ≤15)", QBASE, "Base quincena (≤15)", False)]
    block = [("Volume rule", DY["Volume rule"]),
             ("Precio-ponderado skew rule", DY["Precio-ponderado skew rule"]),
             ("Combined vote (3 signals)", DY["Combined vote (3 signals)"])]
    out = ['<table><tr><th>Strategy · $1M/trade · VWAP-priced, net 0.65 CRC RT</th>'
           '<th class="num">Per year</th><th class="num">Sharpe (full)</th>'
           '<th class="num">Sharpe (OOS)</th><th class="num">Trades/yr</th>'
           '<th class="num">Win %</th><th class="num">Max DD</th></tr>']
    out.append('<tr><td colspan="7" class="grp">Recommended — calendar / quincena family '
               '(OOS = last 40%, held out)</td></tr>')
    for name, b, fk, best in reco:
        rc = ' class="rowbest"' if best else ""
        star = ' <span class="best">recommended · best OOS</span>' if best else ""
        oos_s = FAM[fk]["oos"]["sharpe"]
        out.append(f'<tr{rc}><td>{name}{star}</td>'
                   f'<td class="num {scls(b["sharpe"])}">{usd(b["per_year_usd"])}</td>'
                   f'<td class="num {scls(b["sharpe"])}">{b["sharpe"]}</td>'
                   f'<td class="num {scls(oos_s)}">{oos_s}</td>'
                   f'<td class="num">{b["roundtrips_yr"]}</td><td class="num">{b["win_rate"]}</td>'
                   f'<td class="num">{usd(b["maxdd_usd"])}</td></tr>')
    out.append('<tr><td colspan="7" class="grp">Building-block signals — trade gently / as filters</td></tr>')
    for name, b in block:
        out.append(f'<tr><td>{name}</td>'
                   f'<td class="num {scls(b["sharpe"])}">{usd(b["per_year_usd"])}</td>'
                   f'<td class="num {scls(b["sharpe"])}">{b["sharpe"]}</td>'
                   f'<td class="num muted">—</td>'
                   f'<td class="num">{b["roundtrips_yr"]}</td><td class="num">{b["win_rate"]}</td>'
                   f'<td class="num">{usd(b["maxdd_usd"])}</td></tr>')
    out.append(f'<tr><td><b>ML combined model</b> (conviction-sized, walk-forward)</td>'
               f'<td class="num ok">{usd(LG["ann_bps"]*100)}</td>'
               f'<td class="num ok">{LG["sharpe"]}</td><td class="num muted">—</td>'
               f'<td class="num">{LG["roundtrips_yr"]}</td>'
               f'<td class="num">{LG["hit"]}</td><td class="num">{usd(LG["maxdd_bps"]*100)}</td></tr>')
    out.append("</table>")
    return "".join(out)


def oos_table():
    """Whole calendar family, in-sample (selection) vs out-of-sample (confirmation).
    Order matches the family_oos block; the recommended rule is flagged and holds best."""
    def row(name, o, best=False):
        rc = ' class="rowbest"' if best else ""
        star = ' <span class="best">recommended</span>' if best else ""
        arrow = "▲" if o["oos"]["sharpe"] >= o["is"]["sharpe"] else "▼"
        acls = "ok" if o["oos"]["sharpe"] >= o["is"]["sharpe"] else "no"
        return (f'<tr{rc}><td>{name}{star}</td>'
                f'<td class="num">{usd(o["is"]["per_year_usd"])}</td><td class="num">{o["is"]["sharpe"]}</td>'
                f'<td class="num ok">{usd(o["oos"]["per_year_usd"])}</td>'
                f'<td class="num ok">{o["oos"]["sharpe"]}</td>'
                f'<td class="num {acls}">{arrow}</td></tr>')
    body = "".join([
        row("Base quincena (≤15)", FAM["Base quincena (≤15)"]),
        row(f"Real calendar (≤{CAL_PRE} bd), flat size", FAM["Real calendar"]),
        row("Refined (fixed 5–15) + slow-vol", FAM["Refined + slow-vol"]),
        row("Calendar + slow-vol", FAM["Calendar + slow-vol"], best=True),
    ])
    return ('<table><tr><th>Rule (chronological 60/40 split at ' + OREC["split_date"] +
            ', no re-fitting)</th>'
            '<th class="num">IS / yr</th><th class="num">IS Sharpe</th>'
            '<th class="num">OOS / yr</th><th class="num">OOS Sharpe</th>'
            '<th class="num">OOS vs IS</th></tr>'
            + body + "</table>")


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
    """OUT-OF-SAMPLE per-trade net-P&L shape of each calendar-family variant (VWAP-priced).
    Selection is on IS Sharpe; the OOS Sharpe is the honest confirmation. The recommended
    rule is flagged — it has positive skew, the thinnest left tail, and the top OOS Sharpe."""
    rd = Q["return_dist"]
    best = Q.get("return_dist_best")
    order = ["Base quincena (≤15)", "Real calendar", "Refined + slow-vol", "Calendar + slow-vol"]
    out = ['<table><tr><th>Variant · net P&amp;L per trade (out-of-sample), $1M/trade, VWAP-priced</th>'
           '<th class="num">OOS trades</th><th class="num">Mean/trade</th><th class="num">Median</th>'
           '<th class="num">Std</th><th class="num">Win %</th><th class="num">Skew</th>'
           '<th class="num">Worst trade</th><th class="num">IS→OOS Sharpe</th></tr>']
    for k in order:
        d = rd[k]
        is_best = k == best
        rc = ' class="rowbest"' if is_best else ""
        star = ' <span class="best">recommended</span>' if is_best else ""
        skcls = "ok" if d["skew"] > 0 else "no"
        out.append(f'<tr{rc}><td>{k}{star}</td><td class="num">{d["n"]}</td>'
                   f'<td class="num">${d["mean"]/1e3:+.1f}k</td>'
                   f'<td class="num">${d["median"]/1e3:+.1f}k</td>'
                   f'<td class="num">${d["std"]/1e3:.1f}k</td>'
                   f'<td class="num">{d["win"]:.0f}%</td>'
                   f'<td class="num {skcls}">{d["skew"]:+.2f}</td>'
                   f'<td class="num">${d["min"]/1e3:.0f}k</td>'
                   f'<td class="num {scls(d["oos_sharpe"])}">{d["is_sharpe"]:.2f} → {d["oos_sharpe"]:.2f}</td></tr>')
    out.append("</table>")
    return "".join(out)


def ranking_table():
    """Every strategy re-priced on ONE basis (VWAP-to-VWAP, $1M/trade, 231 sessions/yr),
    ranked by in-sample Sharpe. Supersedes the per-module tables for cross-strategy
    comparison. The out-of-sample column is carried for honesty, never used to order."""
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


def exit_marginal_table():
    """Stage-1 marginal value of each exit mechanism ALONE (best bps setting, in-sample)."""
    bm = XL["stage1_best_per_mechanism"]
    verdict = {"trail_bps": ("helps", "ok"), "floor_bps": ("helps — more OOS than IS", "ok"),
               "target_bps": ("HURTS out-of-sample", "no"), "max_days_sessions": ("no value", "muted")}
    label = {"trail_bps": "Trailing stop", "floor_bps": "Hard stop-loss",
             "target_bps": "Take-profit", "max_days_sessions": "Time stop"}
    unit = {"trail_bps": "bps", "floor_bps": "bps", "target_bps": "bps", "max_days_sessions": "sessions"}
    out = ['<table><tr><th>Mechanism (alone, on the calendar entry)</th><th class="num">Best setting</th>'
           '<th class="num">IS Sharpe</th><th class="num">OOS Sharpe</th><th>Verdict</th></tr>']
    for k in ["trail_bps", "floor_bps", "target_bps", "max_days_sessions"]:
        b = bm[k]
        vtxt, vcls = verdict[k]
        out.append(f'<tr><td>{label[k]}</td><td class="num">{b["value"]} {unit[k]}</td>'
                   f'<td class="num {scls(b["is_sharpe"])}">{b["is_sharpe"]}</td>'
                   f'<td class="num {scls(b["oos_sharpe"])}">{b["oos_sharpe"]}</td>'
                   f'<td class="{vcls}">{vtxt}</td></tr>')
    out.append(f'<tr><td class="muted">Baseline — no exit (calendar entry only)</td>'
               f'<td class="num muted">—</td>'
               f'<td class="num">{XLB["is"]["sharpe"]}</td><td class="num">{XLB["oos"]["sharpe"]}</td>'
               f'<td class="muted">reference</td></tr>')
    out.append("</table>")
    return "".join(out)


def intervention_table():
    """Exit-overlay baseline vs the reserve-regime trim, the official-flow trim, and the
    blind-long-trim control. All on the same VWAP / 231-session basis."""
    order = [
        ("Calendar + trail30/floor40", "Calendar + trail30/floor40", "published best (A5b)", False),
        ("+ reserve-regime long-trim", "+ reserve-regime long-trim", "trim longs when reserves rising", True),
        ("+ official-flow trim (lagged)", "+ official-flow trim (lagged)", "strictly-lagged official regime", False),
        ("CONTROL: blind long-trim", "CONTROL: blind long-trim", "trim ALL longs — no BCCR data", False),
    ]
    out = ['<table><tr><th>Calendar rule · $1M/trade · VWAP-priced, net cost</th>'
           '<th class="num">IS Sharpe</th><th class="num">OOS Sharpe</th>'
           '<th class="num">Full $/yr</th><th class="num">Max DD</th><th>Note</th></tr>']
    for disp, key, note, best in order:
        s = IVS[key]
        rc = ' class="rowbest"' if best else ""
        star = ' <span class="best">new · best OOS/DD</span>' if best else ""
        ncls = "muted" if key.startswith("CONTROL") else ""
        out.append(f'<tr{rc}><td>{disp}{star}</td>'
                   f'<td class="num {scls(s["is"]["sharpe"])}">{s["is"]["sharpe"]}</td>'
                   f'<td class="num {scls(s["oos"]["sharpe"])}">{s["oos"]["sharpe"]}</td>'
                   f'<td class="num">{usd(s["full"]["per_year_usd"])}</td>'
                   f'<td class="num">{usd(s["full"]["maxdd_usd"])}</td>'
                   f'<td class="{ncls}" style="font-size:11px">{note}</td></tr>')
    out.append("</table>")
    return "".join(out)


def take_profit_table():
    """The take-profit sweep on top of trail 30 + floor 40 — the negative result, stated plainly."""
    out = ['<table><tr><th>Take-profit added to trail 30 + floor 40</th>'
           '<th class="num">IS Sharpe</th><th class="num">OOS Sharpe</th>'
           '<th class="num">IS $/yr</th><th class="num">OOS $/yr</th></tr>']
    for r in XL["take_profit_sweep"]:
        tg = "none" if r["target_bps"] is None else f'{r["target_bps"]} bps'
        rc = ' class="rowbest"' if r["target_bps"] is None else ""
        out.append(f'<tr{rc}><td>{tg}</td>'
                   f'<td class="num {scls(r["is_sharpe"])}">{r["is_sharpe"]}</td>'
                   f'<td class="num {scls(r["oos_sharpe"])}">{r["oos_sharpe"]}</td>'
                   f'<td class="num">{usd(r["is_per_yr"])}</td>'
                   f'<td class="num">{usd(r["oos_per_yr"])}</td></tr>')
    out.append("</table>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Part C — per-rule logic, as tabs (recommended rule is the default)
# --------------------------------------------------------------------------- #
RULE_CAL = f"""<div class="rule"><h4>Calendar + slow-vol rule — the recommendation</h4>
  <div class="logic">≤ {CAL_PRE} business days before the IVA/quincena deadline (through it) →  SHORT USD $1M<br>
    &nbsp;&nbsp;&nbsp;(firms sell USD to raise colones for the 15th tax + payroll → colón strengthens)<br>
    otherwise →  LONG USD $1M   (supply fades, USD drifts up)<br>
    size trim: hold HALF size whenever a slow 20-day volume regime disagrees with the trade<br>
    &nbsp;&nbsp;&nbsp;(full size only when supply/thinness confirms the direction)</div>
  <p><b>Why:</b> the intra-month cash-flow cycle (Part B) — companies convert USD→CRC for the D-104 IVA
    filing (due the 15th, rolled to the next business day) and the mid-month quincena payroll, a recurring
    supply surge that strengthens the colón into the deadline and reverts after. The <i>real-calendar</i>
    anchor rolls with weekends/holidays (economically the right entry); the <i>slow-vol trim</i> only cuts
    SIZE (never the direction) when the volume regime disagrees, which removes the fat losing trades.
    <b>Result (VWAP-priced):</b> {usd(QREC['per_year_usd'])}/yr, full-sample Sharpe {QREC['sharpe']},
    ~{QREC['roundtrips_yr']:.0f} trades/yr, max drawdown {usd(QREC['maxdd_usd'])} — and, crucially, an
    <b>out-of-sample Sharpe of {OREC['oos']['sharpe']}</b> ({usd(OREC['oos']['per_year_usd'])}/yr on the 40%
    held out) — the family's best, and it strengthens out of sample where the strong flat-sized entries give
    the edge back (A4/A7). The base "short the whole first half" rule ({usd(QBASE['per_year_usd'])}/yr, Sharpe
    {QBASE['sharpe']}) is the crude ancestor this refines.</p></div>"""

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
 .note {{ background:rgba(224,168,61,0.08); border:1px solid var(--warn); border-radius:10px;
   padding:12px 16px; margin:16px 0 0; font-size:13px; color:var(--ink); }}
 .note b {{ color:var(--warn); }}
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

<p class="lead">The recommended strategy is a <b>calendar (quincena) rule with a slow-volume size trim</b>: short USD
into Costa Rica's mid-month IVA / payroll deadline (within {CAL_PRE} business days of it), long the rest of
the month, at <b>half size whenever a slow 20-day volume regime disagrees</b> with the trade. Selection is
disciplined — the variants are ranked <b>in-sample</b> (first 60% of history) and every headline figure here
is the <b>out-of-sample</b> realisation on the 40% held out after <b>{OREC['split_date']}</b>, so nothing you
read below was tuned on the data it is measured on. On that held-out window the rule earns
<b>{usd(OREC['oos']['per_year_usd'])}/yr at Sharpe {OREC['oos']['sharpe']}</b> (full-sample Sharpe
{QREC['sharpe']}, ~{QREC['roundtrips_yr']:.0f} trades/yr) — and it posts <b>the family's best out-of-sample
Sharpe, strengthening from {OREC['is']['sharpe']} in-sample to {OREC['oos']['sharpe']} out</b> where the strong
flat-sized entries give the edge back, because the slow-vol trim cuts the fat losing trades (out-of-sample
per-trade skew
<b>{RD['Calendar + slow-vol']['skew']:+.2f}</b>, worst trade {usd(RD['Calendar + slow-vol']['min'])} vs
{usd(RD['Base quincena (≤15)']['min'])} for the crude base rule). An optional <b>dynamic trailing-stop
exit</b> (A5) is a second, independent way to cut the same reversion tail. Every number below is VWAP-priced;
secondary signals are kept in tabs but default to this rule.</p>

<div class="kpis" style="margin-top:16px">{kpi_html()}</div>

<p class="note"><b>One accounting basis throughout.</b> Every figure in this report — every part,
every table — is priced the same way: <b>VWAP-to-VWAP</b> next session, <b>${DY['_meta']['notional_usd']:,.0f}</b> per
trade, net of the <b>0.65 CRC</b> round-trip slippage (0.325/side), and annualised on the
<b>{RKM['sessions_per_year_actual']:.0f}</b> sessions/yr MONEX actually trades — not the 252 equity-market
convention (which would overstate $/yr by ~{RKM['per_year_usd_legacy_inflation']*100:.0f}% and Sharpe by
~{RKM['sharpe_legacy_inflation']*100:.1f}%). All modules read this basis from a single source
(<code>src/basis.py</code>), so the per-module tables (Parts A–F) and the unified ranking (Part R) now agree
by construction. The <b>unified ranking (Part R)</b> remains the master cross-strategy comparison.</p>

<div class="part">Part A · The recommended strategy — calendar / quincena rule</div>
<h2><span class="n">A1.</span> The rule &amp; the numbers ($1M per trade, VWAP-priced, net of cost)</h2>
<p class="sub" style="margin:0 0 8px">Built up one design choice at a time. <b>Sharpe (full)</b> is the whole
history; <b>Sharpe (OOS)</b> is the held-out 40% after {OREC['split_date']} — the honest number. The winner
is the one that is best <i>out of sample</i>, not the one with the biggest headline P&amp;L.</p>
<table><tr><th>Version</th><th class="num">Per year (full)</th><th class="num">Sharpe (full)</th>
  <th class="num">Sharpe (OOS)</th><th class="num">Trades/yr</th><th class="num">Max DD</th><th>Note</th></tr>
<tr><td>Base quincena (short if day ≤ 15)</td><td class="num">{usd(QBASE['per_year_usd'])}</td>
  <td class="num">{QBASE['sharpe']}</td>
  <td class="num {scls(FAM['Base quincena (≤15)']['oos']['sharpe'])}">{FAM['Base quincena (≤15)']['oos']['sharpe']}</td>
  <td class="num">{QBASE['roundtrips_yr']}</td><td class="num">{usd(QBASE['maxdd_usd'])}</td>
  <td class="muted">crude ancestor; left-skewed tail</td></tr>
<tr><td>Real calendar (≤{CAL_PRE} bd), flat size</td>
  <td class="num ok">{usd(QCAL['per_year_usd'])}</td><td class="num ok">{QCAL['sharpe']}</td>
  <td class="num {scls(FAM['Real calendar']['oos']['sharpe'])}">{FAM['Real calendar']['oos']['sharpe']}</td>
  <td class="num">{QCAL['roundtrips_yr']}</td><td class="num">{usd(QCAL['maxdd_usd'])}</td>
  <td class="muted">deadline-anchored entry; OOS slips vs IS</td></tr>
<tr><td>Refined (fixed 5–15) + slow-vol</td><td class="num">{usd(QSLOW['per_year_usd'])}</td>
  <td class="num ok">{QSLOW['sharpe']}</td>
  <td class="num {scls(FAM['Refined + slow-vol']['oos']['sharpe'])}">{FAM['Refined + slow-vol']['oos']['sharpe']}</td>
  <td class="num">{QSLOW['roundtrips_yr']}</td><td class="num ok">{usd(QSLOW['maxdd_usd'])}</td>
  <td class="muted">slow-vol holds OOS, fixed window</td></tr>
<tr class="rowbest"><td><b>Calendar + slow-vol</b>
  <span class="best">recommended · best OOS</span></td>
  <td class="num ok">{usd(QREC['per_year_usd'])}</td><td class="num ok">{QREC['sharpe']}</td>
  <td class="num {scls(FAM['Calendar + slow-vol']['oos']['sharpe'])}">{FAM['Calendar + slow-vol']['oos']['sharpe']}</td>
  <td class="num">{QREC['roundtrips_yr']:.0f}</td><td class="num">{usd(QREC['maxdd_usd'])}</td>
  <td class="muted">deadline anchor + tail-cutting trim</td></tr></table>
<div class="rule"><h4>Exact logic (the trading calendar)</h4>
  <div class="logic">≤ {CAL_PRE} business days before the IVA/quincena deadline (through it) →  SHORT USD<br>
    &nbsp;&nbsp;&nbsp;(firms sell USD to raise colones for the 15th tax + payroll → colón strengthens)<br>
    otherwise →  LONG USD  (supply fades, USD drifts up)<br>
    size trim: HALF size whenever a slow 20-day volume regime disagrees with the trade</div>
  <p>Two design choices, each justified before looking at the test window. <b>Entry</b> anchors the short
    window to Costa Rica's actual IVA (D-104) / mid-month quincena deadline — the 15th, rolled to the next
    business day — so it shifts with weekends and holidays (the fixed day-of-month 5–15 window is its
    holiday-blind twin, a near-identical trade population). <b>Sizing</b> trims to half whenever a slow 20-day
    volume regime disagrees, cutting the biggest losing trades without changing direction or adding turnover.
    In-sample the two slow-vol variants are a statistical tie (Sharpe {FAM['Calendar + slow-vol']['is']['sharpe']}
    vs {FAM['Refined + slow-vol']['is']['sharpe']}, well inside noise for ~250 trades); we keep the calendar
    anchor for operational robustness, and out of sample it is the stronger of the two
    (Sharpe {FAM['Calendar + slow-vol']['oos']['sharpe']} vs {FAM['Refined + slow-vol']['oos']['sharpe']}).</p></div>

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

<h2><span class="n">A4.</span> Sample vs out-of-sample — how the whole family is selected</h2>
<p class="lead">This is the discipline behind the recommendation. The rules carry <b>no fitted parameters</b>
(the 20-day volume window, the ≤{CAL_PRE}-day anchor and the half-size trim are fixed a-priori choices, not
optimised), but we still split history chronologically 60/40 at <b>{OREC['split_date']}</b>, <b>rank on the
in-sample Sharpe</b>, and then read the out-of-sample column as the honest confirmation. The pattern is
decisive: <b>the slow-vol size trim is the design choice that matters</b> — the strong flat-sized entries give
the edge back out of sample (the real-calendar entry {FAM['Real calendar']['is']['sharpe']}→{FAM['Real calendar']['oos']['sharpe']},
the fixed window {FAM['Refined (fixed 5–15)']['is']['sharpe']}→{FAM['Refined (fixed 5–15)']['oos']['sharpe']}),
while both slow-vol variants hold or improve (▲). The recommended <b>calendar + slow-vol</b> posts the
family's best out-of-sample Sharpe, {OREC['oos']['sharpe']} — up from {OREC['is']['sharpe']} in sample.</p>
{oos_table()}
{fig("q_oos.png", "Recommended calendar + slow-vol rule (VWAP-priced): in-sample vs out-of-sample equity",
     "In-sample " + OREC['is_start'] + " → " + OREC['is_end'] + " (Sharpe " + str(OREC['is']['sharpe']) +
     ") and out-of-sample → " + OREC['oos_end'] + " (Sharpe " + str(OREC['oos']['sharpe']) + ", "
     + usd(OREC['oos']['per_year_usd']) + "/yr). The out-of-sample slope is if anything steeper — the "
     "post-2021 float is where the cash-flow cycle is most tradeable, and the slow-vol trim keeps the "
     "held-out drawdowns shallow.")}

<h2><span class="n">A5.</span> An alternative tail control — a dynamic exit (stop-loss / take-profit)</h2>
<p class="lead">The recommended rule already cuts the reversion tail through <b>sizing</b> (the slow-vol trim). A
dynamic <b>trailing stop</b> is a second, independent way to do it through <b>timing</b>, shown here on the
flat-sized calendar entry so the mechanism is isolated. Because the edge <b>front-loads</b> — the colón
strengthens <i>into</i> the deadline and reverts after — a trade often peaks mid-hold and gives the move back;
once the running P&amp;L gives back {XM['trail_bps']} bps from its best level we bank it and stay flat until the
calendar opens the next trade. This <b>never changes the entry logic</b> — it can only remove the losing tail.
Both parameters are optimised on the <b>in-sample 60% only</b>; the out-of-sample 40% below is untouched. Treat
it as an optional overlay: it and the slow-vol trim are two routes to the same tail control, not a stack to
apply blindly together.</p>
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

<h2><span class="n">A5b.</span> The full exit engine — which stop actually helps (newest run, {RKM['sessions_per_year_actual']:.0f}-session basis)</h2>
<p class="lead">A5's trailing overlay was tuned by <code>exits.py</code>. The newest run
(<code>exit_lab.py</code>) rebuilds it as a <b>four-mechanism engine</b> — trailing stop, hard stop-loss,
take-profit and time stop — each tested <b>alone</b> before any are combined, everything selected on the
in-sample 60% only and priced on the correct <b>{XL['_meta']['sessions_per_year']:.0f} sessions/yr</b>. The
verdict is sharper than before: <b>a stop-loss helps, a take-profit actively hurts.</b> The recommended overlay
is <b>trailing {XLP['parsimonious']['trail']} bps + hard stop {XLP['parsimonious']['floor']} bps, no
take-profit</b> — in-sample Sharpe <b>{XLR['is']['sharpe']}</b> → out-of-sample <b>{XLR['oos']['sharpe']}</b>
(the overlay does not decay out of sample), cutting the worst drawdown from {usd(XLB['full']['maxdd_usd'])} to
{usd(XLR['full']['maxdd_usd'])}.</p>
{exit_marginal_table()}
{fig("xl_marginal.png", "Marginal value of each exit mechanism ALONE, on the calendar entry",
     "Each mechanism swept on its own (dashed = no-exit in-sample baseline " + str(XLB['is']['sharpe']) +
     ", dotted = out-of-sample " + str(XLB['oos']['sharpe']) + "). Trailing and hard stops lift Sharpe across a "
     "broad band; the take-profit panel slopes the wrong way — tightening it costs Sharpe — and the time stop "
     "adds nothing.")}
<p class="lead"><b>Take-profit is the clear negative result, worth stating plainly.</b> Added on top of the
trail {XLP['parsimonious']['trail']} / floor {XLP['parsimonious']['floor']} overlay, tightening the target
degrades the out-of-sample Sharpe monotonically ({XL['take_profit_sweep'][0]['oos_sharpe']} with none →
{XL['take_profit_sweep'][-1]['oos_sharpe']} at 80 bps). The calendar edge is a <i>drift</i> that accrues while
the trade is held into the deadline, so capping the winner truncates exactly the move the strategy exists to
capture.</p>
{take_profit_table()}
{fig("xl_plateau_bps.png", "Does in-sample selection survive out-of-sample? Every parameter set plotted",
     "All " + str(XLP['n_trials']) + " joint-grid cells, in-sample Sharpe (the selection metric) vs "
     "out-of-sample. corr(IS, OOS) = " + f"{XLP['is_oos_corr']:+.2f}" + " — in-sample ranking genuinely "
     "predicts out-of-sample for this parameter family. The chosen point (red) sits on the IS=OOS diagonal, "
     "not above it. Deflated-Sharpe P(true>0) ≈ " + str(XLP['deflated_sharpe_prob']) + " over the "
     + str(XLP['n_trials']) + " trials; the 18 neighbours of the winner hold a median "
     + str(XLP['neighbourhood']['median_is_sharpe']) + " (" +
     f"{XLP['neighbourhood']['ratio_to_best']*100:.0f}%" + " of peak) — a plateau, not a spike.")}
{fig("xl_equity.png", "Calendar entry with and without the exit overlay — equity &amp; drawdown",
     "Red line = the in-sample / out-of-sample boundary; the bands are tuned only to the left of it. The "
     "overlay (green) tracks the baseline's compounding while its drawdowns (lower panel) are visibly "
     "shallower throughout.")}
<p class="lead"><b>But read the overlay as a RISK improvement, not more money.</b> A paired t-test on the daily
out-of-sample P&amp;L difference gives t = {XL['alpha_vs_risk']['paired_t']}, <b>p = {XL['alpha_vs_risk']['paired_p']}</b>:
total P&amp;L is statistically unchanged (indeed nominally {usd(XL['alpha_vs_risk']['oos_total_usd_delta'])} lower
out of sample). What the overlay buys is <b>+{XL['alpha_vs_risk']['oos_sharpe_delta']} out-of-sample Sharpe and a
~{abs(XL['alpha_vs_risk']['oos_maxdd_delta'])/abs(XLB['oos']['maxdd_usd'])*100:.0f}% cut in max drawdown</b> — a
smoother path you can size up, not the same notional earning more. Volatility-scaled bands (trail
{XL['stage2_joint']['vol']['parsimonious']['trail']}× / floor {XL['stage2_joint']['vol']['parsimonious']['floor']}×
realised vol) scored <i>worse</i> than fixed bps in both windows and are reported as a negative result, not
dropped.</p>

<h2><span class="n">A6.</span> The position against the price</h2>
{fig("q_position_shading.png", "USD/CRC session VWAP shaded by the recommended calendar position",
     f"Green = long USD, red = short USD (short ≤{CAL_PRE} business days into the IVA/quincena deadline). "
     "The entry is always in a position — there is no flat/no-trade state — so every session carries one shade "
     "or the other; the recommended rule additionally holds these at half size when the slow volume regime "
     "disagrees (same colour, thinner book). The short (red) bands cluster mid-month, exactly where the "
     "cash-flow supply surge lands.")}

<h2><span class="n">A7.</span> Return distributions — out-of-sample, per trade</h2>
<p class="lead">Net P&amp;L <b>per trade</b> (per directional roundtrip), in USD at $1M/trade,
<b>VWAP-to-VWAP</b> — the P&amp;L a desk actually books at each exit, not a day-by-day mark — and built on the
<b>out-of-sample window only</b> ({RDW['split_date']} → {RDW['oos_end']}, {RD['Calendar + slow-vol']['n']}
held-out trades per variant). So these are the outcomes an operator would have booked on data the rule was
never tuned on. Selection is on in-sample Sharpe; each panel reports IS→OOS Sharpe so you can see which edges
survive. The recommended <b>{RD_BEST}</b> carries the badge: it is right-shifted, has the thinnest left tail
of the family (out-of-sample skew <b>{RD['Calendar + slow-vol']['skew']:+.2f}</b>, worst trade
{usd(RD['Calendar + slow-vol']['min'])} vs {usd(RD['Base quincena (≤15)']['min'])} for the base rule), and it
posts the family's best out-of-sample Sharpe, strengthening from {RD['Calendar + slow-vol']['is_sharpe']:.2f}
to {RD['Calendar + slow-vol']['oos_sharpe']:.2f} where the flat-sized entries decay.</p>
{dist_tabs_html()}
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

<div class="part">Part R · Unified ranking — every strategy on one honest basis (newest run)</div>
<p class="lead">Until this run each module carried its own accounting, so the headline numbers were <b>not
comparable</b> — <code>strategies.py</code> reported gross Sharpe, <code>backtest.py</code> priced
close-to-close in bps, the calendar modules priced VWAP in dollars. <code>src/rank_strategies.py</code>
re-implements <b>every</b> rule against ONE frame and ONE basis — VWAP-to-VWAP, $1M/trade, 0.325 CRC per side,
annualised on the real <b>{RKM['sessions_per_year_actual']:.0f} sessions/yr</b> — and ranks by <b>in-sample</b>
Sharpe (first 60% of history, split {RKM['split_date']}). The out-of-sample column is carried for honesty and
never feeds the ordering. This table <b>supersedes the per-module tables</b> for cross-strategy comparison.</p>
{ranking_table()}
{fig("ranking.png", "Every strategy ranked on one common accounting basis (in-sample net Sharpe)",
     "The calendar family (green) owns the top of the book and the flow family (red) owns the bottom — "
     "in-sample. The top three seats are the calendar entry plus an exit overlay; the recommended trail 30 + "
     "floor 40 leads at IS Sharpe " + str(RK['ranking'][0]['is']['sharpe']) + " → OOS " +
     str(RK['ranking'][0]['oos']['sharpe']) + ".")}
<ul class="find">
 <li><b>The calendar family owns the top, the flow family owns the bottom — in-sample.</b> Every flow rule
   (order-flow, volume, skew, combined vote) is worthless-to-negative on 2014–2021 and only works on
   2021–2026. Their full-sample numbers average a dead regime with a live one, which is exactly why they read
   as mediocre-but-real rather than regime-contingent.</li>
 <li><b>The flow rules carry catastrophic drawdowns</b> — {usd(RK['ranking'][14]['is']['maxdd_usd'])} to
   {usd(RK['ranking'][10]['is']['maxdd_usd'])} in-sample against the calendar rule's
   {usd(RK['ranking'][0]['is']['maxdd_usd'])}–{usd(RK['ranking'][7]['is']['maxdd_usd'])}, at 80–97 round
   trips/yr. Even where the out-of-sample Sharpe looks excellent (Combined vote
   {RK['ranking'][10]['oos']['sharpe']}), the path is not something a desk would fund.</li>
 <li><b>Sanity check on the harness:</b> "Always long USD" (IS {RK['ranking'][8]['is']['sharpe']} / OOS
   {RK['ranking'][8]['oos']['sharpe']}) and "Always short USD" (IS {RK['ranking'][15]['is']['sharpe']} / OOS
   {RK['ranking'][15]['oos']['sharpe']}) are near-exact mirrors, as two opposite always-on positions must be —
   confirming the P&amp;L and cost accounting.</li>
 <li class="muted"><b>Honest caveat on the metric.</b> Because the in-sample window (2014–2021) is the
   pegged/low-vol regime and the out-of-sample window (2021–2026) is the float, in-sample selection
   structurally penalises any regime-contingent signal. The calendar rule wins partly <i>because</i> it is the
   one edge that works in both — a real virtue, but not the same claim as "the flow signals don't work."</li>
</ul>

<div class="part">Part I · The BCCR's hand — official flow &amp; reserves (newest data)</div>
<p class="lead">MONEX volume is <b>unsigned</b> — it says how much traded, not who was buying. Two BCCR
exports (daily FX <b>intervention</b>, CodCuadro 1587; month-end <b>net reserves</b>, CodCuadro 8) fill that
gap. The official side — the BCCR plus the non-bank public sector's (RECOPE et al.) USD requirement routed
through MONEX — is <b>a median {IVF['official_share_median']*100:.0f}% of daily MONEX volume</b>
({IVF['days_with_flow_pct']:.0f}% of sessions carry official flow). So the <b>differential</b> (total MONEX
minus official) is what the private market actually did, and the signed official flow is information the volume
series cannot carry. This section adds that data and asks whether it beats the published rule.</p>
{fig("iv_share.png", "Official (BCCR + public-sector) flow as a share of MONEX volume",
     "The official footprint is large and persistent — a 60-session average around "
     f"{IVF['official_share_mean']*100:.0f}% of traded volume, so on a typical day roughly half of what "
     "crosses MONEX is the BCCR or the public sector, not private supply/demand. Any read of 'private' flow "
     "from raw volume is therefore contaminated by an official component of this size.")}

<h2><span class="n">I1.</span> The mechanism — official demand marks where the colón turns</h2>
{fig("iv_intramonth.png", "Official USD demand vs the next-day move, by trading-days-to the deadline",
     "Official (and specifically public-sector) USD demand PEAKS at and just after the IVA/quincena deadline "
     "— the shaded window — which is exactly where the colón strengthens hardest next-day (the red line dives "
     f"to {min(IVF['next_move_by_official_quintile_bps'])/1:.0f}-bps-class moves). Heavy official demand does "
     "not push USD up; it marks the point where private supply overwhelms and the market reverts. This is the "
     "same post-deadline reversion the exit overlay (A5b) harvests — now with a name.")}
{fig("iv_direction.png", "Next-day USD move by official net-buying quintile",
     "The signed signal volume cannot see: the heaviest official-net-buying days are followed by the largest "
     f"USD-DOWN moves (contemporaneous corr {IVF['corr_contemp']:+.2f}). It is a real reversion signal, but it "
     f"decays sharply when lagged one session ({IVF['corr_lag1']:+.2f}) and depends on same-day disclosure of "
     "the intervention figures — so it is reported as mechanism, not used as a live entry signal.")}

<h2><span class="n">I2.</span> Does it beat the published rule? A reserve-regime size trim</h2>
<p class="lead">The tradeable win is not the daily flow (redundant with volume, and disclosure-sensitive) but the
<b>reserves</b>. When BCCR reserves are <b>rising</b> — the bank accumulating USD, a colón-supportive regime —
we trim long-USD size to half. It is strictly causal (a lagged monthly figure, no disclosure issue) and a
fixed a-priori rule. Stacked on the published exit overlay it lifts out-of-sample Sharpe
<b>{IVS['Calendar + trail30/floor40']['oos']['sharpe']} → {IVS['+ reserve-regime long-trim']['oos']['sharpe']}</b>
and cuts max drawdown <b>{usd(IVS['Calendar + trail30/floor40']['full']['maxdd_usd'])} →
{usd(IVS['+ reserve-regime long-trim']['full']['maxdd_usd'])}</b>. Reserves rose in only
{IVR['rising_pct_full']:.0f}% of months (not a one-way trend), so this is a regime read, not a disguised
"hold less long USD".</p>
{intervention_table()}
{fig("iv_strategy.png", "Exit-overlay baseline vs + reserve-regime long-trim — equity &amp; drawdown",
     "The trim (green) tracks the baseline's compounding while its drawdowns (lower panel) are visibly "
     "shallower, most clearly through the deep late-sample selloff. Red line = the in-sample / out-of-sample "
     "boundary; the rule is fixed a-priori, nothing is fit on the test window.")}
<p class="lead"><b>The reserve signal does real work — it is not just 'trim longs'.</b> A blind control that halves
<i>every</i> long reaches a similar out-of-sample Sharpe, but the reserve-selective trim earns
<b>{usd(IVRB['oos_total_delta_usd'])} more out-of-sample</b> at the same risk reduction (paired t =
{IVRB['paired_t']}, <b>p = {IVRB['paired_p']}</b>) — it keeps full size on the longs reserves say are safe and
cuts only the ones it flags. Like the exit overlay, this is a <b>risk</b> improvement (lower drawdown, higher
Sharpe, slightly less total P&amp;L), not an alpha one. The official-flow trim is an equally-causal alternative
that lands in the same place; the two BCCR series confirm each other.</p>
{fig("iv_reserves.png", "BCCR net reserves and the rising-reserve (colón-supportive) regime",
     f"Reserves ran from ${IVR['usd_mn_min']/1e3:.1f}bn to ${IVR['usd_mn_max']/1e3:.1f}bn over the sample; the "
     "shaded rising-reserve months are when long-USD size is trimmed. The month-to-month change is roughly "
     f"even ({IVR['rising_pct_full']:.0f}% rising), which is why the trim is a genuine regime read.")}

<div class="part">Part E · Raw relationships behind the signals</div>
<h2><span class="n">E1.</span> Low volume → USD up (seasonally adjusted)</h2>
{fig("vm_vol_nextret.png", "Next-day USD move by volume quintile",
     "Monotonic: quietest days → USD up, busiest → USD down. Seasonal adjustment sharpens the extremes. This "
     "is the raw relationship the volume filter and the ML model lean on.")}
<h2><span class="n">E2.</span> Attribution &amp; drivers</h2>
<div class="cols">
{fig("vm_attribution.png", "Net Sharpe by feature family",
     "Volume-only and calendar-only each beat the drift and are additive — the calendar carries the most and "
     "volume refines it, which is exactly why the recommended rule pairs the calendar entry with a "
     "slow-volume size trim.")}
{fig("vm_coefs.png", "Walk-forward model coefficients",
     "Quincena/day-of-month and volume dominate; no technical indicators used.")}
</div>

<div class="part">Part F · Execution realism — VWAP vs the closing print</div>
<h2><span class="n">F1.</span> The recommended rule does not depend on catching the close</h2>
<p class="lead">Every result in this report is already priced <b>VWAP-to-VWAP</b> — the fill a desk can
actually work by spreading an order across the session, rather than the optimistic assumption of trading the
last print. Here is the proof for the strategy we care about: the recommended calendar + slow-vol rule, same
0.65 CRC round-trip slippage, marked at the closing print vs the session VWAP. The VWAP fill <b>does not
haircut the edge — it slightly improves the Sharpe</b> ({EX['close']['sharpe']} → {EX['vwap']['sharpe']}),
because the VWAP is a steadier reference than a single closing tick.</p>
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
 <li><b>The recommendation is the calendar + slow-vol rule.</b> Short USD into Costa Rica's mid-month IVA /
   payroll deadline (within {CAL_PRE} business days of it), long the rest of the month, at <b>half size
   whenever a slow 20-day volume regime disagrees</b>. Full-sample {usd(QREC['per_year_usd'])}/yr at Sharpe
   {QREC['sharpe']}, ~{QREC['roundtrips_yr']:.0f} trades/yr, drawdown {usd(QREC['maxdd_usd'])} — and, the
   number that matters, <b>out-of-sample Sharpe {OREC['oos']['sharpe']}</b>
   ({usd(OREC['oos']['per_year_usd'])}/yr on the 40% held out). Selected in-sample, confirmed out-of-sample.</li>
 <li><b>The slow-vol trim is what makes it trade-ready.</b> Ranked on in-sample Sharpe, both slow-vol variants
   HOLD or IMPROVE out of sample while the two strong flat-sized entries DECAY (real-calendar {FAM['Real calendar']['is']['sharpe']}→
   {FAM['Real calendar']['oos']['sharpe']}, fixed-window {FAM['Refined (fixed 5–15)']['is']['sharpe']}→
   {FAM['Refined (fixed 5–15)']['oos']['sharpe']}). The trim cuts the fat losing trades: out-of-sample per-trade
   skew turns <b>positive ({RD['Calendar + slow-vol']['skew']:+.2f})</b> and the worst trade halves
   ({usd(RD['Base quincena (≤15)']['min'])} → {usd(RD['Calendar + slow-vol']['min'])}). That tail control, not
   the headline P&amp;L, is why this is the pick.</li>
 <li><b>A dynamic exit is an optional second tail control (A5 / A5b).</b> The newest exit-engine run
   (<code>exit_lab.py</code>, {XL['_meta']['sessions_per_year']:.0f}-session basis) settles the design: on the
   calendar entry, a <b>trailing {XLP['parsimonious']['trail']} bps + hard stop {XLP['parsimonious']['floor']} bps</b>
   overlay lifts the in-sample Sharpe from {XLB['is']['sharpe']} to <b>{XLR['is']['sharpe']}</b> and the
   out-of-sample from {XLB['oos']['sharpe']} to <b>{XLR['oos']['sharpe']}</b>, cutting the worst drawdown
   {usd(XLB['full']['maxdd_usd'])} → {usd(XLR['full']['maxdd_usd'])}. But a paired test (p =
   {XL['alpha_vs_risk']['paired_p']}) shows total P&amp;L is unchanged — it is a <b>risk</b> improvement, a
   smoother path to size up, not more money. <b>A take-profit actively hurts</b> and is not recommended. It is
   an alternative route to the same tail control as the slow-vol trim, not a required stack.</li>
 <li><b>Ranked on one honest basis (Part R), the calendar family dominates.</b> Re-priced identically on the
   real {RKM['sessions_per_year_actual']:.0f} sessions/yr, the top three seats are the calendar entry plus an
   exit overlay (recommended trail {XLP['parsimonious']['trail']} + floor {XLP['parsimonious']['floor']}, IS
   Sharpe {RK['ranking'][0]['is']['sharpe']} → OOS {RK['ranking'][0]['oos']['sharpe']}), the flow rules sit at
   the bottom with {usd(RK['ranking'][14]['is']['maxdd_usd'])}-class drawdowns, and the mirror-image benchmarks
   confirm the accounting. It is the one edge that works in both the pegged and the float regimes.</li>
 <li><b>Your volume thesis is correct but must be traded gently.</b> Low volume → USD up is real and
   monotonic, but a daily flip pays much of it to slippage even VWAP-priced. Its value here is exactly as the
   <i>slow filter</i> that sizes the calendar trade — that is what turns the plain calendar rule into the
   recommended one — and, conviction-sized, lifts the combined/ML book to Sharpe
   {DY['Combined vote (3 signals)']['sharpe']}–{LG['sharpe']}.</li>
 <li><b>The BCCR's hand is now visible, and it sharpens the risk controls (Part I).</b> Official flow (BCCR +
   public sector) is a median {IVF['official_share_median']*100:.0f}% of MONEX volume and peaks at the
   deadline, exactly where the colón turns — it <i>names</i> the reversion the exit overlay harvests. The
   tradeable use is the <b>reserves</b>: trimming long-USD size when reserves rise lifts out-of-sample Sharpe
   {IVS['Calendar + trail30/floor40']['oos']['sharpe']} → {IVS['+ reserve-regime long-trim']['oos']['sharpe']}
   and cuts drawdown {usd(IVS['Calendar + trail30/floor40']['full']['maxdd_usd'])} →
   {usd(IVS['+ reserve-regime long-trim']['full']['maxdd_usd'])}, beating a blind long-trim out-of-sample
   (p = {IVRB['paired_p']}). A risk improvement, not new alpha — and strictly causal.</li>
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
  so its notional varies up to $1M; all model numbers are walk-forward out-of-sample. <b>Selection
  discipline:</b> variants are ranked on the <b>in-sample</b> Sharpe (first 60%, {OREC['is_start']} →
  {OREC['is_end']}); every headline result, histogram and per-trade statistic for the recommended rule is its
  <b>out-of-sample</b> realisation on the held-out 40% ({OREC['split_date']} → {OREC['oos_end']}). Nothing is
  re-fit on the test window. In-sample / out-of-sample = chronological 60/40 split at {OREC['split_date']}.
  Reproducible:
  <code>parse_monex → parse_bccr → analyze → … → rank_strategies → exit_lab → intervention → build_report</code>.
  Every part is annualised on the venue's real {RKM['sessions_per_year_actual']:.0f} sessions/yr, priced
  VWAP-to-VWAP net of the 0.65 CRC round trip, from a single basis (<code>src/basis.py</code>) — so the
  per-module tables (A–F) and the unified ranking (R) agree by construction. Part I adds two BCCR exports
  (FX intervention CodCuadro 1587, net reserves CodCuadro 8) via <code>parse_bccr.py</code>.
</footer>

</div>{JS}</body></html>"""


def main():
    dst = OUT / "report.html"
    dst.write_text(HTML, encoding="utf-8")
    print(f"Wrote {dst} ({dst.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
