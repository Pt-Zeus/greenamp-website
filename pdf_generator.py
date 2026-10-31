"""GreenAmp Energy Solutions — one-page Lender Card (ReportLab), built from engine.run_forecast() output."""
from __future__ import annotations

import datetime
import os
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DARK, GREEN, SLATE, BORDER, MUTED = (colors.HexColor(c) for c in ("#070F1E", "#10B981", "#334155", "#CBD5E1", "#64748B"))


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_lender_pdf(res: dict, site_name: str) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm)
    ss = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=DARK)
    sub = ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica", fontSize=8, leading=11, textColor=MUTED)
    head = ParagraphStyle("h", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=9.5, leading=12,
                          textColor=GREEN, spaceBefore=2 * mm, spaceAfter=2 * mm)
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=8, leading=10.5, textColor=SLATE)
    tiny = ParagraphStyle("x", parent=body, fontSize=6.5, leading=8.5, textColor=MUTED)
    P = lambda t, s=body: Paragraph(t, s)

    def table(rows, widths, header=True):
        t = Table([[P(c) if isinstance(c, str) else c for c in r] for r in rows], colWidths=widths)
        style = [("GRID", (0, 0), (-1, -1), 0.5, BORDER), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("TOPPADDING", (0, 0), (-1, -1), 1.5 * mm), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5 * mm)]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")))
        t.setStyle(TableStyle(style))
        return t

    T, opt_s, opt_o = res["target"], res["options"]["staged"], res["options"]["one_shot"]
    yi = min(T["years_since_switch_on"], 5) - 1
    inp, g, sol, aff = res["inputs"], res["grade"], res["solar"], res["affordability"]
    story = []

    # Header
    logo = next((f for f in ("company_logo.jpeg", "company_logo.jpg", "company_logo.png") if os.path.exists(f)), None)
    head_txt = [P("<b>GREENAMP ENERGY SOLUTIONS</b>", title),
                P("Mini-grid pre-feasibility · Lender Card (screening, not an engineering design)", sub),
                P(f"Generated {datetime.datetime.now():%d %b %Y %H:%M}", sub)]
    if logo:
        h = Table([[Image(logo, width=24 * mm, height=24 * mm), head_txt]], colWidths=[28 * mm, 154 * mm])
        h.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(h)
    else:
        story += head_txt
    story.append(HRFlowable(width="100%", thickness=1.5, color=GREEN, spaceBefore=2 * mm, spaceAfter=3 * mm))

    story.append(table([[f"<b>Site:</b> {_esc(site_name)}",
                         f"<b>Coordinates:</b> {inp['lat']:.4f}, {inp['lon']:.4f}",
                         f"<b>Confidence:</b> {g['grade']}"],
                        [f"<b>Switch-on:</b> {inp['switch_on_year']}",
                         f"<b>Target year:</b> {T['year']}" + (" (extrapolated)" if T["extrapolated"] else ""),
                         f"<b>Feature source:</b> {_esc(res['features'].get('source'))}"]],
                       [60 * mm, 62 * mm, 60 * mm], header=False))

    story.append(P(f"1. DEMAND RANGE IN {T['year']}", head))
    s, pc = T["site_kwh_month"], T["per_customer_kwh_month"]
    story.append(table([
        ["Case", "Site demand (kWh/month)", "Per customer (kWh/month)", "Meaning"],
        ["P90 (low)", f"{s['P90']:,.0f}", f"{pc['P90']}", "Exceeded in 9 of 10 futures: size debt on this"],
        ["P50 (likely)", f"{s['P50']:,.0f}", f"{pc['P50']}", "Median: build phase 1 for this"],
        ["P10 (high)", f"{s['P10']:,.0f}", f"{pc['P10']}", "Exceeded in 1 of 10 futures: plan expansion and inverter headroom"],
    ], [26 * mm, 40 * mm, 40 * mm, 76 * mm]))

    story.append(P("2. SIZING AND REPAYMENT RISK", head))
    story.append(table([
        ["", "Staged (P50, year 2)", "One-shot (P10, year 5)"],
        ["Solar PV / Battery / Inverter", f"{opt_s['pv_kwp']} kWp / {opt_s['battery_kwh']} kWh / {opt_s['inverter_kw']} kW",
         f"{opt_o['pv_kwp']} kWp / {opt_o['battery_kwh']} kWh / {opt_o['inverter_kw']} kW"],
        ["Capex (USD, after grants)", f"{opt_s['capex_usd']:,.0f}", f"{opt_o['capex_usd']:,.0f}"],
        ["Break-even demand (kWh/month)", f"{opt_s['break_even_kwh_month']:,.0f}", f"{opt_o['break_even_kwh_month']:,.0f}"],
        [f"Chance below break-even (year {yi + 1})", f"{opt_s['p_shortfall_by_year'][yi]:.0%}",
         f"{opt_o['p_shortfall_by_year'][yi]:.0%}"],
    ], [62 * mm, 60 * mm, 60 * mm]))
    story.append(Spacer(1, 1.5 * mm))
    story.append(P(f"Staged build saves <b>${res['staged_saving_usd']:,.0f}</b> upfront. {_esc(opt_s['expansion_trigger'])}. "
                   f"Solar resource {sol['psh']} kWh/m²/day ({_esc(sol['psh_source'])}); tilt {sol['tilt_deg']}°. "
                   f"Tariff ${res['tariff_usd_kwh']:.2f}/kWh. Break-even = (operating cost + debt service) / (12 × tariff)."))
    story.append(Spacer(1, 1 * mm))
    story.append(P(f"<b>Affordability:</b> typical household bill ${aff['bill_usd_month']:.2f}/month vs "
                   f"${aff['current_spend_range'][0]:.0f}-{aff['current_spend_range'][1]:.0f} spent on energy today. "
                   f"{_esc(aff['verdict'])}."))

    story.append(P("3. WHAT DROVE THE FORECAST", head))
    for line in (res["drivers"] or ["No matched-village comparison loaded (demo mode)."])[:6]:
        story.append(P("• " + _esc(line)))
    story.append(P(f"• Starting level: {_esc(res['level']['source'])}"))
    if res["growth"]["matches"]:
        names = ", ".join(f"{_esc(m['name'])} ({m['source']})" for m in res["growth"]["matches"][:5])
        story.append(P(f"• Learned growth from: {names}"))
    story.append(P("• Grade reasons: " + _esc("; ".join(g["reasons"]))))

    story.append(P("4. HONESTY LABELS", head))
    L = res["labels"]
    story.append(table([
        ["Calibrated", "Assumed", "Not yet proven"],
        ["<br/>".join(_esc(x) for x in L["calibrated"]) or "Nothing yet (demo mode)",
         "<br/>".join(_esc(x.split(":")[0]) for x in L["assumed"]) + "<br/><i>Details in the app</i>",
         "<br/>".join(_esc(x) for x in L["not_yet_proven"])],
    ], [60 * mm, 62 * mm, 60 * mm]))

    v = res.get("validation")
    story.append(P("5. VALIDATION", head))
    if v:
        best = v["level"]["best_method"]
        bm = v["level"]["methods"][best]
        flat = v["level"]["methods"].get("flat_amda") or {}
        story.append(P(f"Leave-one-site-out on {v['n_sites']} metered mini-grids: best method <b>{_esc(best)}</b>, "
                       f"mean absolute error {bm['mae']} kWh/customer/month (flat AMDA benchmark: {flat.get('mae', 'n/a')}). "
                       f"Share of sites inside the stated range: {v['level']['interval_coverage_country_range']:.0%}."))
    else:
        story.append(P("Not yet run on real meters. Planned: leave-one-site-out on the Selby / Renewvia smart-meter "
                       "dataset (24 mini-grids, Kenya and Nigeria)."))

    story.append(Spacer(1, 3 * mm))
    story.append(P("Sources: " + _esc(" · ".join(res["sources"])), tiny))
    story.append(P("Disclaimer: screening estimate from open data and published benchmarks. Not investment advice and "
                   "not an engineering design. Site surveys and community consultation must precede any investment.", tiny))
    doc.build(story)
    return buf.getvalue()
