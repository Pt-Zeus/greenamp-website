"""
GreenAmp Energy Solutions — Mini-Grid Demand Forecasting (Streamlit UI)

All calculations live in engine/ (run_forecast). This file only collects inputs and shows results.
Runs in DEMO MODE until the data pipeline (pipeline/01-04) has written files into data/.
"""
from __future__ import annotations

import base64
import datetime as dt
import os

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st

from engine import run_forecast
from engine import store
from engine.features import get_features
from pdf_generator import generate_lender_pdf

st.set_page_config(page_title="GreenAmp · Mini-Grid Demand Forecast", page_icon="⚡", layout="wide")

THIS_YEAR = dt.date.today().year

# ─── Replay sites: real mini-grids from the Selby / Renewvia dataset ─────────
# Coordinates and feature values below are TEAM ESTIMATES used only until pipeline/02
# writes real satellite features to data/site_features.parquet.
REPLAY_SITES = {
    "Akipelai (Bayelsa, Nigeria)":      dict(lat=4.7215, lon=6.3411, country="Nigeria", bld=320, comm=11.5, grid=38, ntl=0.8, tt=3.5, wp=35),
    "Balep (Cross River, Nigeria)":     dict(lat=5.8642, lon=8.7124, country="Nigeria", bld=240, comm=8.0, grid=29, ntl=0.5, tt=4.0, wp=28),
    "Oloibiri (Bayelsa, Nigeria)":      dict(lat=4.6851, lon=6.3218, country="Nigeria", bld=190, comm=14.0, grid=18, ntl=1.2, tt=2.5, wp=42),
    "Ekong Anaku (Cross River, Nigeria)": dict(lat=5.2910, lon=8.5420, country="Nigeria", bld=275, comm=9.5, grid=45, ntl=0.4, tt=4.5, wp=25),
    "Kakuma 3A (Turkana, Kenya)":       dict(lat=3.7120, lon=34.8510, country="Kenya", bld=410, comm=16.5, grid=65, ntl=1.5, tt=5.5, wp=22),
    "Kalobeyei Village (Turkana, Kenya)": dict(lat=3.7650, lon=34.7200, country="Kenya", bld=350, comm=13.0, grid=58, ntl=0.9, tt=5.0, wp=20),
    "Katiko (Turkana, Kenya)":          dict(lat=3.2450, lon=35.1200, country="Kenya", bld=150, comm=6.0, grid=52, ntl=0.3, tt=6.0, wp=18),
    "Ngurunit (Marsabit, Kenya)":       dict(lat=1.7480, lon=37.2940, country="Kenya", bld=280, comm=9.0, grid=74, ntl=0.4, tt=7.0, wp=20),
}

# ─── Styling (GreenAmp dark theme) ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.2rem; max-width: 1500px; }
.ga-strip { display:grid; grid-template-columns: repeat(5, 1fr); background:#0B1629; border:1px solid rgba(16,185,129,.3);
  border-top:3px solid #10B981; border-radius:8px; overflow:hidden; margin: .6rem 0 1.2rem; }
.ga-cell { padding: 1rem 1.2rem; border-right:1px solid rgba(255,255,255,.08); }
.ga-cell:last-child { border-right:none; }
.ga-t { font-size:.68rem; font-weight:600; color:#94A3B8; letter-spacing:.05em; text-transform:uppercase; }
.ga-n { font-family:'DM Mono',monospace; font-size:1.7rem; font-weight:500; line-height:1.2; }
.ga-d { font-size:.72rem; color:#94A3B8; margin-top:.2rem; }
.ga-card { background:#0B1629; border:1px solid rgba(255,255,255,.08); border-radius:8px; padding:1rem 1.1rem; margin-bottom:1rem; color:#E2E8F0; }
.ga-h { font-size:.78rem; font-weight:700; color:#10B981; text-transform:uppercase; letter-spacing:.04em;
  border-bottom:1px solid rgba(255,255,255,.08); padding-bottom:.45rem; margin-bottom:.7rem; }
.ga-banner { border-radius:6px; padding:.65rem 1rem; font-size:.85rem; margin-bottom:.8rem; }
.ga-small { font-size:.78rem; color:#94A3B8; line-height:1.55; }
</style>
""", unsafe_allow_html=True)


def logo_data_uri() -> str | None:
    for fn in ("company_logo.jpeg", "company_logo.jpg", "company_logo.png"):
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                return f"data:image/{fn.split('.')[-1]};base64,{base64.b64encode(f.read()).decode()}"
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def geocode(q: str):
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": q, "format": "json", "limit": 1},
                         headers={"User-Agent": "GreenAmp-hackathon-demo/1.0"}, timeout=10)
        j = r.json()
        if j:
            return float(j[0]["lat"]), float(j[0]["lon"]), j[0].get("display_name", q)
    except Exception:
        pass
    return None


def data_status() -> dict:
    return {
        "site features (satellite)": store.site_features() is not None,
        "growth library": store.growth_library() is not None,
        "meter-based starting level": store.selby_site_year1() is not None,
        "validation on real meters": store.validation() is not None,
    }


# ─── Sidebar: inputs ─────────────────────────────────────────────────────────
logo = logo_data_uri()
with st.sidebar:
    if logo:
        st.markdown(f'<div style="text-align:center"><img src="{logo}" style="width:120px;border-radius:8px"></div>',
                    unsafe_allow_html=True)
    st.markdown("### Location")
    mode = st.radio("Choose", ["Replay a real mini-grid", "Enter coordinates", "Search a place"],
                    label_visibility="collapsed")

    preset = None
    country = None
    if mode == "Replay a real mini-grid":
        name = st.selectbox("Real site (Selby / Renewvia dataset)", list(REPLAY_SITES))
        preset = REPLAY_SITES[name]
        lat, lon, country, site_name = preset["lat"], preset["lon"], preset["country"], name.split(" (")[0]
        st.caption("Replay: the forecast ignores this site's own data (within 2 km), so it is a fair test.")
    elif mode == "Enter coordinates":
        lat = st.number_input("Latitude", value=9.0820, format="%.4f", min_value=-60.0, max_value=60.0)
        lon = st.number_input("Longitude", value=8.6753, format="%.4f", min_value=-180.0, max_value=180.0)
        country = st.selectbox("Country (for benchmarks)", ["", "Nigeria", "Kenya", "Other"]) or None
        site_name = f"Site {lat:.3f}, {lon:.3f}"
    else:
        q = st.text_input("Village or town", value="Rokota, Niger State, Nigeria")
        hit = geocode(q) if q else None
        if hit:
            lat, lon, site_name = hit
            st.caption(f"Found: {site_name[:80]}")
            site_name = q.split(",")[0]
        else:
            st.warning("Place not found; using 9.08, 8.68.")
            lat, lon, site_name = 9.0820, 8.6753, q or "Unknown"
        country = st.selectbox("Country (for benchmarks)", ["", "Nigeria", "Kenya", "Other"]) or None

    st.markdown("### Forecast")
    switch_on_year = st.number_input("Expected switch-on year", value=THIS_YEAR + 1, min_value=2015, max_value=2040, step=1)
    target_year = st.number_input("Target year", value=THIS_YEAR + 3, min_value=2015, max_value=2050, step=1)

    st.markdown("### Finance")
    tariff = st.slider("Tariff (USD per kWh)", 0.10, 1.00, 0.45 if country != "Kenya" else 0.35, 0.01)
    grant = st.number_input("Grant per new connection (USD)", value=0, min_value=0, max_value=2000, step=50,
                            help="Example: Nigeria NEP Performance-Based Grant, USD 350 per connection")

    covered = get_features(lat, lon)["covered"]          # real satellite features for this spot?
    with st.expander("Site features (what-if)", expanded=not covered):
        st.caption("Used when no satellite data exists for this location yet. "
                   "Replay-site defaults are team estimates, not satellite values.")
        d = preset or dict(bld=250, comm=10.0, grid=30, ntl=0.5, tt=4.0, wp=30)
        key = f"{lat:.4f}{lon:.4f}"
        use_manual = st.toggle("Use these values", value=not covered, key=f"man{key}")
        bld = st.slider("Buildings in the 1 km area", 20, 1500, int(d["bld"]), 10, key=f"b{key}")
        comm = st.slider("Share of large buildings (%)", 0.0, 40.0, float(d["comm"]), 0.5, key=f"c{key}")
        ntl = st.slider("Night-light level (VIIRS)", 0.0, 5.0, float(d["ntl"]), 0.1, key=f"n{key}")
        tt = st.slider("Travel time to a city (hours)", 0.5, 12.0, float(d["tt"]), 0.5, key=f"t{key}")
        grid = st.slider("Distance to grid (km)", 1, 150, int(d["grid"]), key=f"g{key}")
        market = st.toggle("Market or clinic nearby", value=True, key=f"m{key}")
        wp = st.slider("Relative wealth (percentile)", 5, 95, int(d["wp"]), 5, key=f"w{key}")

overrides = None
if use_manual:
    overrides = {"building_count": bld, "large_building_share": comm / 100.0, "nightlight": ntl,
                 "travel_time_min": tt * 60, "dist_grid_km": grid, "market_km": 1.0 if market else 10.0,
                 "rwi": wp / 100 * 3.5 - 1.5, "country": country}
elif country:
    overrides = {"country": country}

# ─── Header ──────────────────────────────────────────────────────────────────
logo_tag = f'<img src="{logo}" style="height:48px;border-radius:6px">' if logo else ""
st.markdown(f"""
<div style="display:flex;align-items:center;gap:1rem;border-bottom:1px solid rgba(16,185,129,.25);padding-bottom:.8rem">
  {logo_tag}
  <div><div style="font-size:1.35rem;font-weight:700">GreenAmp <span style="color:#38BDF8;font-weight:400">Energy Solutions</span></div>
  <div class="ga-small">Pre-feasibility demand forecasting for mini-grids · screening tool, not an engineering design</div></div>
</div>""", unsafe_allow_html=True)

status = data_status()
if not any(status.values()):
    st.markdown('<div class="ga-banner" style="background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.45);color:#FBBF24">'
                '<b>Demo mode.</b> No pipeline data loaded yet: site features are entered by hand, the starting level '
                'uses published benchmarks, and growth uses an assumed curve. Values are illustrative until '
                '<code>data/</code> is filled by the pipeline.</div>', unsafe_allow_html=True)
else:
    missing = [k for k, v in status.items() if not v]
    st.markdown('<div class="ga-banner" style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.4);color:#34D399">'
                f'<b>Live data loaded.</b> {"Still missing: " + ", ".join(missing) if missing else "All data layers present."}'
                '</div>', unsafe_allow_html=True)

# ─── Run the engine ──────────────────────────────────────────────────────────
with st.spinner("Running 5,000 futures..."):
    res = run_forecast(lat, lon, int(target_year), int(switch_on_year), overrides=overrides,
                       tariff_usd_kwh=tariff, grant_usd_per_connection=float(grant))

if res["status"] != "ok":
    st.info(res["message"] + "  Turn on **Site features (what-if) → Use these values** in the sidebar.")
    st.stop()

T = res["target"]
site_q, pc_q = T["site_kwh_month"], T["per_customer_kwh_month"]
opt_s, opt_o = res["options"]["staged"], res["options"]["one_shot"]
yi = min(T["years_since_switch_on"], 5) - 1
p_short = opt_s["p_shortfall_by_year"][yi]
g = res["grade"]
gcol = {"A": "#10B981", "B": "#38BDF8", "C": "#F87171"}[g["grade"]]
ext = " · extrapolated beyond year 5" if T["extrapolated"] else ""

st.markdown(f"""
<div class="ga-strip">
  <div class="ga-cell"><div class="ga-t">P90 · low case</div><div class="ga-n" style="color:#F87171">{site_q['P90']:,.0f}</div>
    <div class="ga-d">kWh/month in {T['year']} · exceeded 9 years in 10</div></div>
  <div class="ga-cell" style="background:rgba(16,185,129,.06)"><div class="ga-t">P50 · likely</div><div class="ga-n" style="color:#10B981">{site_q['P50']:,.0f}</div>
    <div class="ga-d">kWh/month · {pc_q['P50']} kWh per customer{ext}</div></div>
  <div class="ga-cell"><div class="ga-t">P10 · high case</div><div class="ga-n" style="color:#38BDF8">{site_q['P10']:,.0f}</div>
    <div class="ga-d">kWh/month · exceeded 1 year in 10</div></div>
  <div class="ga-cell"><div class="ga-t">Chance below break-even</div><div class="ga-n" style="color:{'#F87171' if p_short > .3 else '#FBBF24' if p_short > .1 else '#10B981'}">{p_short:.0%}</div>
    <div class="ga-d">staged build, year {yi + 1} · break-even {opt_s['break_even_kwh_month']:,.0f} kWh/mo</div></div>
  <div class="ga-cell"><div class="ga-t">Confidence</div><div class="ga-n" style="color:{gcol}">{g['grade']}</div>
    <div class="ga-d">{g['reasons'][0]}</div></div>
</div>""", unsafe_allow_html=True)

# ─── Map + sizing ────────────────────────────────────────────────────────────
c1, c2 = st.columns([1.3, 1])
with c1:
    st.markdown(f'<div class="ga-h">Location · {site_name}</div>', unsafe_allow_html=True)
    pts = [{"lat": lat, "lon": lon, "label": site_name, "color": [16, 185, 129, 255], "r": 900}]
    lib = store.growth_library()
    if lib is not None and {"lat", "lon"} <= set(lib.columns):
        names = {m["name"] for m in res["growth"]["matches"]}
        for r in lib[lib["name"].isin(names)].dropna(subset=["lat", "lon"]).itertuples():
            pts.append({"lat": r.lat, "lon": r.lon, "label": f"Matched: {r.name}", "color": [56, 189, 248, 220], "r": 1500})
    zoom = 11 if len(pts) == 1 else 4
    st.pydeck_chart(pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
        layers=[pdk.Layer("ScatterplotLayer", data=pd.DataFrame(pts), get_position=["lon", "lat"],
                          get_fill_color="color", get_radius="r", radius_min_pixels=6, pickable=True)],
        tooltip={"text": "{label}"}), height=330)
    st.caption("Green: forecast site. Blue: matched villages it learned from (when the growth library is loaded).")

with c2:
    sol = res["solar"]
    st.markdown('<div class="ga-h">Sizing · staged vs one-shot</div>', unsafe_allow_html=True)
    sizing = pd.DataFrame({
        "": ["Solar PV (kWp)", "Battery (kWh)", "Inverter (kW)", "Capex (USD)", "Break-even (kWh/month)",
             f"Chance below break-even, year {yi + 1}"],
        "Staged (build for P50, year 2)": [str(opt_s["pv_kwp"]), str(opt_s["battery_kwh"]), str(opt_s["inverter_kw"]),
                                           f"{opt_s['capex_usd']:,.0f}", f"{opt_s['break_even_kwh_month']:,.0f}",
                                           f"{opt_s['p_shortfall_by_year'][yi]:.0%}"],
        "One-shot (build for P10, year 5)": [str(opt_o["pv_kwp"]), str(opt_o["battery_kwh"]), str(opt_o["inverter_kw"]),
                                             f"{opt_o['capex_usd']:,.0f}", f"{opt_o['break_even_kwh_month']:,.0f}",
                                             f"{opt_o['p_shortfall_by_year'][yi]:.0%}"],
    })
    st.dataframe(sizing, hide_index=True, width="stretch")
    st.markdown(f'<div class="ga-small">Staged build saves <b>${res["staged_saving_usd"]:,.0f}</b> upfront. '
                f'{opt_s["expansion_trigger"]}.<br>Sun: {sol["psh"]} kWh/m²/day ({sol["psh_source"]}) · '
                f'tilt {sol["tilt_deg"]}° {sol["azimuth"]} · load factor {sol["load_factor"]:.0%} (assumed rule)</div>',
                unsafe_allow_html=True)
    a = res["affordability"]
    ok = a["bill_usd_month"] <= a["current_spend_range"][1]
    st.markdown(f'<div class="ga-banner" style="margin-top:.7rem;background:{"rgba(16,185,129,.1)" if ok else "rgba(239,68,68,.1)"};'
                f'border:1px solid {"rgba(16,185,129,.4)" if ok else "rgba(239,68,68,.4)"}">Typical household bill '
                f'<b>${a["bill_usd_month"]:.2f}/month</b> vs ${a["current_spend_range"][0]:.0f}-{a["current_spend_range"][1]:.0f} '
                f'spent on energy today. {a["verdict"]}.</div>', unsafe_allow_html=True)

# ─── Fan chart + context ─────────────────────────────────────────────────────
c3, c4 = st.columns([1.3, 1])
with c3:
    st.markdown('<div class="ga-h">Five-year demand range (site, kWh/month)</div>', unsafe_allow_html=True)
    fan = pd.DataFrame(res["fan"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fan["calendar_year"], y=fan["P10"], mode="lines", name="P10 high case",
                             line=dict(color="rgba(56,189,248,.6)", dash="dot")))
    fig.add_trace(go.Scatter(x=fan["calendar_year"], y=fan["P90"], mode="lines", name="P90 low case", fill="tonexty",
                             fillcolor="rgba(16,185,129,.10)", line=dict(color="rgba(248,113,113,.7)", dash="dot")))
    fig.add_trace(go.Scatter(x=fan["calendar_year"], y=fan["P50"], mode="lines+markers", name="P50 likely",
                             line=dict(color="#10B981", width=3)))
    fig.add_hline(y=opt_s["break_even_kwh_month"], line_dash="dash", line_color="#FBBF24",
                  annotation_text="Break-even (staged)", annotation_font_color="#FBBF24")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"), yaxis_title="kWh / month",
                      xaxis=dict(dtick=1), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption(res["convention"])

with c4:
    st.markdown('<div class="ga-h">What drove this forecast</div>', unsafe_allow_html=True)
    lines = res["drivers"] or ["Matched-village comparison appears once the growth library is loaded."]
    st.markdown("".join(f"<div class='ga-small'>• {x}</div>" for x in lines), unsafe_allow_html=True)
    st.markdown(f"<div class='ga-small' style='margin-top:.6rem'><b>Starting level:</b> {res['level']['source']} "
                f"({res['level']['low']}-{res['level']['high']} kWh per customer per month)</div>", unsafe_allow_html=True)
    if res["growth"]["matches"]:
        m = pd.DataFrame(res["growth"]["matches"])[["name", "source", "country", "weight", "growth_year3"]]
        m.columns = ["Matched place", "Evidence", "Country", "Weight", "Year-3 growth"]
        st.dataframe(m.head(6), hide_index=True, width="stretch")
    else:
        st.markdown("<div class='ga-small'><b>Growth:</b> assumed curve (no matched villages loaded yet).</div>",
                    unsafe_allow_html=True)
    st.markdown("<div class='ga-small' style='margin-top:.6rem'><b>Grade reasons:</b> " + "; ".join(g["reasons"]) + "</div>",
                unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["Honesty labels", "Validation on real meters", "Data sharing", "Method"])
with t1:
    L = res["labels"]
    a1, a2, a3 = st.columns(3)
    a1.markdown("**Calibrated**\n\n" + ("\n".join(f"- {x}" for x in L["calibrated"]) or "- Nothing yet (demo mode)"))
    a2.markdown("**Assumed**\n\n" + "\n".join(f"- {x}" for x in L["assumed"]))
    a3.markdown("**Not yet proven**\n\n" + "\n".join(f"- {x}" for x in L["not_yet_proven"]))
with t2:
    v = res["validation"]
    if not v:
        st.info("Not run yet. After `pipeline/04_validate.py` runs on the Selby smart-meter sites, this tab shows "
                "leave-one-site-out error against the flat AMDA benchmark, interval coverage, and whether "
                "night-light growth tracks metered kWh.")
    else:
        meth = v["level"]["methods"]
        st.markdown(f"**{v['n_sites']} metered sites**, each predicted from the others only. Target: {v['level']['target']}.")
        st.dataframe(pd.DataFrame([{"Method": k, **(x or {})} for k, x in meth.items()]), hide_index=True,
                     width="stretch")
        st.markdown(f"Best method: **{v['level']['best_method']}** · share of sites inside the stated range: "
                    f"**{v['level']['interval_coverage_country_range']:.0%}** (target ~80%)")
        if "light_vs_kwh" in v and "median_spearman" in v["light_vs_kwh"]:
            lv = v["light_vs_kwh"]
            st.markdown(f"Night light vs metered kWh: median correlation **{lv['median_spearman']}** "
                        f"across {lv['n_sites']} sites.")
        st.caption(v["dataset"] + " · " + " ".join(v.get("caveats", [])))
with t3:
    st.markdown("""
- **Operators share** monthly kWh totals by customer type and 24-hour load shapes. **No personal data:** names, phone numbers and IDs are removed at the operator before upload.
- **A neutral host** (for example AMDA) receives exact site locations, computes each site's features, then **deletes the coordinates** and keeps only features and totals.
- **Published results** use aggregates from at least 5 operators, with small random noise added. Pilots run under one-to-one agreements until then.
- **In return,** operators get free forecasts and benchmarks for their own pipeline of sites.
""")
with t4:
    st.markdown("""
1. **Features:** buildings, large-building share, night lights, travel time, grid distance and market access for the site's 1 km area (open satellite and map data).
2. **Matching:** past villages that most resembled this one *before* they were electrified (Selby smart-meter sites and night-light switch-ons); their real growth paths are used unchanged.
3. **Starting level:** year-1 kWh per customer from metered sites (or AMDA benchmark in demo mode), placed by relative wealth.
4. **Monte Carlo:** 5,000 futures combine level × connection rate × growth path → P90 / P50 / P10 for the chosen year.
5. **Finance:** break-even demand per sizing option, chance of falling below it, and the staged-build saving. Solar resource from NASA POWER.
""")

# ─── PDF ─────────────────────────────────────────────────────────────────────
st.markdown("---")
try:
    pdf = generate_lender_pdf(res, site_name)
    st.download_button("⬇ Download Lender Card (PDF)", data=pdf,
                       file_name=f"GreenAmp_{site_name.replace(' ', '_')}_{T['year']}.pdf", mime="application/pdf")
except Exception as e:  # keep the app usable even if the PDF fails
    st.error(f"PDF could not be generated: {e}")

st.markdown('<div class="ga-small" style="margin-top:1.5rem">GreenAmp Energy Solutions · Data: Selby / Renewvia (2026), '
            'Zenodo DOI 10.5281/zenodo.22149102 (CC BY-SA 4.0); Google Open Buildings; NOAA VIIRS; Oxford/MAP; '
            'OpenStreetMap; NASA POWER; AMDA benchmarks. Screening tool: not investment advice or an engineering design.</div>',
            unsafe_allow_html=True)
