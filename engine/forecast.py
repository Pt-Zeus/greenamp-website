"""Single entry point: run_forecast(lat, lon, target_year, ...) -> dict.

The returned dict is JSON-serialisable and is everything the UI, the lender PDF and a
future paid report need. No Streamlit code lives here.
"""
from __future__ import annotations

import datetime as _dt
import math

import numpy as np

from . import finance, growth, montecarlo, sizing, store
from .features import get_features
from .grade import confidence_grade

YEARS = 5


# ── starting level (kWh per customer per month, year 1) ─────────────────────
def _wealth_percentile(rwi: float) -> tuple[float | None, str]:
    if rwi != rwi:  # NaN
        return None, "no wealth data"
    pool = []
    for df in (store.site_features(), store.growth_library()):
        if df is not None and "rwi" in df.columns:
            pool.extend(df["rwi"].dropna().tolist())
    if len(pool) >= 10:
        return float((np.array(pool) < rwi).mean()), "percentile within the data library"
    # Fallback mapping of raw RWI (roughly -1.5..+2) to 0..1
    return float(np.clip((rwi + 1.5) / 3.5, 0, 1)), "approximate (raw RWI scale)"


def _exclude_near(df, lat, lon, km):
    """Drop Selby sites within km of the query (replay without leakage)."""
    sites = store.selby_sites()
    if lat is None or sites is None or not {"lat", "lon", "name"} <= set(sites.columns):
        return df
    d = store.haversine_km(lat, lon, sites["lat"].to_numpy(), sites["lon"].to_numpy())
    near = set(sites.loc[d <= km, "name"])
    return df.loc[~df["name"].isin(near)] if near else df


def _starting_level(feats: dict, country: str | None, lat=None, lon=None, exclude_km: float = 2.0) -> dict:
    pct, pct_note = _wealth_percentile(feats.get("rwi", math.nan))

    model = store.level_model()
    if model is not None and feats.get("covered"):
        lo, mode, hi = model.predict_one(feats)
        return {"low": lo, "mode": mode, "high": hi, "source": "model (trained on Selby smart-meter sites)",
                "wealth_percentile": pct, "wealth_note": pct_note}

    y1 = store.selby_site_year1()
    if y1 is not None:
        y1 = _exclude_near(y1, lat, lon, exclude_km)
    if y1 is not None and len(y1) >= 5:
        vals = y1["kwh_per_customer_month_y1"].dropna()
        scope = "all Selby sites"
        if country and "country" in y1.columns:
            same = y1.loc[y1["country"].astype(str).str.lower() == country.lower(), "kwh_per_customer_month_y1"].dropna()
            if len(same) >= 3:
                vals, scope = same, f"Selby sites in {country}"
        lo, hi = float(vals.min()), float(vals.max())
        q = 0.5 if pct is None else float(np.clip(pct, 0.1, 0.9))
        mode = float(np.quantile(vals, q))
        return {"low": lo, "mode": mode, "high": hi,
                "source": f"meter (year-1 kWh per customer at {len(vals)} {scope}; placed by wealth)",
                "wealth_percentile": pct, "wealth_note": pct_note}

    b = store.benchmarks()["kwh_per_customer_month"]
    mode = b["mode"] if pct is None else b["mode"] * (0.5 + pct)
    return {"low": b["low"], "mode": float(np.clip(mode, b["low"], b["high"])), "high": b["high"],
            "source": "benchmark (AMDA average; Brief 03 range)" + ("" if pct is None else "; shifted by wealth (assumed rule)"),
            "wealth_percentile": pct, "wealth_note": pct_note}


def _r(x, nd=0):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


# ── main entry point ────────────────────────────────────────────────────────
def run_forecast(lat: float, lon: float, target_year: int,
                 switch_on_year: int | None = None,
                 overrides: dict | None = None,
                 tariff_usd_kwh: float | None = None,
                 grant_usd_per_connection: float | None = None,
                 n_draws: int = 5000,
                 exclude_own_site_km: float = 2.0,
                 include_draws: bool = False) -> dict:
    bm = store.benchmarks()
    today = _dt.date.today()
    switch_on_year = switch_on_year or today.year + 1

    feats = get_features(lat, lon, overrides)
    country = feats.get("country") or (overrides or {}).get("country")
    out: dict = {"inputs": {"lat": lat, "lon": lon, "target_year": target_year,
                            "switch_on_year": switch_on_year, "country": country},
                 "features": {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in feats.items()}}

    buildings = feats.get("building_count", math.nan)
    if not feats["covered"] or buildings != buildings or buildings <= 0:
        out["status"] = "outside_coverage"
        out["message"] = ("No satellite features for this location yet. Enter building count and other "
                          "features manually, or add this area to the data pipeline.")
        return out

    years_since = target_year - switch_on_year + 1
    if years_since < 1:
        out["status"] = "before_switch_on"
        out["message"] = "Target year is before the expected switch-on year."
        return out

    # 1. starting level and uptake
    level = _starting_level(feats, country, lat, lon, exclude_own_site_km)
    up = bm["uptake_share_of_buildings"]
    ramp = store.uptake_ramp()

    # 2. matched growth paths
    g = growth.match(feats, country=country, exclude_near=(lat, lon), exclude_km=exclude_own_site_km)

    # 3. Monte Carlo
    sim = montecarlo.simulate((level["low"], level["mode"], level["high"]), (up["low"], up["high"]),
                              buildings, g["paths"], g["weights"], ramp=ramp, n=n_draws)
    site = sim["site_draws"]

    # 4. target year
    if years_since <= YEARS:
        tgt_draws = site[:, years_since - 1]
        tgt_pc = sim["per_customer_draws"][:, years_since - 1]
        extrapolated = False
    else:
        tgt_draws = montecarlo.extrapolate(site[:, -1], years_since - YEARS)
        tgt_pc = montecarlo.extrapolate(sim["per_customer_draws"][:, -1], years_since - YEARS)
        extrapolated = True
    tq = montecarlo.lender_quantiles(tgt_draws)
    tpc = montecarlo.lender_quantiles(tgt_pc)

    # 5. sizing (real irradiance) — energy on P50, inverter headroom from P10
    psh, psh_src = sizing.peak_sun_hours(lat, lon)
    tilt, az = sizing.tilt_and_azimuth(lat)
    near_market = (feats.get("market_km", math.nan) == feats.get("market_km", math.nan)) and feats["market_km"] <= 2.0
    lf = sizing.load_factor(feats.get("large_building_share", math.nan), near_market)
    y = sim["yearly"]
    staged = sizing.size_system(y[1]["P50"], y[1]["P10"], psh, lf)        # phase 1: P50 in year 2
    oneshot = sizing.size_system(y[-1]["P10"], y[-1]["P10"], psh, lf)     # build for the high case, year 5
    conns = sim["yearly_customers"][-1]["P50"]

    # 6. finance
    tariff = tariff_usd_kwh or bm["tariff_usd_per_kwh"].get(country or "", bm["tariff_usd_per_kwh"]["default"])
    grant = bm["grant_usd_per_connection"]["default"] if grant_usd_per_connection is None else grant_usd_per_connection
    options = {}
    for name, sysd in (("staged", staged), ("one_shot", oneshot)):
        cap = finance.capex(sysd, conns, grant)
        be = finance.break_even_kwh_month(cap, tariff)
        options[name] = {**sysd, "capex_usd": round(cap, 0), **be,
                         "p_shortfall_by_year": finance.shortfall_probability(site, be["break_even_kwh_month"])}
    d = bm["design"]
    phase1_capacity = staged["pv_kwp"] * psh * sizing.DAYS_PER_MONTH / d["pv_oversize"]
    options["staged"]["expansion_trigger"] = (
        f"Expand when metered demand passes {int(d['expansion_trigger_share'] * 100)}% of phase-1 capacity "
        f"(about {phase1_capacity * d['expansion_trigger_share']:,.0f} kWh/month) for 30 days")
    saving = options["one_shot"]["capex_usd"] - options["staged"]["capex_usd"]

    afford = finance.affordability(sim["yearly_per_customer"][0]["P50"], tariff)

    # 7. grade, drivers, honesty labels
    grade = confidence_grade(feats, g, psh_src, level["source"])
    drivers = growth.drivers(feats, g["dims_used"])

    calibrated, assumed = [], []
    (calibrated if level["source"].startswith(("meter", "model")) else assumed).append(
        f"Starting level: {level['source']}")
    if g["source"] == "assumed":
        assumed.append("Growth: assumed curve (no matched villages available)")
    else:
        calibrated.append(f"Growth: {len(g['matches'])} matched places ({g['source']})")
    assumed.append(f"Uptake: {up['source']}")
    assumed.append(f"Costs and debt terms: {bm['costs_usd']['source']}")
    assumed.append(f"Load factor rule: {bm['load_factor_rule']['source']}")
    if not psh_src.startswith("NASA"):
        assumed.append("Solar resource: fallback value")
    unproven = ["kWh accuracy outside the validated sites"]
    if any(m["source"] == "nightlight" for m in g["matches"]):
        unproven.insert(0, "Whether night-light growth tracks kWh growth (night-light paths also mix in new connections)")

    out.update({
        "status": "ok",
        "target": {"year": target_year, "years_since_switch_on": years_since, "extrapolated": extrapolated,
                   "site_kwh_month": {k: _r(v) for k, v in tq.items()},
                   "per_customer_kwh_month": {k: _r(v, 1) for k, v in tpc.items()}},
        "fan": [{"year_index": i + 1, "calendar_year": switch_on_year + i, **{k: _r(v) for k, v in q.items()}}
                for i, q in enumerate(sim["yearly"])],
        "customers": [{k: _r(v) for k, v in q.items()} for q in sim["yearly_customers"]],
        "level": {k: (_r(v, 2) if isinstance(v, float) else v) for k, v in level.items()},
        "growth": {"source": g["source"], "n_good": g["n_good"], "dims_used": g["dims_used"],
                   "excluded_own_site_rows": g.get("n_excluded", 0),
                   "matches": g["matches"][:10]},
        "solar": {"psh": round(psh, 2), "psh_source": psh_src, "tilt_deg": tilt, "azimuth": az,
                  "load_factor": round(lf, 3)},
        "options": options,
        "staged_saving_usd": round(saving, 0),
        "tariff_usd_kwh": tariff,
        "grant_usd_per_connection": grant,
        "affordability": afford,
        "grade": grade,
        "drivers": drivers,
        "labels": {"calibrated": calibrated, "assumed": assumed, "not_yet_proven": unproven},
        "validation": store.validation(),
        "convention": "P90 = exceeded in 9 of 10 futures (low case); P50 = median; P10 = high case.",
        "sources": [
            "Selby / Renewvia Energy (2026). High-Frequency, Multi-Domain Operational Dataset for Community "
            "Solar Mini-Grids in Kenya and Nigeria. Zenodo. DOI 10.5281/zenodo.22149102 (CC BY-SA 4.0)",
            "NASA POWER climatology (ALLSKY_SFC_SW_DWN)",
            "AMDA, Benchmarking Africa's Minigrids",
            "Google Open Buildings; NOAA VIIRS; Oxford/MAP accessibility; OpenStreetMap",
        ],
    })
    if include_draws:
        out["_site_draws"] = site
    return out
