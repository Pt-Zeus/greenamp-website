"""Solar PV and battery sizing with real irradiance (NASA POWER), cached."""
from __future__ import annotations

import json

from . import store

DAYS_PER_MONTH = 30.4


def peak_sun_hours(lat: float, lon: float) -> tuple[float, str]:
    """Annual mean all-sky irradiance (kWh/m2/day) = peak sun hours. Returns (value, source)."""
    cache_path = store.DATA_DIR / "cache_psh.json"
    key = f"{round(lat, 2)},{round(lon, 2)}"
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    if key in cache:
        return float(cache[key]), "NASA POWER (cached)"
    try:
        import requests
        url = ("https://power.larc.nasa.gov/api/temporal/climatology/point"
               f"?parameters=ALLSKY_SFC_SW_DWN&community=RE&latitude={lat:.4f}&longitude={lon:.4f}&format=JSON")
        r = requests.get(url, timeout=15, headers={"User-Agent": "GreenAmp-hackathon/1.0"})
        r.raise_for_status()
        val = float(r.json()["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]["ANN"])
        if val <= 0:
            raise ValueError("bad value")
        cache[key] = val
        try:
            store.DATA_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))
        except Exception:
            pass
        return val, "NASA POWER"
    except Exception:
        return float(store.benchmarks()["design"]["psh_fallback"]), "assumed (NASA POWER unreachable)"


def tilt_and_azimuth(lat: float) -> tuple[float, str]:
    tilt = max(12.0, round(abs(lat) * 0.95, 1))
    return tilt, ("True North (0°)" if lat < 0 else "True South (180°)")


def load_factor(large_building_share: float, near_market: bool) -> float:
    r = store.benchmarks()["load_factor_rule"]
    share = 0.0 if large_building_share != large_building_share else large_building_share
    return min(r["cap"], r["base"] + r["per_large_building_share"] * share + (r["market_bonus"] if near_market else 0.0))


def size_system(energy_kwh_month: float, peak_kwh_month: float, psh: float, lf: float) -> dict:
    """Energy sized on `energy_kwh_month` (P50); inverter/peak headroom from `peak_kwh_month` (P10)."""
    d = store.benchmarks()["design"]
    daily = energy_kwh_month / DAYS_PER_MONTH
    pv_kwp = daily / psh * d["pv_oversize"]
    batt_kwh = daily * d["night_share"] / (d["depth_of_discharge"] * d["round_trip_efficiency"])
    peak_kw = (peak_kwh_month / DAYS_PER_MONTH / 24.0) / max(lf, 0.05)
    return {"daily_kwh": round(daily, 1), "pv_kwp": round(pv_kwp, 1),
            "battery_kwh": round(batt_kwh, 1), "inverter_kw": round(peak_kw, 1)}
