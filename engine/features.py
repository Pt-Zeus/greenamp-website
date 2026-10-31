"""Look up real features for a coordinate.

Order of lookup:
  1. data/site_features.parquet  (named demo / replay sites, nearest within max_km)
  2. data/tiles/<lat>_<lon>.parquet (candidate 1 km cells, nearest within max_km)
  3. nothing -> "outside coverage" (the app should say so, not guess)

User overrides (the sidebar "what-if" sliders) are applied on top and flagged.
"""
from __future__ import annotations

import math

from . import store

# Matching features (one role each: they choose which past villages count as similar).
MATCH_FEATURES = [
    "building_count",        # buildings in the 1 km cell (year before switch-on for library rows)
    "large_building_share",  # share of footprints above 65 m2 (productive-use proxy)
    "nightlight",            # VIIRS mean radiance, nW/cm2/sr
    "travel_time_min",       # Oxford/MAP travel time to nearest city
    "dist_grid_km",          # distance to grid line (OSM power lines, GridFinder fallback)
    "market_km",             # distance to nearest marketplace (OSM)
]
# Starting-level feature only (never used for matching).
LEVEL_FEATURE = "rwi"        # Relative Wealth Index (raw), converted to a percentile in forecast.py

ALL_FEATURES = MATCH_FEATURES + [LEVEL_FEATURE]


def _nearest(df, lat, lon, max_km):
    d = store.haversine_km(lat, lon, df["lat"].to_numpy(), df["lon"].to_numpy())
    i = int(d.argmin())
    if d[i] > max_km:
        return None, float(d[i])
    return df.iloc[i], float(d[i])


def get_features(lat: float, lon: float, overrides: dict | None = None, max_km: float = 1.5) -> dict:
    out: dict = {k: math.nan for k in ALL_FEATURES}
    out.update({"source": "none", "matched_point_km": None, "site_name": None, "country": None})

    row, dist = None, None
    sites = store.site_features()
    if sites is not None:
        row, dist = _nearest(sites, lat, lon, max_km)
    if row is None:
        t = store.tile(*store.floor_tile(lat, lon))
        if t is not None:
            row, dist = _nearest(t, lat, lon, max_km)

    if row is not None:
        for k in ALL_FEATURES:
            if k in row.index and row[k] == row[k]:  # not NaN
                out[k] = float(row[k])
        out["source"] = "satellite"
        out["matched_point_km"] = round(dist, 2)
        out["site_name"] = row.get("name") if "name" in row.index else None
        out["country"] = row.get("country") if "country" in row.index else None

    if overrides:
        used = {k: float(v) for k, v in overrides.items() if k in ALL_FEATURES and v is not None}
        if used:
            out.update(used)
            out["overridden"] = sorted(used)
            out["source"] = "user-entered" if out["source"] == "none" else "satellite + user overrides"
        if overrides.get("country"):
            out["country"] = overrides["country"]

    out["covered"] = out["source"] != "none"
    return out
