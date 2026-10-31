"""Real features for a list of points (Selby sites, demo pins, candidate cells).

Each point is described by its 1 km cell (about +/-0.0045 degrees), the same unit used for the
growth library, so query and library compare like with like.

Feature                 Source                                                         Unit
building_count          Open Buildings 2.5D Temporal, year = ref_year (2016-2023)      count
building_count_v3       Open Buildings V3 polygons (confidence >= 0.70), cross-check   count
large_building_share    Open Buildings V3, share of footprints > 65 m2                 0-1
nightlight              VIIRS monthly (stray-light corrected) avg_rad, mean of ref_year nW/cm2/sr
travel_time_min         Oxford/MAP accessibility to cities 2015                        minutes
dist_grid_km            GridFinder (if --grid-asset given) else OSM power lines        km
market_km               OSM amenity=marketplace (Overpass API)                         km
rwi                     Relative Wealth Index (GEE community catalogue) - CHECK LICENCE  index

Requirements: pip install earthengine-api requests pandas pyarrow ; earthengine authenticate
Earth Engine: free tier is NON-COMMERCIAL only. Register a non-commercial Cloud project.

Usage:
  python pipeline/02_site_features.py --points data/selby_sites.parquet --project YOUR_GCP_PROJECT \
         --out data/selby_features.parquet --series data/selby_viirs_series.parquet
  python pipeline/02_site_features.py --points demo_points.csv --project YOUR_GCP_PROJECT \
         --out data/site_features.parquet
Points file needs columns: name, lat, lon; optional: country, ref_year (default = latest year).
Untested against Earth Engine in the authoring environment: run on 3 points first and
compare building_count with building_count_v3 before trusting it.
"""
from __future__ import annotations

import argparse
import math
import time

import pandas as pd

HALF = 0.0045
OB_V3 = "GOOGLE/Research/open-buildings/v3/polygons"
OB_T = "GOOGLE/Research/open-buildings-temporal/v1"
VIIRS = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
ACCESS = "Oxford/MAP/accessibility_to_cities_2015_v1_0"
RWI = "projects/sat-io/open-datasets/facebook/relative_wealth_index"
OVERPASS = "https://overpass-api.de/api/interpreter"
LATEST_YEAR = 2023        # last year in Open Buildings Temporal
VIIRS_LATEST = 2025


def hav_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ── Earth Engine features ───────────────────────────────────────────────────
def ee_features(ee, lat, lon, ref_year, grid_asset=None):
    cell = ee.Geometry.Rectangle([lon - HALF, lat - HALF, lon + HALF, lat + HALF])
    pt = ee.Geometry.Point([lon, lat])
    by = int(min(max(ref_year, 2016), LATEST_YEAR))
    vy = int(min(max(ref_year, 2013), VIIRS_LATEST))

    # Temporal building count: fractional counts summed at native 0.5 m resolution.
    temporal = (ee.ImageCollection(OB_T).filterBounds(cell)
                .filter(ee.Filter.calendarRange(by, by, "year")).mosaic()
                .select("building_fractional_count"))
    t_count = temporal.reduceRegion(ee.Reducer.sum(), cell, scale=0.5, maxPixels=1e10, tileScale=4) \
        .get("building_fractional_count")

    v3 = ee.FeatureCollection(OB_V3).filterBounds(cell).filter(ee.Filter.gte("confidence", 0.70))
    v3_n = v3.size()
    v3_large = v3.filter(ee.Filter.gt("area_in_meters", 65)).size()

    ntl = (ee.ImageCollection(VIIRS).filterDate(f"{vy}-01-01", f"{vy + 1}-01-01").select("avg_rad").mean()
           .reduceRegion(ee.Reducer.mean(), cell, scale=500).get("avg_rad"))
    acc = ee.Image(ACCESS).select("accessibility").reduceRegion(ee.Reducer.mean(), cell, scale=1000) \
        .get("accessibility")

    rwi_fc = ee.FeatureCollection(RWI).filterBounds(pt.buffer(3000)) \
        .map(lambda f: f.set("d", f.geometry().distance(pt)))
    rwi = ee.Algorithms.If(rwi_fc.size().gt(0), ee.Feature(rwi_fc.sort("d").first()).get("rwi"), None)

    out = {"t_count": t_count, "v3_n": v3_n, "v3_large": v3_large, "ntl": ntl, "acc": acc, "rwi": rwi}
    if grid_asset:
        dist = ee.FeatureCollection(grid_asset).distance(searchRadius=150000, maxError=50) \
            .reduceRegion(ee.Reducer.first(), pt, scale=100).get("distance")
        out["grid_m"] = dist
    return ee.Dictionary(out).getInfo()


# ── OpenStreetMap features (Overpass) ───────────────────────────────────────
def overpass(query: str, retries: int = 3):
    import requests
    for i in range(retries):
        try:
            r = requests.post(OVERPASS, data={"data": query}, timeout=90,
                              headers={"User-Agent": "GreenAmp-hackathon/1.0"})
            if r.status_code == 200:
                return r.json().get("elements", [])
        except Exception:
            pass
        time.sleep(5 * (i + 1))
    return None


def nearest_km(elements, lat, lon):
    best = None
    for e in elements or []:
        pts = []
        if "lat" in e:
            pts.append((e["lat"], e["lon"]))
        if "center" in e:
            pts.append((e["center"]["lat"], e["center"]["lon"]))
        for g in e.get("geometry", []) or []:
            pts.append((g["lat"], g["lon"]))
        for p in pts:
            d = hav_km(lat, lon, p[0], p[1])
            best = d if best is None or d < best else best
    return best


def osm_features(lat, lon, radius_km=30):
    r = int(radius_km * 1000)
    lines = overpass(f'[out:json][timeout:80];(way["power"~"^(line|minor_line)$"](around:{r},{lat},{lon}););out geom;')
    time.sleep(1.5)
    mk = overpass(f'[out:json][timeout:60];(node["amenity"="marketplace"](around:{r},{lat},{lon});'
                  f'way["amenity"="marketplace"](around:{r},{lat},{lon}););out center;')
    time.sleep(1.5)
    grid = nearest_km(lines, lat, lon)
    mkt = nearest_km(mk, lat, lon)
    return (grid if grid is not None else (float(radius_km) if lines is not None else math.nan),
            mkt if mkt is not None else (float(radius_km) if mk is not None else math.nan))


# ── VIIRS monthly series (for switch-on detection and the light-vs-kWh test) ─
def viirs_series(ee, points: pd.DataFrame, start="2012-04-01", end="2026-01-01"):
    feats = [ee.Feature(ee.Geometry.Rectangle([r.lon - HALF, r.lat - HALF, r.lon + HALF, r.lat + HALF]),
                        {"name": str(r.name_)}) for r in points.assign(name_=points["name"]).itertuples()]
    fc = ee.FeatureCollection(feats)
    col = ee.ImageCollection(VIIRS).filterDate(start, end).select("avg_rad")

    def per_image(img):
        month = ee.Date(img.get("system:time_start")).format("YYYY-MM")
        return img.reduceRegions(fc, ee.Reducer.mean(), 500).map(lambda f: f.set("month", month))

    rows = []
    years = range(int(start[:4]), int(end[:4]))
    for y in years:                                    # chunk by year to stay under getInfo limits
        sub = col.filterDate(f"{y}-01-01", f"{y + 1}-01-01")
        info = ee.FeatureCollection(sub.map(per_image)).flatten().getInfo()
        for f in info["features"]:
            p = f["properties"]
            rows.append({"name": p.get("name"), "month": p.get("month"), "rad": p.get("mean")})
        print(f"  VIIRS series {y}: {len(info['features'])} rows", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="parquet or csv with name, lat, lon [, country, ref_year]")
    ap.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid-asset", default=None, help="GridFinder lines uploaded as an EE FeatureCollection asset")
    ap.add_argument("--no-osm", action="store_true")
    ap.add_argument("--series", default=None, help="also write monthly VIIRS series to this parquet")
    a = ap.parse_args()

    import ee
    ee.Initialize(project=a.project)

    pts = pd.read_parquet(a.points) if a.points.endswith(".parquet") else pd.read_csv(a.points)
    if "ref_year" not in pts.columns:
        pts["ref_year"] = LATEST_YEAR
    rows = []
    for r in pts.itertuples():
        print(f"[{r.name}] {r.lat:.4f}, {r.lon:.4f} (ref year {int(r.ref_year)})", flush=True)
        try:
            e = ee_features(ee, float(r.lat), float(r.lon), int(r.ref_year), a.grid_asset)
        except Exception as ex:
            print(f"   Earth Engine error: {ex}")
            e = {}
        grid_km, market_km = (math.nan, math.nan) if a.no_osm else osm_features(float(r.lat), float(r.lon))
        if e.get("grid_m") is not None:
            grid_km, grid_src = e["grid_m"] / 1000.0, "GridFinder"
        else:
            grid_src = "OSM power lines" if not a.no_osm else "missing"
        v3n = e.get("v3_n") or 0
        rows.append({
            "name": r.name, "lat": float(r.lat), "lon": float(r.lon),
            "country": getattr(r, "country", None), "ref_year": int(r.ref_year),
            "building_count": e.get("t_count"),
            "building_count_v3": v3n,
            "large_building_share": (e.get("v3_large") or 0) / v3n if v3n else math.nan,
            "nightlight": e.get("ntl"),
            "travel_time_min": e.get("acc"),
            "dist_grid_km": grid_km, "dist_grid_source": grid_src,
            "market_km": market_km,
            "rwi": e.get("rwi"),
        })
        print(f"   buildings {rows[-1]['building_count']} (V3: {v3n}), light {rows[-1]['nightlight']}, "
              f"grid {grid_km} km", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(a.out, index=False)
    print(f"Wrote {len(out)} rows to {a.out}")

    if a.series:
        s = viirs_series(ee, pts)
        s.to_parquet(a.series, index=False)
        print(f"Wrote VIIRS series ({len(s)} rows) to {a.series}")


if __name__ == "__main__":
    main()
