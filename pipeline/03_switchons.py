"""Build the growth library from (a) night-light switch-ons and (b) Selby smart-meter sites.

(a) Night-light switch-ons
    Input: monthly VIIRS series for 1 km cells  (columns: name, month 'YYYY-MM', rad)
           + a cells file with name, lat, lon [, country]
    A cell "switches on" when, between 2017 and 2020, a sustained step up follows at least
    24 months of darkness. Its growth path g1..g5 is the yearly mean radiance after switch-on
    divided by year 1 (months 3-14 after switch-on, matching the Selby convention).
    Night-light paths mix new connections with more use per customer: labelled as a proxy.

(b) Selby meter paths
    From data/selby_site_year1.parquet (pipeline/01): g1..g5 = yearly kWh per active customer
    divided by year 1. Features come from data/selby_features.parquet (pipeline/02, ref_year =
    year before start).

Two-step use for night-light cells:
  1) python pipeline/03_switchons.py --series cells_series.parquet --cells cells.parquet --out data
     -> writes data/switchon_points.parquet (cells that switched on, with ref_year)
  2) python pipeline/02_site_features.py --points data/switchon_points.parquet --project P \
         --out data/switchon_features.parquet
  3) rerun step 1 with --cell-features data/switchon_features.parquet to attach features.

Optional: --export-bbox W S E N --project P  starts an Earth Engine export (to Google Drive) of
monthly VIIRS series for built-up 1 km cells in that box (untested here; try a small box first).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

DARK = 0.5          # nW/cm2/sr: mean radiance below this counts as dark
JUMP = 0.5          # minimum rise in mean radiance after switch-on
PRE_MONTHS = 24
ONSET_YEARS = (2017, 2020)
SKIP = 2


def detect_onset(s: pd.Series) -> int | None:
    """Return the index of the switch-on month in s (monthly, sorted), or None."""
    x = s.rolling(3, center=True, min_periods=1).median().to_numpy()
    n = len(x)
    if n < PRE_MONTHS + 18:
        return None
    candidates = []
    try:
        import ruptures as rpt
        sig = np.nan_to_num(x, nan=np.nanmedian(x))
        pen = 2 * np.log(n) * max(np.var(sig), 1e-3)
        candidates = [b for b in rpt.Pelt(model="l2", min_size=6).fit(sig).predict(pen=pen) if b < n]
    except Exception:
        pass
    if not candidates:
        candidates = list(range(PRE_MONTHS, n - 12))
    for i in sorted(candidates):
        if i < PRE_MONTHS or i > n - 12:
            continue
        pre = np.nanmean(x[i - PRE_MONTHS:i])
        post = np.nanmean(x[i:i + 12])
        if pre <= DARK and post - pre >= JUMP and post >= 2 * pre:
            return i
    return None


def growth_path(values: np.ndarray, start: int) -> list[float]:
    yearly = []
    for y in range(5):
        lo, hi = start + SKIP + 12 * y, start + SKIP + 12 * (y + 1)
        w = values[lo:hi]
        yearly.append(float(np.nanmean(w)) if len(w) >= 10 and np.isfinite(w).sum() >= 8 else np.nan)
    if not yearly[0] or not np.isfinite(yearly[0]) or yearly[0] < 0.2:
        return [np.nan] * 5
    return [v / yearly[0] if np.isfinite(v) else np.nan for v in yearly]


def nightlight_rows(series: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, grp in series.groupby("name"):
        g = grp.sort_values("month").set_index("month")["rad"].astype(float)
        months = pd.period_range(g.index.min(), g.index.max(), freq="M").strftime("%Y-%m")
        g = g.reindex(months)
        i = detect_onset(g)
        if i is None:
            continue
        onset = g.index[i]
        if not (ONSET_YEARS[0] <= int(onset[:4]) <= ONSET_YEARS[1]):
            continue
        path = growth_path(g.to_numpy(), i)
        if not np.isfinite(path[0]):
            continue
        rows.append({"name": name, "onset_month": onset, "ref_year": int(onset[:4]) - 1,
                     "source": "nightlight", **{f"g{k + 1}": v for k, v in enumerate(path)}})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.merge(cells, on="name", how="left")
    return out


def meter_rows(year1: pd.DataFrame, feats: pd.DataFrame | None, sites: pd.DataFrame | None) -> pd.DataFrame:
    keep = ["name", "country"] + [f"g{i}" for i in range(1, 6)]
    m = year1[[c for c in keep if c in year1.columns]].copy()
    m["source"] = "meter"
    if sites is not None:
        m = m.merge(sites[["name", "lat", "lon"]], on="name", how="left")
    if feats is not None:
        fcols = [c for c in feats.columns if c not in ("lat", "lon", "country", "ref_year")]
        m = m.merge(feats[fcols], on="name", how="left")
    return m


def export_bbox(project: str, bbox: list[float], name: str):
    """Start an EE export of monthly VIIRS means for built-up 1 km cells in bbox (W,S,E,N)."""
    import ee
    ee.Initialize(project=project)
    region = ee.Geometry.Rectangle(bbox)
    built = (ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1").filterBounds(region)
             .filter(ee.Filter.calendarRange(2016, 2016, "year")).mosaic().select("building_presence").gt(0.5))
    frac = built.reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:4326", scale=1000)
    cells = frac.gt(0.002).selfMask().reduceToVectors(geometry=region, scale=1000, geometryType="centroid",
                                                      maxPixels=1e10)
    cells = cells.map(lambda f: f.set("name", f.id()))
    col = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate("2012-04-01", "2026-01-01") \
        .select("avg_rad")

    def per_image(img):
        month = ee.Date(img.get("system:time_start")).format("YYYY-MM")
        return img.reduceRegions(cells.map(lambda f: f.buffer(500).bounds()), ee.Reducer.mean(), 500) \
            .map(lambda f: f.set("month", month))

    table = ee.FeatureCollection(col.map(per_image)).flatten()
    ee.batch.Export.table.toDrive(table, description=f"viirs_{name}", fileFormat="CSV",
                                  selectors=["name", "month", "mean"]).start()
    ee.batch.Export.table.toDrive(cells.map(lambda f: f.set("lon", f.geometry().coordinates().get(0),
                                                             "lat", f.geometry().coordinates().get(1))),
                                  description=f"cells_{name}", fileFormat="CSV",
                                  selectors=["name", "lat", "lon"]).start()
    print("Exports started; download the two CSVs from Google Drive, rename column 'mean' to 'rad'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", help="monthly VIIRS series parquet/csv: name, month, rad")
    ap.add_argument("--cells", help="cells parquet/csv: name, lat, lon [, country]")
    ap.add_argument("--cell-features", help="pipeline/02 output for switch-on cells")
    ap.add_argument("--selby-year1", default="data/selby_site_year1.parquet")
    ap.add_argument("--selby-features", default="data/selby_features.parquet")
    ap.add_argument("--selby-sites", default="data/selby_sites.parquet")
    ap.add_argument("--out", default="data")
    ap.add_argument("--export-bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--project")
    a = ap.parse_args()

    if a.export_bbox:
        export_bbox(a.project, a.export_bbox, "region")
        return

    read = lambda p: (pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)) if p and os.path.exists(p) else None
    parts = []

    series, cells = read(a.series), read(a.cells)
    if series is not None:
        if "rad" not in series.columns and "mean" in series.columns:
            series = series.rename(columns={"mean": "rad"})
        nl = nightlight_rows(series, cells if cells is not None else pd.DataFrame({"name": []}))
        print(f"Night-light switch-ons (2017-2020, dark 24 months before): {len(nl)} of {series['name'].nunique()} cells")
        if len(nl):
            nl[["name", "lat", "lon", "ref_year"] + (["country"] if "country" in nl.columns else [])] \
                .to_parquet(os.path.join(a.out, "switchon_points.parquet"), index=False)
            feats = read(a.cell_features)
            if feats is not None:
                fcols = [c for c in feats.columns if c not in ("lat", "lon", "country", "ref_year")]
                nl = nl.merge(feats[fcols], on="name", how="left")
            parts.append(nl)

    y1 = read(a.selby_year1)
    if y1 is not None:
        mr = meter_rows(y1, read(a.selby_features), read(a.selby_sites))
        print(f"Selby meter growth paths: {len(mr)} sites")
        parts.append(mr)

    if not parts:
        raise SystemExit("Nothing to build: give --series/--cells and/or run pipeline/01 first.")
    lib = pd.concat(parts, ignore_index=True)
    lib.insert(0, "id", [f"{s}:{n}" for s, n in zip(lib["source"], lib["name"])])
    lib.to_parquet(os.path.join(a.out, "growth_library.parquet"), index=False)
    print(f"Wrote growth library: {len(lib)} rows "
          f"({(lib['source'] == 'meter').sum()} meter, {(lib['source'] == 'nightlight').sum()} night-light)")


if __name__ == "__main__":
    main()
