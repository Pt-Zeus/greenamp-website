"""Validation on REAL smart-meter data (Selby / Renewvia sites), leave-one-site-out.

Replaces benchmark_data.py, which tested the model on synthetic data it generated itself.

Tests
  A. Year-1 kWh per customer per month (the brief's Impact metric: error vs metered demand)
     methods compared, each predicting a held-out site from the OTHER sites only:
       flat_amda      - AMDA continental average (6.1 kWh/customer/month)
       pooled_median  - median of the other sites
       country_median - median of other sites in the same country (pooled if < 3)
       quantile_model - small quantile model on satellite features (only if >= 10 sites have features)
     plus interval coverage of the stated low-high range (should be about 80%).
  B. Growth: year-2 and year-3 index (kWh per customer vs year 1), compared with flat growth and 5%/yr.
  C. Night-light vs kWh: does yearly VIIRS radiance move with yearly metered kWh at the same sites?

Usage:
  python pipeline/04_validate.py --data data [--series data/selby_viirs_series.parquet]
Writes data/validation.json (shown in the app) and, if it wins, data/level_model.joblib.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.quantile_model import QuantileLevelModel  # noqa: E402

AMDA_AVG = 6.1
DOI = "10.5281/zenodo.22149102"


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0)
    y, p = y[ok], p[ok]
    if not len(y):
        return None
    ape = np.abs(p - y) / y
    return {"n": int(len(y)), "mae": round(float(np.mean(np.abs(p - y))), 2),
            "mape_pct": round(float(np.mean(ape) * 100), 1),
            "median_ape_pct": round(float(np.median(ape) * 100), 1)}


def tri_quantiles(lo, mode, hi, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    mode = min(max(mode, lo), hi)
    x = rng.triangular(lo, mode, hi, n) if hi > lo else np.full(n, mode)
    return np.percentile(x, 10), np.percentile(x, 90)


def level_tests(y1: pd.DataFrame, feats: pd.DataFrame | None) -> tuple[dict, object]:
    d = y1.dropna(subset=["kwh_per_customer_month_y1"]).reset_index(drop=True)
    if feats is not None:
        fcols = [c for c in feats.columns if c not in ("lat", "lon", "country", "ref_year")]
        d = d.merge(feats[fcols], on="name", how="left")
    y = d["kwh_per_customer_month_y1"].to_numpy()
    preds = {k: [] for k in ("flat_amda", "pooled_median", "country_median")}
    cover, rows = [], []

    model_ok = feats is not None and d.get("building_count") is not None and d["building_count"].notna().sum() >= 10
    if model_ok:
        preds["quantile_model"] = []
    model_cover = []

    for i in range(len(d)):
        others = d.drop(index=i)
        ov = others["kwh_per_customer_month_y1"]
        same = others.loc[others["country"] == d.loc[i, "country"], "kwh_per_customer_month_y1"]
        ref = same if len(same) >= 3 else ov
        preds["flat_amda"].append(AMDA_AVG)
        preds["pooled_median"].append(float(ov.median()))
        preds["country_median"].append(float(ref.median()))
        lo, hi = tri_quantiles(float(ref.min()), float(ref.median()), float(ref.max()))
        cover.append(lo <= y[i] <= hi)
        if model_ok:
            m = QuantileLevelModel().fit(others, others["kwh_per_customer_month_y1"])
            q = m.predict_one(d.loc[i].to_dict())
            preds["quantile_model"].append(q[1])
            model_cover.append(q[0] <= y[i] <= q[2])
        rows.append({"site": d.loc[i, "name"], "country": d.loc[i, "country"], "actual": round(float(y[i]), 2),
                     **{k: round(v[-1], 2) for k, v in preds.items()}})

    res = {"target": "year-1 mean kWh per active customer per month (months 3-14 after start)",
           "methods": {k: metrics(y, v) for k, v in preds.items()},
           "interval_coverage_country_range": round(float(np.mean(cover)), 2),
           "per_site": rows}
    if model_cover:
        res["interval_coverage_quantile_model"] = round(float(np.mean(model_cover)), 2)

    final_model = None
    if model_ok:
        mm, cm = res["methods"]["quantile_model"], res["methods"]["country_median"]
        if mm and cm and mm["mae"] < cm["mae"]:
            final_model = QuantileLevelModel().fit(d, d["kwh_per_customer_month_y1"])
            res["level_model_saved"] = True
        else:
            res["level_model_saved"] = False
            res["level_model_note"] = "Quantile model did not beat the country median; engine keeps the prior."
    best = min((k for k, v in res["methods"].items() if v), key=lambda k: res["methods"][k]["mae"])
    res["best_method"] = best
    return res, final_model


def growth_tests(y1: pd.DataFrame) -> dict:
    out = {}
    for col, years in (("g2", 1), ("g3", 2)):
        d = y1.dropna(subset=[col]).reset_index(drop=True)
        if len(d) < 4:
            out[col] = {"n": int(len(d)), "note": "too few sites with this many years of data"}
            continue
        y = d[col].to_numpy()
        loo = [float(d.drop(index=i)[col].median()) for i in range(len(d))]
        out[col] = {
            "meaning": f"kWh per customer in year {years + 1} relative to year 1",
            "actual_median": round(float(np.median(y)), 3),
            "leave_one_out_median_of_others": metrics(y, loo),
            "flat_growth": metrics(y, np.ones(len(y))),
            "five_pct_per_year": metrics(y, np.full(len(y), 1.05 ** years)),
        }
    return out


def light_vs_kwh(monthly: pd.DataFrame, series: pd.DataFrame) -> dict:
    s = series.rename(columns={"mean": "rad"}) if "rad" not in series.columns else series
    s = s.assign(year=s["month"].str[:4]).groupby(["name", "year"])["rad"].mean()
    m = monthly.assign(year=monthly["month"].str[:4],
                       kwh=monthly["kwh_customer_meters"] + monthly["kwh_pue"])
    m = m.groupby(["name", "year"]).agg(kwh=("kwh", "sum"), n=("month", "count"))
    m = m[m["n"] >= 10]["kwh"]
    j = pd.concat([s, m], axis=1, join="inner").dropna().reset_index()
    per_site = []
    for name, g in j.groupby("name"):
        if len(g) >= 3 and g["rad"].std() > 0 and g["kwh"].std() > 0:
            per_site.append({"site": name, "years": int(len(g)),
                             "spearman": round(float(g["rad"].corr(g["kwh"], method="spearman")), 2)})
    if not per_site:
        return {"note": "not enough overlapping years"}
    rho = [p["spearman"] for p in per_site]
    return {"question": "Does yearly night-light radiance move with yearly metered kWh at the same site?",
            "n_sites": len(per_site), "median_spearman": round(float(np.median(rho)), 2),
            "share_positive": round(float(np.mean(np.array(rho) > 0)), 2), "per_site": per_site}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--series", default=None, help="VIIRS series for Selby sites (pipeline/02 --series)")
    a = ap.parse_args()
    p = lambda f: os.path.join(a.data, f)
    y1 = pd.read_parquet(p("selby_site_year1.parquet"))
    feats = pd.read_parquet(p("selby_features.parquet")) if os.path.exists(p("selby_features.parquet")) else None

    level, model = level_tests(y1, feats)
    out = {"dataset": f"Selby / Renewvia Energy (2026), Zenodo DOI {DOI}, CC BY-SA 4.0",
           "n_sites": int(y1["kwh_per_customer_month_y1"].notna().sum()),
           "level": level, "growth": growth_tests(y1),
           "caveats": ["Small sample (one operator, two countries); treat as indicative.",
                       "Sites were already built, so features may include post-electrification change "
                       "unless pipeline/02 was run with ref_year = year before start."]}
    if a.series and os.path.exists(a.series):
        out["light_vs_kwh"] = light_vs_kwh(pd.read_parquet(p("selby_monthly.parquet")), pd.read_parquet(a.series))

    with open(p("validation.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    if model is not None:
        import joblib
        joblib.dump(model, p("level_model.joblib"))

    print(json.dumps({k: out["level"]["methods"][k] for k in out["level"]["methods"]}, indent=2))
    print("best:", out["level"]["best_method"], "| coverage:", out["level"]["interval_coverage_country_range"])


if __name__ == "__main__":
    main()
