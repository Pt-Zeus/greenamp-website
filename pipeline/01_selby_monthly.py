"""Selby / Renewvia smart-meter data -> monthly kWh per site and per customer.

Data: "High-Frequency, Multi-Domain Operational Dataset for Community Solar Mini-Grids
in Kenya and Nigeria", Zenodo DOI 10.5281/zenodo.22149102, CC BY-SA 4.0.
(Derived data you publish must keep the CC BY-SA 4.0 licence and cite the DOI.)

Download from Zenodo into one folder (SELBY_DIR):
  minigridprojects.parquet, customers.parquet, meteringbasestations.parquet  (small)
  cleaned readings, either  sparkmeterreadings_clean/<site>.parquet
                     or     sparkmeterreadings_clean_<site>.parquet
Start with 5-10 of the SMALLER cleaned files; the full set is ~72 GB with raw files.

Usage:
  python pipeline/01_selby_monthly.py --selby-dir /path/to/selby --out data
  python pipeline/01_selby_monthly.py --selby-dir ... --sites akipelai balep   # subset

Outputs (in --out):
  selby_sites.parquet       one row per site: coordinates, installed PV/battery, capex, dates
  selby_monthly.parquet     site x month: active customers, kWh (customer / productive-use / totalizer)
  selby_site_year1.parquet  site: year-1 kWh per customer, yearly growth g1..g5, customer ramp c1..c5
  selby_ramp.json           median share of eventual customers active in years 1..5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLS = ["meter_customer_code", "meter_type", "slot_start", "energy_kwh"]
START_MIN_CUSTOMERS = 5      # site "starts" in the first month with >= 5 active customers
SKIP_MONTHS = 2              # ignore the first 2 months (ramp-up), so year 1 = months 3-14
ACTIVE_KWH = 0.05            # a customer counts as active in a month above this kWh


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_clean_files(selby_dir: str) -> dict[str, str]:
    files = glob.glob(os.path.join(selby_dir, "sparkmeterreadings_clean", "*.parquet"))
    files += glob.glob(os.path.join(selby_dir, "sparkmeterreadings_clean_*.parquet"))
    out = {}
    for f in files:
        base = os.path.basename(f).replace(".parquet", "")
        site = base.replace("sparkmeterreadings_clean_", "")
        out[site] = f
    return out


def map_site_to_project(site: str, projects: pd.DataFrame, stations: pd.DataFrame | None) -> str | None:
    n = norm(site)
    pnorm = {norm(p): p for p in projects["projectName"]}
    if n in pnorm:
        return pnorm[n]
    if stations is not None:
        for _, r in stations.iterrows():
            if n in (norm(r.get("meteringBaseStation", "")), norm(r.get("meteringSiteId", ""))):
                return r.get("projectName")
    for k, p in pnorm.items():                      # loose match (e.g. "kalobeyeisettlementvillage1a")
        if k and (k in n or n in k):
            return p
    return None


def aggregate_site(path: str) -> pd.DataFrame:
    """Stream one cleaned readings file and return month x meter_type x customer kWh."""
    acc: dict = defaultdict(float)
    pf = pq.ParquetFile(path)
    cols = [c for c in COLS if c in pf.schema.names]
    if set(COLS) - set(cols):
        raise ValueError(f"{path}: missing columns {set(COLS) - set(cols)}")
    for batch in pf.iter_batches(columns=cols, batch_size=2_000_000):
        b = batch.to_pandas()
        b["month"] = pd.to_datetime(b["slot_start"], utc=True).dt.strftime("%Y-%m")
        g = b.groupby(["month", "meter_type", "meter_customer_code"], dropna=False)["energy_kwh"].sum()
        for k, v in g.items():
            acc[k] += float(v)
    s = pd.Series(acc)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["month", "meter_type", "customer"])
    return s.rename("kwh").reset_index()


def site_monthly(cust_month: pd.DataFrame, res_codes: set | None) -> pd.DataFrame:
    cm = cust_month
    rows = []
    for month, grp in cm.groupby("month"):
        c = grp[grp["meter_type"].eq("customer")]
        active = c[c["kwh"] > ACTIVE_KWH]
        row = {"month": month,
               "active_customers": int(active["customer"].nunique()),
               "kwh_customer_meters": float(c["kwh"].sum()),
               "kwh_pue": float(grp.loc[grp["meter_type"].eq("pue"), "kwh"].sum()),
               "kwh_totalizer": float(grp.loc[grp["meter_type"].eq("totalizer"), "kwh"].sum())}
        if res_codes is not None:
            res = active["customer"].isin(res_codes)
            row["active_residential"] = int(res.sum())
            row["kwh_residential"] = float(active.loc[res, "kwh"].sum())
            row["kwh_non_residential"] = float(active.loc[~res, "kwh"].sum())
        rows.append(row)
    m = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    m["kwh_per_active_customer"] = m["kwh_customer_meters"] / m["active_customers"].replace(0, np.nan)
    return m


def yearly_summary(m: pd.DataFrame):
    """Return (summary dict, monthly frame from start) or None if the site never started."""
    start_idx = m.index[m["active_customers"] >= START_MIN_CUSTOMERS]
    if len(start_idx) == 0:
        return None
    m = m.loc[start_idx[0]:].reset_index(drop=True)
    m["months_since_start"] = np.arange(len(m))
    out = {"start_month": m["month"].iloc[0], "months_observed": int(len(m))}
    pc, cust = [], []
    for y in range(1, 6):
        lo, hi = SKIP_MONTHS + 12 * (y - 1), SKIP_MONTHS + 12 * y
        w = m[(m["months_since_start"] >= lo) & (m["months_since_start"] < hi)]
        full = len(w) >= 10
        pc.append(float(w["kwh_per_active_customer"].mean()) if full else np.nan)
        cust.append(float(w["active_customers"].mean()) if full else np.nan)
    out["kwh_per_customer_month_y1"] = pc[0]
    for i, v in enumerate(pc, 1):
        out[f"g{i}"] = v / pc[0] if pc[0] and pc[0] == pc[0] and v == v else np.nan
    max_c = np.nanmax(cust) if not all(np.isnan(cust)) else np.nan
    for i, v in enumerate(cust, 1):
        out[f"c{i}"] = v / max_c if max_c == max_c and max_c > 0 and v == v else np.nan
    out["active_customers_y1"] = cust[0]
    return out, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selby-dir", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--sites", nargs="*", help="optional subset of site names (normalised match)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    projects = pd.read_parquet(os.path.join(a.selby_dir, "minigridprojects.parquet"))
    st_path = os.path.join(a.selby_dir, "meteringbasestations.parquet")
    stations = pd.read_parquet(st_path) if os.path.exists(st_path) else None
    cu_path = os.path.join(a.selby_dir, "customers.parquet")
    customers = pd.read_parquet(cu_path) if os.path.exists(cu_path) else None

    files = find_clean_files(a.selby_dir)
    if a.sites:
        want = {norm(s) for s in a.sites}
        files = {k: v for k, v in files.items() if norm(k) in want or any(w in norm(k) for w in want)}
    if not files:
        raise SystemExit("No cleaned readings files found. Check --selby-dir and file names.")

    monthly_all, year1_rows = [], []
    for site, path in sorted(files.items()):
        proj = map_site_to_project(site, projects, stations)
        print(f"[{site}] -> project {proj!r}: aggregating {os.path.getsize(path) / 1e6:,.0f} MB ...", flush=True)
        cm = aggregate_site(path)

        res_codes = None
        if customers is not None and proj is not None and "customerType" in customers.columns:
            c = customers[customers["projectName"] == proj]
            codes = set(cm.loc[cm["meter_type"].eq("customer"), "customer"].astype(str))
            joined = c["customerAccountNumber"].astype(str).isin(codes)
            if len(codes) and joined.sum() / len(codes) >= 0.5:
                res_codes = set(c.loc[joined & c["customerType"].astype(str).str.lower().eq("residential"),
                                      "customerAccountNumber"].astype(str))
            else:
                print(f"   customer-type join matched {joined.sum()}/{len(codes)} meters; skipping type split")

        m = site_monthly(cm, res_codes)
        res = yearly_summary(m)
        if res is None:
            print("   no month reached the active-customer threshold; skipped")
            continue
        summary, m = res
        m.insert(0, "name", proj or site)
        monthly_all.append(m)
        summary.update({"name": proj or site, "file_site": site})
        year1_rows.append(summary)
        print(f"   start {summary['start_month']}, y1 = {summary['kwh_per_customer_month_y1']:.1f} kWh/customer/month")

    monthly = pd.concat(monthly_all, ignore_index=True)
    year1 = pd.DataFrame(year1_rows)

    keep = [c for c in ["projectName", "country", "lat", "long", "sizePv", "sizeBattery", "typeBattery", "capex",
                        "currency", "powerOnDate", "powerSoldDate", "peakLoadCapacity", "dieselPowerOutputkva"]
            if c in projects.columns]
    sites = projects[keep].rename(columns={"projectName": "name", "long": "lon"})
    sites = sites.merge(year1[["name", "start_month"]], on="name", how="inner")
    sites["ref_year"] = sites["start_month"].str[:4].astype(int) - 1      # features as of the year before start
    year1 = year1.merge(sites[["name", "country"]], on="name", how="left")

    sites.to_parquet(os.path.join(a.out, "selby_sites.parquet"), index=False)
    monthly.to_parquet(os.path.join(a.out, "selby_monthly.parquet"), index=False)
    year1.to_parquet(os.path.join(a.out, "selby_site_year1.parquet"), index=False)

    ramp_cols = [f"c{i}" for i in range(1, 6)]
    ramp = [float(np.nanmedian(year1[c])) if year1[c].notna().any() else None for c in ramp_cols]
    ramp = [r for r in ramp if r is not None]
    with open(os.path.join(a.out, "selby_ramp.json"), "w") as f:
        json.dump({"ramp": ramp, "n_sites": int(len(year1)),
                   "note": "Median share of eventual active customers in years 1..5 (Selby sites)."}, f, indent=2)
    print(f"\nWrote {len(sites)} sites, {len(monthly)} site-months to {a.out}/")


if __name__ == "__main__":
    main()
