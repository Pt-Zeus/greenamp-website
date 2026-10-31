"""Growth paths learned from places that already switched on.

growth_library.parquet (pipeline/03) has one row per switched-on place:
  id, name, source ('meter' = Selby smart-meter kWh per customer,
                    'nightlight' = VIIRS switch-on cell), country, lat, lon,
  the MATCH_FEATURES (as of the year before switch-on), and g1..g5
  (yearly index, g1 = 1.0; NaN where not yet observed).

Matching uses only MATCH_FEATURES, standardised over the library, ignoring
dimensions that are missing for either side. Paths are used unadjusted.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import store
from .features import MATCH_FEATURES

YEARS = 5
GCOLS = [f"g{i}" for i in range(1, YEARS + 1)]


def _fill_paths(paths: np.ndarray) -> np.ndarray:
    """Hold the last observed value flat for unobserved years (conservative)."""
    out = paths.copy()
    for r in range(out.shape[0]):
        last = 1.0
        for c in range(out.shape[1]):
            if np.isnan(out[r, c]):
                out[r, c] = last
            else:
                last = out[r, c]
    return out


def fallback_paths(n: int = 7) -> tuple[np.ndarray, list[dict]]:
    """Assumed S-curve with a spread, used only when no library is available."""
    fb = store.benchmarks()["growth_fallback"]
    base = np.array(fb["per_year_index"][:YEARS], dtype=float)
    spread = fb.get("spread", 0.15)
    mult = np.linspace(1 - spread, 1 + spread, n)
    paths = np.vstack([1 + (base - 1) * m for m in mult])
    return paths, []


def match(features: dict, k: int = 20, country: str | None = None,
          exclude_near: tuple[float, float] | None = None, exclude_km: float = 2.0) -> dict:
    """Return matched growth paths, weights and a description of the matches.

    exclude_near=(lat, lon) drops library rows within exclude_km of the query, so a replay of a
    known site never learns from its own data (no leakage).
    """
    lib = store.growth_library()
    n_excluded = 0
    if lib is not None and exclude_near is not None and {"lat", "lon"} <= set(lib.columns):
        d = store.haversine_km(exclude_near[0], exclude_near[1], lib["lat"].to_numpy(), lib["lon"].to_numpy())
        keep = ~(d <= exclude_km)
        n_excluded = int((~keep).sum())
        lib = lib.loc[keep]
    if lib is None or not any(c in lib.columns for c in MATCH_FEATURES):
        paths, _ = fallback_paths()
        return {"paths": paths, "weights": np.full(len(paths), 1 / len(paths)),
                "matches": [], "source": "assumed", "n_good": 0, "dims_used": [], "n_excluded": n_excluded}

    dims = [c for c in MATCH_FEATURES if c in lib.columns
            and not math.isnan(features.get(c, math.nan)) and lib[c].notna().sum() >= 3]
    lib = lib.copy()
    if dims:
        mu = lib[dims].mean()
        sd = lib[dims].std(ddof=0).replace(0, 1.0)
        z_lib = (lib[dims] - mu) / sd
        z_q = (pd.Series({d: features[d] for d in dims}) - mu) / sd
        sq = (z_lib - z_q) ** 2
        lib["dist"] = np.sqrt(sq.mean(axis=1, skipna=True))
        lib.loc[sq.notna().sum(axis=1) == 0, "dist"] = np.nan
    else:
        lib["dist"] = np.nan

    if lib["dist"].isna().all():
        # No comparable features: use the whole library equally, flagged as unmatched.
        lib["dist"] = 2.0
    lib = lib.dropna(subset=["dist"]).nsmallest(k, "dist")

    w = 1.0 / (lib["dist"].to_numpy() + 0.25)
    if country is not None and "country" in lib.columns:
        w *= np.where(lib["country"].astype(str).str.lower() == str(country).lower(), 1.5, 1.0)
    if "source" in lib.columns:
        w *= np.where(lib["source"] == "meter", 2.0, 1.0)  # measured kWh beats light proxy
    w = w / w.sum()

    gcols = [c for c in GCOLS if c in lib.columns]
    paths = _fill_paths(lib[gcols].to_numpy(dtype=float))
    if paths.shape[1] < YEARS:
        paths = np.hstack([paths, np.repeat(paths[:, -1:], YEARS - paths.shape[1], axis=1)])

    matches = []
    for (_, r), wi in zip(lib.iterrows(), w):
        matches.append({
            "name": r.get("name", r.get("id")),
            "source": r.get("source", "unknown"),
            "country": r.get("country"),
            "distance": round(float(r["dist"]), 2),
            "weight": round(float(wi), 3),
            "growth_year3": None if pd.isna(r.get("g3")) else round(float(r["g3"]), 2),
            "years_observed": int(r[gcols].notna().sum()),
        })
    matches.sort(key=lambda m: -m["weight"])
    n_good = int((lib["dist"] < 1.0).sum())
    src = "meter+nightlight" if set(lib.get("source", [])) >= {"meter", "nightlight"} else \
        (lib["source"].iloc[0] if "source" in lib.columns and len(lib) else "library")
    return {"paths": paths, "weights": w, "matches": matches, "source": src,
            "n_good": n_good, "dims_used": dims, "n_excluded": n_excluded}


def drivers(features: dict, matches_dims: list[str]) -> list[str]:
    """Plain-language Context-panel lines comparing the site with the library."""
    lib = store.growth_library()
    if lib is None:
        return []
    labels = {
        "building_count": ("buildings", "{:.0f}"),
        "large_building_share": ("share of large buildings", "{:.0%}"),
        "nightlight": ("night-light level", "{:.2f}"),
        "travel_time_min": ("travel time to a city", "{:.0f} min"),
        "dist_grid_km": ("distance to grid", "{:.0f} km"),
        "market_km": ("distance to a market", "{:.1f} km"),
    }
    lines = []
    for d in matches_dims:
        if d not in lib.columns:
            continue
        pct = float((lib[d] < features[d]).mean())
        name, fmt = labels[d]
        side = "higher than" if pct >= 0.66 else "lower than" if pct <= 0.33 else "similar to"
        lines.append(f"{name.capitalize()}: {fmt.format(features[d])} ({side} most places in the library)")
    return lines
