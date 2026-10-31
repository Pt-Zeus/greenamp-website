"""Paths and cached loaders for the files the pipeline produces.

Set GREENAMP_DATA to point at another data folder; default is ./data next to app.py.
Every loader returns None (not an error) when a file is missing, so the app still runs
and the honesty labels say which inputs fell back to assumptions.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GREENAMP_DATA", ROOT / "data"))
CONFIG_DIR = ROOT / "config"


def _read_parquet(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / name
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df if len(df) else None


@lru_cache(maxsize=1)
def benchmarks() -> dict:
    with open(CONFIG_DIR / "benchmarks.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def site_features() -> pd.DataFrame | None:
    """Features for named demo/replay sites (pipeline/02 output)."""
    return _read_parquet("site_features.parquet")


@lru_cache(maxsize=256)
def tile(lat_floor: int, lon_floor: int) -> pd.DataFrame | None:
    """Optional 1x1 degree tiles of candidate cells: data/tiles/<lat>_<lon>.parquet."""
    return _read_parquet(f"tiles/{lat_floor}_{lon_floor}.parquet")


@lru_cache(maxsize=1)
def growth_library() -> pd.DataFrame | None:
    return _read_parquet("growth_library.parquet")


@lru_cache(maxsize=1)
def selby_sites() -> pd.DataFrame | None:
    return _read_parquet("selby_sites.parquet")


@lru_cache(maxsize=1)
def selby_site_year1() -> pd.DataFrame | None:
    return _read_parquet("selby_site_year1.parquet")


@lru_cache(maxsize=1)
def uptake_ramp() -> list[float] | None:
    path = DATA_DIR / "selby_ramp.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        ramp = json.load(f).get("ramp")
    return ramp or None


@lru_cache(maxsize=1)
def validation() -> dict | None:
    path = DATA_DIR / "validation.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def level_model():
    path = DATA_DIR / "level_model.joblib"
    if not path.exists():
        return None
    import joblib
    return joblib.load(path)


def clear_caches() -> None:
    for fn in (benchmarks, site_features, tile, growth_library, selby_sites,
               selby_site_year1, uptake_ramp, validation, level_model):
        fn.cache_clear()


def haversine_km(lat1, lon1, lat2, lon2):
    """Works with scalars or pandas Series / numpy arrays for the second point."""
    import numpy as np
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def floor_tile(lat: float, lon: float) -> tuple[int, int]:
    return math.floor(lat), math.floor(lon)
