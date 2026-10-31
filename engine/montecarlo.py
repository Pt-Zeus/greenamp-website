"""Monte Carlo demand range.

Each draw samples ONE starting level, ONE uptake share and ONE matched growth path,
and keeps them for all five years, so the yearly ranges stay consistent.

Lender convention: P90 = value exceeded in 90% of draws (the LOW case),
P50 = median, P10 = exceeded in only 10% of draws (the HIGH case).
"""
from __future__ import annotations

import numpy as np


def lender_quantiles(x: np.ndarray) -> dict:
    return {"P90": float(np.percentile(x, 10)),
            "P50": float(np.percentile(x, 50)),
            "P10": float(np.percentile(x, 90))}


def simulate(level: tuple[float, float, float],
             uptake: tuple[float, float],
             buildings: float,
             paths: np.ndarray,
             weights: np.ndarray,
             ramp: list[float] | None = None,
             n: int = 5000,
             seed: int = 42) -> dict:
    """
    level   : (low, mode, high) kWh per customer per month in year 1
    uptake  : (low, high) final share of buildings that connect
    paths   : (m, 5) growth index per year, year 1 = 1.0 (kWh per customer)
    ramp    : optional share of final connections active in years 1..5
    Returns draws (n x 5) of SITE kWh/month and per-customer draws, plus yearly quantiles.
    """
    rng = np.random.default_rng(seed)
    lo, mode, hi = level
    mode = min(max(mode, lo), hi)
    lvl = rng.triangular(lo, mode, hi, n) if hi > lo else np.full(n, mode)
    upt = rng.uniform(uptake[0], uptake[1], n)
    idx = rng.choice(len(paths), size=n, p=weights)
    g = paths[idx, :]                                         # n x 5
    years = g.shape[1]
    r = np.array((ramp or [1.0] * years)[:years], dtype=float)
    if len(r) < years:
        r = np.concatenate([r, np.full(years - len(r), r[-1])])

    per_customer = lvl[:, None] * g                           # kWh/customer/month
    customers = buildings * upt[:, None] * r[None, :]         # connected customers
    site = per_customer * customers                           # kWh/month

    yearly = [lender_quantiles(site[:, y]) for y in range(years)]
    yearly_pc = [lender_quantiles(per_customer[:, y]) for y in range(years)]
    cust = [lender_quantiles(customers[:, y]) for y in range(years)]
    return {"site_draws": site, "per_customer_draws": per_customer,
            "customers_draws": customers, "yearly": yearly,
            "yearly_per_customer": yearly_pc, "yearly_customers": cust}


def extrapolate(draws_last_year: np.ndarray, extra_years: int, seed: int = 7) -> np.ndarray:
    """Beyond year 5: hold the year-5 draws and widen them (2-sided, ~10% per extra year)."""
    rng = np.random.default_rng(seed)
    sigma = 0.10 * np.sqrt(max(extra_years, 0))
    return draws_last_year * rng.lognormal(0.0, sigma, size=draws_last_year.shape)
