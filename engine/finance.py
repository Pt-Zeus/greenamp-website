"""Break-even demand, probability of shortfall, staged-build saving, affordability."""
from __future__ import annotations

import numpy as np

from . import store


def annuity(principal: float, rate: float, years: int) -> float:
    if principal <= 0:
        return 0.0
    if rate == 0:
        return principal / years
    return principal * rate / (1 - (1 + rate) ** -years)


def capex(system: dict, connections: float, grant_per_connection: float) -> float:
    c = store.benchmarks()["costs_usd"]
    gross = (system["pv_kwp"] * c["pv_per_kwp"]
             + system["battery_kwh"] * c["battery_per_kwh"]
             + connections * c["distribution_per_connection"])
    return max(0.0, gross - grant_per_connection * connections)


def break_even_kwh_month(capex_usd: float, tariff: float) -> dict:
    """Site demand (kWh/month) at which tariff revenue covers operating cost + debt service."""
    c = store.benchmarks()["costs_usd"]
    opex = c["opex_share_of_capex_per_year"] * capex_usd
    debt = annuity(capex_usd * c["debt_share_of_capex"], c["interest_rate"], c["loan_years"])
    be = (opex + debt) / (12.0 * tariff) if tariff > 0 else float("inf")
    return {"break_even_kwh_month": round(be, 0), "opex_usd_year": round(opex, 0),
            "debt_service_usd_year": round(debt, 0)}


def shortfall_probability(site_draws: np.ndarray, be: float) -> list[float]:
    """Share of futures in which demand is below break-even, per year."""
    return [round(float((site_draws[:, y] < be).mean()), 3) for y in range(site_draws.shape[1])]


def affordability(kwh_per_customer_p50: float, tariff: float) -> dict:
    f = store.benchmarks()["fuel_spend_usd_month"]
    bill = kwh_per_customer_p50 * tariff
    if bill <= f["low"]:
        verdict = "Below what households typically spend on candles, kerosene and batteries today"
    elif bill <= f["high"]:
        verdict = "Within the range households typically spend on energy today"
    else:
        verdict = "Above typical current energy spending: risk of low uptake"
    return {"bill_usd_month": round(bill, 2), "current_spend_range": [f["low"], f["high"]],
            "verdict": verdict, "source": f["source"]}
