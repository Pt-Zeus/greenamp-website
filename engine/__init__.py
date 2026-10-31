"""GreenAmp forecast engine. Import the single entry point:

    from engine import run_forecast
    res = run_forecast(lat, lon, target_year=2030)
"""
from .forecast import run_forecast  # noqa: F401
