# GreenAmp Energy Solutions ⚡
### Pre-feasibility demand forecasting for mini-grids

Enter a village's location and a target year. GreenAmp returns that village's electricity demand range for that year (P90 / P50 / P10), the chance the project falls below break-even, a staged solar + battery sizing plan, and a one-page Lender Card.

It is a **screening tool**, not an engineering design or investment advice.

## How it works

1. **Site features** from open satellite and map data for the village's 1 km area: buildings, share of large buildings, night lights, travel time to a city, distance to the grid, market access.
2. **Matching:** the forecast learns growth from places that already switched on and most resembled this village *before* electrification. These are real smart-meter mini-grids (Selby / Renewvia dataset) and night-light switch-ons.
3. **Starting level:** year-1 kWh per customer from metered sites, placed by relative wealth (published benchmarks in demo mode).
4. **Monte Carlo:** 5,000 futures combine starting level × connection rate × growth path.
5. **Finance and sizing:** break-even demand per sizing option, chance of shortfall, staged-build saving and affordability. Solar resource comes from NASA POWER.

**Reading the numbers:** P90 is exceeded in 9 of 10 futures (the low case lenders use). P50 is the median. P10 is the high case.

## Status

The app runs in **demo mode** until the data pipeline has filled `data/` (see `DATA_PIPELINE.md`). In demo mode:
- site features are entered by hand
- the starting level uses published benchmarks
- growth uses an assumed curve

Every result lists what is calibrated, what is assumed and what is not yet proven.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

| Path | What it is |
|---|---|
| `app.py` | Website (Streamlit) |
| `engine/` | Forecast engine: `run_forecast(lat, lon, target_year)` |
| `pdf_generator.py` | Lender Card PDF |
| `config/benchmarks.json` | Every assumption, with its source |
| `pipeline/` | Scripts that build real data into `data/` |
| `data/` | Pipeline outputs (empty = demo mode) |

## Data and credits

- Selby / Renewvia Energy (2026). *High-Frequency, Multi-Domain Operational Dataset for Community Solar Mini-Grids in Kenya and Nigeria.* Zenodo. DOI 10.5281/zenodo.22149102. CC BY-SA 4.0. Derived data keeps this licence.
- Google Open Buildings · NOAA VIIRS · Oxford/MAP accessibility · OpenStreetMap · NASA POWER · AMDA benchmarking reports.
- Earth Engine is used under its free **non-commercial** tier. A commercial launch would use direct data downloads or commercial terms.
