# Data pipeline: turning demo mode into real data

`app.py` is already wired to the engine. The app runs in demo mode until these scripts write files into `data/`.
Upload the small files they create in `data/` to GitHub; the website switches to real data automatically.

Extra packages for the pipeline (on your laptop only): `pip install ruptures earthengine-api`

## 2. Run the pipeline (in this order)

```bash
# Download from Zenodo (DOI 10.5281/zenodo.22149102): minigridprojects, customers,
# meteringbasestations, and 5-10 of the SMALLER cleaned reading files.
python pipeline/01_selby_monthly.py --selby-dir ~/selby --out data

# Real features for the Selby sites, as of the year before each site started (no leakage),
# plus their monthly night-light series for the light-vs-kWh test.
python pipeline/02_site_features.py --points data/selby_sites.parquet --project YOUR_GCP_PROJECT \
       --out data/selby_features.parquet --series data/selby_viirs_series.parquet

# Features for the app's demo / replay points (latest year). A CSV with name, lat, lon, country.
python pipeline/02_site_features.py --points demo_points.csv --project YOUR_GCP_PROJECT \
       --out data/site_features.parquet

# Growth library (Selby meter paths; add night-light cells later with --series/--cells)
python pipeline/03_switchons.py --out data

# Validation on real meters -> data/validation.json (+ level_model.joblib if it wins)
python pipeline/04_validate.py --data data --series data/selby_viirs_series.parquet
```

Run step 02 on three points first and compare `building_count` (temporal) with
`building_count_v3` before trusting it. The Earth Engine code was not run in the authoring environment.

## 3. Claims already fixed in this version

| File | Line(s) | Change |
|---|---|---|
| app.py | 3, 351, 371-376, 677 | Remove "calibrated on 471M records" until pipeline/04 has run; then say "validated on N Selby sites (Zenodo DOI 10.5281/zenodo.22149102)" |
| app.py | 594-603 | Show `res["validation"]`; delete "Zero spatial leakage"; axis label = metered year-1 kWh per customer |
| pdf_generator.py | 95, 170-172, 195, 213 | Same; disclaimer: "screening tool, not an engineering design or investment advice" |
| README.md | 10, 24, 30-36 | Remove "700% error", "$6,000-15,000 saved", the hard-coded MAPE and R²; cite the dataset |

## 4. Where assumptions still live

Everything not yet measured is in `config/benchmarks.json`, each with a `source` and `status`.
Replace before submission where you can:
- **costs:** use `minigridprojects.capex` and `sizePv` from Selby
- **tariffs:** use the Selby `tariffs` table converted to USD
- **uptake:** active customers ÷ building count at Selby sites

The engine reports which values were assumed in `res["labels"]["assumed"]`.
