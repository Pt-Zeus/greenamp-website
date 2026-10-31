"""Quantile level model (moved from the original model_engine.py).

It is trained ONLY on real meter labels (pipeline/04 on Selby sites). With roughly
20-30 sites it is deliberately small: shallow trees, few estimators, few features.
Until it is trained, forecast.py uses the benchmark / Selby-country prior instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

DEFAULT_FEATURES = ["building_count", "large_building_share", "nightlight",
                    "travel_time_min", "dist_grid_km", "rwi"]


class QuantileLevelModel:
    """Predicts year-1 kWh per customer per month at the 10th, 50th and 90th percentiles."""

    def __init__(self, features=None, n_estimators=60, max_depth=2, random_state=42):
        self.features = list(features or DEFAULT_FEATURES)
        kw = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
                  min_samples_leaf=3, random_state=random_state)
        self.models = {a: GradientBoostingRegressor(loss="quantile", alpha=a, **kw)
                       for a in (0.10, 0.50, 0.90)}
        self.medians_: dict = {}

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.features].astype(float).copy()
        return x.fillna(pd.Series(self.medians_))

    def fit(self, df: pd.DataFrame, y) -> "QuantileLevelModel":
        self.medians_ = df[self.features].astype(float).median().to_dict()
        x = self._x(df)
        ylog = np.log(np.asarray(y, dtype=float))
        for m in self.models.values():
            m.fit(x, ylog)
        return self

    def predict_one(self, feats: dict) -> tuple[float, float, float]:
        row = pd.DataFrame([{f: feats.get(f, np.nan) for f in self.features}])
        x = self._x(row)
        q10, q50, q90 = (float(np.exp(self.models[a].predict(x)[0])) for a in (0.10, 0.50, 0.90))
        lo, mid, hi = sorted([q10, q50, q90])
        return lo, mid, hi
