"""
model.py -- the shared base forecaster.

One small, fixed, untuned quantile-LightGBM (median) used by every driver. The paper's
claim is calibration, not accuracy, so this configuration is identical everywhere and is
never tuned; only the conformal calibration layer varies across experiments. Centralising
it here keeps the 9 drivers from drifting apart.
"""
from __future__ import annotations
from lightgbm import LGBMRegressor


def make_base_model(seed, quantile: float = 0.5):
    """Quantile LightGBM, 400 trees, untuned, deterministic given seed. quantile=0.5 is the median
    point forecaster used everywhere (default -> identical to before); the CQR contrast
    (run_cqr.py) uses 0.05 and 0.95 for a central 90% interval, holding every other setting fixed."""
    return LGBMRegressor(objective="quantile", alpha=quantile, n_estimators=400,
                         learning_rate=0.05, num_leaves=31, min_child_samples=40,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         random_state=seed, n_jobs=-1, verbosity=-1)
