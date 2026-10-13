"""
test_leakage.py -- guardrail tests for the depth-conditional conformal pipeline.

Two properties the calibration claim rests on:
  1. NO TARGET LEAKAGE: every feature at time t uses only data strictly before t,
     and the target y is the temperature h days AFTER t.
  2. CONFORMAL COVERAGE SANITY: on exchangeable synthetic data, split-conformal
     empirical coverage matches the nominal 1 - alpha level.

Run:  python src/test_leakage.py   (prints PASS lines; raises on failure)
Also importable as pytest tests (the test_* functions).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from features import build_features, FEATURE_COLS, LAGS, ROLL_WINDOW
from conformal import conformal_quantile, coverage_and_width


def test_no_target_leakage():
    """A strictly increasing, gap-free synthetic series makes any leak detectable:
    temp_c on day index d equals d, so lag_k must be d-k and the target must be d+h."""
    h = 30
    dates = pd.date_range("2000-01-01", periods=900, freq="D")
    long_df = pd.DataFrame({"site_id": "T", "date": dates, "depth_m": 2.0,
                            "temp_c": np.arange(len(dates), dtype=float)})
    feat = build_features(long_df, horizon=h)
    assert len(feat) > 0, "no feature rows built"
    idx = (feat["date"] - dates[0]).dt.days.values.astype(float)  # day index of each row

    # target is h days ahead
    assert np.allclose(feat["y"].values, idx + h), "y must equal temp at t+h"
    # each lag is strictly in the past
    for L in LAGS:
        assert np.allclose(feat[f"lag_{L}"].values, idx - L), f"lag_{L} must equal temp at t-{L}"
    # rolling mean uses shift(1): mean of days [t-ROLL_WINDOW .. t-1] = idx - (ROLL_WINDOW+1)/2
    exp_mean = idx - (ROLL_WINDOW + 1) / 2.0
    assert np.allclose(feat["roll_mean_30"].values, exp_mean), "roll_mean_30 must exclude the current day"
    # no temp-derived feature may reach the current/future value (idx)
    for col in [f"lag_{L}" for L in LAGS] + ["roll_mean_30"]:
        assert (feat[col].values < idx).all(), f"{col} leaks current/future information"
    print(f"  [PASS] no target leakage ({len(feat)} rows; lags/rolling strictly past, y at t+{h})")


def test_conformal_coverage_sanity():
    """On i.i.d. (exchangeable) residuals, |resid| split conformal gives ~1-alpha coverage."""
    rng = np.random.default_rng(0)
    alpha = 0.1
    cal = np.abs(rng.standard_normal(5000))
    test = np.abs(rng.standard_normal(5000))
    Q = conformal_quantile(cal, alpha)
    cov = float(np.mean(test <= Q))
    assert abs(cov - (1 - alpha)) < 0.02, f"coverage {cov:.3f} far from nominal {1 - alpha}"

    # coverage_and_width agrees on a symmetric interval around a zero forecast
    y = rng.standard_normal(5000)
    yp = np.zeros_like(y)
    Q2 = conformal_quantile(np.abs(rng.standard_normal(5000)), alpha)
    c, w = coverage_and_width(y, yp, Q2)
    assert abs(c - (1 - alpha)) < 0.03, f"coverage_and_width {c:.3f} off nominal"
    assert w > 0, "interval width must be positive"
    print(f"  [PASS] conformal coverage sanity (empirical {cov:.3f} ~ nominal {1 - alpha})")


def test_conformal_quantile_empty():
    """Degenerate inputs do not crash: empty calibration -> nan quantile."""
    assert np.isnan(conformal_quantile(np.array([]), 0.1))
    print("  [PASS] empty-calibration edge case")


if __name__ == "__main__":
    print("Running pipeline guardrail tests ...")
    test_no_target_leakage()
    test_conformal_coverage_sanity()
    test_conformal_quantile_empty()
    print("All tests passed.")
