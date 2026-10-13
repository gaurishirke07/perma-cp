"""
features.py — build forecasting features for the depth-conditional conformal seed.

Reads the canonical long table (data/borehole_long.parquet: site_id|date|depth_m|temp_c)
and produces one tabular row per (depth, date) with lagged-target, seasonality, depth,
trend and rolling features, plus the h-step-ahead target y = temp_c at t+h.

Per 02_IMPLEMENTATION_SPEC.md §3. Lags are computed on a per-depth DAILY-reindexed
series so calendar gaps (~17% missing at this site) don't corrupt the offsets.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

HORIZON_DEFAULT = 30           # forecast lead time in days (h)
LAGS = [1, 7, 30, 365]         # same-depth lagged target features
ROLL_WINDOW = 30               # rolling mean/std window (days)


def depth_band(depth: float) -> str:
    """Mondrian calibration groups. Coarse enough to hold calibration points per band."""
    if depth < 1:
        return "shallow(<1m)"
    elif depth < 5:
        return "mid(1-5m)"
    elif depth < 20:
        return "deep_seasonal(5-20m)"
    else:
        return "deep_trend(>=20m)"


def build_features(long_df: pd.DataFrame, horizon: int = HORIZON_DEFAULT,
                   require_lags: bool = True) -> pd.DataFrame:
    """Long table -> feature table. One (site,depth) series at a time on a daily index."""
    frames = []
    for (site, depth), g in long_df.groupby(["site_id", "depth_m"], sort=True):
        g = g.sort_values("date")
        # daily-reindex so lag/lead offsets are true calendar days
        s = g.set_index("date")["temp_c"]
        s = s[~s.index.duplicated(keep="first")]
        full = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s = s.reindex(full)
        df = pd.DataFrame({"temp_c": s})
        df["site_id"] = site
        df["depth_m"] = depth
        df["band"] = depth_band(depth)
        # lagged target (same depth)
        for L in LAGS:
            df[f"lag_{L}"] = s.shift(L)
        # rolling stats over the recent window (shifted so only past is used)
        df["roll_mean_30"] = s.shift(1).rolling(ROLL_WINDOW).mean()
        df["roll_std_30"] = s.shift(1).rolling(ROLL_WINDOW).std()
        # seasonality
        doy = df.index.dayofyear.values
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
        # trend / time index
        df["year"] = df.index.year
        df["time_idx"] = (df.index - df.index.min()).days
        # target: temp h days ahead
        df["y"] = s.shift(-horizon)
        df["date"] = df.index
        frames.append(df.reset_index(drop=True))

    feat = pd.concat(frames, ignore_index=True)
    # keep rows with target + all lag features present (the default, paper behaviour).
    # require_lags=False keeps every row with a target and lets the model see NaN lags
    # (used only by the missing-data robustness check, run_missing_robustness.py).
    if require_lags:
        need = ["y"] + [f"lag_{L}" for L in LAGS] + ["roll_mean_30", "roll_std_30"]
    else:
        need = ["y"]
    feat = feat.dropna(subset=need).reset_index(drop=True)
    return feat


FEATURE_COLS = (
    [f"lag_{L}" for L in LAGS]
    + ["roll_mean_30", "roll_std_30", "sin_doy", "cos_doy", "depth_m", "year", "time_idx"]
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    long_df = pd.read_parquet(root / "data" / "borehole_long.parquet")
    feat = build_features(long_df)
    print(f"feature rows: {len(feat):,}")
    print(f"depths: {feat.depth_m.nunique()} | date range {feat.date.min()} -> {feat.date.max()}")
    print(feat.band.value_counts())
    print("feature cols:", FEATURE_COLS)
