"""
run_baselines.py -- B5: naive forecast baselines for base-model context (ACCURACY only).

Persistence (last value = lag_1) and seasonal-naive (value one year ago = lag_365) per-band
RMSE vs the LightGBM base model, across 25 boreholes on the same split. Confirms the base
forecaster adds value over trivial baselines. This is an accuracy sanity check, NOT calibration;
LightGBM RMSE here must match analyze_domain. Output: results/baselines.csv (+ console).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from features import build_features, FEATURE_COLS
from model import make_base_model
from run_seed import time_split, SEED, HORIZON

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def run_one(sdf):
    feat = build_features(sdf, horizon=HORIZON)
    tr, cal, te, *_ = time_split(feat)
    if len(tr) < 500 or len(te) < 200:
        return None
    m = make_base_model(SEED)
    m.fit(tr[FEATURE_COLS], tr["y"])
    te = te.assign(pred_lgbm=m.predict(te[FEATURE_COLS]))
    rows = []
    for b, g in te.groupby("band"):
        rows.append({"band": b,
                     "rmse_lgbm": rmse(g.y, g.pred_lgbm),
                     "rmse_persistence": rmse(g.y, g["lag_1"]),        # last value
                     "rmse_seasonal_naive": rmse(g.y, g["lag_365"]),   # value ~1 yr ago
                     "n": len(g)})
    return pd.DataFrame(rows)


def main():
    long_df = pd.read_parquet(ROOT / "data" / "borehole_all_long.parquet")
    rows = []
    for site, sdf in long_df.groupby("site_id"):
        r = run_one(sdf)
        if r is None:
            continue
        r["site_id"] = site; rows.append(r); print(f"  {site:10} done")
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(RES / "baselines.csv", index=False)
    agg = res.groupby("band")[["rmse_lgbm", "rmse_persistence", "rmse_seasonal_naive"]].mean().reindex(BANDS)
    pd.set_option("display.width", 200)
    print("\n=== per-band RMSE (degC): LightGBM vs naive baselines, across 25 boreholes ===")
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
