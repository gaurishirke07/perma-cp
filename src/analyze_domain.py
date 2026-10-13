"""
analyze_domain.py — per-depth-band POINT accuracy of our base forecaster across all 25
qualifying boreholes, so our calibration result can be compared honestly against published
permafrost borehole methods (which report RMSE, not coverage).

Output: results/accuracy_by_band.csv  (+ console table vs literature anchors)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from model import make_base_model

from features import build_features, FEATURE_COLS
from run_seed import time_split, SEED, HORIZON

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b): return float(np.mean(np.abs(a - b)))


def main():
    long_df = pd.read_parquet(ROOT / "data" / "borehole_all_long.parquet")
    rows = []
    for site, sdf in long_df.groupby("site_id"):
        feat = build_features(sdf, horizon=HORIZON)
        train, cal, test, *_ = time_split(feat)
        if len(train) < 500 or len(test) < 200:
            continue
        m = make_base_model(SEED)
        m.fit(train[FEATURE_COLS], train["y"])
        test = test.assign(pred=m.predict(test[FEATURE_COLS]))
        for b, g in test.groupby("band"):
            rows.append({"site_id": site, "band": b, "rmse": rmse(g.y, g.pred),
                         "mae": mae(g.y, g.pred), "n": len(g)})
    acc = pd.DataFrame(rows)
    acc.to_csv(RES / "accuracy_by_band.csv", index=False)

    # aggregate across sites (per-site RMSE, then mean/median over sites)
    agg = (acc.groupby("band")
              .agg(rmse_mean=("rmse", "mean"), rmse_median=("rmse", "median"),
                   rmse_sd=("rmse", "std"), mae_mean=("mae", "mean"), n_sites=("site_id", "nunique"))
              .reindex(BANDS))
    print("=== our per-band POINT accuracy across all boreholes (h=30d), degC ===")
    print(agg.round(3).to_string())
    print("\nLiterature anchors (verified):")
    print("  Chance 2024 (Earth Sci Inf 17:6055): GB RMSE 0.13 (permafrost/deep), RF 1.10 (active layer/shallow)")
    print("  PI-LSTM Liu 2023 (Cold Reg Sci Tech 216): borehole ground temp, point-only, no coverage reported")


if __name__ == "__main__":
    main()
