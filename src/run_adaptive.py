"""
run_adaptive.py — METHOD A: depth-adaptive conformal across all 25 boreholes.

Compares three calibrators per depth band, on the test set:
  global  |  Mondrian(depth)  |  Mondrian+ACI (depth-adaptive, the proposed method)
Question: does adding ACI drift-adaptation lift the stuck deep-trend band toward nominal?

Output: results/coverage_adaptive_bands.csv (figure drawn by make_figures.py)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from model import make_base_model

from features import build_features, FEATURE_COLS
from conformal import global_conformal, mondrian_conformal, mondrian_aci
from run_seed import time_split, SEED, ALPHA, HORIZON, NOMINAL

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]


def run_one(sdf):
    feat = build_features(sdf, horizon=HORIZON)
    train, cal, test, *_ = time_split(feat)
    if len(train) < 500 or len(cal) < 200 or len(test) < 200:
        return None
    m = make_base_model(SEED)
    m.fit(train[FEATURE_COLS], train["y"])
    cal = cal.assign(y_pred=m.predict(cal[FEATURE_COLS]))
    test = test.assign(y_pred=m.predict(test[FEATURE_COLS]))
    cal["abs_resid"] = (cal["y"] - cal["y_pred"]).abs()
    a = global_conformal(cal, test, ALPHA); a["variant"] = "global"
    b = mondrian_conformal(cal, test, ALPHA); b["variant"] = "mondrian"
    c = mondrian_aci(cal, test, ALPHA); c["variant"] = "mondrian+ACI"
    return pd.concat([a, b, c], ignore_index=True)


def main():
    long_df = pd.read_parquet(ROOT / "data" / "borehole_all_long.parquet")
    rows = []
    for site, sdf in long_df.groupby("site_id"):
        r = run_one(sdf)
        if r is None:
            print(f"  {site:10} skip"); continue
        r["site_id"] = site; rows.append(r); print(f"  {site:10} done")
    res = pd.concat(rows, ignore_index=True)

    def band_cov(df):
        return (df.groupby(["site_id", "variant", "band"])
                  .apply(lambda x: np.average(x.coverage, weights=x.n_test), include_groups=False)
                  .rename("coverage").reset_index())
    def band_wid(df):
        return (df.groupby(["site_id", "variant", "band"])
                  .apply(lambda x: np.average(x.mean_width, weights=x.n_test), include_groups=False)
                  .rename("width").reset_index())
    bc = band_cov(res); bw = band_wid(res)
    bc.merge(bw, on=["site_id", "variant", "band"]).to_csv(RES / "coverage_adaptive_bands.csv", index=False)

    piv = bc.pivot_table(index="band", columns="variant", values="coverage", aggfunc="mean").reindex(BANDS)
    print("\n=== per-band coverage across 25 boreholes (nominal {:.0%}) ===".format(NOMINAL))
    print(piv[["global", "mondrian", "mondrian+ACI"]].round(3).to_string())


if __name__ == "__main__":
    main()
