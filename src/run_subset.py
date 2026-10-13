"""
run_subset.py — multi-site replication of the depth-conditional conformal result.

Runs the same pipeline as run_seed.py independently on each of 5 bedrock/crest PERMOS
boreholes, to test whether the depth-structured coverage miss (global conformal) and the
Mondrian recovery REPLICATE across sites. Per-site model (sites differ geologically).

Output: results/coverage_subset_by_site.csv, results/coverage_subset_bands.csv
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from model import make_base_model

from features import build_features, FEATURE_COLS, depth_band
from conformal import global_conformal, mondrian_conformal
from run_seed import time_split, SEED, ALPHA, HORIZON, NOMINAL

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]


def run_one(site_df):
    feat = build_features(site_df, horizon=HORIZON)
    train, cal, test, *_ = time_split(feat)
    if len(train) < 500 or len(cal) < 200 or len(test) < 200:
        return None
    m = make_base_model(SEED)
    m.fit(train[FEATURE_COLS], train["y"])
    cal = cal.assign(y_pred=m.predict(cal[FEATURE_COLS]))
    test = test.assign(y_pred=m.predict(test[FEATURE_COLS]))
    cal["abs_resid"] = (cal["y"] - cal["y_pred"]).abs()
    g = global_conformal(cal, test, ALPHA); g["variant"] = "global"
    mo = mondrian_conformal(cal, test, ALPHA); mo["variant"] = "mondrian"
    return pd.concat([g, mo], ignore_index=True)


def main():
    long_df = pd.read_parquet(ROOT / "data" / "borehole_subset_long.parquet")
    all_rows = []
    for site, g in long_df.groupby("site_id"):
        print(f"  {site} ...", end=" ")
        r = run_one(g)
        if r is None:
            print("skipped (too few points)"); continue
        r["site_id"] = site
        all_rows.append(r)
        print("done")
    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(RES / "coverage_subset_by_site.csv", index=False)

    # per-band coverage per site (weighted by n_test)
    def band_cov(df):
        return (df.groupby(["site_id", "variant", "band"])
                  .apply(lambda x: np.average(x.coverage, weights=x.n_test), include_groups=False)
                  .rename("coverage").reset_index())
    bc = band_cov(res)
    bc.to_csv(RES / "coverage_subset_bands.csv", index=False)

    piv = bc.pivot_table(index="band", columns="variant", values="coverage", aggfunc=["mean", "std"])
    print("\n=== per-band coverage across 5 sites (nominal {:.0%}) ===".format(NOMINAL))
    print(piv.reindex(BANDS).round(3).to_string())
    print("\nsites run:", res.site_id.nunique())


if __name__ == "__main__":
    main()
