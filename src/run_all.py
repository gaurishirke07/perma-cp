"""
run_all.py — full-population run over ALL 25 qualifying PERMOS boreholes.

Removes the rock-glacier selection: includes every borehole meeting the depth/record
INCLUSION criterion (>=4 depths <1.5m..>=10m, >=10yr daily), then stratifies results by
landform (crest/talus/rock glacier/moraine). Tests (a) whether the depth-structured global
miss is UNIVERSAL (it should be — variance decays with depth everywhere) and (b) whether the
deep-drift under-coverage depends on landform (the honest heterogeneity).

Output: results/coverage_all_by_site.csv, results/coverage_all_bands.csv (figures drawn by make_figures.py)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

from features import build_features, FEATURE_COLS
from conformal import global_conformal, mondrian_conformal
from run_seed import time_split, SEED, ALPHA, HORIZON, NOMINAL
from model import make_base_model

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
    long_df = pd.read_parquet(ROOT / "data" / "borehole_all_long.parquet")
    morph = long_df.groupby("site_id")["morphology"].first()
    rows = []
    for site, g in long_df.groupby("site_id"):
        r = run_one(g)
        if r is None:
            print(f"  {site:10} skipped"); continue
        r["site_id"] = site; r["morphology"] = morph[site]
        rows.append(r)
        print(f"  {site:10} done ({morph[site]})")
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(RES / "coverage_all_by_site.csv", index=False)

    def band_cov(df):
        return (df.groupby(["site_id", "morphology", "variant", "band"])
                  .apply(lambda x: np.average(x.coverage, weights=x.n_test), include_groups=False)
                  .rename("coverage").reset_index())
    bc = band_cov(res)
    bc.to_csv(RES / "coverage_all_bands.csv", index=False)
    n_sites = res.site_id.nunique()

    print(f"\n=== per-band coverage across ALL {n_sites} boreholes (nominal {NOMINAL:.0%}) ===")
    piv = bc.pivot_table(index="band", columns="variant", values="coverage", aggfunc=["mean", "std"])
    print(piv.reindex(BANDS).round(3).to_string())
    print("\n=== deep_trend(>=20m) coverage by landform ===")
    dt = bc[bc.band == "deep_trend(>=20m)"].pivot_table(index="morphology", columns="variant",
                                                        values="coverage", aggfunc="mean")
    print(dt.round(3).to_string())


if __name__ == "__main__":
    main()
