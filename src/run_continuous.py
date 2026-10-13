"""
run_continuous.py — METHOD B: continuous depth-normalized conformal across all 25 boreholes.
Compares Mondrian+ACI (A) vs depth-normalized (B) vs depth-normalized+ACI (B+drift).
Output: results/coverage_continuous.csv (figure drawn by make_figures.py)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from model import make_base_model

from features import build_features, FEATURE_COLS
from conformal import mondrian_aci, depth_normalized_conformal
from run_seed import time_split, SEED, ALPHA, HORIZON, NOMINAL

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]


def run_one(sdf):
    feat = build_features(sdf, horizon=HORIZON)
    tr, cal, te, *_ = time_split(feat)
    if len(tr) < 500 or len(cal) < 200 or len(te) < 200:
        return None
    m = make_base_model(SEED)
    m.fit(tr[FEATURE_COLS], tr["y"])
    cal = cal.assign(y_pred=m.predict(cal[FEATURE_COLS])); te = te.assign(y_pred=m.predict(te[FEATURE_COLS]))
    cal["abs_resid"] = (cal["y"] - cal["y_pred"]).abs()
    a = mondrian_aci(cal, te, ALPHA); a["variant"] = "A: mondrian+ACI"
    b = depth_normalized_conformal(cal, te, ALPHA, aci=False); b["variant"] = "B: depth-norm"
    c = depth_normalized_conformal(cal, te, ALPHA, aci=True); c["variant"] = "B+ACI: depth-norm+drift"
    return pd.concat([a, b, c], ignore_index=True)


def main():
    long_df = pd.read_parquet(ROOT / "data" / "borehole_all_long.parquet")
    rows = []
    for site, sdf in long_df.groupby("site_id"):
        r = run_one(sdf)
        if r is not None:
            r["site_id"] = site; rows.append(r); print(f"  {site:10} done")
    res = pd.concat(rows, ignore_index=True)
    bc = (res.groupby(["site_id", "variant", "band"])
             .apply(lambda x: np.average(x.coverage, weights=x.n_test), include_groups=False)
             .rename("coverage").reset_index())
    bc.to_csv(RES / "coverage_continuous.csv", index=False)

    piv = bc.pivot_table(index="band", columns="variant", values="coverage", aggfunc="mean").reindex(BANDS)
    print("\n=== per-band coverage (nominal {:.0%}) ===".format(NOMINAL))
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
