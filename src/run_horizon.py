"""
run_horizon.py -- horizon robustness: PermaCP per-band coverage at h=30 (paper) and h=90.

Holds the contract fixed and varies ONLY the forecast horizon. h=30 reproduces run_adaptive;
h=90 shows the method also holds at a longer operational horizon. These are reported as separate
horizons (a longer-horizon problem is harder), not a point-for-point comparison to h=30.
Output: results/horizon.csv (+ console).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from features import build_features, FEATURE_COLS
from conformal import mondrian_aci
from model import make_base_model
from run_seed import time_split, SEED, ALPHA, NOMINAL

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]
HORIZONS = [30, 90]  # 30 = paper baseline


def run_site(sdf, h):
    feat = build_features(sdf, horizon=h)
    tr, cal, te, *_ = time_split(feat)
    if len(tr) < 500 or len(cal) < 200 or len(te) < 200:
        return None
    m = make_base_model(SEED)
    m.fit(tr[FEATURE_COLS], tr["y"])
    cal = cal.assign(y_pred=m.predict(cal[FEATURE_COLS])); cal["abs_resid"] = (cal.y - cal.y_pred).abs()
    te = te.assign(y_pred=m.predict(te[FEATURE_COLS]))
    return mondrian_aci(cal, te, ALPHA)


def main():
    long_df = pd.read_parquet(ROOT / "data" / "borehole_all_long.parquet")
    rows = []
    for h in HORIZONS:
        for site, sdf in long_df.groupby("site_id"):
            r = run_site(sdf, h)
            if r is None:
                continue
            r["site_id"] = site; r["horizon"] = h; rows.append(r)
        print(f"  h={h} done")
    res = pd.concat(rows, ignore_index=True)
    sitecov = (res.groupby(["horizon", "site_id", "band"])
                  .apply(lambda x: np.average(x.coverage, weights=x.n_test), include_groups=False)
                  .rename("coverage").reset_index())
    tbl = sitecov.groupby(["horizon", "band"]).coverage.mean().reset_index()
    tbl.to_csv(RES / "horizon.csv", index=False)
    piv = tbl.pivot_table(index="horizon", columns="band", values="coverage").reindex(columns=BANDS)
    pd.set_option("display.width", 200)
    print(f"\n=== PermaCP per-band coverage by horizon (nominal {NOMINAL:.0%}; h=30 = paper) ===")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
