"""
run_missing_robustness.py -- B4: does dropping missing-lag rows (MAR) bias the coverage?

Baseline drops any row missing the target or a lag (the paper). The variant keeps every row
with a target and lets LightGBM handle NaN lag features natively, so the previously dropped rows
(concentrated at the gappy deep sensors) re-enter. To stay comparable, the SAME per-site
60/20/20 date boundaries are used for both variants. If PermaCP still restores ~0.90, the drop
is not hiding a problem. The drop(baseline) variant must reproduce run_adaptive.
Output: results/missing_robustness.csv (+ console).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from features import build_features, FEATURE_COLS
from conformal import mondrian_aci
from model import make_base_model
from run_seed import time_split, SEED, ALPHA, HORIZON, NOMINAL

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]


def split_on(feat, t_cut, c_cut):
    return (feat[feat.date < t_cut], feat[(feat.date >= t_cut) & (feat.date < c_cut)],
            feat[feat.date >= c_cut])


def run_one(sdf):
    # baseline (drop) feature set defines the shared split dates
    base = build_features(sdf, horizon=HORIZON, require_lags=True)
    tr0, cal0, te0, t_cut, c_cut = time_split(base)
    if len(tr0) < 500 or len(cal0) < 200 or len(te0) < 200:
        return None
    out = []
    for name, req in [("drop(baseline)", True), ("keep-NaN", False)]:
        feat = base if req else build_features(sdf, horizon=HORIZON, require_lags=False)
        tr, cal, te = split_on(feat, t_cut, c_cut)
        if len(tr) < 500 or len(cal) < 200 or len(te) < 200:
            return None
        m = make_base_model(SEED)
        m.fit(tr[FEATURE_COLS], tr["y"])
        cal = cal.assign(y_pred=m.predict(cal[FEATURE_COLS])); cal["abs_resid"] = (cal.y - cal.y_pred).abs()
        te = te.assign(y_pred=m.predict(te[FEATURE_COLS]))
        r = mondrian_aci(cal, te, ALPHA); r["variant"] = name
        out.append(r)
    return pd.concat(out, ignore_index=True)


def main():
    long_df = pd.read_parquet(ROOT / "data" / "borehole_all_long.parquet")
    rows = []
    for site, sdf in long_df.groupby("site_id"):
        r = run_one(sdf)
        if r is None:
            print(f"  {site:10} skip"); continue
        r["site_id"] = site; rows.append(r); print(f"  {site:10} done")
    res = pd.concat(rows, ignore_index=True)
    sitecov = (res.groupby(["site_id", "variant", "band"])
                  .apply(lambda x: np.average(x.coverage, weights=x.n_test), include_groups=False)
                  .rename("coverage").reset_index())
    tbl = sitecov.groupby(["variant", "band"]).coverage.mean().reset_index()
    tbl.to_csv(RES / "missing_robustness.csv", index=False)
    piv = tbl.pivot_table(index="variant", columns="band", values="coverage").reindex(columns=BANDS)
    pd.set_option("display.width", 200)
    print(f"\n=== PermaCP per-band coverage: drop vs keep-NaN missing rows (nominal {NOMINAL:.0%}) ===")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
