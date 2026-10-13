"""
run_gamma_sweep.py -- B3: robustness of PermaCP to the ACI step size gamma.

Holds the full parameter contract fixed (same 25 boreholes, 60/20/20 split, base model,
alpha=0.10, a-priori bands, seed 42) and varies ONLY gamma. The paper uses gamma=0.03; this
shows per-band coverage is stable across a range of gamma, so it is not a tuned knob. The model
is fit ONCE per site (gamma only affects the online ACI step), so this stays comparable.

Baseline gamma=0.03 MUST reproduce run_adaptive PermaCP numbers.
Output: results/gamma_sweep.csv (+ console).
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
GAMMAS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]  # 0.03 = paper baseline


def run_one(sdf):
    feat = build_features(sdf, horizon=HORIZON)
    tr, cal, te, *_ = time_split(feat)
    if len(tr) < 500 or len(cal) < 200 or len(te) < 200:
        return None
    m = make_base_model(SEED)
    m.fit(tr[FEATURE_COLS], tr["y"])
    cal = cal.assign(y_pred=m.predict(cal[FEATURE_COLS])); cal["abs_resid"] = (cal.y - cal.y_pred).abs()
    te = te.assign(y_pred=m.predict(te[FEATURE_COLS]))
    out = []
    for g in GAMMAS:
        a = mondrian_aci(cal, te, ALPHA, gamma=g); a["gamma"] = g
        out.append(a)
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
    sitecov = (res.groupby(["site_id", "gamma", "band"])
                  .apply(lambda x: np.average(x.coverage, weights=x.n_test), include_groups=False)
                  .rename("coverage").reset_index())
    tbl = sitecov.groupby(["gamma", "band"]).coverage.mean().reset_index()
    tbl.to_csv(RES / "gamma_sweep.csv", index=False)
    piv = tbl.pivot_table(index="gamma", columns="band", values="coverage").reindex(columns=BANDS)
    pd.set_option("display.width", 200)
    print("\n=== PermaCP per-band coverage by ACI gamma (nominal {:.0%}; 0.03 = paper) ===".format(NOMINAL))
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
