"""
run_band_sensitivity.py -- B1: robustness of the result to the a-priori depth-band edges.

Holds the full parameter contract fixed (same 25 boreholes, 60/20/20 split, base model,
alpha=0.10, ACI gamma=0.03, seed 42) and varies ONLY the band cut-points. If PermaCP restores
~0.90 in every band and Mondrian still leaves the deep band under-covered across several edge
choices, the fixed bands are not a tuned artefact. The model is fit ONCE per site (it does not
depend on the bands); only the calibration is re-banded per edge set, so this stays comparable.

Baseline edges = (1, 5, 20) m (the paper's) and MUST reproduce run_adaptive numbers.
Output: results/band_sensitivity.csv (+ console).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from features import build_features, FEATURE_COLS
from conformal import mondrian_conformal, mondrian_aci
from model import make_base_model
from run_seed import time_split, SEED, ALPHA, HORIZON, NOMINAL

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
BANDS = ["shallow", "mid", "deep_seasonal", "deep_trend"]
# (e1,e2,e3): shallow<e1, mid<e2, deep_seasonal<e3, deep_trend>=e3
EDGE_SETS = {
    "baseline(1,5,20)":    (1.0, 5.0, 20.0),
    "shallower(0.5,4,18)": (0.5, 4.0, 18.0),
    "deeper(1.5,6,25)":    (1.5, 6.0, 25.0),
    "ZAA18(1,5,18)":       (1.0, 5.0, 18.0),
}


def band_of(depth, e):
    if depth < e[0]: return "shallow"
    if depth < e[1]: return "mid"
    if depth < e[2]: return "deep_seasonal"
    return "deep_trend"


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
    for name, e in EDGE_SETS.items():
        c2 = cal.copy(); c2["band"] = c2["depth_m"].map(lambda d: band_of(d, e))
        t2 = te.copy();  t2["band"] = t2["depth_m"].map(lambda d: band_of(d, e))
        mo = mondrian_conformal(c2, t2, ALPHA); mo["variant"] = "mondrian"; mo["edges"] = name
        a = mondrian_aci(c2, t2, ALPHA);        a["variant"] = "permacp";  a["edges"] = name
        out.append(pd.concat([mo, a], ignore_index=True))
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
    # per (site, edges, variant, band): weight per-depth coverage by n_test
    sitecov = (res.groupby(["site_id", "edges", "variant", "band"])
                  .apply(lambda x: np.average(x.coverage, weights=x.n_test), include_groups=False)
                  .rename("coverage").reset_index())
    tbl = sitecov.groupby(["edges", "variant", "band"]).coverage.mean().reset_index()
    tbl.to_csv(RES / "band_sensitivity.csv", index=False)
    piv = tbl.pivot_table(index=["edges", "variant"], columns="band", values="coverage").reindex(columns=BANDS)
    pd.set_option("display.width", 200)
    print("\n=== per-band coverage by band-edge set (nominal {:.0%}) ===".format(NOMINAL))
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
