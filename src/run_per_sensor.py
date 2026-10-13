"""
run_per_sensor.py -- I1: per-SENSOR coverage, not just 4-band averages.

Reviewer FS2c + AC meta: "near-90% average within four depth bands does not establish equivalent
coverage at every sensor." This keeps per-(site, depth) coverage for global / Mondrian / PermaCP
across the 25 boreholes, holding the full contract. It shows PermaCP centres each sensor near 0.90
and tightens the wide per-sensor spread the baselines show. Band-aggregates reproduce run_adaptive.
Output: results/per_sensor_coverage.csv (+ console honesty stats).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from features import build_features, FEATURE_COLS
from conformal import global_conformal, mondrian_conformal, mondrian_aci
from model import make_base_model
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
    cal = cal.assign(y_pred=m.predict(cal[FEATURE_COLS])); cal["abs_resid"] = (cal.y - cal.y_pred).abs()
    te = te.assign(y_pred=m.predict(te[FEATURE_COLS]))
    g = global_conformal(cal, te, ALPHA); g["variant"] = "global"
    mo = mondrian_conformal(cal, te, ALPHA); mo["variant"] = "mondrian"
    a = mondrian_aci(cal, te, ALPHA); a["variant"] = "permacp"
    cols = ["depth_m", "band", "coverage", "n_test", "variant"]
    return pd.concat([g[cols], mo[cols], a[cols]], ignore_index=True)


def main():
    long_df = pd.read_parquet(ROOT / "data" / "borehole_all_long.parquet")
    rows = []
    for site, sdf in long_df.groupby("site_id"):
        r = run_one(sdf)
        if r is None:
            print(f"  {site:10} skip"); continue
        r["site_id"] = site; rows.append(r); print(f"  {site:10} done")
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(RES / "per_sensor_coverage.csv", index=False)
    pd.set_option("display.width", 200)
    pc = res[res.variant == "permacp"]
    print(f"\n=== PermaCP per-SENSOR coverage: {len(pc)} sensors across 25 boreholes (nominal {NOMINAL:.0%}) ===")
    print(f"  sensors within +-0.05 of 0.90: {(pc.coverage.sub(NOMINAL).abs() <= 0.05).mean():.1%}; "
          f"within +-0.10: {(pc.coverage.sub(NOMINAL).abs() <= 0.10).mean():.1%}")
    print("\n=== per-band per-sensor coverage spread (min / median / max over sensors) ===")
    for v in ["global", "mondrian", "permacp"]:
        print(f"\n[{v}]")
        print(res[res.variant == v].groupby("band").coverage.agg(["min", "median", "max"]).reindex(BANDS).round(3).to_string())


if __name__ == "__main__":
    main()
