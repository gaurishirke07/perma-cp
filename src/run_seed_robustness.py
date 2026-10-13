"""
run_seed_robustness.py -- B2: PermaCP per-band coverage across random seeds.

Holds the contract fixed (same 25 boreholes, 60/20/20 split, features, bands, alpha=0.10,
gamma=0.03) and varies ONLY the LightGBM seed. Shows the ~0.90 result is not a lucky seed.
Re-fits per (seed, site). Seed 42 = paper baseline and must reproduce run_adaptive.
Output: results/seed_robustness.csv (+ console mean+-sd over seeds).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from features import build_features, FEATURE_COLS
from conformal import mondrian_aci
from model import make_base_model
from run_seed import time_split, ALPHA, HORIZON, NOMINAL

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]
SEEDS = [42, 1, 7, 123, 2024]  # 42 = paper baseline


def run_site(sdf, seed):
    feat = build_features(sdf, horizon=HORIZON)
    tr, cal, te, *_ = time_split(feat)
    if len(tr) < 500 or len(cal) < 200 or len(te) < 200:
        return None
    m = make_base_model(seed)
    m.fit(tr[FEATURE_COLS], tr["y"])
    cal = cal.assign(y_pred=m.predict(cal[FEATURE_COLS])); cal["abs_resid"] = (cal.y - cal.y_pred).abs()
    te = te.assign(y_pred=m.predict(te[FEATURE_COLS]))
    return mondrian_aci(cal, te, ALPHA)


def main():
    long_df = pd.read_parquet(ROOT / "data" / "borehole_all_long.parquet")
    rows = []
    for seed in SEEDS:
        for site, sdf in long_df.groupby("site_id"):
            r = run_site(sdf, seed)
            if r is None:
                continue
            r["site_id"] = site; r["seed"] = seed; rows.append(r)
        print(f"  seed {seed} done")
    res = pd.concat(rows, ignore_index=True)
    sitecov = (res.groupby(["seed", "site_id", "band"])
                  .apply(lambda x: np.average(x.coverage, weights=x.n_test), include_groups=False)
                  .rename("coverage").reset_index())
    perseed = sitecov.groupby(["seed", "band"]).coverage.mean().reset_index()
    perseed.to_csv(RES / "seed_robustness.csv", index=False)
    agg = perseed.groupby("band").coverage.agg(["mean", "std", "min", "max"]).reindex(BANDS)
    pd.set_option("display.width", 200)
    print(f"\n=== PermaCP per-band coverage across seeds {SEEDS} (nominal {NOMINAL:.0%}) ===")
    print(agg.round(4).to_string())
    print("\n=== per-seed (42 = paper) ===")
    print(perseed.pivot_table(index="seed", columns="band", values="coverage").reindex(columns=BANDS).round(3).to_string())


if __name__ == "__main__":
    main()
