"""
run_coverage_ci.py — confidence intervals on per-band empirical coverage (nominal 90%).

Coverage's uncertainty has TWO sources, so we report two CIs per band:
  1. WILSON binomial CI on pooled test points. Within-sample; assumes independent Bernoulli trials
     -> OPTIMISTIC here, because residuals are temporally autocorrelated (rho up to ~0.85 shallow)
     and clustered by site/sensor, so the effective sample size << n_points.
  2. CROSS-SITE bootstrap CI over the per-site coverages. The unit of replication is the BOREHOLE
     (25 sites); this is the honest population-level CI and the one to report in the paper.
A band's coverage is "significantly different from 0.90" iff 0.90 falls outside the cross-site CI.

Methods compared: global | Mondrian(depth) | A (Mondrian+ACI).  CPU-only, fixed seed.
Output: results/coverage_ci.csv (+ console)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from model import make_base_model
from statsmodels.stats.proportion import proportion_confint

from features import build_features, FEATURE_COLS
from conformal import conformal_quantile, aci_stream
from run_seed import time_split, SEED, ALPHA, HORIZON, NOMINAL

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]
N_BOOT = 2000
RNG = np.random.default_rng(SEED)


def per_point(cal, test, alpha=ALPHA, gamma=0.03):
    """Per test point: covered (bool) under global / Mondrian / Method A. Mirrors stress_test.py."""
    Qg = conformal_quantile(cal["abs_resid"].values, alpha)
    Qb = {b: conformal_quantile(g["abs_resid"].values, alpha) for b, g in cal.groupby("band")}
    scores = {b: np.sort(g["abs_resid"].values) for b, g in cal.groupby("band")}
    rec = []
    for depth, g in test.groupby("depth_m"):
        b = g["band"].iloc[0]; gg = g.sort_values("date")
        y, yp = gg["y"].values, gg["y_pred"].values
        e = np.abs(y - yp)
        hm = Qb.get(b, np.nan)
        covA, _ = aci_stream(scores.get(b, np.array([])), e, alpha, gamma)
        for i in range(len(gg)):
            rec.append({"band": b, "cov_global": bool(e[i] <= Qg),
                        "cov_mond": bool(e[i] <= hm), "cov_A": bool(covA[i])})
    return pd.DataFrame(rec)


def main():
    long_df = pd.read_parquet(ROOT / "data" / "borehole_all_long.parquet")
    parts = []
    for site, sdf in long_df.groupby("site_id"):
        feat = build_features(sdf, horizon=HORIZON)
        tr, cal, te, *_ = time_split(feat)
        if len(tr) < 500 or len(cal) < 200 or len(te) < 200:
            continue
        m = make_base_model(SEED)
        m.fit(tr[FEATURE_COLS], tr["y"])
        cal = cal.assign(y_pred=m.predict(cal[FEATURE_COLS])); cal["abs_resid"] = (cal.y - cal.y_pred).abs()
        te = te.assign(y_pred=m.predict(te[FEATURE_COLS]))
        pp = per_point(cal, te); pp["site"] = site
        parts.append(pp); print(f"  {site:10} done")
    d = pd.concat(parts, ignore_index=True)

    methods = [("global", "cov_global"), ("mondrian", "cov_mond"), ("A(Mondrian+ACI)", "cov_A")]
    rows = []
    for b in BANDS:
        db = d[d.band == b]
        for name, col in methods:
            k = int(db[col].sum()); n = int(len(db)); cov = k / n
            lo_w, hi_w = proportion_confint(k, n, alpha=0.05, method="wilson")  # pooled binomial
            site_cov = db.groupby("site")[col].mean().values                    # unit = borehole
            boot = np.array([RNG.choice(site_cov, len(site_cov), replace=True).mean()
                             for _ in range(N_BOOT)])
            lo_s, hi_s = np.percentile(boot, [2.5, 97.5])
            rows.append({"band": b, "method": name, "n_sites": len(site_cov), "n_points": n,
                         "coverage": round(cov, 3),
                         "wilson_lo": round(float(lo_w), 3), "wilson_hi": round(float(hi_w), 3),
                         "site_ci_lo": round(float(lo_s), 3), "site_ci_hi": round(float(hi_s), 3),
                         "sig_diff_from_0.90": not (lo_s <= NOMINAL <= hi_s)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "coverage_ci.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== per-band coverage with 95% CIs (nominal {:.2f}) ===".format(NOMINAL))
    print(out.to_string(index=False))
    print("\nwilson_*  = pooled-point binomial CI (OPTIMISTIC: assumes independent points).")
    print("site_ci_* = cross-site bootstrap CI (HONEST: unit of replication = borehole) -> report this.")
    print("sig_diff_from_0.90 = True when 0.90 lies OUTSIDE the cross-site CI.")


if __name__ == "__main__":
    main()
