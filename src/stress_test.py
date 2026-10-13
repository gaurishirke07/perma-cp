"""
stress_test.py — extreme-case evaluation across all 25 boreholes.
Coverage on: ALL test / warmest-decile days / near-0C thaw zone (|y|<=0.5) / worst-site min.
Methods: Global vs Mondrian vs A (Mondrian+ACI). Plus Winkler interval score.
Output: results/stress_test.csv (+ console)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from model import make_base_model
from features import build_features, FEATURE_COLS
from conformal import conformal_quantile, aci_stream
from run_seed import time_split, SEED, ALPHA, HORIZON

ROOT = Path(__file__).resolve().parents[1]
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]


def per_point(cal, test, alpha=ALPHA, gamma=0.03):
    Qg = conformal_quantile(cal["abs_resid"].values, alpha)
    Qb = {b: conformal_quantile(g["abs_resid"].values, alpha) for b, g in cal.groupby("band")}
    scores = {b: np.sort(g["abs_resid"].values) for b, g in cal.groupby("band")}
    rec = []
    for depth, g in test.groupby("depth_m"):
        b = g["band"].iloc[0]; gg = g.sort_values("date")
        y, yp = gg["y"].values, gg["y_pred"].values
        e = np.abs(y - yp)
        hm = Qb.get(b, np.nan)
        covA, qa = aci_stream(scores.get(b, np.array([])), e, alpha, gamma)
        for i in range(len(gg)):
            rec.append({"band": b, "y": y[i], "abs_err": e[i],
                        "cov_global": e[i] <= Qg, "w_global": 2 * Qg,
                        "cov_mond": e[i] <= hm, "w_mond": 2 * hm,
                        "cov_A": covA[i], "w_A": 2 * qa[i]})
    return pd.DataFrame(rec)


def winkler(df, cov_col, w_col, alpha=ALPHA):
    # interval score for symmetric interval: width + (2/alpha)*shortfall
    hw = df[w_col] / 2
    short = (df["abs_err"] - hw).clip(lower=0)
    return float((df[w_col] + (2 / alpha) * short).mean())


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
        parts.append(pp)
        print(f"  {site:10} done")
    d = pd.concat(parts, ignore_index=True)

    # masks
    warm = d["y"] >= d.groupby(["site", "band"])["y"].transform(lambda s: s.quantile(0.9))
    thaw = d["y"].abs() <= 0.5
    masks = {"ALL": np.ones(len(d), bool), "warmest-10%": warm.values, "thaw-zone|y|<=0.5C": thaw.values}

    print("\n=== COVERAGE by scenario (all bands pooled, nominal 90%) ===")
    rows = []
    for name, mk in masks.items():
        sub = d[mk]
        r = {"scenario": name, "n": len(sub),
             "global": sub.cov_global.mean(), "mondrian": sub.cov_mond.mean(), "A": sub.cov_A.mean()}
        rows.append(r); print(f"{name:20} n={len(sub):6}  global={r['global']:.3f}  mond={r['mondrian']:.3f}  A={r['A']:.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "stress_test.csv", index=False)

    print("\n=== WORST-SITE coverage (min over sites, deep_trend band) ===")
    dt = d[d.band == "deep_trend(>=20m)"]
    ws = dt.groupby("site")[["cov_global", "cov_mond", "cov_A"]].mean().min()
    print(f"  global={ws.cov_global:.3f}  mondrian={ws.cov_mond:.3f}  A={ws.cov_A:.3f}")

    print("\n=== WINKLER interval score (lower=better; jointly scores width+miscoverage) ===")
    for b in BANDS:
        db = d[d.band == b]
        print(f"  {b:22} global={winkler(db,'cov_global','w_global'):7.3f}  "
              f"mond={winkler(db,'cov_mond','w_mond'):7.3f}  A={winkler(db,'cov_A','w_A'):7.3f}")


if __name__ == "__main__":
    main()
