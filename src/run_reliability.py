"""
run_reliability.py — reliability sweep / calibration curve across nominal coverage levels.

For each nominal level (1-alpha), measure per-band empirical coverage (cross-site mean over the 25
boreholes) for global | Mondrian | A (Mondrian+ACI). A well-calibrated method tracks the diagonal
(empirical == nominal) at EVERY level, not just 90%. The grid is dense in the high-assurance region
(0.95, 0.99) that matters for hazard decisions. No re-fitting per level: the base model + calibration
residuals are computed once, each level just re-quantiles them (ACI re-runs its cheap online loop).

Output: results/reliability_sweep.csv (figure drawn by make_figures.py)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from model import make_base_model
from features import build_features, FEATURE_COLS
from conformal import conformal_quantile, aci_stream
from run_seed import time_split, SEED, HORIZON

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]
NOMINALS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
GAMMA = 0.03


def aci_coverage(test_e, cal_scores, alpha, gamma=GAMMA):
    """Online ACI coverage on a date-ordered abs-error stream, using band-local calibration scores."""
    if len(cal_scores) == 0 or len(test_e) == 0:
        return np.nan
    cov, _ = aci_stream(cal_scores, test_e, alpha, gamma)
    return float(cov.mean())


def main():
    long_df = pd.read_parquet(ROOT / "data" / "borehole_all_long.parquet")
    sites = []
    for site, sdf in long_df.groupby("site_id"):
        feat = build_features(sdf, horizon=HORIZON)
        tr, cal, te, *_ = time_split(feat)
        if len(tr) < 500 or len(cal) < 200 or len(te) < 200:
            continue
        m = make_base_model(SEED)
        m.fit(tr[FEATURE_COLS], tr["y"])
        cal = cal.assign(y_pred=m.predict(cal[FEATURE_COLS])); cal["abs_resid"] = (cal.y - cal.y_pred).abs()
        te = te.assign(y_pred=m.predict(te[FEATURE_COLS])); te["abs_err"] = (te.y - te.y_pred).abs()
        rec = {"site": site, "cal_abs": cal["abs_resid"].values, "band_cal": {}, "band_te": {}}
        for b in BANDS:
            rec["band_cal"][b] = np.sort(cal[cal.band == b]["abs_resid"].values)
            rec["band_te"][b] = te[te.band == b].sort_values("date")["abs_err"].values
        sites.append(rec); print(f"  {site:10} done")

    rows = []
    for nl in NOMINALS:
        alpha = round(1 - nl, 4)
        for b in BANDS:
            cg, cm, ca = [], [], []
            for s in sites:
                te_b = s["band_te"][b]
                if len(te_b) == 0:
                    continue
                Qg = conformal_quantile(s["cal_abs"], alpha)
                cg.append(float(np.mean(te_b <= Qg)))
                Qb = conformal_quantile(s["band_cal"][b], alpha)
                cm.append(float(np.mean(te_b <= Qb)))
                ca.append(aci_coverage(te_b, s["band_cal"][b], alpha))
            rows.append({"nominal": nl, "band": b, "global": np.nanmean(cg),
                         "mondrian": np.nanmean(cm), "A": np.nanmean(ca), "n_sites": len(cg)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "reliability_sweep.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== empirical coverage by nominal level and band (cross-site mean) ===")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
