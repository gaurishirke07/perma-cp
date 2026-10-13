"""
run_cqr.py -- B6: CQR (conformalized quantile regression) contrast vs the paper's abs-residual score.

The paper wraps a MEDIAN forecaster with a symmetric absolute-residual score. CQR (Romano et al.
2019) uses two quantile models (lower alpha/2, upper 1-alpha/2) and an asymmetric score, giving
adaptive-width intervals -- the paper's own stated future-work direction for the one-sided deep bias.
Everything else is held to the contract (same 25 boreholes, 60/20/20 split, bands, alpha=0.10,
gamma=0.03, seed 42, same LightGBM config; only the quantile target changes).

Four variants, so we can tell apart the two questions:
  - abs-Mondrian / CQR-Mondrian  (no ACI): does the asymmetric SCORE fix Mondrian's deep under-coverage?
  - abs-PermaCP  / CQR-PermaCP   (+ ACI):  does CQR improve the FINAL method?
abs-Mondrian and abs-PermaCP must reproduce run_all / run_adaptive. CQR score: E=max(qlo-y, y-qhi);
interval [qlo-Q, qhi+Q]; covered iff E<=Q, so the same conformal-quantile / ACI machinery applies.
Output: results/cqr.csv (+ console).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from features import build_features, FEATURE_COLS
from conformal import mondrian_conformal, mondrian_aci, aci_stream, conformal_quantile
from model import make_base_model
from run_seed import time_split, SEED, ALPHA, HORIZON, NOMINAL

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]
ORDER = ["abs-Mondrian", "CQR-Mondrian", "abs-PermaCP(paper)", "CQR-PermaCP"]
GAMMA = 0.03


def run_one(sdf):
    feat = build_features(sdf, horizon=HORIZON)
    tr, cal, te, *_ = time_split(feat)
    if len(tr) < 500 or len(cal) < 200 or len(te) < 200:
        return None
    out = []
    # --- abs-residual variants (paper): median model + symmetric score ---
    med = make_base_model(SEED, 0.5); med.fit(tr[FEATURE_COLS], tr["y"])
    cal_a = cal.assign(y_pred=med.predict(cal[FEATURE_COLS])); cal_a["abs_resid"] = (cal_a.y - cal_a.y_pred).abs()
    te_a = te.assign(y_pred=med.predict(te[FEATURE_COLS]))
    am = mondrian_conformal(cal_a, te_a, ALPHA)[["band", "coverage", "mean_width", "n_test"]].copy(); am["variant"] = "abs-Mondrian"
    ap = mondrian_aci(cal_a, te_a, ALPHA, gamma=GAMMA)[["band", "coverage", "mean_width", "n_test"]].copy(); ap["variant"] = "abs-PermaCP(paper)"
    out += [am, ap]
    # --- CQR variants: lower/upper quantile models (same config; quantile = 0.05 / 0.95) ---
    qlo = make_base_model(SEED, ALPHA / 2);     qlo.fit(tr[FEATURE_COLS], tr["y"])
    qhi = make_base_model(SEED, 1 - ALPHA / 2); qhi.fit(tr[FEATURE_COLS], tr["y"])
    cc = cal.copy(); cc["qlo"] = qlo.predict(cal[FEATURE_COLS]); cc["qhi"] = qhi.predict(cal[FEATURE_COLS])
    cc["E"] = np.maximum(cc["qlo"] - cc["y"], cc["y"] - cc["qhi"])
    tt = te.copy(); tt["qlo"] = qlo.predict(te[FEATURE_COLS]); tt["qhi"] = qhi.predict(te[FEATURE_COLS])
    tt["E"] = np.maximum(tt["qlo"] - tt["y"], tt["y"] - tt["qhi"])
    cm_rows, cp_rows = [], []
    for b in BANDS:
        cb = cc[cc.band == b]; tb = tt[tt.band == b].sort_values("date")
        if len(tb) == 0 or len(cb) == 0:
            continue
        base_w = tb["qhi"].values - tb["qlo"].values
        # CQR-Mondrian (no ACI): static per-band quantile of the CQR score
        Qb = conformal_quantile(cb["E"].values, ALPHA)
        cm_rows.append({"band": b, "coverage": float(np.mean(tb["E"].values <= Qb)),
                        "mean_width": float(np.mean(base_w + 2 * Qb)), "n_test": len(tb)})
        # CQR-PermaCP (+ ACI): online-adapted quantile of the CQR score
        cov, Qt = aci_stream(np.sort(cb["E"].values), tb["E"].values, ALPHA, GAMMA)
        cp_rows.append({"band": b, "coverage": float(cov.mean()),
                        "mean_width": float(np.mean(base_w + 2 * Qt)), "n_test": len(tb)})
    cm = pd.DataFrame(cm_rows); cm["variant"] = "CQR-Mondrian"
    cp = pd.DataFrame(cp_rows); cp["variant"] = "CQR-PermaCP"
    out += [cm, cp]
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
    sc = (res.groupby(["site_id", "variant", "band"])
             .apply(lambda x: pd.Series({"coverage": np.average(x.coverage, weights=x.n_test),
                                         "width": np.average(x.mean_width, weights=x.n_test)}),
                    include_groups=False).reset_index())
    out = sc.groupby(["variant", "band"])[["coverage", "width"]].mean().reset_index()
    out.to_csv(RES / "cqr.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== coverage by variant (nominal {NOMINAL:.0%}) ===")
    print(out.pivot_table(index="variant", columns="band", values="coverage").reindex(ORDER, columns=BANDS).round(3).to_string())
    print("\n=== mean interval width (degC) ===")
    print(out.pivot_table(index="variant", columns="band", values="width").reindex(ORDER, columns=BANDS).round(3).to_string())


if __name__ == "__main__":
    main()
