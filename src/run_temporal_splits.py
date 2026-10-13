"""
run_temporal_splits.py -- I2: PermaCP coverage across different temporal splits (rolling-origin).

Addresses the AC's suggestion to evaluate across different temporal splits. Holds the contract
(same boreholes, features, bands, model, alpha, gamma, seed); varies ONLY the train/cal/test date
blocks: three sliding test windows (60-80 / 70-90 / 80-100% of each borehole's timeline), each with
an expanding train and a 20% calibration block before it. The 'recent (80-100%)' fold is exactly the
paper's 60/20/20 split and must reproduce run_adaptive. Output: results/temporal_splits.csv.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from features import build_features, FEATURE_COLS
from conformal import mondrian_aci
from model import make_base_model
from run_seed import SEED, ALPHA, HORIZON, NOMINAL

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
BANDS = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]
# (train_end, cal_end, test_end) as fractions of unique dates; cal=[train_end,cal_end], test=[cal_end,test_end]
FOLDS = {
    "early(test 60-80%)":   (0.40, 0.60, 0.80),
    "mid(test 70-90%)":     (0.50, 0.70, 0.90),
    "recent(test 80-100%)": (0.60, 0.80, 1.00),  # == paper 60/20/20
}


def split_blocks(feat, te, ce, tee):
    dates = np.sort(feat["date"].unique()); n = len(dates)
    d_t = dates[int(n * te)]; d_c = dates[int(n * ce)]; d_e = dates[min(int(n * tee), n - 1)]
    return (feat[feat.date < d_t],
            feat[(feat.date >= d_t) & (feat.date < d_c)],
            feat[(feat.date >= d_c) & (feat.date <= d_e)])


def run_one(sdf):
    feat = build_features(sdf, horizon=HORIZON)
    out = []
    for name, (te, ce, tee) in FOLDS.items():
        tr, cal, test = split_blocks(feat, te, ce, tee)
        if len(tr) < 500 or len(cal) < 200 or len(test) < 200:
            continue
        m = make_base_model(SEED)
        m.fit(tr[FEATURE_COLS], tr["y"])
        cal = cal.assign(y_pred=m.predict(cal[FEATURE_COLS])); cal["abs_resid"] = (cal.y - cal.y_pred).abs()
        test = test.assign(y_pred=m.predict(test[FEATURE_COLS]))
        r = mondrian_aci(cal, test, ALPHA); r["fold"] = name
        out.append(r)
    return pd.concat(out, ignore_index=True) if out else None


def main():
    long_df = pd.read_parquet(ROOT / "data" / "borehole_all_long.parquet")
    rows = []
    for site, sdf in long_df.groupby("site_id"):
        r = run_one(sdf)
        if r is None:
            print(f"  {site:10} skip"); continue
        r["site_id"] = site; rows.append(r); print(f"  {site:10} done")
    res = pd.concat(rows, ignore_index=True)
    sitecov = (res.groupby(["fold", "site_id", "band"])
                  .apply(lambda x: np.average(x.coverage, weights=x.n_test), include_groups=False)
                  .rename("coverage").reset_index())
    tbl = sitecov.groupby(["fold", "band"]).coverage.mean().reset_index()
    tbl.to_csv(RES / "temporal_splits.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== PermaCP per-band coverage across temporal splits (nominal {NOMINAL:.0%}) ===")
    print(tbl.pivot_table(index="fold", columns="band", values="coverage").reindex(columns=BANDS).round(3).to_string())
    print("\nsites per fold:", dict(res.groupby("fold").site_id.nunique()))


if __name__ == "__main__":
    main()
