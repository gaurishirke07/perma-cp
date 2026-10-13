"""
run_seed.py — entry point for the depth-conditional conformal seed experiment.

load parquet -> features -> TIME split 60/20/20 (test = most recent/warmest) ->
quantile LightGBM point forecaster -> Option A (global conformal) + Option B (Mondrian by
depth) + AR(1)-whitened robustness variant -> results CSVs (figures via make_figures.py).

CPU-only. Fixed seed. Run:  python src/run_seed.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from model import make_base_model

from features import build_features, FEATURE_COLS
from conformal import (global_conformal, mondrian_conformal, mondrian_ar1_whitened,
                       ar1_rho)

SEED = 42
ALPHA = 0.10          # nominal 90% coverage
HORIZON = 30          # days
NOMINAL = 1 - ALPHA

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def time_split(feat: pd.DataFrame, frac_train=0.6, frac_cal=0.2):
    """Split by DATE (same boundaries across all depths). Test = most recent window."""
    dates = np.sort(feat["date"].unique())
    t_cut = dates[int(len(dates) * frac_train)]
    c_cut = dates[int(len(dates) * (frac_train + frac_cal))]
    train = feat[feat.date < t_cut]
    cal = feat[(feat.date >= t_cut) & (feat.date < c_cut)]
    test = feat[feat.date >= c_cut]
    return train, cal, test, pd.Timestamp(t_cut), pd.Timestamp(c_cut)


def main():
    np.random.seed(SEED)
    print("Loading data + building features ...")
    long_df = pd.read_parquet(ROOT / "data" / "borehole_long.parquet")
    feat = build_features(long_df, horizon=HORIZON)
    train, cal, test, t_cut, c_cut = time_split(feat)
    print(f"  train {len(train):,} (<{t_cut.date()}) | cal {len(cal):,} | "
          f"test {len(test):,} (>= {c_cut.date()})")

    # warming check: is the test window warmer at depth than calibration? (drives the break)
    deep = feat[feat.depth_m >= 20]
    print(f"  deep (>=20m) mean temp: cal={deep[(deep.date>=t_cut)&(deep.date<c_cut)].y.mean():.3f} "
          f"test={deep[deep.date>=c_cut].y.mean():.3f} degC")

    print("Fitting quantile LightGBM (point / median) ...")
    model = make_base_model(SEED)
    model.fit(train[FEATURE_COLS], train["y"])

    cal = cal.assign(y_pred=model.predict(cal[FEATURE_COLS]))
    test = test.assign(y_pred=model.predict(test[FEATURE_COLS]))
    cal["resid"] = cal["y"] - cal["y_pred"]
    cal["abs_resid"] = cal["resid"].abs()
    cal["abs_resid_signed"] = cal["resid"]

    # ---- Option A: global conformal ----
    print("Option A: global (pooled) split conformal ...")
    g = global_conformal(cal, test, ALPHA)
    g.to_csv(RES / "coverage_global.csv", index=False)

    # ---- Option B: Mondrian (depth-stratified) ----
    print("Option B: Mondrian (depth-stratified) conformal ...")
    m = mondrian_conformal(cal, test, ALPHA)
    # AR(1)-whitened robustness variant
    w = mondrian_ar1_whitened(cal, test, ALPHA)
    compare = pd.concat([g, m], ignore_index=True)
    compare.to_csv(RES / "coverage_compare.csv", index=False)
    w.to_csv(RES / "coverage_ar1_whitened.csv", index=False)

    # residual autocorrelation per band (exchangeability diagnostic)
    diag = []
    for b, gg in cal.sort_values("date").groupby("band"):
        diag.append({"band": b, "cal_resid_ar1_rho": ar1_rho(gg["resid"].values),
                     "resid_std": gg["resid"].std(), "n_cal": len(gg)})
    diag = pd.DataFrame(diag)
    diag.to_csv(RES / "residual_autocorr_by_band.csv", index=False)

    _summary(g, m, w, diag)
    print("\nDone. Tables in results/.")


def _summary(g, m, w, diag):
    def band_cov(df):
        return df.groupby("band").apply(
            lambda x: np.average(x.coverage, weights=x.n_test), include_groups=False)
    print("\n=== per-band coverage (test), nominal {:.0%} ===".format(NOMINAL))
    tbl = pd.DataFrame({"global": band_cov(g), "mondrian": band_cov(m),
                        "mondrian+AR1": band_cov(w)})
    order = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]
    print(tbl.reindex(order).round(3).to_string())
    print("\n=== residual autocorrelation by band (exchangeability diagnostic) ===")
    print(diag.set_index("band").reindex(order).round(3).to_string())
    tbl.reindex(order).to_csv(RES / "per_band_coverage_summary.csv")


if __name__ == "__main__":
    main()
