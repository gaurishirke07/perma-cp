"""
conformal.py — split conformal calibration, global vs depth-stratified (Mondrian).

Core claim of the paper lives here: a single GLOBAL conformal quantile (pooled over all
depths) mis-covers in a depth-structured way; a MONDRIAN quantile (per depth band) restores
per-band coverage. Optional AR(1) residual whitening (after Berk 2025) addresses the temporal
autocorrelation that violates conformal exchangeability — reported as a robustness variant.

Method: point base forecaster -> absolute-residual split conformal.
  half-width Q = quantile_{1-alpha}(|r_cal|),  interval = yhat +/- Q.
  global:   Q from pooled calibration residuals.
  mondrian: Q_b from residuals within depth band b.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def conformal_quantile(abs_resid: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile of |residual| at level 1-alpha."""
    n = len(abs_resid)
    if n == 0:
        return np.nan
    # ceil((n+1)(1-alpha))/n rank (Vovk finite-sample correction)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(abs_resid, level, method="higher"))


def aci_stream(cal_scores, test_abs_err, alpha, gamma: float = 0.03):
    """Online Adaptive Conformal Inference (Gibbs & Candes 2021) over a date-ordered stream of
    absolute errors, using a fixed band-local calibration score set. Returns per-step
    (covered, half_width). The caller sets the stream's grain: per sensor in the method drivers,
    per band in the reliability sweep. Edge cases match the rest of the module: empty scores give
    an infinite interval; a level >= 1 uses the maximum score."""
    sc = np.asarray(cal_scores)
    e = np.asarray(test_abs_err, dtype=float)
    n = len(e)
    cov = np.empty(n, dtype=bool)
    hw = np.empty(n, dtype=float)
    smax = sc.max() if len(sc) else np.inf
    a_t = alpha
    for i in range(n):
        lvl = min(max(1.0 - a_t, 0.0), 1.0)
        if len(sc) == 0:
            q = np.inf
        elif lvl >= 1.0:
            q = smax
        else:
            q = float(np.quantile(sc, lvl, method="higher"))
        c = e[i] <= q
        cov[i] = c
        hw[i] = q
        a_t = a_t + gamma * (alpha - (0 if c else 1))
    return cov, hw


def ar1_rho(resid_series: np.ndarray) -> float:
    """Lag-1 autocorrelation of a time-ordered residual series (exchangeability diagnostic)."""
    x = np.asarray(resid_series, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return 0.0
    return float(np.sum(x[1:] * x[:-1]) / denom)


def ar1_whiten(resid_series: np.ndarray, rho: float) -> np.ndarray:
    """Whiten an AR(1) residual series: e'_t = (e_t - rho e_{t-1}) / sqrt(1-rho^2)."""
    x = np.asarray(resid_series, float)
    if abs(rho) >= 0.999:
        rho = np.sign(rho) * 0.999
    w = (x[1:] - rho * x[:-1]) / np.sqrt(1 - rho ** 2)
    return w


def coverage_and_width(y_true, y_pred, half_width) -> tuple[float, float]:
    """Empirical coverage and mean interval width for a symmetric interval."""
    lo = y_pred - half_width
    hi = y_pred + half_width
    covered = (y_true >= lo) & (y_true <= hi)
    return float(np.mean(covered)), float(np.mean(2 * half_width))


def global_conformal(cal: pd.DataFrame, test: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Single pooled quantile; report coverage+width per depth on the test set."""
    Q = conformal_quantile(cal["abs_resid"].values, alpha)
    rows = []
    for depth, g in test.groupby("depth_m"):
        cov, w = coverage_and_width(g["y"].values, g["y_pred"].values, Q)
        rows.append({"depth_m": depth, "band": g["band"].iloc[0],
                     "coverage": cov, "mean_width": w, "half_width": Q, "n_test": len(g)})
    out = pd.DataFrame(rows).sort_values("depth_m")
    out["method"] = "global"
    return out


def mondrian_conformal(cal: pd.DataFrame, test: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Per-band quantile (Mondrian); report coverage+width per depth on the test set."""
    band_Q = {b: conformal_quantile(g["abs_resid"].values, alpha)
              for b, g in cal.groupby("band")}
    rows = []
    for depth, g in test.groupby("depth_m"):
        b = g["band"].iloc[0]
        Q = band_Q.get(b, np.nan)
        cov, w = coverage_and_width(g["y"].values, g["y_pred"].values, Q)
        rows.append({"depth_m": depth, "band": b, "coverage": cov,
                     "mean_width": w, "half_width": Q, "n_test": len(g)})
    out = pd.DataFrame(rows).sort_values("depth_m")
    out["method"] = "mondrian(depth)"
    return out


def mondrian_aci(cal: pd.DataFrame, test: pd.DataFrame, alpha: float, gamma: float = 0.03) -> pd.DataFrame:
    """METHOD A — depth-adaptive conformal: Mondrian-by-depth + ACI drift adaptation.

    Per depth band, run Adaptive Conformal Inference (Gibbs & Candes 2021) online over the
    date-ordered test stream, using the band's calibration |residual| distribution as the score.
    The effective miscoverage alpha_t is nudged each step toward the target, so the interval
    widens when the (warming-driven) deep band starts missing. Combines Mondrian's depth
    conditioning with ACI's drift tracking — neither alone does both.
    """
    band_scores = {b: np.sort(g["abs_resid"].values) for b, g in cal.groupby("band")}
    rows = []
    for depth, g in test.groupby("depth_m"):
        b = g["band"].iloc[0]
        gg = g.sort_values("date")
        e = np.abs(gg["y"].values - gg["y_pred"].values)
        cov, hw = aci_stream(band_scores.get(b, np.array([])), e, alpha, gamma)
        rows.append({"depth_m": depth, "band": b, "coverage": float(cov.mean()),
                     "mean_width": float(np.mean(2 * hw)), "n_test": len(gg)})
    out = pd.DataFrame(rows).sort_values("depth_m")
    out["method"] = "mondrian+ACI(depth-adaptive)"
    return out


def depth_normalized_conformal(cal: pd.DataFrame, test: pd.DataFrame, alpha: float,
                               aci: bool = False, gamma: float = 0.03) -> pd.DataFrame:
    """METHOD B — continuous depth-normalized conformal (optionally + ACI).

    Generalizes Mondrian's discrete bands to a smooth depth scale s(depth): normalize each
    nonconformity score by s(depth), take ONE pooled conformal quantile on the normalized
    scores, then rescale per test depth -> interval half-width = Q * s(depth). No bins.
    With aci=True, ACI adapts the pooled miscoverage online over the date-ordered test stream
    (A's drift term, carried over unchanged). s(depth) = per-depth calibration residual std,
    with a global floor so sparse depths borrow strength.
    """
    cal = cal.copy()
    scale = cal.groupby("depth_m")["abs_resid"].std()
    gscale = float(cal["abs_resid"].std()) or 1.0
    def s_of(d):
        v = scale.get(d, np.nan)
        return float(v) if (v == v and v > 1e-6) else gscale
    u = np.sort((cal["abs_resid"] / cal["depth_m"].map(s_of)).values)
    umax = u.max() if len(u) else np.inf

    if not aci:
        Q = conformal_quantile(u, alpha)
        rows = []
        for depth, g in test.groupby("depth_m"):
            hw = Q * s_of(depth)
            cov, w = coverage_and_width(g["y"].values, g["y_pred"].values, hw)
            rows.append({"depth_m": depth, "band": g["band"].iloc[0], "coverage": cov,
                         "mean_width": w, "n_test": len(g)})
        out = pd.DataFrame(rows).sort_values("depth_m"); out["method"] = "depth-normalized"
        return out

    tt = test.sort_values("date")
    y, yp, dep = tt["y"].values, tt["y_pred"].values, tt["depth_m"].values
    band, a_t = tt["band"].values, alpha
    rec = []
    for i in range(len(tt)):
        lvl = min(max(1.0 - a_t, 0.0), 1.0)
        qn = umax if lvl >= 1.0 else float(np.quantile(u, lvl, method="higher"))
        hw = qn * s_of(dep[i])
        c = (y[i] >= yp[i] - hw) and (y[i] <= yp[i] + hw)
        rec.append((dep[i], band[i], c, 2 * hw))
        a_t = a_t + gamma * (alpha - (0 if c else 1))
    d = pd.DataFrame(rec, columns=["depth_m", "band", "cov", "wid"])
    out = (d.groupby(["depth_m", "band"]).agg(coverage=("cov", "mean"), mean_width=("wid", "mean"),
                                              n_test=("cov", "size")).reset_index().sort_values("depth_m"))
    out["method"] = "depth-normalized+ACI"
    return out


def mondrian_ar1_whitened(cal: pd.DataFrame, test: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Robustness variant: per-band conformal quantile computed on AR(1)-whitened calibration
    residuals (restores exchangeability under temporal autocorrelation, after Berk 2025).
    The whitening rescales the residual spread; we map the whitened quantile back to the raw
    scale via the ratio of raw-to-whitened std so the interval stays interpretable."""
    band_Q, band_rho = {}, {}
    for b, g in cal.sort_values("date").groupby("band"):
        r = g["abs_resid_signed"].values
        rho = ar1_rho(r)
        w = ar1_whiten(r, rho)
        # quantile on |whitened|, rescaled to raw residual scale
        raw_sd = np.nanstd(r) or 1.0
        wht_sd = np.nanstd(w) or 1.0
        Qw = conformal_quantile(np.abs(w), alpha) * (raw_sd / wht_sd)
        band_Q[b] = Qw
        band_rho[b] = rho
    rows = []
    for depth, g in test.groupby("depth_m"):
        b = g["band"].iloc[0]
        Q = band_Q.get(b, np.nan)
        cov, wdt = coverage_and_width(g["y"].values, g["y_pred"].values, Q)
        rows.append({"depth_m": depth, "band": b, "coverage": cov, "mean_width": wdt,
                     "half_width": Q, "cal_resid_ar1_rho": band_rho.get(b, np.nan),
                     "n_test": len(g)})
    out = pd.DataFrame(rows).sort_values("depth_m")
    out["method"] = "mondrian+AR1whiten"
    return out
