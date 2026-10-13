"""
make_figures.py -- the single, consistent generator for every data-chart figure.

Reads the committed result CSVs and writes all figures in ONE visual style (shared fonts, palette,
helpers). Clean descriptive titles (no round codes). Pure plotting (no model fitting). This is the
single figure generator for the repository; the drivers only write their result CSVs.
Run after the eval drivers. Designed to stay legible at ~half column width.

Figure index (-> paper location):
  coverage_adaptive_bands  main result (paper Fig 3)      | coverage_continuous   Method B (App)
  coverage_all_by_landform landform breakdown (App)       | reliability_sweep     reliability (App)
  robustness: band_sensitivity, gamma_sweep, seed_robustness, missing_robustness, baselines,
  horizon, cqr, per_sensor_coverage, temporal_splits
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10, "figure.dpi": 150,
})
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"; FIG = ROOT / "figures"
NOM = 0.90
SHORT = {"shallow(<1m)": "shallow", "mid(1-5m)": "mid", "deep_seasonal(5-20m)": "deep-seas",
         "deep_trend(>=20m)": "deep-trend", "shallow": "shallow", "mid": "mid",
         "deep_seasonal": "deep-seas", "deep_trend": "deep-trend"}
BORDER = ["shallow", "mid", "deep-seas", "deep-trend"]
PAL = ["#c1121f", "#2a9d8f", "#e9c46a", "#5a7d9a", "#8e7dbe", "#6a994e"]
# consistent colour per named method across every figure
VCOL = {"global": "#c1121f", "mondrian": "#2a9d8f", "Mondrian": "#2a9d8f",
        "mondrian+ACI": "#e9c46a", "PermaCP": "#e9c46a",
        "A: mondrian+ACI": "#e9c46a", "B: depth-norm": "#8ab17d", "B+ACI: depth-norm+drift": "#5a7d9a"}


def _bc(df):
    df = df.copy(); df["b"] = df["band"].map(SHORT); return df


def cov_bars(piv, title, fname, ylim=(0.70, 1.0), labels=True, colors=None, ylabel="empirical coverage",
             nominal=True, logy=False, fmt="%.2f"):
    piv = piv.reindex(columns=BORDER); groups = list(piv.index); n = len(groups)
    fig, ax = plt.subplots(figsize=(5.4, 3.6)); x = np.arange(len(BORDER)); w = 0.82 / max(n, 1)
    for i, g in enumerate(groups):
        col = (colors or {}).get(str(g), PAL[i % len(PAL)])
        bars = ax.bar(x + (i - (n - 1) / 2) * w, piv.loc[g].values, w, color=col, label=str(g),
                      alpha=.92, edgecolor="white", linewidth=.4)
        if labels and not logy:
            ax.bar_label(bars, fmt=fmt, fontsize=7.2, padding=1.5, rotation=90)
    if nominal:
        ax.axhline(NOM, ls="--", c="k", lw=1.1, zorder=0)
        ax.text(len(BORDER) - 0.5, NOM, " 0.90", va="center", ha="left", fontsize=9)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(BORDER); ax.set_ylabel(ylabel)
    if not logy:
        ax.set_ylim(*ylim)
    ax.set_title(title); ax.legend(loc="lower left", ncol=min(n, 2), framealpha=.9, handlelength=1.2)
    fig.tight_layout(); fig.savefig(FIG / fname, bbox_inches="tight"); plt.close(fig); print("saved", fname)


# ---------- core paper figures (from CSVs) ----------
def fig_adaptive_bands():
    """Paper Fig 3: 2-panel (coverage | width) across 25 boreholes, from committed CSVs; no model re-run.
    Left bars = cross-site mean coverage; whiskers = cross-site BOOTSTRAP CI (coverage_ci.csv, site_ci) --
    the honest uncertainty of the mean (borehole series are autocorrelated/clustered, so a naive SEM would
    be too tight). Bars are unchanged from before; only the whiskers switch from SD to CI."""
    bands = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]
    variants = [("global", "global", "global"), ("mondrian", "Mondrian", "mondrian"),
                ("mondrian+ACI", "PermaCP", "A(Mondrian+ACI)")]
    d = pd.read_csv(RES / "coverage_adaptive_bands.csv")
    ci = pd.read_csv(RES / "coverage_ci.csv")   # cross-site bootstrap CI (Round 3): columns site_ci_lo/hi
    x = np.arange(len(bands)); w = 0.26
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, (v, lab, cm) in enumerate(variants):
        sub = d[d.variant == v]
        mean = np.array([sub[sub.band == b].coverage.mean() for b in bands])
        cr = ci[ci.method == cm].set_index("band").reindex(bands)
        lo = np.nan_to_num(np.clip(mean - cr["site_ci_lo"].values, 0, None))
        hi = np.nan_to_num(np.clip(cr["site_ci_hi"].values - mean, 0, None))   # CI brackets the mean
        ax1.bar(x + (i - 1) * w, mean, w, yerr=[lo, hi], capsize=3, color=VCOL[v],
                label=lab, alpha=.9, edgecolor="white", linewidth=.4)
    ax1.axhline(NOM, ls="--", c="k", lw=1, label=f"nominal {NOM:.0%}")
    ax1.set_xticks(x); ax1.set_xticklabels(BORDER, fontsize=9)
    ax1.set_ylabel("empirical coverage"); ax1.set_ylim(0, 1.0)
    ax1.set_title("Coverage (whiskers: 95% cross-site bootstrap CI)", fontsize=11); ax1.legend(fontsize=8, loc="lower right")
    for i, (v, lab, _) in enumerate(variants):
        sub = d[d.variant == v]
        mean = [sub[sub.band == b].width.mean() for b in bands]
        ax2.bar(x + (i - 1) * w, mean, w, color=VCOL[v], label=lab, alpha=.9, edgecolor="white", linewidth=.4)
    ax2.set_xticks(x); ax2.set_xticklabels(BORDER, fontsize=9)
    ax2.set_yscale("log"); ax2.set_ylabel("mean interval width (°C, log)")
    ax2.set_title("Width cost of adaptation", fontsize=11); ax2.legend(fontsize=8)
    fig.suptitle("Depth-adaptive conformal (Mondrian + ACI) across 25 boreholes", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "coverage_adaptive_bands.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved coverage_adaptive_bands.png")


def fig_landform():
    """App: deep-trend coverage by landform (global vs Mondrian), 15 boreholes with a deep-trend sensor.
    No whiskers (some landforms have n=1, so a cross-site SD is not meaningful) -> matches the submitted figure."""
    d = pd.read_csv(RES / "coverage_all_bands.csv")
    dt = d[d.band == "deep_trend(>=20m)"]
    piv = dt.pivot_table(index="variant", columns="morphology", values="coverage").reindex(["global", "mondrian"])
    lf = list(piv.columns)
    n = dt[dt.variant == "global"].groupby("morphology").site_id.nunique().reindex(lf)
    lab = {"global": "global", "mondrian": "Mondrian"}
    fig, ax = plt.subplots(figsize=(6.4, 3.8)); x = np.arange(len(lf)); w = 0.38
    for i, v in enumerate(["global", "mondrian"]):
        ax.bar(x + (i - .5) * w, piv.loc[v].values, w, color=VCOL[v], label=lab[v], alpha=.92, edgecolor="white", linewidth=.4)
    ax.axhline(NOM, ls="--", c="k", lw=1.1, label=f"nominal {NOM:.0%}")
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={int(n[l])})" for l in lf], fontsize=9)
    ax.set_ylabel("deep-band (≥20m) coverage"); ax.set_ylim(0, 1.05); ax.set_title("Deep-trend coverage by landform")
    ax.legend(framealpha=.9, loc="lower right"); fig.tight_layout(); fig.savefig(FIG / "coverage_all_by_landform.png", bbox_inches="tight")
    plt.close(fig); print("saved coverage_all_by_landform.png")


def fig_continuous():
    """App: Method B (continuous depth-normalized) vs PermaCP, 25 boreholes.
    Grouped bars with cross-site mean +/- SD whiskers, clipped to [0,1] -> matches the submitted figure."""
    bands = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]
    order = ["A: mondrian+ACI", "B: depth-norm", "B+ACI: depth-norm+drift"]
    d = pd.read_csv(RES / "coverage_continuous.csv")
    x = np.arange(len(bands)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, v in enumerate(order):
        sub = d[d.variant == v]
        mean = np.array([sub[sub.band == b].coverage.mean() for b in bands])
        sd = np.nan_to_num(np.array([sub[sub.band == b].coverage.std(ddof=1) for b in bands]))
        lo = np.minimum(sd, mean); hi = np.minimum(sd, 1.0 - mean)   # clip whiskers to the [0,1] coverage bound
        ax.bar(x + (i - 1) * w, mean, w, yerr=[lo, hi], capsize=3, color=VCOL[v], label=v, alpha=.9, edgecolor="white", linewidth=.4)
    ax.axhline(NOM, ls="--", c="k", lw=1, label=f"nominal {NOM:.0%}")
    ax.set_xticks(x); ax.set_xticklabels(BORDER); ax.set_ylabel("empirical coverage"); ax.set_ylim(0, 1.05)
    ax.set_title("Method B: continuous depth-normalized conformal (25 boreholes)")
    ax.legend(loc="lower left", framealpha=.9, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "coverage_continuous.png", bbox_inches="tight"); plt.close(fig); print("saved coverage_continuous.png")


def fig_reliability():
    """App: reliability sweep, empirical vs nominal coverage per depth band (25 boreholes), 2x2.
    Lines = global / Mondrian / PermaCP against the ideal diagonal -> matches the submitted figure."""
    bands = ["shallow(<1m)", "mid(1-5m)", "deep_seasonal(5-20m)", "deep_trend(>=20m)"]
    d = pd.read_csv(RES / "reliability_sweep.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.6), sharex=True, sharey=True)
    for ax, b in zip(axes.ravel(), bands):
        s = d[d.band == b].sort_values("nominal")
        ax.plot([0.45, 1.0], [0.45, 1.0], ls="--", c="k", lw=1, label="ideal")
        ax.plot(s.nominal, s["global"], "o-", c=VCOL["global"], ms=5, label="global")
        ax.plot(s.nominal, s["mondrian"], "^-", c=VCOL["mondrian"], ms=5, label="Mondrian")
        ax.plot(s.nominal, s["A"], "s-", c=VCOL["PermaCP"], ms=5, label="PermaCP")
        ax.set_title(b, fontsize=11); ax.set_xlim(0.45, 1.0); ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=8, loc="upper left")
    axes[1, 0].set_xlabel("nominal coverage"); axes[1, 1].set_xlabel("nominal coverage")
    axes[0, 0].set_ylabel("empirical coverage"); axes[1, 0].set_ylabel("empirical coverage")
    fig.suptitle("Reliability sweep: empirical vs nominal coverage per depth band (25 boreholes)", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "reliability_sweep.png", bbox_inches="tight"); plt.close(fig); print("saved reliability_sweep.png")


# ---------- robustness rounds ----------
def fig_robustness():
    d = _bc(pd.read_csv(RES / "band_sensitivity.csv"))
    cov_bars(d[d.variant == "permacp"].pivot_table(index="edges", columns="b", values="coverage"),
             "PermaCP coverage vs depth-band edges", "band_sensitivity.png", ylim=(0.80, 0.96))

    d = _bc(pd.read_csv(RES / "gamma_sweep.csv"))
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, b in enumerate(BORDER):
        s = d[d.b == b].sort_values("gamma"); ax.plot(s.gamma, s.coverage, "o-", color=PAL[i], label=b, ms=6, lw=1.8)
    ax.axhline(NOM, ls="--", c="k", lw=1.1); ax.axvline(0.03, ls=":", c="grey", lw=1)
    ax.text(0.032, 0.862, "paper γ=0.03", fontsize=9, color="grey")
    ax.set_xlabel("ACI step size γ"); ax.set_ylabel("PermaCP coverage"); ax.set_ylim(0.86, 0.915)
    ax.set_title("PermaCP coverage vs ACI step size γ"); ax.legend(ncol=2, framealpha=.9, handlelength=1.4)
    fig.tight_layout(); fig.savefig(FIG / "gamma_sweep.png", bbox_inches="tight"); plt.close(fig); print("saved gamma_sweep.png")

    d = _bc(pd.read_csv(RES / "seed_robustness.csv"))
    agg = d.groupby("b").coverage.agg(["mean", "std"]).reindex(BORDER)
    fig, ax = plt.subplots(figsize=(5.0, 3.6)); x = np.arange(len(BORDER))
    m = agg["mean"].values; s = np.nan_to_num(agg["std"].values)
    yerr = np.vstack([np.minimum(s, m), np.minimum(s, 1.0 - m)])   # clip whiskers to the [0,1] coverage bound
    bars = ax.bar(x, m, 0.6, yerr=yerr, capsize=5, color=PAL[1], alpha=.9, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", fontsize=8.5, padding=3); ax.axhline(NOM, ls="--", c="k", lw=1.1)
    ax.set_xticks(x); ax.set_xticklabels(BORDER); ax.set_ylabel("PermaCP coverage (mean ± SD, 5 seeds)")
    ax.set_ylim(0.86, 0.915); ax.set_title("PermaCP coverage across random seeds")
    fig.tight_layout(); fig.savefig(FIG / "seed_robustness.png", bbox_inches="tight"); plt.close(fig); print("saved seed_robustness.png")

    d = _bc(pd.read_csv(RES / "missing_robustness.csv"))
    cov_bars(d.pivot_table(index="variant", columns="b", values="coverage"),
             "Coverage: dropping vs keeping missing rows", "missing_robustness.png", ylim=(0.80, 0.96))

    d = _bc(pd.read_csv(RES / "baselines.csv"))
    agg = d.groupby("b")[["rmse_lgbm", "rmse_persistence", "rmse_seasonal_naive"]].mean().reindex(BORDER).T
    agg.index = ["LightGBM", "persistence", "seasonal-naive"]
    cov_bars(agg, "Forecaster vs naive baselines (RMSE)", "baselines.png",
             ylim=(0, agg.values.max() * 1.22), nominal=False, ylabel="RMSE (°C)", fmt="%.2f")

    d = _bc(pd.read_csv(RES / "horizon.csv"))
    piv = d.pivot_table(index="horizon", columns="b", values="coverage"); piv.index = [f"h={h}" for h in piv.index]
    cov_bars(piv, "PermaCP coverage at h=30 vs h=90", "horizon.png", ylim=(0.80, 0.96))

    d = _bc(pd.read_csv(RES / "cqr.csv"))
    order = [v for v in ["abs-Mondrian", "CQR-Mondrian", "abs-PermaCP(paper)", "CQR-PermaCP"] if v in set(d.variant)]
    cov_bars(d.pivot_table(index="variant", columns="b", values="coverage").reindex(order),
             "CQR vs absolute-residual score", "cqr.png", ylim=(0.70, 1.0))

    if (RES / "per_sensor_coverage.csv").exists():
        d = pd.read_csv(RES / "per_sensor_coverage.csv")
        fig, ax = plt.subplots(figsize=(5.6, 3.7)); cmap = {"global": "#c1121f", "mondrian": "#2a9d8f", "permacp": "#5a7d9a"}
        for v, lab in [("global", "global"), ("mondrian", "Mondrian"), ("permacp", "PermaCP")]:
            s = d[d.variant == v]
            ax.scatter(s.depth_m.clip(lower=0.1), s.coverage, s=np.sqrt(s.n_test.clip(upper=400)) + 4,
                       c=cmap[v], alpha=.45, edgecolor="none", label=lab)
        ax.axhline(NOM, ls="--", c="k", lw=1.1); ax.set_xscale("log")
        ax.set_xlabel("sensor depth (m, log)"); ax.set_ylabel("per-sensor coverage"); ax.set_ylim(0, 1.04)
        ax.set_title("Per-sensor coverage (all 25 boreholes)")
        ax.text(0.98, 0.04, "96% of PermaCP sensors\nwithin ±0.05 of 0.90", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=.9))
        ax.legend(loc="lower left", framealpha=.9, markerscale=1.5)
        fig.tight_layout(); fig.savefig(FIG / "per_sensor_coverage.png", bbox_inches="tight"); plt.close(fig); print("saved per_sensor_coverage.png")

    if (RES / "temporal_splits.csv").exists():
        d = _bc(pd.read_csv(RES / "temporal_splits.csv"))
        cov_bars(d.pivot_table(index="fold", columns="b", values="coverage"),
                 "PermaCP coverage across temporal splits", "temporal_splits.png", ylim=(0.80, 0.96))


def main():
    # Paper figures (Fig 3 + App) and all robustness/reviewer figures, from committed CSVs. No model re-run.
    fig_adaptive_bands()   # Fig 3: coverage | width, 2-panel, clipped whiskers
    fig_continuous()       # App: Method B continuous depth-normalized
    fig_landform()         # App: deep-trend coverage by landform
    fig_reliability()      # App: reliability sweep 2x2
    fig_robustness()       # 9 robustness/reviewer figures


if __name__ == "__main__":
    main()
