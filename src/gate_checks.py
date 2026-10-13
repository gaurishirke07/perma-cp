"""
Gate 1 + Gate 2 checks for the permafrost depth-conditional conformal project.
Source: PERMOS Database 2025 (DOI:10.13093/permos-2025-01)
Chosen site: Stockhorn (STO_6000), borehole_id=7, Upper Valais, crest, bedrock (schists)
Depths: 0.3m - 98.3m, 25 sensors, daily resolution, 2001-2024 (23yr)

Run from the directory containing bht_day_*.csv, borehole_*.csv, site_*.csv
(i.e. PERMOS-DOI-2025-01/Data/)
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(".")  # run from the raw PERMOS Data/ dir (holds bht_day_*.csv)
OUT_DIR = Path(__file__).resolve().parents[1]  # repo root: writes to results/ and data/
BOREHOLE_ID = 7
COMMON_START = "2002-01-01"   # window where nearly all depths have data
COMMON_END = "2023-12-31"
MISSING_THRESHOLD = 0.20      # 20% max missing per depth

# depth bands per 02_IMPLEMENTATION_SPEC.md: shallow <1m, mid 1-5m, deep >=5m
# for this site's genuine ZAA (~68m observed), also track a "very deep / trend" band
def depth_band(depth):
    if depth < 1:
        return "shallow"
    elif depth < 5:
        return "mid"
    elif depth < 20:
        return "deep_seasonal_tail"
    else:
        return "deep_trend"


def load_borehole(bht_day_csv, borehole_id):
    df = pd.read_csv(bht_day_csv, parse_dates=["time"])
    d = df[df.borehole_id == borehole_id].copy()
    d["band"] = d["depth"].apply(depth_band)
    return d


def check1_granularity(d):
    """Median gap in days per depth. Expect 1.0 for daily data."""
    rows = []
    for depth, g in d.groupby("depth"):
        g = g.sort_values("time")
        gap = g["time"].diff().dt.days.dropna().median()
        rows.append({"depth": depth, "median_gap_days": gap, "n_rows": len(g)})
    out = pd.DataFrame(rows).sort_values("depth")
    bad = out[out.median_gap_days > 1.0]
    status = "PASS" if bad.empty else f"FAIL — {len(bad)} depths with gap > 1 day"
    return out, status


def check2_missing(d, start=COMMON_START, end=COMMON_END):
    """% missing per depth over the common coverage window."""
    sub = d[(d.time >= start) & (d.time <= end)]
    full_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    rows = []
    for depth, g in sub.groupby("depth"):
        n = len(g)
        miss_pct = 100 * (1 - n / full_days)
        rows.append({"depth": depth, "n_rows": n, "pct_missing": round(miss_pct, 1)})
    out = pd.DataFrame(rows).sort_values("depth")
    bad = out[out.pct_missing > MISSING_THRESHOLD * 100]
    status = "PASS" if bad.empty else f"FLAG — {len(bad)} depths over {MISSING_THRESHOLD*100:.0f}% missing"
    return out, status


def check3_deep_trend(d, deep_depth, start=COMMON_START, end=COMMON_END):
    """Linear slope of annual mean temp at the deepest sensor. Want slope > 0."""
    sub = d[(d.time >= start) & (d.time <= end) & (d.depth == deep_depth)].copy()
    sub["year"] = sub.time.dt.year
    annual = sub.groupby("year")["temp"].mean()
    annual = annual.iloc[1:-1]  # drop partial edge years
    slope, intercept = np.polyfit(annual.index.values.astype(float), annual.values, 1)
    status = "PASS" if slope > 0 else "FAIL — no warming trend at deep sensor"
    return annual, slope, status


def check4_variance_vs_depth(d, depths, start=COMMON_START, end=COMMON_END):
    """Std dev of temp per depth. Want high shallow, decaying with depth."""
    sub = d[(d.time >= start) & (d.time <= end) & (d.depth.isin(depths))]
    rows = []
    for depth, g in sub.groupby("depth"):
        rows.append({
            "depth": depth,
            "std": g["temp"].std(),
            "range": g["temp"].max() - g["temp"].min(),
            "mean": g["temp"].mean(),
        })
    out = pd.DataFrame(rows).sort_values("depth")
    # PASS if monotonic-ish decay from shallow to mid-depth (allow uptick at very deep = trend regime)
    decaying = out["std"].iloc[:-1].is_monotonic_decreasing
    status = "PASS" if decaying else "CHECK MANUALLY — variance not monotonically decaying"
    return out, status


def build_canonical_parquet(d, site_id="STO_6000", out_path=None):
    """Long format: site_id | date | depth_m | temp_c"""
    long_df = d[["time", "depth", "temp"]].rename(
        columns={"time": "date", "depth": "depth_m", "temp": "temp_c"}
    )
    long_df.insert(0, "site_id", site_id)
    if out_path:
        long_df.to_parquet(out_path, index=False)
    return long_df


if __name__ == "__main__":
    bht_day = list(DATA_DIR.glob("bht_day_*.csv"))[0]
    d = load_borehole(bht_day, BOREHOLE_ID)

    print("=== CHECK 1: granularity ===")
    g1, s1 = check1_granularity(d)
    print(g1.to_string(), f"\nstatus: {s1}\n")

    print("=== CHECK 2: missing % ===")
    g2, s2 = check2_missing(d)
    print(g2.to_string(), f"\nstatus: {s2}\n")

    print("=== CHECK 3: deep sensor trend (98.3m) ===")
    g3, slope, s3 = check3_deep_trend(d, deep_depth=98.3)
    print(g3.round(3).to_string(), f"\nslope: {slope:+.4f} degC/yr\nstatus: {s3}\n")

    print("=== CHECK 4: variance vs depth ===")
    check_depths = [0.3, 0.8, 1.8, 3.3, 5.3, 9.3, 18.3, 28.3, 48.3, 68.3, 98.3]
    g4, s4 = check4_variance_vs_depth(d, check_depths)
    print(g4.round(4).to_string(), f"\nstatus: {s4}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results").mkdir(exist_ok=True)
    g1.to_csv(OUT_DIR / "results" / "gate2_granularity.csv", index=False)
    g2.to_csv(OUT_DIR / "results" / "gate2_missing.csv", index=False)
    g3.to_csv(OUT_DIR / "results" / "gate2_deep_trend.csv")
    g4.to_csv(OUT_DIR / "results" / "gate2_variance_vs_depth.csv", index=False)

    (OUT_DIR / "data").mkdir(exist_ok=True)
    long_df = build_canonical_parquet(d, out_path=OUT_DIR / "data" / "borehole_long.parquet")
    print(f"canonical parquet saved: {len(long_df)} rows")
