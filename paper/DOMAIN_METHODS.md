# Domain Methods — Permafrost / Ground-Temperature ML, ranked

> Every ML (or hybrid physics-ML) method for permafrost ground temperature / active layer / thaw /
> infrastructure risk that we found. Ranked by **relevance to our contribution** (borehole temperature
> forecasting + whether they leave the *calibration* gap open) first, then **recency**.
> Provenance: `primary` = confirmed on the source; `secondary` = search-summary lead (verify if cited).
> **The through-line: not one of these reports calibrated, depth-resolved coverage.** That is our gap.

## Ranked table

| # | Method / paper | Year | Target | Architecture | UQ? | Depth axis? | Effectiveness (reported) | Prov. |
|---|---|---|---|---|---|---|---|---|
| 1 | **PI-LSTM** — Liu et al., *Cold Regions Sci & Tech* 216:104009 | 2023 | **Borehole ground-temp profile** | Physics (GIPL2) pretrain + LSTM fine-tune | ❌ point | partial (predicts profile, not calibrated) | +27% vs LSTM, +69% vs GIPL2 | primary |
| 2 | **Chance et al.** — *Earth Sci Informatics* 17:6055–6073 | 2024 | Subsurface soil temp (Alaska) | kNN/RF/GBM/SVR/NN + MERRA-2 | ❌ point | error differs by depth, not calibrated | GB RMSE **0.13 °C** deep / RF **1.10 °C** active layer | primary |
| 3 | **Gou et al. digital twin** — *JGR Earth Surface* 131:e2025JF008787 | 2026 | Ground-temp field + soil properties (road embankment) | Differentiable physics + NN + DTS | ⚠ property inference, not coverage | continuous (physics), not calibrated | accurate spatiotemporal reconstruction | primary |
| 4 | **Bayesian inverse thermal state** — *The Cryosphere* 17:3505 | 2023 | Permafrost thermal state | Physics-based Bayesian inversion | ✅ **Bayesian UQ** (but physics, slow, site-specific) | no | posterior on thermal params | secondary |
| 5 | **Pan-Arctic hybrid physics-ML risk** — arXiv:2510.02189 | 2025 | Permafrost fraction + infra risk (spatial) | Stacked RF+HGB+ElasticNet + physical sensitivity, 2.9M obs | ⚠ spatial uncertainty maps, no coverage | no (spatial, not depth) | R²=0.980, RMSE 5.01 pp | primary |
| 6 | **ALT across Alaska (ML + Stefan)** — *Sci Reports* s41598-025-26586-w | 2025 | Active layer thickness | ML + physically-based Stefan + CMIP6 | ❌ projections | no | ALT projections to 2100 (SSP2-4.5/5-8.5) | secondary |
| 7 | **RF-LSTM + Stefan regularization** — (ALT, physics-informed) | 2025 | Active layer thickness | Hybrid RF-LSTM + Stefan loss | ❌ point | no | ~15% RMSE reduction vs baseline | secondary |
| 8 | **Tri-hybrid RF-LSTM-XGBoost** — Qinghai-Xizang highway (S0165232X24002222) | 2024 | Permafrost table depth | RF + LSTM + XGBoost ensemble | ❌ point | no | outperforms single models | secondary |
| 9 | **ML climate-memory in degradation** — *J Hydrology* (S0022169425016129) | 2025 | Permafrost degradation drivers | ML + lagged-feature attribution | ❌ point | no | identifies multi-year precip/temp memory | secondary |
| 10 | **LSTM vs CNN vs RF thaw depth** — QTP (S0048969722029837) | 2022 | Permafrost thaw depth | Comparative LSTM/CNN/RF | ❌ point | no | LSTM best for interannual thaw depth | secondary |
| 11 | **MT-InSAR + LSTM** — Tuotuo River basin | 2023 | Surface deformation / degradation | InSAR + LSTM | ❌ point | no (surface) | deformation tracking | secondary |
| 12 | **ML permafrost degradation review** — *AI in Civil Eng* 10.1007/s43503-025-00080-8 | 2025 | (survey) | — | flags UQ as *emerging/open* | — | crowding/gap evidence | secondary |

## Reading of the ranking

- **Tier 1 (our direct neighbours, #1–3):** forecast/reconstruct *borehole/subsurface temperature*.
  All are **point/accuracy** work — the exact niche we extend with calibration. #1 PI-LSTM is the
  closest and most-cited; #2 Chance gives our RMSE anchor; #3 Gou is the newest and the "why-physics
  is slow/site-specific" foil.
- **Tier 2 (UQ exists but not ours, #4–5):** #4 does real UQ but physics-Bayesian (slow, site-bound);
  #5 does spatial uncertainty maps but no coverage and no depth. Together they let us say "UQ exists,
  but not *distribution-free, depth-resolved, fast* UQ."
- **Tier 3 (adjacent targets, #6–12):** ALT, thaw depth, degradation, deformation, risk — related but
  a different predictand. Cite 1–2 to show breadth; none touches calibrated depth-resolved coverage.

## Recency snapshot
2026: #3 · 2025: #5, #6, #7, #9, #12 · 2024: #2, #8 · 2023: #1, #4, #11 · 2022: #10.
The field is **actively moving toward hybrid physics-ML and infrastructure risk** — but the
**calibration/coverage axis is still empty**, and the newest work (#5, 2025) explicitly leaves it open.

## Verify-before-cite queue (Phase-3 style, for the ones we'd actually reference)
- #5 arXiv:2510.02189 — confirm authors/venue status (currently primary via abstract fetch).
- #6 Sci Reports s41598-025-26586-w — confirm it's ALT (not ground temp) + authors.
- #7/#8/#9/#10 — resolve exact citation + lead author before use (secondary today).
