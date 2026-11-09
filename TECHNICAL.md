# Technical notes

How every number in `outputs/results.json` is produced. The code (`rul.py`) is the authority; this file explains it.

## 1. Questions

1. **SOH-based target (main result).** For a cell never seen in training, can per-cycle electrical features predict the number of cycles left until the cell's state of health (SOH) falls below 80%, better than the cycle counter alone?
2. **Benchmark audit.** The processed dataset ships with a remaining-useful-life (RUL) label. What does that label measure, and how much of the accuracy reported on it comes from the cycle counter?

## 2. Data — `download_data.py`, `rul.load`

- Hawaii Natural Energy Institute (HNEI) cycling data of 14 commercial NMC-LCO 18650 cells, one temperature, one current (Devie, Baure & Dubarry, Energies 11(5), 1031, 2018).
- Processed copy from [ignavinuales/Battery_RUL_Prediction](https://github.com/ignavinuales/Battery_RUL_Prediction), file `Datasets/HNEI_Processed/Final Database.csv`, pinned to commit `10320b0` and checked by SHA-256. The CSV stays out of the repository because the source has no licence file.
- 15,064 rows, one per cycle: `Cycle_Index`, eight features computed by the dataset authors from voltage–time curves (discharge time, time for the 3.6→3.4 V decrement, maximum discharge voltage, minimum charge voltage, time at 4.15 V, constant-current time, charging time, total time) and the label `RUL`.

**Cell reconstruction.** The file concatenates the cells with no cell column. `Cycle_Index` restarts at 1 for each new cell, so the cell id is the cumulative count of drops in `Cycle_Index`: 13 resets, 14 cells, 1,051–1,081 rows per cell. 25–61 cycle numbers per cell are absent from the source file.

**Cleaning.** Two sanity rules, applied after reconstruction:

| Rule | Reason | Rows dropped |
|---|---|---|
| Total cycle time > 24 h | median cycle 2.76 h; longer rows are logging artefacts or long rests | 182 |
| Min. charge voltage > max. discharge voltage | rows start discharge at a lower voltage and discharge for less time: partial cycles after an incomplete charge | 415 |

597 rows dropped (4.0%), 14,467 kept. 355 of the 415 voltage-rule rows (86%) fall in the last 40% of the test.

## 3. SOH proxy — `rul.soh`, `rul.soh_summary`

- SOH proxy per cycle = discharge time / the cell's median discharge time over cycles 10–60. At constant discharge current, discharge time is proportional to the charge delivered at that rate, so the proxy reflects capacity loss and resistance growth together.
- Smoothed with a centred 21-row rolling median (uses later cycles: a retrospective curve, used only to define the target). A trailing version (past cycles only) is computed for the sensitivity check.
- End of life (EoL) = first cycle where the smoothed proxy falls below 80%: cycles 345–419 (mean 381.4, SD 26.2, CV 6.9%).
- End-of-test SOH (median of the unsmoothed proxy over the last 30 cycles): 35.3–45.9%.
- The first ~20 cycles are start-up artefacts (proxy up to about 270%); `soh.png` draws from cycle 20.

## 4. Models and validation

- **LightGBM** (gradient-boosted trees), library defaults: 100 trees, learning rate 0.1, 31 leaves, `random_state = 42`. No tuning, so no validation set is needed and no test information enters model choice.
- **Ridge** (linear regression with an L2 penalty), `alpha = 1.0`, on standardized features.
- **Train mean**: the mean target of the training cells, for every row.
- **Leave-one-cell-out**: train on 13 cells, predict the 14th, rotate through all 14; predictions are pooled before scoring. A random 80/20 split is used only in protocols A–C, to show its effect.
- **Metrics**: MAE and RMSE in cycles; R² is reported in `results.json` and kept out of the conclusions, because the wide range of the targets makes it high for every non-trivial model.

## 5. Main result: RUL to the SOH-defined end of life — `rul.soh_target_study`

**Target.** RUL_SOH = EoL cycle − `Cycle_Index`, for each cell's rows up to and including its EoL cycle: 5,070 rows, 337–394 per cell.

**All nine variants (leave-one-cell-out):**

| | features | model | MAE | RMSE |
|---|---|---|---|---|
| S1 | none | train mean | 93.4 | 108.7 |
| S2 | `Cycle_Index` | Ridge | 22.2 | 27.2 |
| S3 | `Cycle_Index` | LightGBM | 21.2 | 26.7 |
| S4 | electrical (8) | Ridge | 30.1 | 58.3 |
| S5 | electrical (8) | LightGBM | 21.2 | 26.3 |
| S6 | electrical + `Cycle_Index` | LightGBM | 20.5 | 25.6 |
| S7 | electrical without discharge time (7) | LightGBM | 24.7 | 31.9 |
| S8 | S7 + `Cycle_Index` | LightGBM | 22.7 | 28.2 |
| S9 | charge-side (min. charge voltage, time at 4.15 V, constant-current time, charging time) + `Cycle_Index` | LightGBM | 21.3 | 25.6 |

**Each model against the counter (S2), by distance to EoL.** Gain = MAE(S2) − MAE(model), positive when the model is better. Interval: cell-level bootstrap (section 7). p: Wilcoxon signed-rank test on the 14 per-cell MAE differences.

| | overall (5,070 rows) | ≤ 100 cycles left (1,243 rows) | > 200 cycles left (2,443 rows) |
|---|---|---|---|
| S5 | +1.1 [−4.9, 7.6], 7/14, p 0.67 | +7.8 [−0.3, 15.6], 9/14, p 0.12 | −3.3 [−7.8, 2.1], 5/14, p 0.15 |
| **S6** | +1.8 [−4.2, 8.5], 6/14, p 0.76 | **+10.0 [2.3, 17.9], 9/14, p 0.058** | −3.7 [−8.3, 1.5], 3/14, p 0.10 |
| S7 | −2.4 [−8.2, 3.4], 5/14, p 0.43 | +4.6 [−5.8, 13.7], 8/14, p 0.39 | −8.4 [−12.9, −3.6], 1/14, p 0.011 |
| S8 | −0.4 [−6.0, 5.5], 6/14, p 0.95 | +8.3 [−0.2, 16.6], 8/14, p 0.14 | −6.3 [−10.0, −2.6], 3/14, p 0.009 |
| S9 | +1.0 [−4.9, 7.3], 7/14, p 0.86 | +9.2 [0.7, 17.5], 9/14, p 0.12 | −4.2 [−7.9, −0.2], 5/14, p 0.12 |

Cells: number of cells where the model has the lower MAE. Near EoL, S2's MAE is 23.4 and S6's is 13.4.

**Error by current SOH (smoothed proxy), plotted in `outputs/soh_target.png`:**

| current SOH | rows | S2 | S5 | S6 |
|---|---|---|---|---|
| 95%+ | 1,467 | 22.1 | 26.2 | 26.5 |
| 90–95% | 1,124 | 22.0 | 23.0 | 23.7 |
| 85–90% | 1,253 | 22.1 | 18.8 | 17.4 |
| 80–85% | 1,212 | 22.8 | 15.9 | 13.5 |

The 14 EoL rows (proxy just below 80%) fall outside the bands. `results.json` also holds the error by cycles left (0–50 … 301+).

**Sensitivity of S6 vs S2:**

| setting | EoL cycles | overall gain | ≤ 100 cycles left |
|---|---|---|---|
| threshold 75% | 424–480 | +6.1 [1.2, 10.8], 12/14, p 0.042 | +9.6 [4.3, 14.9], 12/14, p 0.011 |
| **threshold 80% (main)** | 345–419 | +1.8 [−4.2, 8.5], 6/14, p 0.76 | +10.0 [2.3, 17.9], 9/14, p 0.058 |
| threshold 85% | 258–323 | −0.6 [−5.9, 4.1], 8/14, p 0.90 | +3.8 [−3.5, 10.6], 9/14, p 0.36 |
| trailing smoothing, 80% | 355–431 | +2.8 [−3.7, 10.3], 6/14, p 0.67 | +10.4 [3.0, 17.8], 11/14, p 0.030 |

**Placebo.** Each cell receives another cell's EoL (10 random shuffles in which no cell keeps its own), and S2 and S6 are refitted. Near EoL, S2 scores 23.3–23.5 and S6 20.1–27.6, against 13.4 with the true EoL. The near-EoL advantage is tied to each cell's real end of life; binning on the target does not produce it.

**Reading:**
1. Overall, every electrical variant ties the cycle counter (all intervals include zero). With 14 near-identical cells, one cycle's measurements say little about how fast a cell will degrade hundreds of cycles later.
2. Near EoL, adding electrical features to the counter halves the error (23.4 → 13.4). The bootstrap interval excludes zero; the Wilcoxon test is borderline (p 0.058).
3. Far from EoL, the electrical variants are worse than the counter, significantly so for S7, S8 and S9.
4. The effect survives the placebo, trailing smoothing and charge-side features only (S9), so it rests on more than the discharge-time feature. Discharge time at constant current is a direct capacity measurement, so its approach to the 80% line is legitimate information for this target.
5. It depends on the threshold: clear at 75%, within noise at 85%.

**Status: exploratory.** The near-EoL comparison (S6 vs S2, ≤ 100 cycles) was selected after an exploratory run that examined several variants and bins; with a Bonferroni correction for about 12 such comparisons it would not reach significance. The sweep, trailing smoothing and placebo were added afterwards to probe it, following an independent audit.

## 6. Benchmark audit: the dataset's RUL label — `rul.main`

**The label.** In every row, RUL = (last cycle of the cell) − `Cycle_Index`, exactly (checked by a unit test). The last cycle is 1,103–1,134 for all cells (mean 1,110.4, SD 7.4, CV 0.7%), while end-of-test discharge time varies 8.5% (CV) across cells: the label counts down to the end of the test.

**Eight protocols:**

| | split | features | model | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| A | random 80/20 | electrical + `Cycle_Index` | LightGBM | 2.8 | 4.1 | 1.000 |
| B | random 80/20 | electrical | LightGBM | 17.4 | 26.6 | 0.993 |
| C | random 80/20 | `Cycle_Index` | LightGBM | 4.4 | 7.1 | 1.000 |
| D | leave one cell out | electrical | LightGBM | 38.1 | 56.3 | 0.969 |
| E | leave one cell out | electrical | Ridge | 44.2 | 59.8 | 0.965 |
| F | leave one cell out | none | train mean | 277.2 | 320.6 | −0.000 |
| G | leave one cell out | `Cycle_Index` | Ridge | 4.8 | 7.6 | 0.999 |
| H | leave one cell out | electrical + `Cycle_Index` | LightGBM | 5.8 | 8.6 | 0.999 |

**Error of D by life stage** (cycle / the cell's last retained cycle):

| life stage | rows | MAE | mean true RUL | MAE / mean RUL |
|---|---|---|---|---|
| 0–20% | 3,006 | 14.8 | 997.9 | 1.5% |
| 20–40% | 2,807 | 21.1 | 782.1 | 2.7% |
| 40–60% | 2,995 | 31.4 | 556.5 | 5.6% |
| 60–80% | 2,847 | 61.2 | 335.3 | 18.3% |
| 80–100% | 2,812 | 63.8 | 110.0 | 58.0% |

**D vs E:** MAE difference 6.1 cycles, 95% interval [0.7, 10.8], D lower in 11 of 14 cells.

**D bias vs health:** each cell's mean signed error (prediction − true RUL) against its end-of-test discharge time (median of the last 30 cycles): Spearman ρ = 0.87 over 14 cells.

**Reading:**
1. The counter generalizes across cells (C 4.4 → G 4.8): every cell ends near cycle 1,110.
2. The random split flatters only the models without the counter (B 17.4 → D 38.1, a factor of 2.2).
3. Electrical features add nothing to the counter (H 5.8 vs G 4.8).
4. D's per-cell errors track health: healthier cells are predicted to have more cycles left. Part of D's error is disagreement with a label that ignores health; this motivates the SOH-based target of section 5.
5. Error grows through life because the cells' health curves fan out: the same discharge time maps to increasingly different cycle numbers in different cells.

## 7. Statistics

- **Cell-level bootstrap.** Draw 14 cells with replacement, recompute the MAE difference as total absolute error over total rows, repeat 2,000 times (`seed = 42`), take the 2.5th and 97.5th percentiles. With 14 clusters the percentile bootstrap tends to give intervals that are too narrow; the Wilcoxon test and the per-cell counts are reported alongside it.
- **Wilcoxon signed-rank test** on the 14 per-cell MAE differences, two-sided.
- **Spearman correlation**: Pearson correlation of the ranks.

## 8. Limits

- 14 cells, one chemistry, one temperature, one current: every between-cell statistic rests on 14 points.
- The SOH proxy mixes capacity loss and resistance growth; separating them needs a slow reference discharge or impedance data, which this dataset lacks.
- The features come from the dataset authors; they were kept as published.
- Per-cycle regression: each row is predicted from one cycle's measurements, with no cycle history.
- The voltage-rule filter removes mostly late-life cycles.
- The SOH-target result is exploratory (section 5) and threshold-dependent.

Next steps: features from the cycle history (trends over recent cycles), early-life prediction of the SOH-defined end of life, and more cells, chemistries and conditions.

## 9. Tests and CI

- `tests/test_pipeline.py` (9 tests): cell reconstruction; the dataset label is the countdown to each cell's last cycle; exact cleaning counts; life fraction within [0, 1]; a recording fake model confirms that no leave-one-cell-out fold trains on its held-out cell; electrical features exclude the counter and the label; the SOH proxy is 100% at its reference cycles and every cell reaches 80%; the SOH target counts down to each cell's SOH-defined end of life; the placebo shuffle moves every cell.
- `.github/workflows/ci.yml` runs on every push and pull request, on Python 3.12 with the pinned versions in `requirements.txt`: `ruff check`, `ruff format --check`, data download, `pytest`, the full pipeline, and a comparison that fails if any value in `outputs/results.json` differs from the committed copy.
- `results.json` is written with LF line endings on every operating system.

## 10. History

- **Sep 2026:** first version; its folder was lost.
- **27 Oct 2026:** rebuilt with LightGBM defaults. An audit added the counter-only control (G), which showed the dataset label is a countdown to the end of the test and corrected the first version's reading of the A–D gap; added the SOH proxy.
- **9 Nov 2026:** SOH-based target study (section 5), checked by an independent audit that reproduced every number and led to the threshold sweep, trailing-smoothing check, placebo test, charge-side variant and Wilcoxon tests reported here.
