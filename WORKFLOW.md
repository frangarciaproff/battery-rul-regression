# Workflow

Step-by-step record of how this project was built and how each number in `outputs/results.json` is produced. The code is the authority; this file explains it.

## 0. Question

Can per-cycle electrical features predict a cell's remaining useful life (RUL, cycles left) for a cell the model has never seen? And how much of the accuracy usually reported on this dataset comes from the cycle counter rather than from the electrical signal?

## 1. Data acquisition — `download_data.py`

- Source: Hawaii Natural Energy Institute (HNEI) cycling data of 14 NMC-LCO 18650 cells (Devie, Baure & Dubarry, Energies 11(5), 1031, 2018). Used in the processed form published at [ignavinuales/Battery_RUL_Prediction](https://github.com/ignavinuales/Battery_RUL_Prediction), file `Datasets/HNEI_Processed/Final Database.csv`.
- Pinned to commit `10320b0` (30 Jan 2023) and checked by SHA-256, so every run reads the same bytes.
- The CSV is not committed: the source repository has no licence file.
- Content: 15,064 rows, one per cycle. Columns: `Cycle_Index`, eight features derived from the voltage–time curve (discharge time, time for the 3.6→3.4 V decrement, maximum discharge voltage, minimum charge voltage, time at 4.15 V, constant-current time, charging time, total time) and the target `RUL`.
- The features were computed by the dataset authors, not re-derived from raw traces here.

## 2. Cell reconstruction and target — `rul.load`

The file concatenates the 14 cells with no cell column. `Cycle_Index` restarts at 1 at the start of each cell, so a cell id is the cumulative count of drops in `Cycle_Index`.

- 13 resets → 14 cells, 1,051 to 1,081 rows per cell. 25 to 61 cycle numbers per cell are absent from the source file.
- Without this step, leave-one-cell-out validation is impossible and any split leaks rows of the same cell into train and test.
- Target: in every cell, RUL = (last cycle of the cell) − `Cycle_Index`, exactly (checked in `tests/test_pipeline.py`). The last cycle is 1,103–1,134 (mean 1,110.4, SD 7.4) for all cells, while end-of-test discharge time, a capacity proxy at constant current, varies 8.5% across cells. The cells stopped at about the same cycle count in different states of health: the label counts down to the end of the test, not to a health-defined end of life.

## 3. Cleaning — `rul.load`

Two sanity rules, applied after cell reconstruction:

| Rule | Why | Rows dropped |
|---|---|---|
| Total cycle time > 24 h | The median cycle takes 2.8 h; longer values are logging artefacts or long rests | 182 |
| Minimum charge voltage > maximum discharge voltage | Flagged rows start discharge at a lower voltage and discharge for less time than the rest, consistent with partial cycles after an incomplete charge | 415 |

597 of 15,064 rows dropped (4.0%); 14,467 used. 355 of the 415 voltage-rule rows (86%) fall in the last 40% of life, so late-life error is measured on the cycles that pass the filter. Cycles of 4–24 h remain in the data.

Life fraction, used only for the error breakdown in step 6, is cycle number over the cell's last retained cycle, computed after cleaning.

## 4. Validation protocols — `rul.main`

Eight rows. A, B and C share the same random 80/20 split (`random_state = 42`); D to H use leave-one-cell-out (14 folds: train on 13 cells, predict the 14th, pool the predictions).

| | split | features | model | purpose |
|---|---|---|---|---|
| A | random | electrical + `Cycle_Index` | LightGBM | the setup usually reported on this dataset |
| B | random | electrical | LightGBM | remove the counter, keep the leaky split |
| C | random | `Cycle_Index` | LightGBM | control: no electrical measurement at all |
| D | leave one cell out | electrical | LightGBM | electrical signal on unseen cells |
| E | leave one cell out | electrical | Ridge (standardized, `alpha = 1.0`) | linear baseline for D |
| F | leave one cell out | none | training-set mean | trivial baseline |
| G | leave one cell out | `Cycle_Index` | Ridge (standardized, `alpha = 1.0`) | counter on unseen cells |
| H | leave one cell out | electrical + `Cycle_Index` | LightGBM | does the electrical signal add to the counter? |

LightGBM uses library defaults (100 trees, learning rate 0.1, 31 leaves) with `random_state = 42`, the same in every protocol. No tuning, so no validation set is needed and no test information enters model choice.

## 5. Metrics — `rul.scores`, `rul.compare_cells`, `rul.bias_vs_health`

- MAE and RMSE in cycles, R² on the pooled predictions. For D to H the 14 folds' predictions are pooled before scoring. R² is reported in `results.json` but not used for conclusions: the RUL range makes every non-trivial model score above 0.96.
- D vs E: per-cell MAE, the number of cells where D is lower, and a 95% interval for the MAE difference from 2,000 bootstrap resamples of whole cells (`seed = 42`).
- D bias vs health: each cell's mean signed error (prediction − true RUL) against its end-of-test discharge time (median of its last 30 retained cycles); Spearman correlation over the 14 cells.

## 6. Error by life stage — `rul.by_life_stage`

Protocol D predictions, split into five 20% bins of life fraction. For each bin: row count, MAE, mean true RUL, and MAE / mean true RUL as relative error.

## 6b. State of health — `rul.soh`, `rul.soh_summary`, `rul.plot_soh`

- SOH proxy per cycle: discharge time over the cell's median discharge time in cycles 10–60. At constant discharge current, discharge time is proportional to the capacity delivered at that rate, so the proxy includes resistance growth, not only capacity loss.
- Smoothed with a centred 21-row rolling median. It uses later cycles, so it is a retrospective monitoring curve, not a real-time estimate.
- End of life: first cycle where the smoothed SOH falls below 80%.
- End-of-test SOH: median of the unsmoothed ratio over each cell's last 30 retained cycles.
- `outputs/soh.png` plots the smoothed curves from cycle 20; the first cycles are start-up artefacts (discharge time up to 3× the reference).

## 7. Results (`outputs/results.json`, 27 Oct 2026)

| | MAE | RMSE | R² |
|---|---|---|---|
| A | 2.8 | 4.1 | 1.000 |
| B | 17.4 | 26.6 | 0.993 |
| C | 4.4 | 7.1 | 1.000 |
| G | 4.8 | 7.6 | 0.999 |
| H | 5.8 | 8.6 | 0.999 |
| D | 38.1 | 56.3 | 0.969 |
| E | 44.2 | 59.8 | 0.965 |
| F | 277.2 | 320.6 | −0.000 |

| life stage | n | MAE | mean true RUL | relative error |
|---|---|---|---|---|
| 0–20% | 3,006 | 14.8 | 997.9 | 1.5% |
| 20–40% | 2,807 | 21.1 | 782.1 | 2.7% |
| 40–60% | 2,995 | 31.4 | 556.5 | 5.6% |
| 60–80% | 2,847 | 61.2 | 335.3 | 18.3% |
| 80–100% | 2,812 | 63.8 | 110.0 | 58.0% |

D vs E: MAE difference 6.1 cycles, 95% interval 0.7–10.8; D lower in 11 of 14 cells. D bias vs end-of-test discharge time: Spearman ρ = 0.87 (14 cells).

Reading:
1. The counter generalizes across cells: C (random) 4.4, G (leave one cell out) 4.8. Because every cell ends near cycle 1,110, RUL ≈ 1,110 − `Cycle_Index` for any cell, seen or not.
2. The random split inflates only models without the counter: B 17.4 vs D 38.1, a factor of 2.2.
3. Electrical features add nothing to the counter (H 5.8 vs G 4.8).
4. On electrical features alone, LightGBM beats Ridge, with an interval that excludes zero, and both beat the trivial baseline by a factor of 6 or more.
5. D's errors are partly the label's fault. Cells that ended the test with a longer discharge time (more remaining capacity) are predicted to have more cycles left. The model is reading health; the label only reads the test schedule.
6. Absolute error grows through life while RUL shrinks, so relative error rises from 1.5% to 58.0% in the last fifth of life.
7. SOH-defined end of life (SOH < 80%) comes at cycles 345–419: mean 381.4, SD 26.2, CV 6.9%, about 10× the 0.7% spread of the test length. At the end of the test, SOH is 35–46%. A health-defined label would give the cells genuinely different lifetimes.

## 8. Limits

14 cells, one chemistry, one temperature, one current. The label ends where the test ended, so there is almost no between-cell variance in lifetime to predict. Features from the dataset authors. No early-life prediction (14 cells are 14 training examples for that task). It is a per-cycle regression, not a forecasting model.

Next step: predict the SOH-defined end of life (step 6b) from early-life cycles.

## 9. Tests and CI

- `tests/test_pipeline.py` checks cell reconstruction (13 resets, 14 cells), that RUL is the countdown to each cell's last cycle, the cleaning counts, the life-fraction range, that no leave-one-cell-out fold trains on its held-out cell, and that the SOH proxy is exactly 100% at its reference cycles and every cell reaches 80% within the test.
- `.github/workflows/ci.yml` runs on every push and pull request with the pinned versions in `requirements.txt` on Python 3.12: install, `ruff check`, download data, `pytest`, run the pipeline, and fail if `outputs/results.json` differs from the committed copy.
- Protocols A to F gave identical figures on Windows (Python 3.9) and Linux (Python 3.13, LightGBM 4.7). The full current `results.json` was produced on Linux (Python 3.10) with the pinned versions.
