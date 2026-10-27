# Battery remaining useful life: what a popular benchmark really measures

Rechargeable batteries lose capacity every time they are charged and discharged. Knowing how many cycles a battery has left, its remaining useful life (RUL), matters to anyone who runs batteries, from phones to grid storage.

This project uses a public dataset of 14 identical lithium-ion cells, cycled in a lab until the experiment ended, and often used to test RUL predictions. It finds that the "cycles left" label counts down to the end of the experiment, which came at almost the same cycle for every cell. The cycle number alone therefore predicts it almost perfectly: on cells the model has never seen, the average error is 4.8 cycles out of about 1,110. A model that reads the cells' electrical behaviour misses by 38 cycles on average, because it picks up how worn each cell is, and the cells wore out at different rates.

Measuring health directly, as the fraction of the original capacity a cell still delivers, tells the real story: the cells reach 80% of their original capacity between cycle 345 and cycle 419, and end the experiment with 35–46% left.

![SOH proxy against cycle for the 14 cells](outputs/soh.png)

## Key results

- RUL = (last cycle of the cell) − `Cycle_Index`, exactly, for every row. The last cycle is 1,103–1,134 for all 14 cells (SD 7.4).
- Leave-one-cell-out (train on 13 cells, test on the 14th): the cycle counter alone gives MAE 4.8 cycles; the eight electrical features alone give 38.1.
- The electrical model tracks degradation: each cell's mean prediction error correlates with its end-of-test discharge time, a capacity proxy (Spearman ρ = 0.87, 14 cells).
- State of health (SOH): end of life at SOH < 80% falls at cycles 345–419 (mean 381, CV 6.9%), against a test-length CV of 0.7%. Monitoring SOH against a fixed threshold is how a capacity guarantee is checked.

## Quick start

```
pip install -r requirements.txt
python download_data.py
python rul.py data/hnei_final_database.csv outputs
python -m pytest
```

`outputs/results.json` holds every figure. `WORKFLOW.md` explains each step.

## Data

Hawaii Natural Energy Institute (HNEI) cycling data of 14 NMC-LCO 18650 cells: Devie, Baure & Dubarry, *Intrinsic variability in the degradation of a batch of commercial 18650 lithium-ion cells*, Energies 11(5), 1031 (2018). Processed copy from [ignavinuales/Battery_RUL_Prediction](https://github.com/ignavinuales/Battery_RUL_Prediction), fetched by `download_data.py` at a pinned commit and checked by hash. 15,064 cycles; cell id reconstructed from 13 `Cycle_Index` resets; 597 rows (4.0%) dropped by two sanity rules; 14,467 used.

## Method

Per-cycle regression of RUL from the eight per-cycle electrical features published with the dataset, the cycle counter, or both. Eight protocols compare a random 80/20 split with leave-one-cell-out validation:

| | split | features | model | MAE (cycles) | RMSE |
|---|---|---|---|---|---|
| A | random 80/20 | electrical + `Cycle_Index` | LightGBM | 2.8 | 4.1 |
| B | random 80/20 | electrical | LightGBM | 17.4 | 26.6 |
| C | random 80/20 | `Cycle_Index` | LightGBM | 4.4 | 7.1 |
| **G** | **leave one cell out** | **`Cycle_Index`** | **Ridge** | **4.8** | **7.6** |
| H | leave one cell out | electrical + `Cycle_Index` | LightGBM | 5.8 | 8.6 |
| D | leave one cell out | electrical | LightGBM | 38.1 | 56.3 |
| E | leave one cell out | electrical | Ridge | 44.2 | 59.8 |
| F | leave one cell out | none | train mean | 277.2 | 320.6 |

- The counter generalizes across cells (C 4.4 → G 4.8). The random split inflates only the models without it (B 17.4 → D 38.1).
- Adding electrical features to the counter makes it worse (G 4.8 → H 5.8).
- LightGBM beats Ridge on electrical features by 6.1 cycles MAE (cell-level bootstrap 95% CI 0.7–10.8; lower error in 11 of 14 cells).
- Protocol D error grows through life: MAE 14.8 cycles in the first 20% of life, 63.8 in the last 20%.

SOH proxy: discharge time at constant current over each cell's early-life median (cycles 10–60), smoothed with a centred 21-cycle rolling median.

## Layout

```
rul.py                      pipeline: load, clean, eight protocols, error breakdowns, SOH
download_data.py            pinned, hash-checked data download
tests/test_pipeline.py      data, target, split and SOH checks
.github/workflows/ci.yml    lint, tests, pipeline run and results check on every push
outputs/results.json        figures from the last run
outputs/soh.png             SOH proxy per cell, from the last run
```

## Limits

- 14 cells, one chemistry, one temperature, one current.
- The RUL label ends where the experiment ended, so lifetimes barely differ between cells (0.7%).
- The electrical features are those published with the dataset.
- 86% of the rows removed by the voltage sanity rule come from the last 40% of life; late-life error is measured on the cycles that pass the rule.
- The SOH proxy is the capacity delivered at the test discharge rate, so resistance growth lowers it as well as capacity loss.

Next step: predict the SOH-defined end of life (cycles 345–419) from early-life cycles, a target on which the cells have genuinely different lifetimes.
