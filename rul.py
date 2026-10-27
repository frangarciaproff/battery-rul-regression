"""Battery remaining-useful-life (RUL) regression on the processed HNEI dataset.

Per-cycle regression with leave-one-cell-out validation. Not a forecasting model.

Data: 14 NMC-LCO 18650 cells (Hawaii Natural Energy Institute), processed copy from
https://github.com/ignavinuales/Battery_RUL_Prediction (Datasets/HNEI_Processed/Final Database.csv).

Usage:
    python rul.py "path/to/Final Database.csv" [output_dir]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "RUL"
CYCLE = "Cycle_Index"
DISCHARGE_TIME = "Discharge Time (s)"
ELECTRICAL = [
    DISCHARGE_TIME,
    "Decrement 3.6-3.4V (s)",
    "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)",
    "Time at 4.15V (s)",
    "Time constant current (s)",
    "Charging time (s)",
    "Total time (s)",
]
# LightGBM defaults (100 trees, learning rate 0.1, 31 leaves), same in every protocol.
LGBM_PARAMS = dict(random_state = SEED, verbose = -1)
N_BOOT = 2000
# State of health (SOH) proxy: discharge time over its early-life median.
SOH_REF_CYCLES = (10, 60)
SOH_WINDOW = 21
EOL_SOH = 0.8


def load(path):
    df = pd.read_csv(path, index_col = 0)
    n_raw = len(df)
    # The published file concatenates the cells with no cell column:
    # Cycle_Index restarts at 1 for each new cell.
    resets = (df[CYCLE].diff() < 0)
    df["cell"] = resets.cumsum()
    n_resets = int(resets.sum())
    cycles_per_cell = df.groupby("cell").size()
    life = df.groupby("cell")[CYCLE].max()
    raw_frac = df[CYCLE] / df.groupby("cell")[CYCLE].transform("max")
    # RUL + Cycle_Index is one constant per cell: the target counts down to the end of the test.
    countdown = bool((df[TARGET] + df[CYCLE]).groupby(df["cell"]).nunique().eq(1).all())

    bad_time = df["Total time (s)"] > 24 * 3600
    bad_volt = df["Min. Voltage Charg. (V)"] > df["Max. Voltage Dischar. (V)"]
    drop = bad_time | bad_volt
    clean = df.loc[~drop].reset_index(drop = True)
    # Life fraction per cell: cycle number over the cell's last retained cycle.
    clean["life_frac"] = clean[CYCLE] / clean.groupby("cell")[CYCLE].transform("max")

    info = dict(
        rows_raw = n_raw,
        cycle_index_resets = n_resets,
        n_cells = int(df["cell"].nunique()),
        rows_per_cell_min = int(cycles_per_cell.min()),
        rows_per_cell_max = int(cycles_per_cell.max()),
        missing_cycles_per_cell_min = int((life - cycles_per_cell).min()),
        missing_cycles_per_cell_max = int((life - cycles_per_cell).max()),
        rul_plus_cycle_constant_per_cell = countdown,
        median_total_time_h = round(float(df["Total time (s)"].median() / 3600), 2),
        dropped_total_time_gt_24h = int(bad_time.sum()),
        dropped_vmin_charge_gt_vmax_discharge = int((bad_volt & ~bad_time).sum()),
        dropped_vmin_rule_in_last_40pct_of_life = int((bad_volt & ~bad_time & (raw_frac > 0.6)).sum()),
        dropped_total = int(drop.sum()),
        dropped_pct = round(100 * drop.mean(), 1),
        rows_used = len(clean),
        life_min_cycles = int(life.min()),
        life_max_cycles = int(life.max()),
        life_mean_cycles = round(float(life.mean()), 1),
        life_sd_cycles = round(float(life.std()), 1),
    )
    return clean, info


def scores(y, p):
    return dict(MAE = round(mean_absolute_error(y, p), 1),
                RMSE = round(float(np.sqrt(mean_squared_error(y, p))), 1),
                R2 = round(r2_score(y, p), 3))


def random_split(df, feats):
    X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df[TARGET],
                                              test_size = 0.2, random_state = SEED)
    model = LGBMRegressor(**LGBM_PARAMS).fit(X_tr, y_tr)
    return scores(y_te, model.predict(X_te))


def leave_one_cell_out(df, feats, make_model):
    preds = pd.Series(index = df.index, dtype = float)
    for cell in sorted(df["cell"].unique()):
        test = df["cell"] == cell
        if make_model is None:
            preds[test] = df.loc[~test, TARGET].mean()
        else:
            model = make_model().fit(df.loc[~test, feats], df.loc[~test, TARGET])
            preds[test] = model.predict(df.loc[test, feats])
    return preds


def by_life_stage(df, preds):
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    stage = pd.cut(df["life_frac"], bins = bins, labels = labels, include_lowest = True)
    out = []
    for lab in labels:
        m = stage == lab
        mae = mean_absolute_error(df.loc[m, TARGET], preds[m])
        mean_rul = df.loc[m, TARGET].mean()
        out.append(dict(stage = lab, n = int(m.sum()), MAE = round(mae, 1),
                        mean_true_RUL = round(mean_rul, 1),
                        relative_error_pct = round(100 * mae / mean_rul, 1)))
    return out


def compare_cells(df, p_ref, p_alt):
    """Per-cell MAE of two leave-one-cell-out predictions, and a cell-level bootstrap
    95% interval for MAE(p_alt) - MAE(p_ref)."""
    abs_ref = (p_ref - df[TARGET]).abs().groupby(df["cell"])
    abs_alt = (p_alt - df[TARGET]).abs().groupby(df["cell"])
    n = df.groupby("cell").size().to_numpy()
    s_ref, s_alt = abs_ref.sum().to_numpy(), abs_alt.sum().to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(n), size = (N_BOOT, len(n)))
    diff = (s_alt[idx].sum(axis = 1) - s_ref[idx].sum(axis = 1)) / n[idx].sum(axis = 1)
    mae_ref, mae_alt = abs_ref.mean(), abs_alt.mean()
    return dict(cells_ref_better = int((mae_ref < mae_alt).sum()),
                n_cells = len(n),
                mae_diff = round(float((s_alt.sum() - s_ref.sum()) / n.sum()), 1),
                mae_diff_ci95 = np.percentile(diff, [2.5, 97.5]).round(1).tolist())


def bias_vs_health(df, preds):
    """Per-cell mean signed error against end-of-test discharge time (capacity proxy at
    constant current): median over each cell's last 30 retained cycles."""
    bias = (preds - df[TARGET]).groupby(df["cell"]).mean()
    end_dt = df.sort_values(CYCLE).groupby("cell")[DISCHARGE_TIME].apply(lambda s: s.tail(30).median())
    return dict(spearman_rho = round(float(bias.corr(end_dt, method = "spearman")), 2),
                n_cells = len(bias),
                end_discharge_time_cv_pct = round(float(100 * end_dt.std() / end_dt.mean()), 1))


def soh(df):
    """SOH proxy per row: discharge time over the cell's median discharge time in cycles 10-60.

    Smoothed with a centred 21-row rolling median, so it uses later cycles: a retrospective
    monitoring curve, not a real-time estimate. At constant discharge current, discharge time
    is proportional to delivered capacity at that rate, which includes resistance growth.
    """
    curves = []
    for _, d in df.sort_values(["cell", CYCLE]).groupby("cell"):
        ref = d.loc[d[CYCLE].between(*SOH_REF_CYCLES), DISCHARGE_TIME].median()
        ratio = d[DISCHARGE_TIME] / ref
        curves.append(pd.DataFrame({"cell": d["cell"], CYCLE: d[CYCLE], "soh_raw": ratio,
                                    "soh": ratio.rolling(SOH_WINDOW, center = True, min_periods = 5).median()}))
    return pd.concat(curves)


def soh_summary(curves):
    # First cycle below the threshold; median of the raw ratio over the last 30 cycles as end-of-test SOH.
    eol = curves.loc[curves["soh"] < EOL_SOH].groupby("cell")[CYCLE].min()
    end = curves.groupby("cell")["soh_raw"].apply(lambda s: s.tail(30).median())
    return dict(eol_threshold = EOL_SOH,
                eol_cycle_per_cell = eol.astype(int).tolist(),
                eol_cycle_mean = round(float(eol.mean()), 1),
                eol_cycle_sd = round(float(eol.std()), 1),
                eol_cycle_cv_pct = round(float(100 * eol.std() / eol.mean()), 1),
                end_of_test_soh_min = round(float(end.min()), 3),
                end_of_test_soh_max = round(float(end.max()), 3)), eol, end


def plot_soh(curves, eol, end, path):
    ink, muted, line, accent = "#1f2328", "#59636e", "#9aa4ae", "#1f6feb"
    fig, ax = plt.subplots(figsize = (8, 4.5), dpi = 150)
    # The first cycles are start-up artefacts (discharge time up to 3x the reference); not drawn.
    for _, d in curves.loc[curves[CYCLE] >= 20].groupby("cell"):
        ax.plot(d[CYCLE], 100 * d["soh"], color = line, linewidth = 1.2, alpha = 0.9)
    ax.axhline(100 * EOL_SOH, color = muted, linewidth = 1, linestyle = (0, (4, 3)))
    ax.text(curves[CYCLE].max(), 100 * EOL_SOH + 1.5, f"end of life: SOH {100 * EOL_SOH:.0f}%",
            ha = "right", va = "bottom", fontsize = 9, color = muted)
    ax.scatter(eol, np.full(len(eol), 100 * EOL_SOH), s = 30, color = accent, zorder = 3,
               edgecolor = "white", linewidth = 1)
    ax.annotate(f"80% reached at cycle {eol.min():.0f}-{eol.max():.0f}",
                xy = (eol.max(), 100 * EOL_SOH), xytext = (eol.max() + 60, 100 * EOL_SOH + 12),
                fontsize = 9, color = ink, arrowprops = dict(arrowstyle = "-", color = muted, linewidth = 0.8))
    last = curves.groupby("cell")[CYCLE].max()
    ax.annotate(f"test stops at cycle {last.min():.0f}-{last.max():.0f}",
                xy = (last.min(), 42), xytext = (last.min() - 360, 26),
                fontsize = 9, color = ink, arrowprops = dict(arrowstyle = "-", color = muted, linewidth = 0.8))
    ax.set_xlabel("Cycle", color = ink)
    ax.set_ylabel("SOH proxy (% of early-life discharge time)", color = ink)
    ax.set_title(f"Same test length, different health: {len(last)} cells end the test at "
                 f"{100 * end.min():.0f}-{100 * end.max():.0f}% SOH",
                 loc = "left", fontsize = 11, color = ink)
    ax.set_ylim(20, 105)
    ax.grid(axis = "y", color = "#d8dee4", linewidth = 0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors = muted)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(path, out_dir):
    df, info = load(path)
    y = df[TARGET]
    results = {"data": info, "lgbm_params": LGBM_PARAMS, "protocols": {}}
    P = results["protocols"]

    P["A_random_electrical_plus_cycle"] = random_split(df, ELECTRICAL + [CYCLE])
    P["B_random_electrical_only"] = random_split(df, ELECTRICAL)
    P["C_random_cycle_only"] = random_split(df, [CYCLE])

    lgbm = lambda: LGBMRegressor(**LGBM_PARAMS)
    ridge = lambda: make_pipeline(StandardScaler(), Ridge(alpha = 1.0))
    p_d = leave_one_cell_out(df, ELECTRICAL, lgbm)
    p_e = leave_one_cell_out(df, ELECTRICAL, ridge)
    P["D_loco_lgbm_electrical"] = scores(y, p_d)
    P["E_loco_ridge_electrical"] = scores(y, p_e)
    P["F_loco_train_mean"] = scores(y, leave_one_cell_out(df, [], None))
    P["G_loco_ridge_cycle_only"] = scores(y, leave_one_cell_out(df, [CYCLE], ridge))
    P["H_loco_lgbm_electrical_plus_cycle"] = scores(y, leave_one_cell_out(df, ELECTRICAL + [CYCLE], lgbm))
    results["D_error_by_life_stage"] = by_life_stage(df, p_d)
    results["D_vs_E_by_cell"] = compare_cells(df, p_d, p_e)
    results["D_bias_vs_health"] = bias_vs_health(df, p_d)
    curves = soh(df)
    results["soh"], eol, end = soh_summary(curves)

    out_dir.mkdir(parents = True, exist_ok = True)
    plot_soh(curves, eol, end, out_dir / "soh.png")
    # LF line endings on every OS, so the committed file matches the CI run byte for byte.
    with open(out_dir / "results.json", "w", newline = "\n") as f:
        f.write(json.dumps(results, indent = 2) + "\n")
    print(json.dumps(results, indent = 2))


if __name__ == "__main__":
    data_path = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs")
    main(data_path, out)
