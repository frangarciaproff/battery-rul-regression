"""Checks on data reconstruction, cleaning and validation splits.

Run from the repository root after `python download_data.py`:
    pytest
"""

from pathlib import Path

import pandas as pd
import pytest

import rul

DATA = Path("data") / "hnei_final_database.csv"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason = "run download_data.py first")


@pytest.fixture(scope = "module")
def loaded():
    return rul.load(DATA)


def test_cell_reconstruction(loaded):
    _, info = loaded
    assert info["rows_raw"] == 15064
    assert info["cycle_index_resets"] == 13
    assert info["n_cells"] == 14
    assert (info["rows_per_cell_min"], info["rows_per_cell_max"]) == (1051, 1081)


def test_cleaning(loaded):
    df, info = loaded
    assert info["dropped_total_time_gt_24h"] == 182
    assert info["dropped_vmin_charge_gt_vmax_discharge"] == 415
    assert info["dropped_total"] == 597
    assert info["rows_used"] == len(df) == 14467
    assert (df["Total time (s)"] <= 24 * 3600).all()
    assert (df["Min. Voltage Charg. (V)"] <= df["Max. Voltage Dischar. (V)"]).all()


def test_life_fraction_in_unit_interval(loaded):
    df, _ = loaded
    assert df["life_frac"].between(0, 1).all()
    assert (df.groupby("cell")["life_frac"].max() == 1).all()


def test_leave_one_cell_out_never_trains_on_test_cell(loaded):
    df, _ = loaded
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(set(df.loc[X.index, "cell"]))
            return self

        def predict(self, X):
            return pd.Series(0.0, index = X.index).to_numpy()

    rul.leave_one_cell_out(df, rul.ELECTRICAL, Spy)
    cells = set(df["cell"])
    assert len(seen) == len(cells)
    for train_cells, held_out in zip(seen, sorted(cells)):
        assert held_out not in train_cells
        assert train_cells == cells - {held_out}


def test_electrical_features_exclude_cycle_and_target():
    assert rul.CYCLE not in rul.ELECTRICAL
    assert rul.TARGET not in rul.ELECTRICAL


def test_target_is_countdown_to_end_of_test(loaded):
    _, info = loaded
    raw = pd.read_csv(DATA, index_col = 0)
    cell = (raw[rul.CYCLE].diff() < 0).cumsum()
    life = raw[rul.CYCLE].groupby(cell).transform("max")
    assert (raw[rul.TARGET] == life - raw[rul.CYCLE]).all()
    assert info["rul_plus_cycle_constant_per_cell"]


def test_soh_proxy(loaded):
    df, _ = loaded
    curves = rul.soh(df)
    ref = curves.loc[curves[rul.CYCLE].between(*rul.SOH_REF_CYCLES)].groupby("cell")["soh_raw"].median()
    assert ((ref - 1).abs() < 1e-9).all()
    summary, eol, _ = rul.soh_summary(curves)
    assert len(eol) == df["cell"].nunique()
    assert all(rul.SOH_REF_CYCLES[1] < c < df[rul.CYCLE].max() for c in summary["eol_cycle_per_cell"])
