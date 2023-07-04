import numpy as np
import pandas as pd

from h5dataframe import H5DataFrame


def test_can_create_from_pandas_DataFrame(hdf5: H5DataFrame) -> None:
    assert isinstance(hdf5, H5DataFrame)


def test_has_correct_columns(hdf5: H5DataFrame) -> None:
    assert np.array_equal(hdf5.columns, ["col_int", "col_str", "col_float"])


def test_has_correct_repr(hdf5: H5DataFrame) -> None:
    assert (
        repr(hdf5) == "   col_int col_str  col_float\n"
        "0        1       a        1.5\n"
        "1        2      bc        2.5\n"
        "2        3     def        3.5\n"
        "[FILE]\n"
        "[3 rows x 3 columns]"
    )


def test_convert_to_pandas(hdf5: H5DataFrame) -> None:
    df = hdf5.copy()

    assert isinstance(df, pd.DataFrame)
    assert np.array_equal(df.index, [0, 1, 2])
    assert np.array_equal(df.columns, ["col_int", "col_str", "col_float"])
    assert np.array_equal(df.col_int, [1, 2, 3])
    assert np.array_equal(df.col_str, ["a", "bc", "def"])


def test_can_get_column_from_getattr(hdf5: H5DataFrame) -> None:
    assert np.array_equal(hdf5.col_str, ["a", "bc", "def"])


def test_can_set_existing_column(hdf5: H5DataFrame) -> None:
    hdf5["col_int"] = -1
    assert np.array_equal(hdf5["col_int"], [-1, -1, -1])


def test_can_set_new_column_str(hdf5: H5DataFrame) -> None:
    hdf5["new"] = "new!"
    assert np.array_equal(hdf5["new"], ["new!", "new!", "new!"])


def test_can_set_new_column_int(hdf5: H5DataFrame) -> None:
    hdf5["new"] = [-1, -2, -3]
    assert np.array_equal(hdf5["new"], [-1, -2, -3])


def test_can_set_column_no_index() -> None:
    hdf5 = H5DataFrame(pd.DataFrame(columns=["value"]))
    hdf5["value"] = [1, 2, 3]

    assert np.array_equal(hdf5["value"], [1, 2, 3])


def test_set_value_sets_in_h5_file(hdf5: H5DataFrame) -> None:
    hdf5.loc[1, "col_str"] = "test"
    assert np.array_equal(hdf5._data_file["arrays"]["col_str"], ["a", "test", "def"])
