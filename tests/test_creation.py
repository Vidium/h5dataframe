import numpy as np
import pandas as pd

from h5dataframe import H5DataFrame


def test_can_create_from_pandas_DataFrame(h5df: H5DataFrame) -> None:
    assert isinstance(h5df, H5DataFrame)


def test_has_correct_columns(h5df: H5DataFrame) -> None:
    assert np.array_equal(h5df.columns, ["col_int", "col_str", "col_float"])


def test_has_correct_repr(h5df: H5DataFrame) -> None:
    assert (
        repr(h5df) == "   col_int col_str  col_float\n"
        "0        1       a        1.5\n"
        "1        2      bc        2.5\n"
        "2        3     def        3.5\n"
        "\n"
        "[3 rows x 3 columns]"
    )


def test_convert_to_pandas(h5df: H5DataFrame) -> None:
    df = h5df.copy()

    assert isinstance(df, pd.DataFrame)
    assert np.array_equal(df.index, [0, 1, 2])
    assert np.array_equal(df.columns, ["col_int", "col_str", "col_float"])
    assert np.array_equal(df.col_int, [1, 2, 3])
    assert np.array_equal(df.col_str, ["a", "bc", "def"])


def test_can_get_column_from_getattr(h5df: H5DataFrame) -> None:
    assert np.array_equal(h5df.col_str, ["a", "bc", "def"])


def test_can_set_existing_column(h5df: H5DataFrame) -> None:
    h5df["col_int"] = -1
    assert np.array_equal(h5df["col_int"], [-1, -1, -1])


def test_can_set_new_column_str(h5df: H5DataFrame) -> None:
    h5df["new"] = "new!"
    assert np.array_equal(h5df["new"], ["new!", "new!", "new!"])


def test_can_set_new_column_int(h5df: H5DataFrame) -> None:
    h5df["new"] = [-1, -2, -3]
    assert np.array_equal(h5df["new"], [-1, -2, -3])


def test_can_set_column_no_index() -> None:
    h5df = H5DataFrame(pd.DataFrame(columns=["value"]))
    h5df["value"] = [1, 2, 3]

    assert np.array_equal(h5df["value"], [1, 2, 3])


def test_set_value_sets_in_h5_file(h5df: H5DataFrame) -> None:
    h5df.loc[1, "col_str"] = "test"
    assert np.array_equal(h5df._data_file["arrays"]["col_str"], ["a", "test", "def"])
