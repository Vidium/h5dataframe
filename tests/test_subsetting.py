import pytest
import numpy as np
import pandas as pd
from h5dataframe.dataframe import H5DataFrame


def test_loc(hdf5: H5DataFrame) -> None:
    assert np.all(hdf5.loc[2].values == np.array([3, "def", 3.5], dtype=object))


def test_loc_multi(hdf5: H5DataFrame) -> None:
    assert np.all(hdf5.loc[[2, 1]].values == np.array([[3, "def", 3.5], [2, "bc", 2.5]], dtype=object))


def test_get_column(hdf5: H5DataFrame) -> None:
    assert hdf5["col_str"].equals(pd.Series(["a", "bc", "def"], name="col_str"))


def test_get_column_regular_df(hdf5: H5DataFrame):
    df = pd.DataFrame({"col_str": ["a", "bc", "def"]})
    assert df["col_str"].equals(pd.Series(["a", "bc", "def"], name="col_str"))

    assert hdf5["col_str"].equals(df["col_str"])


@pytest.mark.xfail
def test_compare_with_regular_df(hdf5: H5DataFrame):
    # TODO: maybe one day find a fix, for now can only use h5dataframe.equals(), pd.DataFrame.equals() fails
    df = pd.DataFrame({"col_str": ["a", "bc", "def"]})
    assert df["col_str"].equals(hdf5["col_str"])


def test_iterrows(hdf5: H5DataFrame):
    _, row = next(hdf5.iterrows())
    assert row.equals(pd.Series([1, "a", 1.5], index=["col_int", "col_str", "col_float"]))
