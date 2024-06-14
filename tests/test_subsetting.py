import numpy as np
import pandas as pd
from h5dataframe.dataframe import H5DataFrame


def test_loc(hdf5: H5DataFrame) -> None:
    assert np.all(hdf5.loc[2].values == np.array([3, "def", 3.5], dtype=object))


def test_loc_multi(hdf5: H5DataFrame) -> None:
    assert np.all(hdf5.loc[[2, 1]].values == np.array([[3, "def", 3.5], [2, "bc", 2.5]], dtype=object))


def test_get_column(hdf5: H5DataFrame) -> None:
    assert hdf5["col_str"].equals(pd.Series(["a", "bc", "def"], name="col_str"))
