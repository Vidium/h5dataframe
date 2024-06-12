import numpy as np
from h5dataframe.dataframe import H5DataFrame


def test_loc(hdf5: H5DataFrame) -> None:
    assert np.all(hdf5.loc[2].values == np.array([3, "def", 3.5], dtype=object))


def test_loc_multi(hdf5: H5DataFrame) -> None:
    assert np.all(hdf5.loc[[2, 1]].values == np.array([[3, "def", 3.5], [2, "bc", 2.5]], dtype=object))
