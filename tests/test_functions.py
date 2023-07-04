import numpy as np

from h5dataframe import H5DataFrame


def test_can_map_function(hdf5: H5DataFrame) -> None:
    res = hdf5.col_int.map(lambda x: x**2)

    assert np.array_equal(res, [1, 4, 9])
