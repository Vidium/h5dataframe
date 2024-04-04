import tempfile
from pathlib import Path

import ch5mpy as ch
import numpy as np
import pandas as pd

from h5dataframe import H5DataFrame


def test_can_write() -> None:
    hdf5 = H5DataFrame(
        pd.DataFrame(
            {"col_int": [1, 2, 3], "col_str": ["a", "bc", "def"], "col_float": [1.5, 2.5, 3.5]}, index=["a", "b", "c"]
        )
    )

    with tempfile.NamedTemporaryFile() as path:
        values = ch.H5Dict(ch.File(Path(path.name), mode=ch.H5Mode.WRITE_TRUNCATE))

        ch.write_object(hdf5, values, "")

        assert set(values.keys()) == {"index", "arrays"}
        assert np.array_equal(values["index"], ["a", "b", "c"])
        assert set(values["arrays"].keys()) == {"col_float", "col_int", "col_str"}
        assert np.array_equal(values["arrays"]["col_int"], [1, 2, 3])


def test_can_read() -> None:
    with tempfile.NamedTemporaryFile() as path:
        values = ch.H5Dict(ch.File(Path(path.name), mode=ch.H5Mode.WRITE_TRUNCATE))

        values["index"] = ["a", "b", "c"]
        values["arrays"] = {"col_int": [1, 2, 3], "col_str": ["a", "bc", "def"]}
        values.attributes["columns_dtype"] = np.str_

        hdf5 = H5DataFrame.read(values)
        assert np.array_equal(hdf5.index, ["a", "b", "c"])
        assert np.array_equal(hdf5.columns, ["col_int", "col_str"])
