import tempfile
import pytest
from pathlib import Path

import ch5mpy as ch

from h5dataframe import Categorical


@pytest.fixture
def cat() -> Categorical:
    return Categorical([0, 1, 1, 0, 0, 1, 2])

    # with tempfile.NamedTemporaryFile() as path:
    #     values = ch.H5Dict(ch.File(Path(path.name), mode=ch.H5Mode.READ_WRITE_CREATE))
    #
    #     hdf5 = H5DataFrame(
    #         pd.DataFrame({"col_int": [1, 2, 3], "col_str": ["a", "bc", "def"], "col_float": [1.5, 2.5, 3.5]})
    #     )
    #
    #     hdf5.write(values)
    #     return hdf5


def test_categorical_creation_from_pandas(cat: Categorical):
    with tempfile.NamedTemporaryFile() as path:
        values = ch.H5Dict(ch.File(Path(path.name), mode=ch.H5Mode.READ_WRITE_CREATE))

        cat.write(values, "test")

        assert isinstance(cat.codes, ch.H5Array)
