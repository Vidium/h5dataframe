import tempfile
from pathlib import Path
from typing import Generator

import ch5mpy as ch
import pandas as pd
import pytest

from h5dataframe import H5DataFrame


@pytest.fixture
def h5df() -> Generator[H5DataFrame, None, None]:
    with tempfile.NamedTemporaryFile() as path:
        values = ch.H5Dict(ch.File(Path(path.name), mode=ch.H5Mode.READ_WRITE_CREATE))

        h5df = H5DataFrame(
            pd.DataFrame({"col_int": [1, 2, 3], "col_str": ["a", "bc", "def"], "col_float": [1.5, 2.5, 3.5]})
        )

        h5df.write(values)
        return h5df
