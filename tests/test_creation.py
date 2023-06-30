import numpy as np
import pandas as pd
import pytest

from h5dataframe import H5DataFrame


@pytest.fixture
def h5df() -> H5DataFrame:
    return H5DataFrame.from_dataframe(
        pd.DataFrame({"col_int": [1, 2, 3], "col_str": ["a", "bc", "def"], "col_float": [1.5, 2.5, 3.5]})
    )


def test_can_create_from_pandas_DataFrame(h5df: H5DataFrame) -> None:
    assert isinstance(h5df, H5DataFrame)


def test_has_correct_columns(h5df: H5DataFrame) -> None:
    assert np.array_equal(h5df.columns, ["col_int", "col_str", "col_float"])


def test_has_correct_column_order(h5df: H5DataFrame) -> None:
    assert np.array_equal(h5df._columns_order, [0, 2, 1])


def test_has_correct_repr(h5df: H5DataFrame) -> None:
    assert (
        repr(h5df) == "  col_int col_str col_float\n"
        "0     1.0       a       1.5\n"
        "1     2.0      bc       2.5\n"
        "2     3.0     def       3.5\n"
        "[3 rows x 3 columns]"
    )
