from typing import Any, cast

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
from pandas.core.array_algos.take import take_1d as pd_take_1d  # type: ignore[import-untyped]

from h5dataframe._typing import NDArrayLike


def take_1d(
    arr: NDArrayLike[Any],
    indexer: npt.NDArray[np.intp],
    fill_value: Any = None,
    allow_fill: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> NDArrayLike[Any]:
    if not isinstance(arr, ch.H5Array):
        return cast(ch.H5Array[Any], pd_take_1d(arr, indexer, fill_value=fill_value, allow_fill=allow_fill, mask=mask))

    if fill_value is None:
        fill_value = np.nan

    if allow_fill:
        assert arr.ndim == 1
        assert mask is not None
        if len(arr) < len(mask):
            np.insert(arr, np.where(mask), fill_value)
    return arr[indexer]
