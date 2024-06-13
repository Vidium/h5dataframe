from __future__ import annotations

from importlib import reload
from typing import Any, Callable

import ch5mpy as ch
import numpy as np
import pandas.core.construction
import pandas.core.dtypes.cast
import pandas.core.internals.blocks
import pandas.io.formats.format as fmt
from pandas import Series
from pandas._typing import ArrayLike


def _repr_series(series: Series[Any]) -> str:
    params = fmt.get_series_repr_params()  # type: ignore[attr-defined]
    return series.copy().to_string(**params)  # type: ignore[no-any-return]


def no_H5Array(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(series: Series[Any], *args: Any, **kwargs: Any) -> Any:
        if isinstance(series.values, ch.H5Array):
            series = series.copy()

        return func(series, *args, **kwargs)

    return wrapper


Series.__repr__ = _repr_series  # type: ignore[assignment]
Series.map = no_H5Array(Series.map)


def no_H5Array_str_sanitization(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(result: np.ndarray | ch.H5Array, data, dtype: np.dtype | None, copy: bool) -> np.ndarray | ch.H5Array:
        if isinstance(result, ch.H5Array):
            return result

        return func(result, data, dtype, copy)

    return wrapper


pandas.core.construction._sanitize_str_dtypes = no_H5Array_str_sanitization(
    pandas.core.construction._sanitize_str_dtypes
)


# NOTE: may cause issues with Series of mixed dtypes ?
def no_H5Array_str_coercing(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(values: ArrayLike) -> ArrayLike:
        if isinstance(values, ch.H5Array):
            return values

        return func(values)

    return wrapper


maybe_coerce_values = no_H5Array_str_coercing(pandas.core.internals.blocks.maybe_coerce_values)
pandas.core.internals.blocks.maybe_coerce_values = maybe_coerce_values


def np_can_hold_element_with_str(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(dtype: np.dtype, element: Any) -> Any:
        if dtype.kind == "U":
            return element

        return func(dtype, element)

    return wrapper


pandas.core.dtypes.cast.np_can_hold_element = np_can_hold_element_with_str(pandas.core.dtypes.cast.np_can_hold_element)

# = RELOAD ====================================================================
reload(pandas.core.internals.array_manager)
reload(pandas.core.internals.blocks)
reload(pandas.core.series)
