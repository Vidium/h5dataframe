from __future__ import annotations

from typing import Any, Callable

import ch5mpy as ch
import pandas.io.formats.format as fmt
from pandas import Series


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
