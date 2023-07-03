from __future__ import annotations

from typing import Any

import pandas.io.formats.format as fmt
from pandas import Series


def _repr_series(series: Series[Any]) -> str:
    params = fmt.get_series_repr_params()  # type: ignore[attr-defined]
    return Series(series.values.copy()).to_string(**params)  # type: ignore[no-any-return]


Series.__repr__ = _repr_series  # type: ignore[assignment]
