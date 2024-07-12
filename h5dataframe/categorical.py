from __future__ import annotations
from typing import Any
from pathlib import Path

import pandas as pd
import pandas._typing as pdt
import numpy as np
import numpy.typing as npt

import ch5mpy as ch


class Categorical(pd.Categorical):
    _stored_array: npt.NDArray[Any] | ch.H5Array[Any]
    _stored_dtype: pd.CategoricalDtype | None

    # region magic methods
    def __init__(
        self,
        values: pdt.ListLike | ch.H5Array[Any],
        dtype: pd.CategoricalDtype | None = None,
    ) -> None:
        if isinstance(values, ch.H5Array):
            self._stored_array = values
            self._stored_dtype = dtype

        else:
            self._stored_array = np.array(values)
            self._stored_dtype = (
                pd.CategoricalDtype(np.unique(values).tolist(), ordered=False)
                if dtype is None
                else pd.CategoricalDtype(ordered=False).update_dtype(dtype)  # type: ignore[attr-defined]
            )

    @classmethod
    def __h5_read__(cls, values: ch.H5Dict[Any]) -> Categorical:
        return cls(values["codes"], pd.CategoricalDtype(values.attributes["categories"], ordered=False))

    def __h5_write__(self, values: ch.H5Dict[Any]) -> None:
        values.attributes["categories"] = self.categories
        values["codes"] = self.codes

    # endregion

    # region attributes
    @property
    def _ndarray(self) -> npt.NDArray[Any] | ch.H5Array[Any]:
        return self._stored_array

    @property
    def _dtype(self) -> pd.CategoricalDtype | None:
        return self._stored_dtype

    # endregion

    # region methods
    def write(self, file: str | Path | ch.File | ch.Group | ch.H5Dict[Any], name: str = "") -> None:
        if not isinstance(file, (ch.H5Dict, ch.Group)):
            file = ch.File(file, mode=ch.H5Mode.WRITE_TRUNCATE)

        if not isinstance(file, ch.H5Dict):
            file = ch.H5Dict(file)

        file = ch.write_object(self, file, name)

        self._stored_array = file["codes"]
        self._stored_dtype = pd.CategoricalDtype(file.attributes["categories"], ordered=False)

    # endregion
